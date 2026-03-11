"""Folder watch sidecar/daemon for RecallForge.

Lightweight polling implementation (no heavy framework):
- include/exclude glob filtering
- debounced batch processing
- start/stop/list/status controls
- integrates with existing storage ingest/index paths
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

logger = logging.getLogger("recallforge.watch_folder")


@dataclass
class WatchConfig:
    folder_path: str
    collection: str = "default"
    recursive: bool = True
    include_globs: List[str] = field(default_factory=lambda: ["**/*"])
    exclude_globs: List[str] = field(default_factory=list)
    debounce_seconds: float = 2.0
    batch_size: int = 32
    scan_interval: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatchConfig":
        return cls(**data)


class WatchFolderDaemon:
    STATE_FILE_NAME = ".recallforge_watch_state.json"

    def __init__(self, storage: Any, backend: Any, store_path: Optional[str] = None):
        self.storage = storage
        self.backend = backend
        self.store_path = store_path or os.path.expanduser("~/.recallforge")

        self._lock = threading.Lock()
        self.watches: Dict[str, Dict[str, Any]] = {}
        self.queues: Dict[str, Queue] = {}
        self.running: Dict[str, threading.Event] = {}
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.scan_threads: Dict[str, threading.Thread] = {}

        self._is_text_file = storage._is_text_file
        self._is_image_file = storage._is_image_file

    def _get_state_path(self) -> Path:
        return Path(self.store_path) / self.STATE_FILE_NAME

    def _save_state(self) -> None:
        try:
            p = self._get_state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"watches": self.watches}, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save watch state: %s", e)

    def _candidate_files(self, root: Path, recursive: bool) -> List[Path]:
        if recursive:
            files = [p for p in root.rglob("*") if p.is_file()]
        else:
            files = [p for p in root.glob("*") if p.is_file()]
        return files

    def _should_process(self, path: Path, config: WatchConfig) -> bool:
        root = Path(config.folder_path).expanduser().resolve()
        try:
            rel = path.resolve().relative_to(root).as_posix()
        except Exception:
            return False

        rel_path = Path(rel)

        def _match(pattern: str) -> bool:
            compact = pattern.replace("**/", "")
            return (
                fnmatch.fnmatch(rel, pattern)
                or fnmatch.fnmatch(rel, compact)
                or rel_path.match(pattern)
                or rel_path.match(compact)
            )

        if config.include_globs:
            if not any(_match(pat) for pat in config.include_globs):
                return False
        if config.exclude_globs:
            if any(_match(pat) for pat in config.exclude_globs):
                return False

        return self._is_text_file(path) or self._is_image_file(path)

    def _build_snapshot(self, config: WatchConfig) -> Dict[str, float]:
        root = Path(config.folder_path).expanduser().resolve()
        snap: Dict[str, float] = {}
        for p in self._candidate_files(root, config.recursive):
            if not self._should_process(p, config):
                continue
            try:
                rel = p.resolve().relative_to(root).as_posix()
                snap[rel] = p.stat().st_mtime
            except Exception:
                continue
        return snap

    def _scanner_loop(self, watch_id: str) -> None:
        config = WatchConfig.from_dict(self.watches[watch_id]["config"])
        root = Path(config.folder_path).expanduser().resolve()
        queue = self.queues[watch_id]
        evt = self.running[watch_id]

        prev = self._build_snapshot(config)
        while evt.is_set():
            current = self._build_snapshot(config)

            # created/modified
            for rel, mtime in current.items():
                if rel not in prev:
                    queue.put({"path": str(root / rel), "type": "created", "timestamp": time.time()})
                elif mtime > prev[rel]:
                    queue.put({"path": str(root / rel), "type": "modified", "timestamp": time.time()})

            # deleted
            for rel in prev.keys() - current.keys():
                queue.put({"path": str(root / rel), "type": "deleted", "timestamp": time.time()})

            prev = current
            time.sleep(max(0.1, config.scan_interval))

    def _worker_loop(self, watch_id: str) -> None:
        queue = self.queues[watch_id]
        evt = self.running[watch_id]
        config = WatchConfig.from_dict(self.watches[watch_id]["config"])

        pending: Dict[str, Dict[str, Any]] = {}
        last_flush = time.time()

        while evt.is_set():
            try:
                item = queue.get(timeout=0.2)
                pending[item["path"]] = item
            except Empty:
                pass

            now = time.time()
            if pending and (now - last_flush >= max(0.05, config.debounce_seconds)):
                self._process_batch(watch_id, pending, config)
                pending.clear()
                last_flush = now

        if pending:
            self._process_batch(watch_id, pending, config)

    def _process_batch(self, watch_id: str, pending: Dict[str, Dict[str, Any]], config: WatchConfig) -> None:
        items = sorted(pending.values(), key=lambda x: x["timestamp"])[: max(1, config.batch_size)]
        for item in items:
            self._process_file_change(item, config)

        self.watches[watch_id]["last_processed"] = time.time()
        self.watches[watch_id]["processed_count"] = self.watches[watch_id].get("processed_count", 0) + len(items)
        self._save_state()

    def _process_file_change(self, item: Dict[str, Any], config: WatchConfig) -> None:
        root = Path(config.folder_path).expanduser().resolve()
        path = Path(item["path"])
        event_type = item["type"]

        try:
            rel_path = path.resolve().relative_to(root).as_posix()
        except Exception:
            return

        if event_type == "deleted":
            self.storage.delete_memory(rel_path, config.collection)
            return

        if not path.exists():
            return

        self.backend._load_embedder()

        if self._is_image_file(path):
            self.storage.index_image(
                path=str(path),
                collection=config.collection,
                embed_func=self.backend.embed_image,
                model="Qwen3-VL-Embedding-2B",
            )
            return

        text = self.storage._read_text_robust(path)
        if text:
            self.storage.upsert_memory(
                path=rel_path,
                text=text,
                collection=config.collection,
                embed_func=self.backend.embed_text,
                model="Qwen3-VL-Embedding-2B",
            )

    def start_watch(self, config: WatchConfig) -> str:
        root = Path(config.folder_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Folder not found: {root}")

        with self._lock:
            for watch_id, info in self.watches.items():
                if Path(info["config"]["folder_path"]).expanduser().resolve() == root and info.get("status") == "running":
                    return watch_id

            watch_id = f"watch_{root.name}_{int(time.time())}"
            self.watches[watch_id] = {
                "config": config.to_dict(),
                "started_at": time.time(),
                "status": "running",
            }
            self.queues[watch_id] = Queue()
            self.running[watch_id] = threading.Event()
            self.running[watch_id].set()

            s = threading.Thread(target=self._scanner_loop, args=(watch_id,), daemon=True)
            w = threading.Thread(target=self._worker_loop, args=(watch_id,), daemon=True)
            s.start()
            w.start()

            self.scan_threads[watch_id] = s
            self.worker_threads[watch_id] = w
            self._save_state()
            return watch_id

    def stop_watch(self, watch_id: str) -> bool:
        if watch_id not in self.watches:
            return False

        with self._lock:
            if watch_id in self.running:
                self.running[watch_id].clear()

            if watch_id in self.scan_threads:
                self.scan_threads[watch_id].join(timeout=2)
                del self.scan_threads[watch_id]

            if watch_id in self.worker_threads:
                self.worker_threads[watch_id].join(timeout=2)
                del self.worker_threads[watch_id]

            self.queues.pop(watch_id, None)
            self.running.pop(watch_id, None)

            self.watches[watch_id]["status"] = "stopped"
            self.watches[watch_id]["stopped_at"] = time.time()
            self._save_state()
            return True

    def stop_all(self) -> int:
        count = 0
        for watch_id in list(self.watches.keys()):
            if self.stop_watch(watch_id):
                count += 1
        return count

    def list_watches(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for watch_id, info in self.watches.items():
            out[watch_id] = {
                "status": info.get("status", "unknown"),
                "folder_path": info["config"]["folder_path"],
                "collection": info["config"].get("collection", "default"),
                "started_at": info.get("started_at"),
                "stopped_at": info.get("stopped_at"),
                "processed_count": info.get("processed_count", 0),
                "last_processed": info.get("last_processed"),
            }
        return out

    def get_watch_status(self, watch_id: str) -> Optional[Dict[str, Any]]:
        return self.list_watches().get(watch_id)


_daemon: Optional[WatchFolderDaemon] = None


def get_daemon(storage: Any, backend: Any, store_path: Optional[str] = None) -> WatchFolderDaemon:
    global _daemon
    if _daemon is None:
        _daemon = WatchFolderDaemon(storage, backend, store_path)
    return _daemon


def reset_daemon() -> None:
    global _daemon
    if _daemon:
        _daemon.stop_all()
    _daemon = None
