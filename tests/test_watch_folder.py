import time
from pathlib import Path

from recallforge.watch_folder import WatchConfig, WatchFolderDaemon


class FakeStorage:
    def __init__(self):
        self.upserts = []
        self.deletes = []

    def _is_text_file(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".txt"}

    def _is_image_file(self, path: Path) -> bool:
        return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}

    def _read_text_robust(self, path: Path):
        return path.read_text(encoding="utf-8")

    def upsert_memory(self, **kwargs):
        self.upserts.append(kwargs)

    def index_image(self, **kwargs):
        # Not used in these tests
        pass

    def delete_memory(self, path, collection):
        self.deletes.append((path, collection))


class FakeBackend:
    def _load_embedder(self):
        return None

    def embed_text(self, text):
        return [0.0] * 8

    def embed_image(self, image):
        return [0.0] * 8


def test_watch_folder_create_and_modify(tmp_path):
    storage = FakeStorage()
    backend = FakeBackend()
    daemon = WatchFolderDaemon(storage, backend, store_path=str(tmp_path / ".store"))

    watched = tmp_path / "watched"
    watched.mkdir()

    config = WatchConfig(
        folder_path=str(watched),
        include_globs=["**/*.md"],
        debounce_seconds=0.1,
        scan_interval=0.1,
    )

    watch_id = daemon.start_watch(config)

    f = watched / "note.md"
    f.write_text("v1", encoding="utf-8")
    time.sleep(0.35)

    f.write_text("v2", encoding="utf-8")
    time.sleep(0.35)

    daemon.stop_watch(watch_id)

    assert len(storage.upserts) >= 2
    assert storage.upserts[0]["path"] == "note.md"
    assert storage.upserts[-1]["text"] == "v2"


def test_watch_folder_delete(tmp_path):
    storage = FakeStorage()
    backend = FakeBackend()
    daemon = WatchFolderDaemon(storage, backend, store_path=str(tmp_path / ".store"))

    watched = tmp_path / "watched"
    watched.mkdir()

    f = watched / "old.md"
    f.write_text("old", encoding="utf-8")

    config = WatchConfig(
        folder_path=str(watched),
        include_globs=["**/*.md"],
        debounce_seconds=0.1,
        scan_interval=0.1,
    )

    watch_id = daemon.start_watch(config)
    time.sleep(0.2)

    f.unlink()
    time.sleep(0.35)

    daemon.stop_watch(watch_id)

    assert ("old.md", "default") in storage.deletes
