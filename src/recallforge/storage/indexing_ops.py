"""Indexing operations service for LanceDB storage backend."""

import fnmatch
import hashlib
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..documents import extract_document_artifacts, is_document_file
from ..video import extract_video_artifacts, is_video_file
from .chunking import chunk_document
from .lancedb_shared import (
    DEFAULT_INDEX_DIR,
    _safe_filter,
    _validate_identifier,
    hash_content,
    hash_file_bytes,
    extract_title,
    trace_log,
)

if TYPE_CHECKING:
    from .lancedb_backend import LanceDBBackend

logger = logging.getLogger("recallforge.storage")

DEFAULT_MAX_FILE_SIZE_MB = 100


class IndexingOps:
    """Service class for indexing operations."""

    def __init__(self, backend: "LanceDBBackend"):
        self._backend = backend

    def _resolve_captioner(self, embed_func, method_name: str):
        """Resolve optional caption/description method from callable or its bound object."""
        if embed_func is None:
            return None
        direct = getattr(embed_func, method_name, None)
        if callable(direct):
            return direct
        owner = getattr(embed_func, "__self__", None)
        candidate = getattr(owner, method_name, None) if owner is not None else None
        if callable(candidate):
            return candidate
        return None

    def _describe_image(self, embed_func, image_path: str, enabled: bool) -> str:
        if not enabled:
            return ""
        describer = self._resolve_captioner(embed_func, "describe_image") or self._resolve_captioner(embed_func, "caption_image")
        if not describer:
            return ""
        try:
            caption = describer(image_path)
            return caption.strip() if isinstance(caption, str) else ""
        except Exception as e:
            logger.warning("index_image: caption generation failed for %s: %s", image_path, e)
            return ""

    def _describe_video(self, embed_image_func, embed_video_func, video_path: str, frame_paths: List[str], enabled: bool) -> str:
        if not enabled:
            return ""

        describer = (
            self._resolve_captioner(embed_video_func, "describe_video")
            or self._resolve_captioner(embed_image_func, "describe_video")
        )
        if describer:
            try:
                caption = describer(video_path, frame_paths=frame_paths)
                if isinstance(caption, str) and caption.strip():
                    return caption.strip()
            except Exception as e:
                logger.warning("index_video: video caption generation failed for %s: %s", video_path, e)

        # Fallback: summarize first keyframes through image captions.
        if frame_paths:
            parts: List[str] = []
            for frame_path in frame_paths[:3]:
                frame_caption = self._describe_image(embed_image_func, frame_path, enabled=True)
                if frame_caption:
                    parts.append(frame_caption)
            return " ".join(parts).strip()

        return ""

    def index_document(
        self,
        path: str,
        text: str,
        collection: str,
        model: str,
        embed_func,
        content_type: str = "text"
    ) -> str:
        """Full document indexing pipeline for text content."""
        if content_type != "text":
            raise ValueError("index_document supports only text content")
        return self.upsert_memory(
            path=path,
            text=text,
            collection=collection,
            embed_func=embed_func,
            model=model,
        )

    def _embed_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        embed_func,
    ) -> List[List[float]]:
        """Embed chunks with batch support and safe fallbacks.

        Supports:
        - embed_func.embed_texts(texts)
        - embed_func(texts)
        - embed_func.embed_text(text) / embed_func(text) per item fallback
        """
        texts = [chunk["text"] for chunk in chunks]

        if not texts:
            return []

        if hasattr(embed_func, "embed_texts"):
            try:
                vectors = embed_func.embed_texts(texts)
                if hasattr(vectors, "tolist"):
                    vectors = vectors.tolist()
                vectors = list(vectors)
                if len(vectors) == len(texts):
                    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
            except Exception as e:
                logger.debug(f"batch embed via embed_texts failed, falling back: {e}")

        try:
            vectors = embed_func(texts)
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            if isinstance(vectors, (list, tuple)) and len(vectors) == len(texts):
                first = vectors[0]
                if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
                    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
        except Exception:
            pass

        single_embed = embed_func.embed_text if hasattr(embed_func, "embed_text") else embed_func
        output: List[List[float]] = []
        for text in texts:
            vector = single_embed(text)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            output.append(list(vector))
        return output

    def upsert_memory(
        self,
        path: str,
        text: str,
        collection: str,
        embed_func,
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        importance: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        _skip_delete: bool = False,

    ) -> str:
        """Create or update a text memory, replacing old vectors for this path.

        Args:
            path: Memory path key within collection
            text: Memory content text
            collection: Collection name
            embed_func: Function/object to embed text into vectors.
                Supports embed_func(text), embed_func.embed_text(text),
                embed_func(texts), or embed_func.embed_texts(texts).
            model: Embedding model name
            importance: Optional importance score (0.0-1.0)
            ttl_seconds: Optional time-to-live in seconds (0 or None = no expiration)
            tags: Optional list of string tags
            _skip_delete: Internal optimization flag for callers that already
                deleted path-scoped vectors in the same namespace.

        Returns:
            Content hash of the stored memory
        """
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path is required")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required")

        trace_log("upsert_memory_start", path=normalized_path, collection=collection,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile,
                  importance=importance, ttl_seconds=ttl_seconds, tags=tags, _skip_delete=_skip_delete)


        content_hash = hash_content(text)
        title = extract_title(text, normalized_path)

        if not _skip_delete:
            # Build namespace filter for deletion
            del_filter = f"{_safe_filter('collection', collection)} AND {_safe_filter('file_path', normalized_path)}"
            if user_id is not None:
                del_filter += f" AND {_safe_filter('user_id', user_id)}"
            if session_id is not None:
                del_filter += f" AND {_safe_filter('session_id', session_id)}"
            if project_id is not None:
                del_filter += f" AND {_safe_filter('project_id', project_id)}"
            if profile is not None:
                del_filter += f" AND {_safe_filter('profile', profile)}"

            # Remove prior vectors for this memory path to prevent duplicate chunks.
            try:
                self._backend._embeddings_table.delete(del_filter)
            except Exception as e:
                logger.warning(f"upsert_memory: failed to delete old vectors for {collection}/{normalized_path}: {e}")

        self._backend.insert_content(content_hash, text, "text")
        self._backend.insert_document(
            collection, normalized_path, title, content_hash, "text",
            user_id=user_id, session_id=session_id, project_id=project_id, profile=profile
        )

        chunks = chunk_document(text)
        vectors = self._embed_chunks_batch(chunks, embed_func)
        for i, chunk in enumerate(chunks):
            self._backend.insert_embedding(
                content_hash=content_hash,
                seq=i,
                pos=chunk["pos"],
                vector=vectors[i],
                model=model,
                collection=collection,
                file_path=normalized_path,
                title=title,
                text_body=chunk["text"],
                content_type="text",
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
                importance=importance,
                ttl_seconds=ttl_seconds,
                tags=tags,

            )

        # Schedule debounced FTS rebuild instead of immediate rebuild
        self._backend._fts.schedule_fts_rebuild()

        trace_log("upsert_memory_done", path=normalized_path, hash=content_hash[:8], chunks=len(chunks))
        return content_hash

    def delete_memory(
        self,
        path: str,
        collection: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deactivate a memory and remove all associated vectors."""
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path is required")

        trace_log("delete_memory_start", path=normalized_path, collection=collection,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        # Build namespace filter
        del_filter = f"{_safe_filter('collection', collection)} AND {_safe_filter('file_path', normalized_path)}"
        if user_id is not None:
            del_filter += f" AND {_safe_filter('user_id', user_id)}"
        if session_id is not None:
            del_filter += f" AND {_safe_filter('session_id', session_id)}"
        if project_id is not None:
            del_filter += f" AND {_safe_filter('project_id', project_id)}"
        if profile is not None:
            del_filter += f" AND {_safe_filter('profile', profile)}"

        removed_vectors = 0
        try:
            removed_vectors = len(
                self._backend._embeddings_table.search()
                .where(del_filter)
                .to_list()
            )
        except Exception as e:
            logger.warning(f"delete_memory: failed to count vectors for {collection}/{normalized_path}: {e}")
            removed_vectors = 0

        try:
            self._backend._embeddings_table.delete(del_filter)
        except Exception as e:
            logger.error(f"delete_memory: failed to delete embeddings for {collection}/{normalized_path}: {e}")

        self._backend.deactivate_document(collection, normalized_path)

        # Schedule debounced FTS rebuild
        self._backend._fts.schedule_fts_rebuild()

        trace_log("delete_memory_done", path=normalized_path, removed_vectors=removed_vectors)
        return {
            "success": True,
            "path": normalized_path,
            "collection": collection,
            "removed_vectors": removed_vectors,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }

    def _video_frames_dir_for_logical_path(self, logical_path: str) -> Path:
        artifact_root = Path(self._backend._store_path or DEFAULT_INDEX_DIR) / "video_frames"
        digest = hashlib.sha1(logical_path.encode("utf-8")).hexdigest()[:16]
        return artifact_root / digest

    def _delete_video_frame_artifacts(self, logical_path: str) -> None:
        output_dir = self._video_frames_dir_for_logical_path(logical_path)
        if not output_dir.exists():
            return

        try:
            trash_bin = shutil.which("trash")
            if trash_bin:
                subprocess.run([trash_bin, str(output_dir)], check=True, capture_output=True, text=True)
            else:
                shutil.rmtree(output_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(
                "delete_path: failed to cleanup video frame artifacts for %s at %s: %s",
                logical_path,
                output_dir,
                e,
            )

    def delete_path(
        self,
        path: str,
        collection: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        include_children: bool = False,
    ) -> Dict[str, Any]:
        """Delete a logical path and optionally all derived child assets."""
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path is required")

        removed_vectors = self._delete_path_entries(
            collection=collection,
            logical_path=normalized_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            include_children=include_children,
        )
        if include_children:
            self._delete_video_frame_artifacts(normalized_path)
        self._backend._fts.schedule_fts_rebuild()
        return {
            "success": True,
            "path": normalized_path,
            "collection": collection,
            "removed_vectors": removed_vectors,
            "include_children": include_children,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }

    def _is_text_file(self, file_path: Path) -> bool:
        """Best-effort text file detection."""
        try:
            with file_path.open("rb") as f:
                sample = f.read(8192)
        except Exception:
            return False

        if b"\x00" in sample:
            return False

        return True

    def _read_text_robust(self, file_path: Path) -> Optional[str]:
        """Read text file using common encodings with replacement fallback."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception:
                return None

        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    def _is_image_file(self, file_path: Path) -> bool:
        """Best-effort image file detection by extension."""
        return file_path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic"
        }

    def _is_video_file(self, file_path: Path) -> bool:
        """Best-effort video file detection by extension."""
        return is_video_file(file_path)

    def _is_document_file(self, file_path: Path) -> bool:
        """Best-effort office-document detection by extension."""
        return is_document_file(file_path)

    def _iter_folder_files(self, root: Path, recursive: bool):
        """Iterate files while pruning common heavyweight directories."""
        if not recursive:
            for child in sorted(root.iterdir()):
                if child.is_file():
                    yield child
            return

        pruned_dirnames = {".git", "node_modules", "__pycache__", ".venv", "venv"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                dirname for dirname in sorted(dirnames)
                if dirname not in pruned_dirnames and not dirname.startswith(".")
            ]
            for filename in sorted(filenames):
                yield Path(dirpath) / filename

    def _namespace_filters(
        self,
        collection: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> List[str]:
        filters = [_safe_filter("collection", collection)]
        if user_id is not None:
            filters.append(_safe_filter("user_id", user_id))
        if session_id is not None:
            filters.append(_safe_filter("session_id", session_id))
        if project_id is not None:
            filters.append(_safe_filter("project_id", project_id))
        if profile is not None:
            filters.append(_safe_filter("profile", profile))
        return filters

    def _delete_path_entries(
        self,
        collection: str,
        logical_path: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        content_type: Optional[str] = None,
        include_children: bool = False,
    ) -> int:
        filters = self._namespace_filters(
            collection=collection,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )
        # Validate and escape the logical_path for LIKE patterns
        validated_path = _validate_identifier(logical_path, "logical_path")
        escaped_path = validated_path.replace("'", "''")
        if include_children:
            filters.append(f"(file_path = '{escaped_path}' OR file_path LIKE '{escaped_path}::%')")
        else:
            filters.append(f"file_path = '{escaped_path}'")
        if content_type is not None:
            filters.append(_safe_filter("content_type", content_type))

        filter_clause = " AND ".join(filters)
        removed_vectors = 0
        try:
            removed_vectors = len(self._backend._embeddings_table.search().where(filter_clause).to_list())
        except Exception as e:
            logger.debug(f"_delete_path_entries: failed to count rows for {logical_path}: {e}")

        try:
            self._backend._embeddings_table.delete(filter_clause)
        except Exception as e:
            logger.debug(f"_delete_path_entries: failed to delete embeddings for {logical_path}: {e}")

        doc_filters = self._namespace_filters(
            collection=collection,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )
        if include_children:
            doc_filters.append(f"(file_path = '{escaped_path}' OR file_path LIKE '{escaped_path}::%')")
        else:
            doc_filters.append(f"file_path = '{escaped_path}'")
        if content_type is not None:
            doc_filters.append(_safe_filter("content_type", content_type))

        try:
            self._backend._documents_table.update(
                where=" AND ".join(doc_filters),
                values={"active": 0, "updated_at": int(time.time() * 1000)},
            )
        except Exception as e:
            logger.debug(f"_delete_path_entries: failed to deactivate documents for {logical_path}: {e}")

        return removed_vectors

    def _matches_globs(self, rel_path: str, include_globs: Optional[List[str]], exclude_globs: Optional[List[str]]) -> bool:
        include = include_globs or ["**/*"]
        exclude = exclude_globs or []
        if include and not any(fnmatch.fnmatch(rel_path, pattern) for pattern in include):
            return False
        if exclude and any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude):
            return False
        return True

    def index_folder(
        self,
        folder_path: str,
        collection: str,
        recursive: bool,
        include_globs: Optional[List[str]],
        exclude_globs: Optional[List[str]],
        embed_func,
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> Dict[str, Any]:
        """Index text files from a folder and return summary counts."""
        root = Path(folder_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Folder not found: {folder_path}")

        trace_log("index_folder_start", folder=str(root), collection=collection, recursive=recursive,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        include = include_globs or ["**/*"]
        exclude = exclude_globs or []

        indexed = 0
        skipped = 0
        errors = 0
        skipped_details: List[Dict[str, str]] = []

        def mark_skipped(item_path: str, reason: str) -> None:
            nonlocal skipped
            skipped += 1
            skipped_details.append({"path": item_path, "reason": reason})

        # Use bulk mode to defer FTS rebuilds until the end
        with self._backend.bulk_mode():
            for file_path in self._iter_folder_files(root, recursive):
                rel = file_path.relative_to(root).as_posix()
                if include and not any(fnmatch.fnmatch(rel, pattern) for pattern in include):
                    mark_skipped(rel, "glob_mismatch")
                    continue
                if exclude and any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
                    mark_skipped(rel, "excluded")
                    continue

                # Check file size before processing
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > max_file_size_mb * 1024 * 1024:
                        logger.warning("Skipping %s: file size %dMB exceeds limit %dMB",
                                       file_path, file_size // (1024 * 1024), max_file_size_mb)
                        mark_skipped(rel, "file_too_large")
                        continue
                except OSError as e:
                    logger.warning("Could not get size for %s: %s", file_path, e)
                    mark_skipped(rel, "unreadable")
                    continue

                if not self._is_text_file(file_path):
                    mark_skipped(rel, "not_text_file")
                    continue

                text = self._read_text_robust(file_path)
                if text is None or not text.strip():
                    mark_skipped(rel, "empty_content")
                    continue

                try:
                    self.upsert_memory(
                        path=rel,
                        text=text,
                        collection=collection,
                        embed_func=embed_func,
                        model=model,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                    )
                    indexed += 1
                except Exception as e:
                    if "already indexed" in str(e).lower():
                        mark_skipped(rel, "dedup")
                        continue
                    logger.error(f"index_folder: failed to index {rel}: {e}")
                    errors += 1
        # FTS rebuild happens once at context exit

        trace_log("index_folder_done", folder=str(root), indexed=indexed, skipped=skipped, errors=errors)
        return {
            "success": True,
            "folder_path": str(root),
            "collection": collection,
            "indexed": indexed,
            "skipped": skipped,
            "errors": errors,
            "total_seen": indexed + skipped + errors,
            "skipped_details": skipped_details,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }

    def ingest(
        self,
        collection: str,
        text: Optional[str],
        path: Optional[str],
        file_path: Optional[str],
        folder_path: Optional[str],
        recursive: bool,
        content_types: List[str],
        include_globs: Optional[List[str]],
        exclude_globs: Optional[List[str]],
        embed_text_func,
        embed_image_func,
        embed_video_func,
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
        enable_captioning: bool = True,
    ) -> Dict[str, Any]:
        """Unified multimodal ingest for text, file, or folder inputs."""
        content_types = content_types or ["text", "image", "video", "document"]
        allowed = set(content_types)
        if not allowed.issubset({"text", "image", "video", "document"}):
            raise ValueError("content_types must be subset of ['text', 'image', 'video', 'document']")

        trace_log("ingest_start", collection=collection, text=text is not None, file_path=file_path, folder_path=folder_path,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        summary = {
            "success": True,
            "collection": collection,
            "indexed_text": 0,
            "indexed_images": 0,
            "indexed_videos": 0,
            "indexed_documents": 0,
            "indexed_document_sections": 0,
            "indexed_video_embeddings": 0,
            "indexed_video_frames": 0,
            "indexed_video_transcripts": 0,
            "skipped": 0,
            "errors": 0,
            "items": [],
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }

        def mark(
            item_path: str,
            item_type: str,
            status: str,
            error: Optional[str] = None,
            reason: Optional[str] = None,
        ) -> None:
            item = {"path": item_path, "type": item_type, "status": status}
            if error:
                item["error"] = error
            if reason:
                item["reason"] = reason
            summary["items"].append(item)

        def ingest_single(candidate: Path, rel_hint: Optional[str] = None) -> None:
            candidate = candidate.expanduser().resolve()
            if not candidate.exists() or not candidate.is_file():
                summary["errors"] += 1
                mark(str(candidate), "unknown", "error", "file not found")
                return

            item_path = rel_hint or str(candidate)
            is_image = self._is_image_file(candidate)
            is_video = self._is_video_file(candidate)
            is_document = self._is_document_file(candidate)

            try:
                if is_image:
                    if "image" not in allowed:
                        summary["skipped"] += 1
                        mark(item_path, "image", "skipped", reason="not_in_content_types")
                        return
                    self.index_image(
                        path=str(candidate),
                        collection=collection,
                        embed_func=embed_image_func,
                        model=model,
                        stored_path=item_path,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                        enable_captioning=enable_captioning,
                    )
                    summary["indexed_images"] += 1
                    mark(item_path, "image", "indexed")
                    return

                if is_video:
                    if "video" not in allowed:
                        summary["skipped"] += 1
                        mark(item_path, "video", "skipped", reason="not_in_content_types")
                        return
                    video_summary = self.index_video(
                        path=str(candidate),
                        collection=collection,
                        embed_text_func=embed_text_func,
                        embed_image_func=embed_image_func,
                        embed_video_func=embed_video_func,
                        model=model,
                        stored_path=item_path,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                        enable_captioning=enable_captioning,
                    )
                    summary["indexed_videos"] += 1
                    summary["indexed_images"] += video_summary["indexed_frames"]
                    summary["indexed_text"] += video_summary["indexed_transcripts"]
                    summary["indexed_video_embeddings"] += video_summary.get("indexed_video_embeddings", 0)
                    summary["indexed_video_frames"] += video_summary["indexed_frames"]
                    summary["indexed_video_transcripts"] += video_summary["indexed_transcripts"]
                    mark(item_path, "video", "indexed")
                    return

                if is_document:
                    if "document" not in allowed:
                        summary["skipped"] += 1
                        mark(item_path, "document", "skipped", reason="not_in_content_types")
                        return
                    document_summary = self.index_document_file(
                        path=str(candidate),
                        collection=collection,
                        embed_func=embed_text_func,
                        embed_image_func=embed_image_func,
                        model=model,
                        stored_path=item_path,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                    )
                    total_sections = document_summary.get("indexed_sections", 0) + document_summary.get("indexed_images", 0)
                    if total_sections == 0:
                        summary["skipped"] += 1
                        mark(item_path, "document", "skipped", reason="empty_content")
                    else:
                        summary["indexed_documents"] += 1
                        summary["indexed_document_sections"] += total_sections
                        summary["indexed_text"] += document_summary.get("indexed_sections", 0)
                        summary["indexed_images"] += document_summary.get("indexed_images", 0)
                        mark(item_path, "document", "indexed")
                    return

                if "text" not in allowed:
                    summary["skipped"] += 1
                    mark(item_path, "text", "skipped", reason="not_in_content_types")
                    return
                if not self._is_text_file(candidate):
                    summary["skipped"] += 1
                    mark(item_path, "text", "skipped", reason="not_text_file")
                    return
                body = self._read_text_robust(candidate)
                if body is None or not body.strip():
                    summary["skipped"] += 1
                    mark(item_path, "text", "skipped", reason="empty_content")
                    return
                self.upsert_memory(
                    path=item_path,
                    text=body,
                    collection=collection,
                    embed_func=embed_text_func,
                    model=model,
                    user_id=user_id,
                    session_id=session_id,
                    project_id=project_id,
                    profile=profile,
                )
                summary["indexed_text"] += 1
                mark(item_path, "text", "indexed")
            except Exception as e:
                logger.error(f"ingest: failed to index {item_path}: {e}")
                summary["errors"] += 1
                if is_image:
                    item_type = "image"
                elif is_video:
                    item_type = "video"
                elif is_document:
                    item_type = "document"
                else:
                    item_type = "text"
                mark(item_path, item_type, "error", str(e))

        # Use bulk mode to defer FTS rebuilds until the end
        with self._backend.bulk_mode():
            if text is not None:
                if "text" not in allowed:
                    summary["skipped"] += 1
                    mark(path or "inline/skipped", "text", "skipped", reason="not_in_content_types")
                else:
                    if path:
                        text_path = path.strip() or "inline"
                    else:
                        import hashlib
                        text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                        text_path = f"inline/{text_hash}"
                    self.upsert_memory(
                        path=text_path,
                        text=text,
                        collection=collection,
                        embed_func=embed_text_func,
                        model=model,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                    )
                    summary["indexed_text"] += 1
                    mark(text_path, "text", "indexed")

            if file_path:
                ingest_single(Path(file_path))

            if folder_path:
                root = Path(folder_path).expanduser().resolve()
                if not root.exists() or not root.is_dir():
                    raise ValueError(f"Folder not found: {folder_path}")
                for candidate in self._iter_folder_files(root, recursive):
                    rel = candidate.relative_to(root).as_posix()
                    if include_globs and not any(fnmatch.fnmatch(rel, pattern) for pattern in include_globs):
                        summary["skipped"] += 1
                        mark(rel, "unknown", "skipped", reason="glob_mismatch")
                        continue
                    if exclude_globs and any(fnmatch.fnmatch(rel, pattern) for pattern in exclude_globs):
                        summary["skipped"] += 1
                        mark(rel, "unknown", "skipped", reason="excluded")
                        continue

                    # Check file size before processing
                    try:
                        file_size = os.path.getsize(candidate)
                        if file_size > max_file_size_mb * 1024 * 1024:
                            logger.warning("Skipping %s: file size %dMB exceeds limit %dMB",
                                           candidate, file_size // (1024 * 1024), max_file_size_mb)
                            summary["skipped"] += 1
                            mark(rel, "unknown", "skipped", reason="file_too_large")
                            continue
                    except OSError as e:
                        logger.warning("Could not get size for %s: %s", candidate, e)
                        summary["skipped"] += 1
                        mark(rel, "unknown", "skipped", reason="unreadable")
                        continue

                    ingest_single(candidate, rel)

            if text is None and not file_path and not folder_path:
                raise ValueError("Provide one of: text, file_path, or folder_path")
        # FTS rebuild happens once at context exit

        summary["total_seen"] = len(summary["items"])
        trace_log("ingest_done", collection=collection, indexed_text=summary["indexed_text"], indexed_images=summary["indexed_images"])
        return summary

    def index_image(
        self,
        path: str,
        collection: str,
        embed_func,
        model: str = "Qwen3-VL-Embedding-2B",
        stored_path: Optional[str] = None,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        enable_captioning: bool = True,
    ) -> str:
        """
        Index an image file.

        Args:
            path: Absolute path to image
            collection: Collection name
            embed_func: Function(path) -> List[float]
            model: Embedding model name

        Returns:
            Content hash (hash of file bytes + mtime)
        """
        actual_path = str(Path(path).expanduser().resolve())
        logical_path = stored_path or actual_path

        # Use file bytes + mtime hash for correctness
        # This ensures re-indexing when file content changes
        try:
            content_hash = hash_file_bytes(actual_path)
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"index_image: failed to hash {path}: {e}")
            raise

        trace_log("index_image_start", path=path, hash=content_hash[:8])

        resolved_title = title or os.path.splitext(os.path.basename(logical_path))[0]
        try:
            modified_at = int(os.path.getmtime(actual_path) * 1000)
            created_at = int(os.path.getctime(actual_path) * 1000)
        except OSError as e:
            logger.warning(f"index_image: failed to get file times for {path}: {e}")
            modified_at = int(time.time() * 1000)
            created_at = modified_at

        # Remove previous image vectors for this logical document path.
        # Deleting only by hash_seq misses changed-content reindex cases.
        self._delete_path_entries(
            collection=collection,
            logical_path=logical_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            content_type="image",
        )

        self._backend.insert_content(content_hash, actual_path, content_type="image")
        self._backend.insert_document(
            collection=collection,
            file_path=logical_path,
            title=resolved_title,
            content_hash=content_hash,
            content_type="image",
            created_at=created_at,
            modified_at=modified_at,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )

        vector = embed_func(actual_path)
        image_caption = self._describe_image(embed_func, actual_path, enabled=enable_captioning)
        self._backend.insert_embedding(
            content_hash=content_hash,
            seq=0,
            pos=0,
            vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
            model=model,
            collection=collection,
            file_path=logical_path,
            title=resolved_title,
            text_body=image_caption,
            content_type="image",
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )

        # Schedule debounced FTS rebuild
        self._backend._fts.schedule_fts_rebuild()

        trace_log("index_image_done", path=logical_path, hash=content_hash[:8])
        return content_hash

    def index_video(
        self,
        path: str,
        collection: str,
        embed_text_func,
        embed_image_func,
        embed_video_func=None,
        model: str = "Qwen3-VL-Embedding-2B",
        stored_path: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        frame_interval_seconds: float = 5.0,
        max_frames: int = 8,
        enable_captioning: bool = True,
    ) -> Dict[str, Any]:
        """Index a video into a top-level video embedding plus derived assets."""
        actual_path = str(Path(path).expanduser().resolve())
        logical_path = stored_path or actual_path
        resolved_title = os.path.splitext(os.path.basename(logical_path))[0]
        video_embed = embed_video_func or embed_image_func

        artifact_root = Path(self._backend._store_path or DEFAULT_INDEX_DIR) / "video_frames"
        digest = hashlib.sha1(logical_path.encode("utf-8")).hexdigest()[:16]
        output_dir = artifact_root / digest

        self._delete_path_entries(
            collection=collection,
            logical_path=logical_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            include_children=True,
        )

        artifacts = extract_video_artifacts(
            video_path=actual_path,
            output_dir=output_dir,
            logical_path=logical_path,
            frame_interval_seconds=frame_interval_seconds,
            max_frames=max_frames,
        )

        try:
            content_hash = hash_file_bytes(actual_path)
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"index_video: failed to hash {path}: {e}")
            raise

        transcript_summary = "\n".join(
            segment.text.strip()
            for segment in artifacts.transcripts
            if isinstance(segment.text, str) and segment.text.strip()
        ).strip()
        frame_paths = [frame.image_path for frame in artifacts.frames]
        video_caption = self._describe_video(
            embed_image_func=embed_image_func,
            embed_video_func=embed_video_func,
            video_path=actual_path,
            frame_paths=frame_paths,
            enabled=enable_captioning,
        )
        parts = [part for part in (video_caption, transcript_summary) if part]
        video_body = "\n\n".join(parts)[:4000]

        try:
            modified_at = int(os.path.getmtime(actual_path) * 1000)
            created_at = int(os.path.getctime(actual_path) * 1000)
        except OSError as e:
            logger.warning(f"index_video: failed to get file times for {path}: {e}")
            modified_at = int(time.time() * 1000)
            created_at = modified_at

        indexed_video_embeddings = 0
        try:
            vector = video_embed(actual_path)
            self._backend.insert_content(content_hash, actual_path, content_type="video")
            self._backend.insert_document(
                collection=collection,
                file_path=logical_path,
                title=resolved_title,
                content_hash=content_hash,
                content_type="video",
                created_at=created_at,
                modified_at=modified_at,
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
            )
            self._backend.insert_embedding(
                content_hash=content_hash,
                seq=0,
                pos=0,
                vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                model=model,
                collection=collection,
                file_path=logical_path,
                title=resolved_title,
                text_body=video_body,
                content_type="video",
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
            )
            indexed_video_embeddings = 1
        except Exception as e:
            logger.warning(
                "index_video: raw video embedding failed for %s; continuing with derived assets: %s",
                actual_path,
                e,
            )

        indexed_frames = 0
        indexed_transcripts = 0

        for frame in artifacts.frames:
            self.index_image(
                path=frame.image_path,
                collection=collection,
                embed_func=embed_image_func,
                model=model,
                stored_path=frame.logical_path,
                title=frame.title,
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
                enable_captioning=enable_captioning,
            )
            indexed_frames += 1

        for segment in artifacts.transcripts:
            self.upsert_memory(
                path=segment.logical_path,
                text=segment.text,
                collection=collection,
                embed_func=embed_text_func,
                model=model,
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
            )
            indexed_transcripts += 1

        return {
            "success": True,
            "path": logical_path,
            "collection": collection,
            "hash": content_hash,
            "indexed_video_embeddings": indexed_video_embeddings,
            "indexed_frames": indexed_frames,
            "indexed_transcripts": indexed_transcripts,
            "duration_seconds": artifacts.duration_seconds,
            "transcript_path": artifacts.transcript_path,
            "ffmpeg_available": artifacts.ffmpeg_available,
        }

    def index_document_file(
        self,
        path: str,
        collection: str,
        embed_func,
        embed_image_func=None,
        model: str = "Qwen3-VL-Embedding-2B",
        stored_path: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract and index a document file into structured text assets."""
        actual_path = str(Path(path).expanduser().resolve())
        logical_path = stored_path or actual_path

        self._delete_path_entries(
            collection=collection,
            logical_path=logical_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            include_children=True,
        )

        artifacts = extract_document_artifacts(actual_path, logical_path)
        indexed_sections = 0
        indexed_images = 0

        # Track temp dirs from PDF vision fallback for cleanup
        _temp_dirs_to_clean: set = set()

        for section in artifacts.sections:
            if section.content_type == "image" and section.image_path:
                # Track the parent temp dir for cleanup after embedding
                import os
                _temp_dirs_to_clean.add(os.path.dirname(section.image_path))

                # Use image embedding for image sections
                image_embed = embed_image_func or embed_func
                if embed_image_func is None:
                    logger.warning(
                        "No image embedder provided for PDF page image %s; "
                        "falling back to text embedder which may produce poor results. "
                        "Pass embed_image_func for proper vision embedding.",
                        section.image_path,
                    )
                self.index_image(
                    path=section.image_path,
                    collection=collection,
                    embed_func=image_embed,
                    model=model,
                    stored_path=section.logical_path,
                    title=section.title,
                    user_id=user_id,
                    session_id=session_id,
                    project_id=project_id,
                    profile=profile,
                )
                # Override content entry to reference source PDF, not temp image.
                # Temp images are cleaned up after this loop; the embedding vector
                # persists and is the primary retrieval artifact.
                from recallforge.storage.lancedb_shared import hash_content
                content_hash = hash_content(f"pdf_page_image:{actual_path}:page:{section.index}")
                self._backend.insert_content(content_hash, actual_path, content_type="pdf_page_image")
                indexed_images += 1
            else:
                # Use text embedding for text sections
                self.upsert_memory(
                    path=section.logical_path,
                    text=section.text,
                    collection=collection,
                    embed_func=embed_func,
                    model=model,
                    user_id=user_id,
                    session_id=session_id,
                    project_id=project_id,
                    profile=profile,
                    _skip_delete=True,
                )
                indexed_sections += 1

        # Clean up temp dirs from PDF page-to-image rendering
        import shutil
        for temp_dir in _temp_dirs_to_clean:
            if temp_dir and "recallforge_pdf_" in temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Ensure FTS rebuild is scheduled even when no sections were indexed,
        # since _delete_path_entries above may have removed stale entries.
        if indexed_sections == 0 and indexed_images == 0:
            self._backend._fts.schedule_fts_rebuild()

        return {
            "success": True,
            "path": logical_path,
            "collection": collection,
            "document_type": artifacts.document_type,
            "extractor": artifacts.extractor,
            "indexed_sections": indexed_sections,
            "indexed_images": indexed_images,
        }
