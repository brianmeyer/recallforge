import time
from pathlib import Path

from recallforge.watch_folder import WatchConfig, WatchFolderDaemon


class FakeStorage:
    def __init__(self):
        self.upserts = []
        self.image_indexes = []
        self.video_indexes = []
        self.document_indexes = []
        self.deletes = []

    def upsert_memory(self, **kwargs):
        self.upserts.append(kwargs)

    def index_image(self, **kwargs):
        self.image_indexes.append(kwargs)

    def index_video(self, **kwargs):
        self.video_indexes.append(kwargs)

    def index_document_file(self, **kwargs):
        self.document_indexes.append(kwargs)

    def delete_memory(self, path, collection):
        self.deletes.append((path, collection))

    def delete_path(self, path, collection, include_children=False):
        self.deletes.append((path, collection, include_children))


class FakeBackend:
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

    assert ("old.md", "default", False) in storage.deletes


def test_watch_folder_non_recursive_skips_nested(tmp_path):
    storage = FakeStorage()
    backend = FakeBackend()
    daemon = WatchFolderDaemon(storage, backend, store_path=str(tmp_path / ".store"))

    watched = tmp_path / "watched"
    nested = watched / "nested"
    nested.mkdir(parents=True)

    config = WatchConfig(
        folder_path=str(watched),
        recursive=False,
        include_globs=["**/*.md", "*.md"],
        debounce_seconds=0.1,
        scan_interval=0.1,
    )

    watch_id = daemon.start_watch(config)

    (nested / "note.md").write_text("nested", encoding="utf-8")
    time.sleep(0.35)

    daemon.stop_watch(watch_id)

    assert storage.upserts == []


def test_watch_folder_image_uses_logical_path(tmp_path):
    storage = FakeStorage()
    backend = FakeBackend()
    daemon = WatchFolderDaemon(storage, backend, store_path=str(tmp_path / ".store"))

    watched = tmp_path / "watched"
    watched.mkdir()

    config = WatchConfig(
        folder_path=str(watched),
        include_globs=["**/*.png", "*.png"],
        debounce_seconds=0.1,
        scan_interval=0.1,
    )

    watch_id = daemon.start_watch(config)

    image_path = watched / "diagram.png"
    image_path.write_bytes(b"fake image bytes")
    time.sleep(1.0)
    image_path.unlink()
    time.sleep(1.0)

    daemon.stop_watch(watch_id)

    assert storage.image_indexes
    assert storage.image_indexes[0]["stored_path"] == "diagram.png"
    assert ("diagram.png", "default", False) in storage.deletes


def test_watch_folder_document_uses_logical_path_and_child_cleanup(tmp_path):
    storage = FakeStorage()
    backend = FakeBackend()
    daemon = WatchFolderDaemon(storage, backend, store_path=str(tmp_path / ".store"))

    watched = tmp_path / "watched"
    watched.mkdir()

    config = WatchConfig(
        folder_path=str(watched),
        include_globs=["**/*.docx", "*.docx"],
        debounce_seconds=0.1,
        scan_interval=0.1,
    )

    watch_id = daemon.start_watch(config)

    document_path = watched / "notes.docx"
    document_path.write_bytes(b"placeholder")
    time.sleep(1.0)
    document_path.unlink()
    time.sleep(1.0)

    daemon.stop_watch(watch_id)

    assert storage.document_indexes
    assert storage.document_indexes[0]["stored_path"] == "notes.docx"
    assert ("notes.docx", "default", True) in storage.deletes
