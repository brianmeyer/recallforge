"""
test_storage.py - Unit tests for StorageBackend ABC and LanceDBBackend.

Tests document indexing, BM25 + vector search, content-type filtering,
and cross-modal search using a real (but temp) LanceDB instance.
Uses a deterministic mock embed_func — NO real model inference.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from typing import List, Optional

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recallforge.storage.base import StorageBackend, SearchResult, Document
from recallforge.storage.lancedb_backend import LanceDBBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mock_embed(text: str) -> List[float]:
    """Deterministic pseudo-random 2048-dim unit vector from text SHA-256."""
    h = hashlib.sha256(text.encode()).hexdigest()
    values = []
    for i in range(2048):
        chunk = h[(i * 2) % 64: (i * 2) % 64 + 4]
        val = int(chunk, 16) / 65536.0 - 0.5
        values.append(val)
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


def mock_embed_array(text: str) -> np.ndarray:
    return np.array(mock_embed(text), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests for StorageBackend ABC
# ---------------------------------------------------------------------------

class TestStorageBackendABC(unittest.TestCase):
    """StorageBackend cannot be instantiated directly."""

    def test_cannot_instantiate_abstract(self):
        with self.assertRaises(TypeError):
            StorageBackend()


# ---------------------------------------------------------------------------
# LanceDBBackend integration tests (no real models)
# ---------------------------------------------------------------------------

class TestLanceDBBackendInit(unittest.TestCase):
    """Tests for LanceDBBackend initialization."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_backend_is_storage_backend(self):
        self.assertIsInstance(self.backend, StorageBackend)

    def test_initial_empty(self):
        self.assertEqual(self.backend.count_documents(), 0)
        self.assertEqual(self.backend.count_embeddings(), 0)

    def test_has_vectors_initially_false(self):
        self.assertFalse(self.backend.has_vectors())


class TestDocumentOperations(unittest.TestCase):
    """Tests for insert_document and find_document."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_insert_and_find_document(self):
        doc_id = self.backend.insert_document(
            collection="test",
            file_path="notes/hello.md",
            title="Hello World",
            content_hash="abc123",
            content_type="text",
        )
        self.assertIsNotNone(doc_id)

        doc = self.backend.find_document("test", "notes/hello.md")
        self.assertIsNotNone(doc)
        self.assertEqual(doc.file_path, "notes/hello.md")
        self.assertEqual(doc.title, "Hello World")
        self.assertEqual(doc.collection, "test")

    def test_find_nonexistent_document_returns_none(self):
        result = self.backend.find_document("test", "no/such/file.md")
        self.assertIsNone(result)

    def test_deactivate_document(self):
        self.backend.insert_document(
            collection="test",
            file_path="notes/active.md",
            title="Active Doc",
            content_hash="xyz789",
        )
        # deactivate should not raise
        self.backend.deactivate_document("test", "notes/active.md")


class TestContentOperations(unittest.TestCase):
    """Tests for insert_content and get_content."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_insert_and_retrieve_content(self):
        body = "This is the document body."
        self.backend.insert_content("hashA", body)
        retrieved = self.backend.get_content("hashA")
        self.assertEqual(retrieved, body)

    def test_get_nonexistent_content_returns_none(self):
        result = self.backend.get_content("no-such-hash")
        self.assertIsNone(result)


class TestCacheOperations(unittest.TestCase):
    """Tests for get_cached and set_cached."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_miss_returns_none(self):
        self.assertIsNone(self.backend.get_cached("nonexistent-key"))

    def test_cache_set_and_get(self):
        self.backend.set_cached("mykey", '{"foo": "bar"}')
        val = self.backend.get_cached("mykey")
        self.assertEqual(val, '{"foo": "bar"}')


class TestIndexAndSearch(unittest.TestCase):
    """Tests for index_document and search_fts / search_vec."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

        # Index three documents
        docs = [
            ("ai-overview.md", "Artificial intelligence AI machines neural networks deep learning."),
            ("memory-systems.md", "Memory systems AI agents episodic semantic graph databases knowledge."),
            ("graph-databases.md", "Graph databases nodes relationships knowledge graphs entities Neo4j."),
        ]
        for path, text in docs:
            self.backend.index_document(
                path=path,
                text=text,
                collection="test",
                model="mock-embedder",
                embed_func=mock_embed,
            )

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_count_documents_after_index(self):
        self.assertEqual(self.backend.count_documents(), 3)

    def test_has_vectors_after_index(self):
        self.assertTrue(self.backend.has_vectors())

    def test_count_embeddings_positive(self):
        self.assertGreater(self.backend.count_embeddings(), 0)

    def test_bm25_search_returns_results(self):
        results = self.backend.search_fts("graph knowledge", limit=10)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], SearchResult)

    def test_bm25_search_scores_positive(self):
        results = self.backend.search_fts("graph databases", limit=10)
        for r in results:
            self.assertGreaterEqual(r.score, 0)

    def test_vector_search_returns_results(self):
        query_vec = mock_embed("how do AI agents remember things")
        results = self.backend.search_vec(query_vec, limit=10)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], SearchResult)

    def test_vector_search_result_fields(self):
        query_vec = mock_embed("graph memory")
        results = self.backend.search_vec(query_vec, limit=5)
        r = results[0]
        self.assertIsNotNone(r.filepath)
        self.assertIsNotNone(r.title)
        self.assertIsNotNone(r.hash)

    def test_collection_filter(self):
        """Results from a different collection should not appear."""
        results = self.backend.search_fts("graph", limit=10, collection="other")
        self.assertEqual(len(results), 0)

    def test_collection_filter_match(self):
        results = self.backend.search_fts("graph", limit=10, collection="test")
        self.assertGreater(len(results), 0)


class BatchEmbedderWithMethod:
    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    def embed_texts(self, texts: List[str]):
        self.batch_calls += 1
        return [mock_embed(t) for t in texts]

    def embed_text(self, text: str):
        self.single_calls += 1
        return mock_embed(text)


class BatchEmbedCallable:
    def __init__(self):
        self.calls = []

    def __call__(self, payload):
        self.calls.append(payload)
        if isinstance(payload, list):
            return [mock_embed(t) for t in payload]
        return mock_embed(payload)


class TestInlineMemoryOperations(unittest.TestCase):
    """Tests for upsert_memory/delete_memory/index_folder behavior."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upsert_memory_replaces_old_embeddings(self):
        self.backend.upsert_memory(
            path="notes/memory.md",
            text="first version\n" + ("a" * 800),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        first_rows = self.backend._embeddings_table.search().where(
            "collection = 'test' AND file_path = 'notes/memory.md'"
        ).to_list()
        first_hashes = {r["content_hash"] for r in first_rows}

        self.backend.upsert_memory(
            path="notes/memory.md",
            text="second version\n" + ("b" * 1200),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        second_rows = self.backend._embeddings_table.search().where(
            "collection = 'test' AND file_path = 'notes/memory.md'"
        ).to_list()
        second_hashes = {r["content_hash"] for r in second_rows}

        self.assertEqual(len(second_hashes), 1)
        self.assertNotEqual(first_hashes, second_hashes)

    def test_upsert_memory_uses_batch_embed_texts_when_available(self):
        embedder = BatchEmbedderWithMethod()

        self.backend.upsert_memory(
            path="notes/batch-method.md",
            text="batch method content " + ("x" * 1200),
            collection="test",
            embed_func=embedder,
            model="mock-embedder",
        )

        self.assertGreaterEqual(embedder.batch_calls, 1)
        self.assertEqual(embedder.single_calls, 0)

    def test_upsert_memory_supports_callable_batch_signature(self):
        embedder = BatchEmbedCallable()

        self.backend.upsert_memory(
            path="notes/batch-callable.md",
            text="batch callable content " + ("y" * 1200),
            collection="test",
            embed_func=embedder,
            model="mock-embedder",
        )

        self.assertTrue(any(isinstance(call, list) for call in embedder.calls))

    def test_upsert_memory_skip_delete_still_prevents_stale_duplicates(self):
        self.backend.upsert_memory(
            path="notes/skip-delete.md",
            text="first version " + ("a" * 900),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        del_filter = "collection = 'test' AND file_path = 'notes/skip-delete.md'"
        self.backend._embeddings_table.delete(del_filter)

        self.backend.upsert_memory(
            path="notes/skip-delete.md",
            text="second version " + ("b" * 1000),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            _skip_delete=True,
        )

        rows = self.backend._embeddings_table.search().where(del_filter).to_list()
        hashes = {r["content_hash"] for r in rows}
        self.assertEqual(len(hashes), 1)
        self.assertGreater(len(rows), 0)

    def test_delete_memory_deactivates_doc_and_removes_embeddings(self):
        self.backend.upsert_memory(
            path="notes/delete-me.md",
            text="delete this memory",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        result = self.backend.delete_memory(path="notes/delete-me.md", collection="test")
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["removed_vectors"], 1)

        rows = self.backend._embeddings_table.search().where(
            "collection = 'test' AND file_path = 'notes/delete-me.md'"
        ).to_list()
        self.assertEqual(len(rows), 0)
        self.assertIsNone(self.backend.find_document("test", "notes/delete-me.md"))

    def test_index_folder_indexes_text_and_skips_binary(self):
        folder = os.path.join(self.temp_dir, "folder")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "a.md"), "w", encoding="utf-8") as f:
            f.write("alpha memory")
        with open(os.path.join(folder, "b.txt"), "w", encoding="utf-8") as f:
            f.write("beta memory")
        with open(os.path.join(folder, "x.bin"), "wb") as f:
            f.write(b"\x00\xFF\x10")

        summary = self.backend.index_folder(
            folder_path=folder,
            collection="test",
            recursive=True,
            include_globs=["**/*.md", "**/*.txt", "*.md", "*.txt"],
            exclude_globs=["*x.bin"],
            embed_func=mock_embed,
            model="mock-embedder",
        )

        self.assertTrue(summary["success"])
        self.assertEqual(summary["indexed"], 2)
        self.assertGreaterEqual(summary["skipped"], 1)
        self.assertIn("skipped_details", summary)
        self.assertIn({"path": "x.bin", "reason": "glob_mismatch"}, summary["skipped_details"])

    def test_ingest_folder_reports_skip_reasons(self):
        folder = os.path.join(self.temp_dir, "ingest_folder")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "keep.md"), "w", encoding="utf-8") as f:
            f.write("keep this text")
        with open(os.path.join(folder, "image.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        with open(os.path.join(folder, "empty.txt"), "w", encoding="utf-8") as f:
            f.write("   ")

        summary = self.backend.ingest(
            collection="test",
            text=None,
            path=None,
            file_path=None,
            folder_path=folder,
            recursive=True,
            content_types=["text"],
            include_globs=["**/*", "*"],
            exclude_globs=["empty.txt"],
            embed_text_func=mock_embed,
            embed_image_func=mock_embed,
            embed_video_func=mock_embed,
            model="mock-embedder",
        )

        self.assertTrue(summary["success"])
        self.assertEqual(summary["indexed_text"], 1)
        skipped = {(item["path"], item.get("reason")) for item in summary["items"] if item["status"] == "skipped"}
        self.assertIn(("image.png", "not_in_content_types"), skipped)
        self.assertIn(("empty.txt", "excluded"), skipped)


class TestContentTypeFilter(unittest.TestCase):
    """Tests for content_type filtering in search methods."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

        # Index a text document
        self.backend.index_document(
            path="ai-overview.md",
            text="Artificial intelligence AI machines neural networks architecture system design.",
            collection="test",
            model="mock-embedder",
            embed_func=mock_embed,
            content_type="text",
        )

        # Insert a mock image embedding directly
        self.backend.insert_embedding(
            content_hash="img_arch_png",
            seq=0,
            pos=0,
            vector=mock_embed("architecture diagram system design"),
            model="mock-embedder",
            collection="test",
            file_path="diagram-architecture.png",
            title="Architecture Diagram",
            text_body="architecture system design network infrastructure",
            content_type="image",
        )
        self.backend.rebuild_fts_index()

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_search_text_only(self):
        results = self.backend.search_fts("architecture", limit=20, content_type="text")
        for r in results:
            self.assertEqual(r.content_type, "text")

    def test_search_image_only(self):
        results = self.backend.search_fts("architecture", limit=20, content_type="image")
        for r in results:
            self.assertEqual(r.content_type, "image")

    def test_vec_search_text_only(self):
        vec = mock_embed("system architecture design")
        results = self.backend.search_vec(vec, limit=20, content_type="text")
        for r in results:
            self.assertEqual(r.content_type, "text")

    def test_vec_search_image_only(self):
        vec = mock_embed("system architecture design")
        results = self.backend.search_vec(vec, limit=20, content_type="image")
        for r in results:
            self.assertEqual(r.content_type, "image")


class TestNamespaceFiltering(unittest.TestCase):
    """Tests for namespace/profile filtering in memory operations and search."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-ns-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upsert_memory_with_namespace(self):
        content_hash = self.backend.upsert_memory(
            path="notes/namespace-test.md",
            text="Namespace isolation test content " + ("x" * 500),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            user_id="user123",
            session_id="sess456",
            project_id="proj789",
            profile="work",
        )
        self.assertIsNotNone(content_hash)

        rows = self.backend._embeddings_table.search().where(
            "collection = 'test' AND file_path = 'notes/namespace-test.md'"
        ).to_list()
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row.get("user_id"), "user123")
            self.assertEqual(row.get("session_id"), "sess456")
            self.assertEqual(row.get("project_id"), "proj789")
            self.assertEqual(row.get("profile"), "work")

    def test_search_with_namespace_filter(self):
        self.backend.upsert_memory(
            path="docs/alice.md",
            text="Alice's document about machine learning " + ("a" * 500),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            user_id="alice",
        )
        self.backend.upsert_memory(
            path="docs/bob.md",
            text="Bob's document about machine learning " + ("b" * 500),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            user_id="bob",
        )
        self.backend.rebuild_fts_index()

        alice_results = self.backend.search_fts("machine learning", limit=10, user_id="alice")
        for r in alice_results:
            self.assertEqual(r.user_id, "alice")

        bob_results = self.backend.search_fts("machine learning", limit=10, user_id="bob")
        for r in bob_results:
            self.assertEqual(r.user_id, "bob")

    def test_vector_search_with_namespace_filter(self):
        self.backend.upsert_memory(
            path="projects/projA.md",
            text="Project A documentation about neural networks " + ("a" * 500),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            project_id="projA",
        )
        self.backend.upsert_memory(
            path="projects/projB.md",
            text="Project B documentation about neural networks " + ("b" * 500),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            project_id="projB",
        )

        vec = mock_embed("neural networks documentation")
        proj_a_results = self.backend.search_vec(vec, limit=10, project_id="projA")
        for r in proj_a_results:
            self.assertEqual(r.project_id, "projA")

        proj_b_results = self.backend.search_vec(vec, limit=10, project_id="projB")
        for r in proj_b_results:
            self.assertEqual(r.project_id, "projB")


class TestMemoryMetadata(unittest.TestCase):
    """Tests for memory importance, TTL, and tags metadata."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-meta-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upsert_memory_with_importance(self):
        self.backend.upsert_memory(
            path="notes/important.md",
            text="Very important memory",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            importance=0.95,
        )
        rows = self.backend._embeddings_table.search().where(
            "collection = 'test' AND file_path = 'notes/important.md'"
        ).to_list()
        self.assertGreaterEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["importance"], 0.95, places=2)

    def test_upsert_memory_with_tags(self):
        tags = ["project", "ai", "urgent"]
        self.backend.upsert_memory(
            path="notes/tagged.md",
            text="Memory with tags",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            tags=tags,
        )
        rows = self.backend._embeddings_table.search().where(
            "collection = 'test' AND file_path = 'notes/tagged.md'"
        ).to_list()
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["tags"], '["project", "ai", "urgent"]')

    def test_search_excludes_expired_entries(self):
        import time
        self.backend.upsert_memory(
            path="notes/short-ttl.md",
            text="This will expire quickly",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            ttl_seconds=1,
        )
        self.backend.upsert_memory(
            path="notes/permanent.md",
            text="This is permanent",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        self.backend.rebuild_fts_index()
        time.sleep(1.5)

        results_expire = self.backend.search_fts("expire", limit=10, collection="test")
        expire_paths = [r.display_path for r in results_expire]
        self.assertNotIn("test/notes/short-ttl.md", expire_paths)

    def test_metadata_backward_compatibility(self):
        self.backend.upsert_memory(
            path="notes/legacy.md",
            text="Legacy memory without metadata",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        rows = self.backend._embeddings_table.search().where(
            "collection = 'test' AND file_path = 'notes/legacy.md'"
        ).to_list()
        self.assertGreaterEqual(len(rows), 1)
        self.assertIsNone(rows[0]["importance"])
        self.assertIsNone(rows[0]["ttl_seconds"])
        self.assertIsNone(rows[0]["tags"])
        self.assertIsNone(rows[0]["expires_at"])


class TestFTSMissFallbackBehavior(unittest.TestCase):
    """Tests for P0: FTS miss fallback behavior - no BM25 fallback on empty results."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-fts-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fts_empty_results_returns_empty_no_bm25_fallback(self):
        """FTS returning empty results (no matches) should NOT trigger BM25 fallback."""
        # Index a document
        self.backend.upsert_memory(
            path="notes/test.md",
            text="This is a document about machine learning algorithms.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        self.backend.rebuild_fts_index()

        # Search for something that doesn't exist - should return empty, not BM25 fallback
        results = self.backend.search_fts("xyzzynonexistentterm12345", limit=10, collection="test")
        self.assertEqual(len(results), 0)

    def test_fts_error_triggers_bm25_fallback(self):
        """FTS errors should still trigger BM25 fallback for resilience."""
        # Index a document with content
        self.backend.upsert_memory(
            path="notes/fallback.md",
            text="Fallback test document about neural networks.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        self.backend.rebuild_fts_index()

        # Normal search should work
        results = self.backend.search_fts("neural", limit=10, collection="test")
        self.assertGreater(len(results), 0)

    def test_fts_match_returns_results(self):
        """FTS matches should return results normally."""
        self.backend.upsert_memory(
            path="notes/match.md",
            text="Document about artificial intelligence and deep learning.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        self.backend.rebuild_fts_index()

        results = self.backend.search_fts("intelligence", limit=10, collection="test")
        self.assertGreater(len(results), 0)
        self.assertIn("test/notes/match.md", [r.display_path for r in results])


class TestBulkModeFTSRebuild(unittest.TestCase):
    """Tests for P0: FTS rebuild scheduling in bulk operations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-bulk-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bulk_mode_defers_rebuilds(self):
        """Bulk mode should defer all FTS rebuilds until context exit."""
        # Reset the rebuild counter
        self.backend._fts_rebuild_pending = 0
        self.backend._fts_needs_rebuild = False

        with self.backend.bulk_mode():
            # During bulk mode, rebuilds should be deferred
            self.backend._schedule_fts_rebuild()
            self.assertTrue(self.backend._fts_needs_rebuild)
            self.assertEqual(self.backend._fts_rebuild_pending, 1)
            self.assertFalse(self.backend._fts_rebuild_pending >= self.backend.FTS_REBUILD_PENDING_THRESHOLD)

        # After exiting bulk mode, rebuild should have happened
        self.assertFalse(self.backend._fts_needs_rebuild)
        self.assertEqual(self.backend._fts_rebuild_pending, 0)

    def test_index_folder_uses_bulk_mode(self):
        """index_folder should trigger only one FTS rebuild at the end."""
        folder = os.path.join(self.temp_dir, "bulk_folder")
        os.makedirs(folder, exist_ok=True)

        # Create multiple files
        for i in range(5):
            with open(os.path.join(folder, f"doc{i}.md"), "w", encoding="utf-8") as f:
                f.write(f"Document {i} content about topic {i}.")

        # Spy on _do_fts_rebuild to count calls
        original_rebuild = self.backend._do_fts_rebuild
        rebuild_count = [0]

        def spy_rebuild():
            rebuild_count[0] += 1
            return original_rebuild()

        self.backend._do_fts_rebuild = spy_rebuild

        summary = self.backend.index_folder(
            folder_path=folder,
            collection="test",
            recursive=True,
            include_globs=["**/*.md", "*.md"],  # Match both root and nested .md files
            exclude_globs=None,
            embed_func=mock_embed,
            model="mock-embedder",
        )

        self.assertTrue(summary["success"])
        self.assertEqual(summary["indexed"], 5)
        # Should have exactly 1 rebuild call at the end (via bulk_mode context exit)
        self.assertEqual(rebuild_count[0], 1)

    def test_upsert_memory_schedules_rebuild_normally(self):
        """Outside bulk mode, upsert_memory should schedule rebuilds."""
        self.backend._fts_rebuild_pending = 0
        self.backend._fts_needs_rebuild = False

        # Single upsert should schedule a rebuild (deferred by threshold logic)
        self.backend.upsert_memory(
            path="notes/single.md",
            text="Single document test.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        # Should have scheduled a rebuild
        self.assertTrue(self.backend._fts_needs_rebuild)
        self.assertGreater(self.backend._fts_rebuild_pending, 0)

    def test_bulk_mode_nested_contexts(self):
        """Nested bulk mode contexts should work correctly."""
        self.backend._fts_rebuild_pending = 0
        self.backend._fts_needs_rebuild = False

        with self.backend.bulk_mode():
            self.backend._schedule_fts_rebuild()
            with self.backend.bulk_mode():
                self.backend._schedule_fts_rebuild()
                # Still in bulk mode, no rebuild yet
                self.assertTrue(self.backend._fts_needs_rebuild)
            # Inner context exit but outer still active
            self.assertTrue(self.backend._bulk_mode)

        # After outer context exit, rebuild should happen
        self.assertFalse(self.backend._fts_needs_rebuild)


class TestListCollectionsAndNamespaces(unittest.TestCase):
    """Tests for list_collections and list_namespaces (REC-29)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-list-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _upsert(self, path, text, collection, **ns_kwargs):
        self.backend.upsert_memory(
            path=path,
            text=text,
            collection=collection,
            embed_func=mock_embed,
            model="mock-embedder",
            **ns_kwargs,
        )

    # --- list_collections ---

    def test_list_collections_empty_store(self):
        result = self.backend.list_collections()
        self.assertEqual(result, [])

    def test_list_collections_returns_unique_sorted(self):
        self._upsert("a.md", "Alpha doc", "colA")
        self._upsert("b.md", "Beta doc", "colB")
        self._upsert("c.md", "Another in colA", "colA")
        result = self.backend.list_collections()
        self.assertEqual(result, ["colA", "colB"])

    def test_list_collections_namespace_filter(self):
        self._upsert("x.md", "User1 doc colA", "colA", user_id="user1")
        self._upsert("y.md", "User2 doc colB", "colB", user_id="user2")
        self._upsert("z.md", "User1 doc colC", "colC", user_id="user1")
        # Only user1's collections
        result = self.backend.list_collections(user_id="user1")
        self.assertEqual(result, ["colA", "colC"])

    def test_list_collections_profile_filter(self):
        self._upsert("p.md", "Profile A doc", "colP", profile="profileA")
        self._upsert("q.md", "Profile B doc", "colQ", profile="profileB")
        result = self.backend.list_collections(profile="profileA")
        self.assertEqual(result, ["colP"])

    # --- list_namespaces ---

    def test_list_namespaces_empty_store(self):
        result = self.backend.list_namespaces()
        self.assertEqual(result, [])

    def test_list_namespaces_no_namespace_fields(self):
        # Documents with no namespace fields should appear as empty dict
        self._upsert("a.md", "No namespace doc", "col1")
        result = self.backend.list_namespaces()
        self.assertEqual(result, [{}])

    def test_list_namespaces_returns_unique_combinations(self):
        self._upsert("a.md", "Doc A", "col1", user_id="u1", session_id="s1")
        self._upsert("b.md", "Doc B", "col1", user_id="u1", session_id="s2")
        self._upsert("c.md", "Doc C", "col2", user_id="u1", session_id="s1")
        # Two chunks from same namespace → still one entry
        self._upsert("d.md", "Doc D " * 300, "col1", user_id="u2", session_id="s1")
        result = self.backend.list_namespaces()
        user_ids = {ns.get("user_id") for ns in result}
        self.assertIn("u1", user_ids)
        self.assertIn("u2", user_ids)
        # Three unique u1 combos + one u2 combo
        u1_entries = [ns for ns in result if ns.get("user_id") == "u1"]
        self.assertEqual(len(u1_entries), 2)  # (u1,s1) and (u1,s2)

    def test_list_namespaces_collection_filter(self):
        self._upsert("x.md", "ColX user1", "colX", user_id="u1")
        self._upsert("y.md", "ColY user2", "colY", user_id="u2")
        result = self.backend.list_namespaces(collection="colX")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].get("user_id"), "u1")

    def test_list_namespaces_omits_empty_fields(self):
        self._upsert("m.md", "Mixed ns", "col1", user_id="u1")
        result = self.backend.list_namespaces()
        # Only non-empty namespace fields should be present
        self.assertTrue(all("session_id" not in ns for ns in result))
        self.assertTrue(all("project_id" not in ns for ns in result))
        self.assertTrue(all("profile" not in ns for ns in result))
        self.assertEqual(result[0]["user_id"], "u1")


if __name__ == "__main__":
    unittest.main()
