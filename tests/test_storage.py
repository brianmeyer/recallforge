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


if __name__ == "__main__":
    unittest.main()
