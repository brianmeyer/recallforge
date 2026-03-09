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


if __name__ == "__main__":
    unittest.main()
