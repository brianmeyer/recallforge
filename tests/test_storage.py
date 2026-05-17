"""
test_storage.py - Unit tests for StorageBackend ABC and LanceDBBackend.

Tests document indexing, BM25 + vector search, content-type filtering,
and cross-modal search using a real (but temp) LanceDB instance.
Uses a deterministic mock embed_func — NO real model inference.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recallforge.storage.base import StorageBackend, SearchResult, Document
from recallforge.storage.lancedb_backend import LanceDBBackend
from recallforge.storage.lancedb_shared import build_memory_id
from recallforge.search import HybridSearcher


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


class EmbedOnlyBackend:
    """Minimal backend for storage-backed hybrid search tests."""

    def get_mode(self):
        return "embed"

    def needs_reranker(self):
        return False

    def embed_text(self, text: str):
        return mock_embed(text)


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

    def test_index_version_bumps_only_for_visible_updates(self):
        initial = int(self.backend.get_index_version())
        self.backend.upsert_memory(
            path="notes/versioned.md",
            text="Visible Acme Robotics memory.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        after_visible = int(self.backend.get_index_version())
        self.assertGreater(after_visible, initial)

        batch_id = self.backend.begin_index_batch()
        self.backend.upsert_memory(
            path="notes/versioned.md",
            text="Hidden Globex Labs replacement.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            _skip_delete=True,
            _active=0,
            _index_batch_id=batch_id,
        )
        self.assertEqual(int(self.backend.get_index_version()), after_visible)

        self.backend.promote_index_batch(
            batch_id=batch_id,
            collection="test",
            logical_path="notes/versioned.md",
        )
        self.assertGreater(int(self.backend.get_index_version()), after_visible)


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

    def test_upsert_memory_populates_memory_identity(self):
        self.backend.upsert_memory(
            path="notes/identity.md",
            text="Memory identity test content " + ("z" * 600),
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        doc = self.backend.find_document("test", "notes/identity.md")
        self.assertIsNotNone(doc)
        self.assertIsNotNone(doc.memory_id)
        self.assertEqual(doc.memory_role, "root")
        self.assertEqual(doc.memory_root_path, "notes/identity.md")

        results = self.backend.search_fts("identity", limit=5, collection="test")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].memory_id, doc.memory_id)
        self.assertEqual(results[0].memory_role, "root")
        self.assertEqual(results[0].memory_root_path, "notes/identity.md")

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

    def test_memory_graph_entities_and_related_memories_track_evidence(self):
        self.backend.upsert_memory(
            path="notes/acme-review.md",
            text="Mira from Acme Robotics tracks the launch review for Project Atlas.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        self.backend.upsert_memory(
            path="notes/acme-budget.md",
            text="The budget memo says Acme Robotics approved new sensors.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        entities = self.backend.list_memory_entities(path="notes/acme-review.md", collection="test")
        entity_keys = {row["entity_key"] for row in entities}
        self.assertIn("acme_robotics", entity_keys)
        self.assertTrue(any(row["evidence"] and "Acme Robotics" in row["evidence"] for row in entities))

        related = self.backend.find_related_memories(path="notes/acme-review.md", collection="test")
        self.assertTrue(related)
        self.assertEqual(related[0]["path"], "notes/acme-budget.md")
        self.assertIn("acme_robotics", {item["entity_key"] for item in related[0]["shared_entities"]})
        self.assertTrue(related[0]["evidence"])

    def test_memory_graph_rows_are_replaced_on_memory_update(self):
        self.backend.upsert_memory(
            path="notes/company.md",
            text="Mira from Acme Robotics owns the launch checklist.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        self.assertTrue(
            self.backend.list_memory_entities(path="notes/company.md", entity="Acme Robotics", collection="test")
        )

        self.backend.upsert_memory(
            path="notes/company.md",
            text="Mira from Globex Labs owns the launch checklist.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        self.assertFalse(
            self.backend.list_memory_entities(path="notes/company.md", entity="Acme Robotics", collection="test")
        )
        self.assertTrue(
            self.backend.list_memory_entities(path="notes/company.md", entity="Globex Labs", collection="test")
        )

    def test_index_batch_rows_are_hidden_until_promoted(self):
        path = "notes/batch-promotion.md"
        self.backend.upsert_memory(
            path=path,
            text="Mira from Acme Robotics owns the launch checklist.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        visible_embeddings_before = self.backend.count_embeddings()
        self.assertEqual(self.backend.count_documents(), 1)

        batch_id = self.backend.begin_index_batch()
        self.backend.upsert_memory(
            path=path,
            text="Mira from Globex Labs owns the launch checklist.",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            _skip_delete=True,
            _active=0,
            _index_batch_id=batch_id,
        )

        self.assertEqual(self.backend.count_documents(), 1)
        self.assertEqual(self.backend.count_embeddings(), visible_embeddings_before)
        memory_before = self.backend.get_memory(path=path, collection="test")
        self.assertIsNotNone(memory_before)
        self.assertIn("Acme Robotics", memory_before["summary"])
        self.assertNotIn("Globex Labs", memory_before["summary"])
        self.assertTrue(
            self.backend.list_memory_entities(path=path, entity="Acme Robotics", collection="test")
        )
        self.assertFalse(
            self.backend.list_memory_entities(path=path, entity="Globex Labs", collection="test")
        )

        hidden_rows = self.backend._embeddings_table.search().where(
            f"collection = 'test' AND file_path = '{path}' AND active = 0"
        ).to_list()
        self.assertGreater(len(hidden_rows), 0)

        promoted = self.backend.promote_index_batch(
            batch_id=batch_id,
            collection="test",
            logical_path=path,
        )
        self.assertGreater(promoted["activated_embeddings"], 0)
        self.assertGreater(promoted["deactivated_embeddings"], 0)

        memory_after = self.backend.get_memory(path=path, collection="test")
        self.assertIsNotNone(memory_after)
        self.assertEqual(self.backend.count_documents(), 1)
        self.assertIn("Globex Labs", memory_after["summary"])
        self.assertNotIn("Acme Robotics", memory_after["summary"])
        self.assertFalse(
            self.backend.list_memory_entities(path=path, entity="Acme Robotics", collection="test")
        )
        self.assertTrue(
            self.backend.list_memory_entities(path=path, entity="Globex Labs", collection="test")
        )

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

    def test_importance_boost_affects_vector_search_order(self):
        vector = mock_embed("same semantic vector")
        self.backend.insert_content("low-hash", "Low priority memory", "text")
        self.backend.insert_content("high-hash", "High priority memory", "text")
        self.backend.insert_embedding(
            content_hash="low-hash",
            seq=0,
            pos=0,
            vector=vector,
            model="mock-embedder",
            collection="test",
            file_path="notes/low.md",
            title="Low",
            text_body="same semantic vector",
            content_type="text",
            importance=0.0,
        )
        self.backend.insert_embedding(
            content_hash="high-hash",
            seq=0,
            pos=0,
            vector=vector,
            model="mock-embedder",
            collection="test",
            file_path="notes/high.md",
            title="High",
            text_body="same semantic vector",
            content_type="text",
            importance=1.0,
        )

        results = self.backend.search_vec(vector, limit=2, collection="test")
        self.assertEqual(results[0].display_path, "test/notes/high.md")
        self.assertAlmostEqual(results[0].importance, 1.0, places=2)


class TestMemoryLookupCompatibility(unittest.TestCase):
    """Regression tests for canonical memory lookup compatibility."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-memory-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_memory_resolves_legacy_null_memory_id_and_path(self):
        path = "notes/legacy-memory.md"
        expected_memory_id = build_memory_id("test", path)
        self.backend.upsert_memory(
            path=path,
            text="Legacy memory lookup regression test",
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        lookup_where = "collection = 'test' AND file_path = 'notes/legacy-memory.md'"
        self.backend._documents_table.update(where=lookup_where, values={"memory_id": None})
        self.backend._embeddings_table.update(where=lookup_where, values={"memory_id": None})

        memories = self.backend.list_memories(collection="test", limit=10)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["memory_id"], expected_memory_id)

        by_id = self.backend.get_memory(expected_memory_id, collection="test")
        self.assertIsNotNone(by_id)
        self.assertEqual(by_id["memory_id"], expected_memory_id)
        self.assertEqual(by_id["path"], path)
        self.assertEqual(by_id["root_document"]["path"], path)
        self.assertGreater(len(by_id["snippets"]), 0)

        by_path = self.backend.get_memory(path=path, collection="test")
        self.assertIsNotNone(by_path)
        self.assertEqual(by_path["memory_id"], expected_memory_id)
        self.assertEqual(by_path["path"], path)
        self.assertEqual(by_path["root_document"]["path"], path)

    def test_memory_lookup_surfaces_derived_summary(self):
        path = "notes/summary-memory.md"
        text = (
            "RecallForge stores memory summaries so agents can inspect memories quickly. "
            "This regression checks that canonical memory reads expose a concise summary."
        )
        self.backend.upsert_memory(
            path=path,
            text=text,
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        memories = self.backend.list_memories(collection="test", limit=10)
        self.assertEqual(len(memories), 1)
        self.assertTrue(memories[0]["summary"].startswith("RecallForge stores memory summaries"))

        memory = self.backend.get_memory(path=path, collection="test")
        self.assertIsNotNone(memory)
        self.assertTrue(memory["summary"].startswith("RecallForge stores memory summaries"))
        self.assertEqual(memory["path"], path)


class TestConversationMemoryIndexing(unittest.TestCase):
    """Tests for first-class conversation memories with turn rollups."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-conversation-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_conversation_creates_root_and_turn_children(self):
        turns = [
            {"role": "user", "content": "Can we renew the customer contract before Q3?"},
            {"role": "assistant", "content": "Yes. The renewal plan depends on pricing approval."},
            {"role": "user", "content": "Please remember the pricing risk and legal review."},
        ]

        result = self.backend.index_conversation(
            path="threads/customer-renewal",
            title="Customer Renewal Thread",
            summary="Discussion about renewal timing, pricing risk, and legal review.",
            turns=turns,
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
            user_id="alice",
            session_id="thread-123",
            tags=["sales"],
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["indexed_turns"], 3)
        self.assertEqual(result["path"], "threads/customer-renewal")
        self.assertEqual(
            result["memory_id"],
            build_memory_id(
                "test",
                "threads/customer-renewal",
                user_id="alice",
                session_id="thread-123",
            ),
        )
        self.assertIn("conversation", result["tags"])
        self.assertIn("sales", result["tags"])

        rows = self.backend._documents_table.search().where(
            "collection = 'test' AND active = 1"
        ).to_list()
        self.assertEqual(len(rows), 4)
        roles = {row["file_path"]: row["memory_role"] for row in rows}
        self.assertEqual(roles["threads/customer-renewal"], "root")
        self.assertEqual(roles["threads/customer-renewal::turn:0001"], "child")
        self.assertEqual(roles["threads/customer-renewal::turn:0002"], "child")
        self.assertEqual(roles["threads/customer-renewal::turn:0003"], "child")
        self.assertEqual({row["memory_id"] for row in rows}, {result["memory_id"]})
        self.assertEqual({row["memory_root_path"] for row in rows}, {"threads/customer-renewal"})

        memory = self.backend.get_memory(path="threads/customer-renewal", collection="test", user_id="alice")
        self.assertIsNotNone(memory)
        self.assertEqual(memory["memory_id"], result["memory_id"])
        self.assertEqual(len(memory["children"]), 3)
        self.assertTrue(any("pricing risk" in snippet["text"] for snippet in memory["snippets"]))

    def test_matching_turns_roll_up_to_parent_conversation(self):
        turns = [
            {"role": "user", "content": "The launch needs pricing approval from finance."},
            {"role": "assistant", "content": "I noted that pricing approval blocks the renewal email."},
            {"role": "user", "content": "Legal review is separate from the pricing approval path."},
        ]
        self.backend.index_conversation(
            path="threads/pricing-approval",
            title="Pricing Approval Thread",
            turns=turns,
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )
        self.backend.rebuild_fts_index()

        searcher = HybridSearcher(
            backend=EmbedOnlyBackend(),
            storage=self.backend,
            collection="test",
            limit=5,
            fts_probe_limit=10,
        )
        results = searcher.search("pricing approval renewal")

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.memory_role, "root")
        self.assertEqual(result.memory_root_path, "threads/pricing-approval")
        self.assertEqual(result.filepath, "recallforge://test/threads/pricing-approval")
        self.assertEqual(result.display_path, "test/threads/pricing-approval")
        self.assertGreaterEqual(result.memory_hit_count, 2)
        evidence_paths = [result.memory_primary_evidence_path] + (result.memory_supporting_paths or [])
        self.assertIn("recallforge://test/threads/pricing-approval::turn:0002", evidence_paths)

    def test_reindex_conversation_replaces_stale_children_as_one_batch(self):
        path = "threads/reindex-consistency"
        self.backend.index_conversation(
            path=path,
            title="Reindex Consistency",
            turns=[
                {"role": "user", "content": "Acme Robotics asked about launch readiness."},
                {"role": "assistant", "content": "Globex Labs is the stale follow-up owner."},
            ],
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        first_memory = self.backend.get_memory(path=path, collection="test")
        self.assertIsNotNone(first_memory)
        self.assertEqual(len(first_memory["children"]), 2)
        self.assertTrue(
            self.backend.list_memory_entities(path=path, entity="Globex Labs", collection="test")
        )

        self.backend.index_conversation(
            path=path,
            title="Reindex Consistency",
            turns=[
                {"role": "user", "content": "Initech is now the only launch readiness owner."},
            ],
            collection="test",
            embed_func=mock_embed,
            model="mock-embedder",
        )

        updated_memory = self.backend.get_memory(path=path, collection="test")
        self.assertIsNotNone(updated_memory)
        self.assertEqual(len(updated_memory["children"]), 1)
        self.assertEqual(updated_memory["children"][0]["path"], f"{path}::turn:0001")
        self.assertFalse(
            self.backend.list_memory_entities(path=path, entity="Globex Labs", collection="test")
        )
        self.assertTrue(
            self.backend.list_memory_entities(path=path, entity="Initech", collection="test")
        )


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


class TestCollectionManagement(unittest.TestCase):
    """Tests for rename_collection and delete_collection (REC-65)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-collections-")
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

    # --- rename_collection ---

    def test_rename_collection_empty_names_raises(self):
        with self.assertRaises(ValueError):
            self.backend.rename_collection("", "new")
        with self.assertRaises(ValueError):
            self.backend.rename_collection("old", "")

    def test_rename_collection_same_name_noop(self):
        result = self.backend.rename_collection("same", "same")
        self.assertTrue(result["success"])
        self.assertEqual(result["embeddings_updated"], 0)
        self.assertEqual(result["documents_updated"], 0)

    def test_rename_collection_updates_embeddings_and_documents(self):
        # Create documents in old collection
        self._upsert("a.md", "Alpha doc in oldcol", "oldcol")
        self._upsert("b.md", "Beta doc in oldcol", "oldcol")
        self._upsert("c.md", "Gamma doc in other", "other")  # Different collection

        # Rename oldcol to newcol
        result = self.backend.rename_collection("oldcol", "newcol")
        self.assertTrue(result["success"])
        self.assertEqual(result["old_name"], "oldcol")
        self.assertEqual(result["new_name"], "newcol")
        self.assertGreater(result["embeddings_updated"], 0)
        self.assertGreater(result["documents_updated"], 0)

        # Rebuild FTS index after rename
        self.backend.rebuild_fts_index()

        # Verify old collection is gone
        collections = self.backend.list_collections()
        self.assertNotIn("oldcol", collections)
        self.assertIn("newcol", collections)
        self.assertIn("other", collections)

        # Verify documents are now in new collection
        doc_a = self.backend.find_document("newcol", "a.md")
        self.assertIsNotNone(doc_a)
        self.assertEqual(doc_a.collection, "newcol")

        # Verify search works on renamed collection
        results = self.backend.search_fts("alpha", limit=10, collection="newcol")
        self.assertEqual(len(results), 1)
        self.assertIn("newcol/a.md", results[0].display_path)

        # Verify old collection returns no results
        results_old = self.backend.search_fts("alpha", limit=10, collection="oldcol")
        self.assertEqual(len(results_old), 0)

    def test_rename_collection_preserves_namespaces(self):
        self._upsert("x.md", "Namespaced doc", "oldcol", user_id="u1", profile="work")

        result = self.backend.rename_collection("oldcol", "newcol")
        self.assertTrue(result["success"])

        # Verify namespace fields are preserved
        rows = self.backend._embeddings_table.search().where(
            "collection = 'newcol' AND file_path = 'x.md'"
        ).to_list()
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row["user_id"], "u1")
            self.assertEqual(row["profile"], "work")

    # --- delete_collection ---

    def test_delete_collection_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.backend.delete_collection("")

    def test_delete_collection_removes_all_data(self):
        # Create documents in collection
        self._upsert("a.md", "Doc A in delcol", "delcol")
        self._upsert("b.md", "Doc B in delcol", "delcol")
        self._upsert("c.md", "Doc C in other", "other")  # Different collection

        # Get initial counts
        initial_embeddings = self.backend.count_embeddings()
        initial_documents = self.backend.count_documents()

        # Delete the collection
        result = self.backend.delete_collection("delcol")
        self.assertTrue(result["success"])
        self.assertEqual(result["name"], "delcol")
        self.assertGreater(result["embeddings_deleted"], 0)
        self.assertGreater(result["documents_deleted"], 0)

        # Verify collection is gone
        collections = self.backend.list_collections()
        self.assertNotIn("delcol", collections)
        self.assertIn("other", collections)

        # Verify documents are gone
        doc_a = self.backend.find_document("delcol", "a.md")
        self.assertIsNone(doc_a)

        # Verify other collection is untouched
        doc_c = self.backend.find_document("other", "c.md")
        self.assertIsNotNone(doc_c)

    def test_delete_collection_cleans_orphans(self):
        # Create a document with unique content
        unique_content = "Unique content for orphan test " + ("x" * 500)
        self._upsert("orphan.md", unique_content, "orphancol")

        # Get the content hash
        doc = self.backend.find_document("orphancol", "orphan.md")
        self.assertIsNotNone(doc)
        content_hash = doc.content_hash

        # Verify content exists
        content_before = self.backend.get_content(content_hash)
        self.assertIsNotNone(content_before)

        # Delete the collection
        result = self.backend.delete_collection("orphancol")
        self.assertTrue(result["success"])
        self.assertGreater(result["orphans_cleaned"], 0)

        # Verify content is cleaned up (orphan removed)
        content_after = self.backend.get_content(content_hash)
        self.assertIsNone(content_after)

    def test_delete_collection_preserves_shared_content(self):
        # Create same content in two collections
        shared_content = "Shared content " + ("s" * 500)
        self._upsert("shared.md", shared_content, "colA")
        self._upsert("shared.md", shared_content, "colB")

        # Get the content hash
        doc_a = self.backend.find_document("colA", "shared.md")
        content_hash = doc_a.content_hash

        # Verify content exists
        self.assertIsNotNone(self.backend.get_content(content_hash))

        # Delete one collection
        result = self.backend.delete_collection("colA")
        self.assertTrue(result["success"])

        # Content should still exist (referenced by colB)
        content_after = self.backend.get_content(content_hash)
        self.assertIsNotNone(content_after)

        # colB should still have its document
        doc_b = self.backend.find_document("colB", "shared.md")
        self.assertIsNotNone(doc_b)

    def test_delete_collection_nonexistent_succeeds(self):
        # Deleting a non-existent collection should succeed with 0 counts
        result = self.backend.delete_collection("nonexistent")
        self.assertTrue(result["success"])
        self.assertEqual(result["embeddings_deleted"], 0)
        self.assertEqual(result["documents_deleted"], 0)
        self.assertEqual(result["orphans_cleaned"], 0)


class CaptioningEmbedder:
    """Mock multimodal embedder with optional caption methods."""

    def __call__(self, path: str) -> List[float]:
        return mock_embed(path)

    def caption_image(self, path: str) -> str:
        return "Neural network diagram with labeled hidden layers."

    def describe_video(self, path: str, frame_paths=None) -> str:
        frame_count = len(frame_paths or [])
        return f"Technical explainer video with {frame_count} keyframes showing diagrams."

    def generate_text(self, prompt: str, max_tokens: int = 60) -> str:
        prompt_lower = prompt.lower()
        if "image memory" in prompt_lower:
            return '["neural network", "diagram", "hidden layers"]'
        if "video memory" in prompt_lower:
            return '["technical explainer", "architecture diagram", "presentation"]'
        return "[]"


class FailingCaptioningEmbedder(CaptioningEmbedder):
    def caption_image(self, path: str) -> str:
        raise RuntimeError("caption failed")

    def describe_video(self, path: str, frame_paths=None) -> str:
        raise RuntimeError("video caption failed")


class TestIngestCaptioning(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-caption-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

        self.image_path = os.path.join(self.temp_dir, "diagram.jpg")
        with open(self.image_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xdbmockjpg")

        self.video_path = os.path.join(self.temp_dir, "demo.mp4")
        with open(self.video_path, "wb") as f:
            f.write(b"mock-video")

        self.frame_path = os.path.join(self.temp_dir, "frame_0001.jpg")
        with open(self.frame_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xdbframe")

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_image_stores_caption_in_text_body(self):
        embedder = CaptioningEmbedder()
        self.backend.index_image(
            path=self.image_path,
            collection="test",
            embed_func=embedder,
            caption_media=True,
        )

        rows = self.backend._embeddings_table.search().where("content_type = 'image'").to_list()
        self.assertEqual(len(rows), 1)
        self.assertIn("Neural network diagram", rows[0].get("text_body") or "")
        self.assertEqual(
            json.loads(rows[0].get("tags") or "[]"),
            ["neural network", "diagram", "hidden layers"],
        )

    def test_index_image_caption_failure_keeps_embedding(self):
        embedder = FailingCaptioningEmbedder()
        self.backend.index_image(
            path=self.image_path,
            collection="test",
            embed_func=embedder,
            caption_media=True,
        )

        rows = self.backend._embeddings_table.search().where("content_type = 'image'").to_list()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("text_body"), "")

    def test_index_video_stores_video_caption_in_text_body(self):
        embedder = CaptioningEmbedder()
        fake_artifacts = SimpleNamespace(
            frames=[
                SimpleNamespace(
                    image_path=self.frame_path,
                    logical_path="demo.mp4::frame:0001@0.00s",
                    title="frame 1",
                    timestamp_seconds=0.0,
                )
            ],
            transcripts=[],
            duration_seconds=4.2,
            transcript_path=None,
            ffmpeg_available=True,
        )

        with patch("recallforge.storage.indexing_ops.extract_video_artifacts", return_value=fake_artifacts):
            self.backend.index_video(
                path=self.video_path,
                collection="test",
                embed_text_func=mock_embed,
                embed_image_func=embedder,
                embed_video_func=embedder,
                caption_media=True,
            )

        video_rows = self.backend._embeddings_table.search().where("content_type = 'video'").to_list()
        self.assertEqual(len(video_rows), 1)
        self.assertIn("Technical explainer video", video_rows[0].get("text_body") or "")
        self.assertEqual(
            json.loads(video_rows[0].get("tags") or "[]"),
            ["technical explainer", "architecture diagram", "presentation"],
        )

    def test_memory_lookup_surfaces_media_tags(self):
        embedder = CaptioningEmbedder()
        self.backend.index_image(
            path=self.image_path,
            collection="test",
            embed_func=embedder,
            caption_media=True,
        )

        memories = self.backend.list_memories(collection="test", limit=10)
        self.assertEqual(len(memories), 1)
        self.assertEqual(
            memories[0]["tags"],
            ["neural network", "diagram", "hidden layers"],
        )

        memory = self.backend.get_memory(path=str(Path(self.image_path).expanduser().resolve()), collection="test")
        self.assertIsNotNone(memory)
        self.assertEqual(
            memory["tags"],
            ["neural network", "diagram", "hidden layers"],
        )

    def test_generated_media_tags_strip_fenced_json(self):
        embedder = CaptioningEmbedder()

        def fenced_json(_prompt: str, max_tokens: int = 60) -> str:
            return '```json\n["diagram", "hidden layers", "neural network"]\n```'

        embedder.generate_text = fenced_json

        self.backend.index_image(
            path=self.image_path,
            collection="test",
            embed_func=embedder,
            caption_media=True,
        )

        rows = self.backend._embeddings_table.search().where("content_type = 'image'").to_list()
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            json.loads(rows[0].get("tags") or "[]"),
            ["diagram", "hidden layers", "neural network"],
        )

    def test_generated_media_tags_extract_fenced_json_from_wrapped_text(self):
        embedder = CaptioningEmbedder()

        def wrapped_fenced_json(_prompt: str, max_tokens: int = 60) -> str:
            return (
                "Here are the tags:\n"
                "```json\n"
                '["diagram", "hidden layers", "neural network"]\n'
                "```"
            )

        embedder.generate_text = wrapped_fenced_json

        self.backend.index_image(
            path=self.image_path,
            collection="test",
            embed_func=embedder,
            caption_media=True,
        )

        rows = self.backend._embeddings_table.search().where("content_type = 'image'").to_list()
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            json.loads(rows[0].get("tags") or "[]"),
            ["diagram", "hidden layers", "neural network"],
        )

    def test_index_video_keeps_parent_memory_and_links_children(self):
        embedder = CaptioningEmbedder()
        logical_path = str(Path(self.video_path).expanduser().resolve())
        expected_tags = ["technical explainer", "architecture diagram", "presentation"]
        fake_artifacts = SimpleNamespace(
            frames=[
                SimpleNamespace(
                    image_path=self.frame_path,
                    logical_path=f"{logical_path}::frame:0001@0.00s",
                    title="frame 1",
                    timestamp_seconds=0.0,
                )
            ],
            transcripts=[
                SimpleNamespace(
                    logical_path=f"{logical_path}::transcript:0001",
                    text="The presenter explains the architecture diagram.",
                )
            ],
            duration_seconds=4.2,
            transcript_path=None,
            ffmpeg_available=True,
        )

        def failing_video_embed(_path: str):
            raise RuntimeError("video embed failed")

        with patch("recallforge.storage.indexing_ops.extract_video_artifacts", return_value=fake_artifacts):
            self.backend.index_video(
                path=self.video_path,
                collection="test",
                embed_text_func=mock_embed,
                embed_image_func=embedder,
                embed_video_func=failing_video_embed,
                caption_media=True,
            )

        root_doc = self.backend.find_document("test", logical_path)
        self.assertIsNotNone(root_doc)
        self.assertEqual(root_doc.memory_role, "root")
        self.assertEqual(root_doc.memory_root_path, logical_path)

        child_rows = self.backend._documents_table.search().where(
            f"collection = 'test' AND file_path LIKE '{logical_path}::%'"
        ).to_list()
        self.assertGreaterEqual(len(child_rows), 2)
        for row in child_rows:
            self.assertEqual(row.get("memory_id"), root_doc.memory_id)
            self.assertEqual(row.get("memory_role"), "child")
            self.assertEqual(row.get("memory_root_path"), logical_path)

        child_embedding_rows = self.backend._embeddings_table.search().where(
            f"collection = 'test' AND file_path LIKE '{logical_path}::%'"
        ).to_list()
        self.assertGreaterEqual(len(child_embedding_rows), 2)
        for row in child_embedding_rows:
            self.assertEqual(json.loads(row.get("tags") or "[]"), expected_tags)

        root_embedding_rows = self.backend._embeddings_table.search().where(
            f"collection = 'test' AND file_path = '{logical_path}'"
        ).to_list()
        self.assertEqual(len(root_embedding_rows), 1)
        self.assertIn("Technical explainer video", root_embedding_rows[0].get("text_body") or "")
        self.assertEqual(json.loads(root_embedding_rows[0].get("tags") or "[]"), expected_tags)

        memories = self.backend.list_memories(collection="test", limit=10)
        self.assertEqual(len(memories), 1)
        self.assertTrue((memories[0].get("summary") or "").startswith("Technical explainer video"))

        memory = self.backend.get_memory(path=logical_path, collection="test")
        self.assertIsNotNone(memory)
        self.assertEqual(memory["tags"], expected_tags)
        self.assertTrue((memory.get("summary") or "").startswith("Technical explainer video"))

    def test_index_document_file_creates_root_memory_and_links_sections(self):
        document_path = os.path.join(self.temp_dir, "report.pdf")
        logical_path = str(Path(document_path).expanduser().resolve())
        with open(document_path, "wb") as f:
            f.write(b"%PDF-1.4 mock")

        fake_artifacts = SimpleNamespace(
            document_type="pdf",
            extractor="unit-test",
            sections=[
                SimpleNamespace(
                    logical_path=f"{logical_path}::section:0001",
                    title="report section 1",
                    text="Budget and launch notes for the memory product.",
                    section_type="section",
                    index=1,
                    content_type="text",
                    image_path=None,
                )
            ],
        )

        with patch("recallforge.storage.indexing_ops.extract_document_artifacts", return_value=fake_artifacts):
            self.backend.index_document_file(
                path=document_path,
                collection="test",
                embed_func=mock_embed,
                model="mock-embedder",
            )

        root_doc = self.backend.find_document("test", logical_path)
        self.assertIsNotNone(root_doc)
        self.assertEqual(root_doc.memory_role, "root")
        self.assertEqual(root_doc.memory_root_path, logical_path)

        child_doc = self.backend.find_document("test", f"{logical_path}::section:0001")
        self.assertIsNotNone(child_doc)
        self.assertEqual(child_doc.memory_id, root_doc.memory_id)
        self.assertEqual(child_doc.memory_role, "child")
        self.assertEqual(child_doc.memory_root_path, logical_path)

        root_embedding_rows = self.backend._embeddings_table.search().where(
            f"collection = 'test' AND file_path = '{logical_path}'"
        ).to_list()
        self.assertEqual(len(root_embedding_rows), 1)
        self.assertIn("Budget and launch notes", root_embedding_rows[0].get("text_body") or "")

        memory = self.backend.get_memory(path=logical_path, collection="test")
        self.assertIsNotNone(memory)
        self.assertTrue((memory.get("summary") or "").startswith("Budget and launch notes"))

    def test_index_document_file_continues_when_root_summary_embedding_fails(self):
        document_path = os.path.join(self.temp_dir, "notes.pdf")
        logical_path = str(Path(document_path).expanduser().resolve())
        with open(document_path, "wb") as f:
            f.write(b"%PDF-1.4 mock")

        fake_artifacts = SimpleNamespace(
            document_type="pdf",
            extractor="unit-test",
            sections=[
                SimpleNamespace(
                    logical_path=f"{logical_path}::section:0001",
                    title="notes section 1",
                    text="First section about memory retrieval.",
                    section_type="section",
                    index=1,
                    content_type="text",
                    image_path=None,
                ),
                SimpleNamespace(
                    logical_path=f"{logical_path}::section:0002",
                    title="notes section 2",
                    text="Second section about multimodal evidence.",
                    section_type="section",
                    index=2,
                    content_type="text",
                    image_path=None,
                ),
            ],
        )

        def embed_except_summary(text: str):
            if "First section about memory retrieval." in text and "Second section about multimodal evidence." in text:
                raise RuntimeError("summary embed failed")
            return mock_embed(text)

        with patch("recallforge.storage.indexing_ops.extract_document_artifacts", return_value=fake_artifacts):
            result = self.backend.index_document_file(
                path=document_path,
                collection="test",
                embed_func=embed_except_summary,
                model="mock-embedder",
            )

        self.assertEqual(result["indexed_sections"], 2)
        child_doc = self.backend.find_document("test", f"{logical_path}::section:0001")
        self.assertIsNotNone(child_doc)

        root_embedding_rows = self.backend._embeddings_table.search().where(
            f"collection = 'test' AND file_path = '{logical_path}'"
        ).to_list()
        self.assertEqual(len(root_embedding_rows), 0)

    def test_index_document_file_preserves_ocr_text_for_image_only_pages(self):
        document_path = os.path.join(self.temp_dir, "scan.pdf")
        logical_path = str(Path(document_path).expanduser().resolve())
        with open(document_path, "wb") as f:
            f.write(b"%PDF-1.4 mock")

        fake_artifacts = SimpleNamespace(
            document_type="pdf",
            extractor="unit-test",
            sections=[
                SimpleNamespace(
                    logical_path=f"{logical_path}::page:0001",
                    title="scan page 1",
                    text="Scanned invoice total due on receipt.",
                    section_type="page",
                    index=1,
                    content_type="image",
                    image_path=self.frame_path,
                )
            ],
        )

        with patch("recallforge.storage.indexing_ops.extract_document_artifacts", return_value=fake_artifacts):
            result = self.backend.index_document_file(
                path=document_path,
                collection="test",
                embed_func=mock_embed,
                embed_image_func=mock_embed,
                model="mock-embedder",
            )

        self.assertEqual(result["indexed_images"], 1)
        self.assertEqual(result["indexed_sections"], 1)

        ocr_doc = self.backend.find_document("test", f"{logical_path}::page:0001::ocr")
        self.assertIsNotNone(ocr_doc)
        self.assertEqual(ocr_doc.memory_role, "child")
        self.assertEqual(ocr_doc.memory_root_path, logical_path)

        ocr_rows = self.backend._embeddings_table.search().where(
            f"collection = 'test' AND file_path = '{logical_path}::page:0001::ocr'"
        ).to_list()
        self.assertGreaterEqual(len(ocr_rows), 1)
        self.assertIn("Scanned invoice total due", ocr_rows[0].get("text_body") or "")

    def test_ingest_caption_media_disabled_skips_image_caption(self):
        embedder = CaptioningEmbedder()
        self.backend.ingest(
            collection="test",
            text=None,
            path=None,
            file_path=self.image_path,
            folder_path=None,
            recursive=True,
            content_types=["image"],
            include_globs=None,
            exclude_globs=None,
            embed_text_func=mock_embed,
            embed_image_func=embedder,
            embed_video_func=embedder,
            model="mock-embedder",
            caption_media=False,
        )

        rows = self.backend._embeddings_table.search().where("content_type = 'image'").to_list()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("text_body"), "")


class TestAudioIngest(unittest.TestCase):
    """Tests for transcript-first audio memories."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="recallforge-test-audio-")
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_index_audio_creates_root_memory_and_transcript_children(self):
        audio_path = Path(self.temp_dir) / "planning.wav"
        audio_path.write_bytes(b"fake audio bytes")
        audio_path.with_suffix("").with_name("planning.transcript.json").write_text(
            json.dumps(
                {
                    "segments": [
                        {"start": 0.0, "end": 2.0, "text": "The team reviews the roadmap decisions."},
                        {"start": 2.0, "end": 5.0, "text": "They assign latency budget follow ups."},
                    ]
                }
            ),
            encoding="utf-8",
        )

        logical_path = str(audio_path.expanduser().resolve())
        summary = self.backend.index_audio(
            path=str(audio_path),
            collection="test",
            embed_text_func=mock_embed,
            model="mock-embedder",
        )

        self.assertTrue(summary["success"])
        self.assertEqual(summary["indexed_transcripts"], 2)

        root_doc = self.backend.find_document("test", logical_path)
        self.assertIsNotNone(root_doc)
        self.assertEqual(root_doc.content_type, "audio")
        self.assertEqual(root_doc.memory_role, "root")
        self.assertEqual(root_doc.memory_root_path, logical_path)

        child_rows = self.backend._embeddings_table.search().where(
            f"collection = 'test' AND file_path LIKE '{logical_path}::transcript:%'"
        ).to_list()
        self.assertEqual(len(child_rows), 2)
        for row in child_rows:
            self.assertEqual(row.get("memory_role"), "child")
            self.assertEqual(row.get("memory_root_path"), logical_path)

        self.backend.rebuild_fts_index()
        audio_results = self.backend.search_fts(
            "roadmap decisions",
            limit=5,
            collection="test",
            content_type="audio",
        )
        self.assertTrue(any(result.display_path == f"test/{logical_path}" for result in audio_results))

        transcript_results = self.backend.search_fts(
            "latency budget",
            limit=5,
            collection="test",
            content_type="text",
        )
        self.assertTrue(any("::transcript:" in result.display_path for result in transcript_results))

        memories = self.backend.list_memories(collection="test", limit=10)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["content_type"], "audio")

    def test_index_audio_requires_transcript_sidecar(self):
        audio_path = Path(self.temp_dir) / "silent.wav"
        audio_path.write_bytes(b"fake audio bytes")

        with self.assertRaisesRegex(ValueError, "transcript-first"):
            self.backend.index_audio(
                path=str(audio_path),
                collection="test",
                embed_text_func=mock_embed,
                model="mock-embedder",
            )


if __name__ == "__main__":
    unittest.main()
