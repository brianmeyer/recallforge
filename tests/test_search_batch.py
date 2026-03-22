"""
test_search_batch.py - Unit tests for search_batch parallel multi-query with RRF merge.

Tests: batch query normalization, parallel execution, RRF merge, deduplication.
"""

import os
import sys
import time
import unittest
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recallforge.search import BatchQuery, BatchSearchResult, search_batch, HybridSearcher
from recallforge.storage.base import SearchResult
from recallforge.backends.base import ModelBackend, BackendInfo


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _make_search_result(filepath: str, score: float = 0.9, source: str = "hybrid",
                       content_type: str = "text") -> SearchResult:
    return SearchResult(
        filepath=filepath,
        display_path=filepath,
        title=os.path.basename(filepath),
        context=None,
        hash=filepath.replace("/", "_"),
        docid=filepath[:6],
        collection="test",
        modified_at="2026-01-01",
        body_length=100,
        score=score,
        source=source,
        content_type=content_type,
        chunk_pos=0,
        body=f"Content of {filepath}",
    )


class StubBackend(ModelBackend):
    """Minimal ModelBackend stub for unit tests."""

    def __init__(self, mode: str = "hybrid"):
        self._mode = mode

    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(2048, dtype=np.float32) / np.sqrt(2048)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed_text(t) for t in texts]

    def embed_image(self, image_path: str) -> np.ndarray:
        return self.embed_text(image_path)

    def embed_images(self, image_paths: List[str]) -> List[np.ndarray]:
        return [self.embed_image(p) for p in image_paths]

    def embed_video(self, video_path: str) -> np.ndarray:
        return self.embed_text(video_path)

    def embed_videos(self, video_paths: List[str]) -> List[np.ndarray]:
        return [self.embed_video(p) for p in video_paths]

    def rerank(self, query: str, documents: List[Dict[str, Any]], **kwargs) -> List[float]:
        return [0.9 - i * 0.05 for i in range(len(documents))]

    def warm_up(self) -> None:
        pass

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            name="stub",
            device="cpu",
            dtype="float32",
            embedder_loaded=True,
            reranker_loaded=True,
        )


class StubStorage:
    """Minimal StorageBackend stub for unit tests."""

    def __init__(self, fts_results=None, vec_results=None):
        self._fts_results = fts_results or []
        self._vec_results = vec_results or []
        self._search_count = 0

    def search_fts(self, query: str, limit: int = 20, **kwargs) -> List[SearchResult]:
        return list(self._fts_results[:limit])

    def search_vec(self, vector, limit: int = 20, **kwargs) -> List[SearchResult]:
        return list(self._vec_results[:limit])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBatchQuery(unittest.TestCase):
    """Test BatchQuery dataclass."""

    def test_batch_query_defaults(self):
        """BatchQuery should have sensible defaults."""
        q = BatchQuery(query="test query")
        self.assertEqual(q.query, "test query")
        self.assertIsNone(q.mode)
        self.assertIsNone(q.intent)
        self.assertEqual(q.weight, 1.0)

    def test_batch_query_custom(self):
        """BatchQuery should accept custom values."""
        q = BatchQuery(query="test", mode="fts", intent="exact_lookup", weight=2.5)
        self.assertEqual(q.query, "test")
        self.assertEqual(q.mode, "fts")
        self.assertEqual(q.intent, "exact_lookup")
        self.assertEqual(q.weight, 2.5)


class TestBatchSearchResult(unittest.TestCase):
    """Test BatchSearchResult dataclass."""

    def test_batch_search_result_fields(self):
        """BatchSearchResult should have all required fields."""
        r = BatchSearchResult(
            filepath="doc.md",
            display_path="doc.md",
            title="doc.md",
            context=None,
            hash="abc123",
            docid="abc",
            collection="test",
            modified_at="2026-01-01",
            body_length=100,
            body="content",
            score=0.85,
            source="0,2",
            query_scores={0: 0.9, 2: 0.75},
        )
        self.assertEqual(r.filepath, "doc.md")
        self.assertEqual(r.score, 0.85)
        self.assertEqual(r.source, "0,2")
        self.assertEqual(r.query_scores[0], 0.9)


class TestSearchBatchEmpty(unittest.TestCase):
    """Test search_batch with empty input."""

    def test_empty_queries_returns_empty(self):
        """search_batch with empty queries list should return empty results."""
        backend = StubBackend()
        storage = StubStorage()
        results = search_batch([], backend=backend, storage=storage)
        self.assertEqual(results, [])


class TestSearchBatchSingleQuery(unittest.TestCase):
    """Test search_batch with a single query."""

    def test_single_string_query(self):
        """search_batch should handle a single string query."""
        backend = StubBackend()
        vec_results = [
            _make_search_result("doc1.md", 0.9, "hybrid"),
            _make_search_result("doc2.md", 0.8, "hybrid"),
        ]
        fts_results = [
            _make_search_result("doc1.md", 0.85, "fts"),
            _make_search_result("doc3.md", 0.7, "fts"),
        ]
        storage = StubStorage(fts_results=fts_results, vec_results=vec_results)

        # Patch HybridSearcher to return predictable results
        with patch.object(HybridSearcher, 'search') as mock_search:
            mock_search.return_value = [
                type('HybridResult', (), {
                    'filepath': 'doc1.md',
                    'display_path': 'doc1.md',
                    'title': 'doc1.md',
                    'context': None,
                    'hash': 'abc',
                    'docid': 'doc1',
                    'collection': 'test',
                    'modified_at': '2026-01-01',
                    'body_length': 100,
                    'body': 'content',
                    'score': 0.9,
                    'source': 'hybrid',
                })
            ]
            results = search_batch(["test query"], backend=backend, storage=storage, limit=10)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].filepath, "doc1.md")


class TestSearchBatchMultipleQueries(unittest.TestCase):
    """Test search_batch with multiple queries."""

    def test_multiple_queries_merge_results(self):
        """search_batch should merge results from multiple queries."""
        backend = StubBackend(mode="embed")
        storage = StubStorage()

        # Create a mock HybridSearcher that returns different results for each query
        call_count = [0]

        def mock_search(self, query):
            call_count[0] += 1
            if call_count[0] == 1:
                # First query finds doc1 and doc2
                return [
                    type('HybridResult', (), {
                        'filepath': 'doc1.md',
                        'display_path': 'doc1.md',
                        'title': 'doc1.md',
                        'context': None,
                        'hash': 'h1',
                        'docid': 'd1',
                        'collection': 'test',
                        'modified_at': '2026-01-01',
                        'body_length': 100,
                        'body': 'content1',
                        'score': 0.9,
                        'source': 'hybrid',
                    }),
                    type('HybridResult', (), {
                        'filepath': 'doc2.md',
                        'display_path': 'doc2.md',
                        'title': 'doc2.md',
                        'context': None,
                        'hash': 'h2',
                        'docid': 'd2',
                        'collection': 'test',
                        'modified_at': '2026-01-01',
                        'body_length': 100,
                        'body': 'content2',
                        'score': 0.8,
                        'source': 'hybrid',
                    }),
                ]
            else:
                # Second query finds doc2 and doc3 (overlap on doc2)
                return [
                    type('HybridResult', (), {
                        'filepath': 'doc2.md',
                        'display_path': 'doc2.md',
                        'title': 'doc2.md',
                        'context': None,
                        'hash': 'h2',
                        'docid': 'd2',
                        'collection': 'test',
                        'modified_at': '2026-01-01',
                        'body_length': 100,
                        'body': 'content2',
                        'score': 0.95,  # Higher score in second query
                        'source': 'hybrid',
                    }),
                    type('HybridResult', (), {
                        'filepath': 'doc3.md',
                        'display_path': 'doc3.md',
                        'title': 'doc3.md',
                        'context': None,
                        'hash': 'h3',
                        'docid': 'd3',
                        'collection': 'test',
                        'modified_at': '2026-01-01',
                        'body_length': 100,
                        'body': 'content3',
                        'score': 0.7,
                        'source': 'hybrid',
                    }),
                ]

        with patch.object(HybridSearcher, '__init__', lambda self, **kwargs: None):
            with patch.object(HybridSearcher, 'search', mock_search):
                results = search_batch(
                    ["query one", "query two"],
                    backend=backend,
                    storage=storage,
                    limit=10,
                )

                # Should have 3 unique documents
                filepaths = [r.filepath for r in results]
                self.assertIn("doc1.md", filepaths)
                self.assertIn("doc2.md", filepaths)
                self.assertIn("doc3.md", filepaths)

                # doc2 should appear in both queries (source should show both indices)
                doc2_result = next((r for r in results if r.filepath == "doc2.md"), None)
                self.assertIsNotNone(doc2_result)
                self.assertIn("0", doc2_result.source)
                self.assertIn("1", doc2_result.source)

    def test_limit_is_respected(self):
        """search_batch should respect the limit parameter."""
        backend = StubBackend(mode="embed")
        storage = StubStorage()

        def mock_search(self, query):
            # Return many results
            return [
                type('HybridResult', (), {
                    'filepath': f'doc{i}.md',
                    'display_path': f'doc{i}.md',
                    'title': f'doc{i}.md',
                    'context': None,
                    'hash': f'h{i}',
                    'docid': f'd{i}',
                    'collection': 'test',
                    'modified_at': '2026-01-01',
                    'body_length': 100,
                    'body': f'content{i}',
                    'score': 0.9 - i * 0.1,
                    'source': 'hybrid',
                })
                for i in range(20)
            ]

        with patch.object(HybridSearcher, '__init__', lambda self, **kwargs: None):
            with patch.object(HybridSearcher, 'search', mock_search):
                results = search_batch(
                    ["single query"],
                    backend=backend,
                    storage=storage,
                    limit=5,
                )
                self.assertLessEqual(len(results), 5)


class TestSearchBatchModes(unittest.TestCase):
    """Test search_batch with different search modes."""

    def test_fts_mode(self):
        """search_batch should support 'fts' mode."""
        backend = StubBackend()
        fts_results = [
            _make_search_result("doc1.md", 0.9, "fts"),
            _make_search_result("doc2.md", 0.8, "fts"),
        ]
        storage = StubStorage(fts_results=fts_results)

        queries = [BatchQuery(query="test", mode="fts")]
        results = search_batch(queries, backend=backend, storage=storage, limit=10)
        self.assertEqual(len(results), 2)

    def test_vec_mode(self):
        """search_batch should support 'vec' mode."""
        backend = StubBackend()
        vec_results = [
            _make_search_result("doc1.md", 0.9, "vec"),
            _make_search_result("doc2.md", 0.8, "vec"),
        ]
        storage = StubStorage(vec_results=vec_results)

        queries = [BatchQuery(query="test", mode="vec")]
        results = search_batch(queries, backend=backend, storage=storage, limit=10)
        self.assertEqual(len(results), 2)

    def test_hybrid_mode(self):
        """search_batch should support 'hybrid' mode (default)."""
        backend = StubBackend()
        storage = StubStorage()

        with patch.object(HybridSearcher, '__init__', lambda self, **kwargs: None):
            with patch.object(HybridSearcher, 'search') as mock_search:
                mock_search.return_value = [
                    type('HybridResult', (), {
                        'filepath': 'doc.md',
                        'display_path': 'doc.md',
                        'title': 'doc.md',
                        'context': None,
                        'hash': 'h',
                        'docid': 'd',
                        'collection': 'test',
                        'modified_at': '2026-01-01',
                        'body_length': 100,
                        'body': 'content',
                        'score': 0.9,
                        'source': 'hybrid',
                    })
                ]
                queries = [BatchQuery(query="test", mode="hybrid")]
                results = search_batch(queries, backend=backend, storage=storage, limit=10)
                self.assertEqual(len(results), 1)


class TestSearchBatchWeights(unittest.TestCase):
    """Test search_batch with query weights."""

    def test_weighted_queries(self):
        """search_batch should apply weights to queries."""
        backend = StubBackend()
        storage = StubStorage()

        results_list = [
            [
                type('HybridResult', (), {
                    'filepath': 'doc1.md',
                    'display_path': 'doc1.md',
                    'title': 'doc1.md',
                    'context': None,
                    'hash': 'h1',
                    'docid': 'd1',
                    'collection': 'test',
                    'modified_at': '2026-01-01',
                    'body_length': 100,
                    'body': 'content1',
                    'score': 0.9,
                    'source': 'hybrid',
                }),
            ],
            [
                type('HybridResult', (), {
                    'filepath': 'doc2.md',
                    'display_path': 'doc2.md',
                    'title': 'doc2.md',
                    'context': None,
                    'hash': 'h2',
                    'docid': 'd2',
                    'collection': 'test',
                    'modified_at': '2026-01-01',
                    'body_length': 100,
                    'body': 'content2',
                    'score': 0.95,
                    'source': 'hybrid',
                }),
            ],
        ]

        call_idx = [0]

        def mock_search(self, query):
            idx = call_idx[0]
            call_idx[0] += 1
            return results_list[idx]

        with patch.object(HybridSearcher, '__init__', lambda self, **kwargs: None):
            with patch.object(HybridSearcher, 'search', mock_search):
                queries = [
                    BatchQuery(query="q1", weight=2.0),  # Higher weight
                    BatchQuery(query="q2", weight=1.0),
                ]
                results = search_batch(queries, backend=backend, storage=storage, limit=10)

                # Both docs should appear
                self.assertEqual(len(results), 2)


class TestSearchBatchDeduplication(unittest.TestCase):
    """Test search_batch deduplication behavior."""

    def test_same_document_different_queries(self):
        """Same document from different queries should be deduplicated with merged scores."""
        backend = StubBackend()
        storage = StubStorage()

        results_list = [
            [
                type('HybridResult', (), {
                    'filepath': 'shared.md',
                    'display_path': 'shared.md',
                    'title': 'shared.md',
                    'context': None,
                    'hash': 'h1',
                    'docid': 'd1',
                    'collection': 'test',
                    'modified_at': '2026-01-01',
                    'body_length': 100,
                    'body': 'shared content',
                    'score': 0.8,
                    'source': 'hybrid',
                }),
            ],
            [
                type('HybridResult', (), {
                    'filepath': 'shared.md',
                    'display_path': 'shared.md',
                    'title': 'shared.md',
                    'context': None,
                    'hash': 'h1',
                    'docid': 'd1',
                    'collection': 'test',
                    'modified_at': '2026-01-01',
                    'body_length': 100,
                    'body': 'shared content',
                    'score': 0.9,
                    'source': 'hybrid',
                }),
            ],
        ]

        call_idx = [0]

        def mock_search(self, query):
            idx = call_idx[0]
            call_idx[0] += 1
            return results_list[idx]

        with patch.object(HybridSearcher, '__init__', lambda self, **kwargs: None):
            with patch.object(HybridSearcher, 'search', mock_search):
                results = search_batch(
                    ["query one", "query two"],
                    backend=backend,
                    storage=storage,
                    limit=10,
                )

                # Should have exactly 1 result (deduplicated)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].filepath, "shared.md")

                # Source should show both query indices
                self.assertIn("0", results[0].source)
                self.assertIn("1", results[0].source)

                # query_scores should have both queries
                self.assertIn(0, results[0].query_scores)
                self.assertIn(1, results[0].query_scores)

    def test_same_document_merges_tags_deterministically(self):
        """Duplicate hits should merge tag sets in stable first-seen order."""
        backend = StubBackend()
        storage = StubStorage()

        results_list = [
            [
                type('HybridResult', (), {
                    'filepath': 'shared.md',
                    'display_path': 'shared.md',
                    'title': 'shared.md',
                    'context': None,
                    'hash': 'h1',
                    'docid': 'd1',
                    'collection': 'test',
                    'modified_at': '2026-01-01',
                    'body_length': 100,
                    'body': 'shared content',
                    'score': 0.8,
                    'source': 'hybrid',
                    'tags': ['alpha', 'shared'],
                }),
            ],
            [
                type('HybridResult', (), {
                    'filepath': 'shared.md',
                    'display_path': 'shared.md',
                    'title': 'shared.md',
                    'context': None,
                    'hash': 'h1',
                    'docid': 'd1',
                    'collection': 'test',
                    'modified_at': '2026-01-01',
                    'body_length': 100,
                    'body': 'shared content',
                    'score': 0.9,
                    'source': 'hybrid',
                    'tags': ['shared', 'beta'],
                }),
            ],
        ]

        def mock_search(self, query):
            if query == "query one":
                time.sleep(0.05)
                return results_list[0]
            return results_list[1]

        with patch.object(HybridSearcher, '__init__', lambda self, **kwargs: None):
            with patch.object(HybridSearcher, 'search', mock_search):
                results = search_batch(
                    ["query one", "query two"],
                    backend=backend,
                    storage=storage,
                    limit=10,
                )

                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].tags, ["alpha", "shared", "beta"])


if __name__ == "__main__":
    unittest.main()
