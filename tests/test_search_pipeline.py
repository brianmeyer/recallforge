"""
test_search_pipeline.py - Unit tests for HybridSearcher pipeline.

All backends and storage are mocked — NO real model inference, NO real LanceDB.
Tests: RRF fusion, score blending, tiered mode branching, strong-signal detection.
"""

import os
import sys
import unittest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch, call

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recallforge.search import HybridSearcher, HybridResult, hybrid_query
from recallforge.storage.base import SearchResult
from recallforge.backends.base import ModelBackend, BackendInfo


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _make_search_result(filepath: str, score: float = 0.9, source: str = "fts",
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

    def __init__(self, mode: str = "full"):
        self._mode = mode

    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(2048, dtype=np.float32) / np.sqrt(2048)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed_text(t) for t in texts])

    def embed_image(self, image_path: str) -> np.ndarray:
        return self.embed_text(image_path)

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        return self.embed_texts(image_paths)

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        # Return descending scores
        return [0.9 - i * 0.05 for i in range(len(documents))]

    def expand_query(self, query: str) -> Dict[str, str]:
        return {
            "lex": query + " keywords",
            "vec": query + " semantic",
            "hyde": "hypothetical document about " + query,
        }

    def warm_up(self) -> None:
        pass

    def get_info(self) -> BackendInfo:
        return BackendInfo(name="stub", device="cpu", dtype="float32",
                          embedder_loaded=True, reranker_loaded=True,
                          expander_loaded=True)


class StubStorage:
    """Minimal StorageBackend stub for unit tests."""

    def __init__(self, fts_results=None, vec_results=None):
        self._fts_results = fts_results or []
        self._vec_results = vec_results or []

    def search_fts(self, query: str, limit: int = 20,
                   collection=None, content_type=None) -> List[SearchResult]:
        return list(self._fts_results[:limit])

    def search_vec(self, vector, limit: int = 20,
                   collection=None, content_type=None) -> List[SearchResult]:
        return list(self._vec_results[:limit])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHybridSearcherInit(unittest.TestCase):
    """Test HybridSearcher construction."""

    def test_creates_with_injected_deps(self):
        backend = StubBackend()
        storage = StubStorage()
        searcher = HybridSearcher(backend=backend, storage=storage, limit=5)
        self.assertEqual(searcher.limit, 5)
        self.assertIs(searcher.backend, backend)
        self.assertIs(searcher.storage, storage)


class TestBM25Probe(unittest.TestCase):
    def test_bm25_probe_delegates_to_storage(self):
        results = [_make_search_result("file1.md", 0.9)]
        backend = StubBackend()
        storage = StubStorage(fts_results=results)
        searcher = HybridSearcher(backend=backend, storage=storage)

        probed = searcher._bm25_probe("query")
        self.assertEqual(len(probed), 1)
        self.assertEqual(probed[0].filepath, "file1.md")


class TestVectorSearch(unittest.TestCase):
    def test_vector_search_delegates_to_storage(self):
        results = [_make_search_result("vec1.md", 0.85, "vec")]
        backend = StubBackend()
        storage = StubStorage(vec_results=results)
        searcher = HybridSearcher(backend=backend, storage=storage)

        found = searcher._vector_search("how do agents remember")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].filepath, "vec1.md")


class TestStrongSignalDetected(unittest.TestCase):
    def test_no_results_no_signal(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage())
        self.assertFalse(searcher._strong_signal_detected([]))

    def test_single_result_no_signal(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage())
        self.assertFalse(searcher._strong_signal_detected([_make_search_result("a.md", 0.99)]))

    def test_clear_winner_is_strong(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage())
        results = [
            _make_search_result("a.md", 0.95),
            _make_search_result("b.md", 0.60),
        ]
        self.assertTrue(searcher._strong_signal_detected(results))

    def test_close_scores_not_strong(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage())
        results = [
            _make_search_result("a.md", 0.70),
            _make_search_result("b.md", 0.68),
        ]
        self.assertFalse(searcher._strong_signal_detected(results))


class TestRRFFusion(unittest.TestCase):
    def _make_searcher(self):
        return HybridSearcher(backend=StubBackend(), storage=StubStorage(), rrf_k=60)

    def test_rrf_deduplicates(self):
        """Same filepath appearing in two lists should be one result."""
        searcher = self._make_searcher()
        r1 = _make_search_result("shared.md", 0.9, "fts")
        r2 = _make_search_result("shared.md", 0.8, "vec")
        all_results = {"original_fts": [r1], "original_vec": [r2]}
        fused = searcher._reciprocal_rank_fusion(all_results)
        filepaths = [r.filepath for r in fused]
        self.assertEqual(len(filepaths), len(set(filepaths)))

    def test_rrf_higher_for_top_ranked(self):
        """Top-ranked item in both lists should have highest RRF score."""
        searcher = self._make_searcher()
        r_top = _make_search_result("top.md", 0.9)
        r_low = _make_search_result("low.md", 0.1)
        all_results = {
            "fts": [r_top, r_low],
            "vec": [r_top, r_low],
        }
        fused = searcher._reciprocal_rank_fusion(all_results)
        self.assertEqual(fused[0].filepath, "top.md")

    def test_rrf_empty_lists(self):
        searcher = self._make_searcher()
        fused = searcher._reciprocal_rank_fusion({})
        self.assertEqual(fused, [])

    def test_rrf_scores_positive(self):
        searcher = self._make_searcher()
        results = [_make_search_result(f"file{i}.md", 0.9 - i * 0.1) for i in range(5)]
        fused = searcher._reciprocal_rank_fusion({"list": results})
        for r in fused:
            self.assertGreater(r.score, 0)


class TestSelectBestChunk(unittest.TestCase):
    def test_returns_dict_with_text_and_filepath(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage())
        r = _make_search_result("doc.md", 0.9)
        chunk = searcher._select_best_chunk(r)
        self.assertIn("text", chunk)
        self.assertIn("filepath", chunk)
        self.assertEqual(chunk["filepath"], "doc.md")

    def test_uses_body_when_available(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage())
        r = _make_search_result("doc.md")
        r.body = "The actual body"
        chunk = searcher._select_best_chunk(r)
        self.assertEqual(chunk["text"], "The actual body")


class TestRerankCandidates(unittest.TestCase):
    def test_returns_filepath_score_dict(self):
        backend = StubBackend(mode="hybrid")
        searcher = HybridSearcher(backend=backend, storage=StubStorage())
        candidates = [
            _make_search_result("a.md", 0.9),
            _make_search_result("b.md", 0.8),
        ]
        scores = searcher._rerank_candidates(candidates, "test query")
        self.assertIn("a.md", scores)
        self.assertIn("b.md", scores)
        for s in scores.values():
            self.assertIsInstance(s, float)

    def test_embed_mode_returns_default_score(self):
        backend = StubBackend(mode="embed")
        searcher = HybridSearcher(backend=backend, storage=StubStorage())
        candidates = [_make_search_result("doc.md", 0.9)]
        scores = searcher._rerank_candidates(candidates, "query")
        # embed mode: backend.needs_reranker() == False → default 0.5
        self.assertAlmostEqual(scores.get("doc.md", 0), 0.5)


class TestBlendScores(unittest.TestCase):
    def test_blend_returns_hybrid_results(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage())
        rrf = [
            _make_search_result("a.md", 0.9),
            _make_search_result("b.md", 0.5),
        ]
        rerank_scores = {"a.md": 0.85, "b.md": 0.40}
        blended = searcher._blend_scores(rrf, rerank_scores)
        self.assertIsInstance(blended[0], HybridResult)

    def test_blend_sorted_descending(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage(), limit=10)
        rrf = [_make_search_result(f"file{i}.md", 1.0 / (i + 1)) for i in range(5)]
        rerank_scores = {r.filepath: 0.5 for r in rrf}
        blended = searcher._blend_scores(rrf, rerank_scores)
        scores = [r.score for r in blended]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_blend_respects_limit(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage(), limit=3)
        rrf = [_make_search_result(f"file{i}.md", 1.0) for i in range(10)]
        rerank_scores = {r.filepath: 0.5 for r in rrf}
        blended = searcher._blend_scores(rrf, rerank_scores)
        self.assertLessEqual(len(blended), 3)


class TestFullSearchPipeline(unittest.TestCase):
    """End-to-end search pipeline with mocked storage."""

    def setUp(self):
        self.fts_results = [
            _make_search_result("file1.md", 0.9),
            _make_search_result("file2.md", 0.7),
        ]
        self.vec_results = [
            _make_search_result("file2.md", 0.85, "vec"),
            _make_search_result("file3.md", 0.65, "vec"),
        ]

    def _make_searcher(self, mode: str = "full") -> HybridSearcher:
        backend = StubBackend(mode=mode)
        storage = StubStorage(
            fts_results=self.fts_results,
            vec_results=self.vec_results,
        )
        return HybridSearcher(backend=backend, storage=storage, limit=5)

    def test_search_returns_hybrid_results(self):
        searcher = self._make_searcher("hybrid")
        results = searcher.search("AI memory systems")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], HybridResult)

    def test_search_results_have_required_fields(self):
        searcher = self._make_searcher("hybrid")
        results = searcher.search("test query")
        r = results[0]
        self.assertIsNotNone(r.filepath)
        self.assertIsNotNone(r.score)
        self.assertIsNotNone(r.rrf_rank)
        self.assertIsNotNone(r.rerank_score)
        self.assertIsNotNone(r.source)

    def test_search_embed_mode(self):
        """embed mode: no reranking, still returns results."""
        searcher = self._make_searcher("embed")
        results = searcher.search("some query")
        self.assertGreater(len(results), 0)

    def test_search_full_mode(self):
        """full mode: expansion + reranking, still returns results."""
        searcher = self._make_searcher("full")
        results = searcher.search("complex query")
        self.assertGreater(len(results), 0)

    def test_search_scores_sorted_descending(self):
        searcher = self._make_searcher("hybrid")
        results = searcher.search("query")
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_deduplicates_filepaths(self):
        searcher = self._make_searcher("hybrid")
        results = searcher.search("shared content")
        filepaths = [r.filepath for r in results]
        self.assertEqual(len(filepaths), len(set(filepaths)))


class TestHybridQueryConvenience(unittest.TestCase):
    """Tests for the hybrid_query() convenience function."""

    def test_hybrid_query_uses_provided_deps(self):
        backend = StubBackend(mode="embed")
        storage = StubStorage(
            fts_results=[_make_search_result("a.md", 0.9)],
            vec_results=[_make_search_result("a.md", 0.8, "vec")],
        )
        results = hybrid_query("test", backend=backend, storage=storage, limit=5)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
