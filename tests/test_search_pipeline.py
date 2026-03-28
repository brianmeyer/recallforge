"""
test_search_pipeline.py - Unit tests for HybridSearcher pipeline.

All backends and storage are mocked — NO real model inference, NO real LanceDB.
Tests: RRF fusion, score blending, tiered mode branching.
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
                         content_type: str = "text", tags: Optional[List[str]] = None) -> SearchResult:
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
        tags=tags,
    )


class StubBackend(ModelBackend):
    """Minimal ModelBackend stub for unit tests."""

    def __init__(self, mode: str = "hybrid", generated_text_response: Optional[str] = None):
        self._mode = mode
        self._generated_text_response = generated_text_response

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

    def caption_image(self, image_path: str) -> str:
        return f"caption for {os.path.basename(image_path)}"

    def generate_text(self, prompt: str, max_tokens: int = 60) -> str:
        if self._generated_text_response is None:
            raise NotImplementedError("generate_text unavailable in stub")
        return self._generated_text_response

    def rerank(self, query: str, documents: List[Dict[str, Any]], **kwargs) -> List[float]:
        # Return descending scores
        return [0.9 - i * 0.05 for i in range(len(documents))]

    def warm_up(self) -> None:
        pass

    def get_info(self) -> BackendInfo:
        return BackendInfo(name="stub", device="cpu", dtype="float32",
                          embedder_loaded=True, reranker_loaded=True)


class StubStorage:
    """Minimal StorageBackend stub for unit tests."""

    def __init__(self, fts_results=None, vec_results=None):
        self._fts_results = fts_results or []
        self._vec_results = vec_results or []
        self.last_fts_query = None
        self.last_fts_content_type = None
        self.last_vec_content_type = None

    def search_fts(self, query: str, limit: int = 20,
                   collection=None, content_type=None,
                   user_id=None, session_id=None, project_id=None, profile=None) -> List[SearchResult]:
        self.last_fts_query = query
        self.last_fts_content_type = content_type
        return list(self._fts_results[:limit])

    def search_vec(self, vector, limit: int = 20,
                   collection=None, content_type=None,
                   user_id=None, session_id=None, project_id=None, profile=None) -> List[SearchResult]:
        self.last_vec_content_type = content_type
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

    def test_rerank_top_k_reads_env_override(self):
        backend = StubBackend()
        storage = StubStorage()
        with patch.dict(os.environ, {"RECALLFORGE_RERANK_TOP_K": "7"}):
            searcher = HybridSearcher(backend=backend, storage=storage)
        self.assertEqual(searcher.rerank_top_k, 7)

    def test_media_reranking_defaults_off(self):
        backend = StubBackend()
        storage = StubStorage()
        searcher = HybridSearcher(backend=backend, storage=storage)
        self.assertFalse(searcher.enable_media_reranking)

    def test_media_reranking_reads_env_override(self):
        backend = StubBackend()
        storage = StubStorage()
        with patch.dict(os.environ, {"RECALLFORGE_ENABLE_MEDIA_RERANKING": "true"}):
            searcher = HybridSearcher(backend=backend, storage=storage)
        self.assertTrue(searcher.enable_media_reranking)


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

    def test_search_video_delegates_to_storage(self):
        results = [_make_search_result("sample_video.mp4", 0.82, "vec", content_type="video")]
        backend = StubBackend()
        storage = StubStorage(vec_results=results)
        searcher = HybridSearcher(backend=backend, storage=storage)

        found = searcher.search_video("sample_video.mp4")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].filepath, "sample_video.mp4")

    def test_search_video_raises_when_backend_lacks_support(self):
        """Backend without callable embed_video should raise NotImplementedError."""

        class NoVideoBackend(StubBackend):
            embed_video = None  # Override to non-callable

        backend = NoVideoBackend()
        storage = StubStorage(vec_results=[])
        searcher = HybridSearcher(backend=backend, storage=storage)

        with self.assertRaises(NotImplementedError):
            searcher.search_video("sample_video.mp4")


class TestRRFFusion(unittest.TestCase):
    def _make_searcher(self):
        return HybridSearcher(backend=StubBackend(), storage=StubStorage(), rrf_k=60)

    def test_rrf_deduplicates(self):
        """Same filepath appearing in two lists should be one result."""
        searcher = self._make_searcher()
        r1 = _make_search_result("shared.md", 0.9, "fts")
        r2 = _make_search_result("shared.md", 0.8, "vec")
        all_results = {"original_fts": [r1], "original_vec": [r2]}
        fused, audit_info = searcher._reciprocal_rank_fusion(all_results)
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
        fused, audit_info = searcher._reciprocal_rank_fusion(all_results)
        self.assertEqual(fused[0].filepath, "top.md")

    def test_rrf_empty_lists(self):
        searcher = self._make_searcher()
        fused, audit_info = searcher._reciprocal_rank_fusion({})
        self.assertEqual(fused, [])

    def test_rrf_scores_positive(self):
        searcher = self._make_searcher()
        results = [_make_search_result(f"file{i}.md", 0.9 - i * 0.1) for i in range(5)]
        fused, audit_info = searcher._reciprocal_rank_fusion({"list": results})
        for r in fused:
            self.assertGreater(r.score, 0)

    def test_text_target_biases_default_rrf_toward_vector(self):
        searcher = HybridSearcher(
            backend=StubBackend(),
            storage=StubStorage(),
            rrf_k=60,
            content_type="text",
        )
        all_results = {
            "original_fts": [_make_search_result("fts.md", 0.9, "fts", "text")],
            "original_vec": [_make_search_result("vec.md", 0.9, "vec", "text")],
        }

        fused, audit_info = searcher._reciprocal_rank_fusion(all_results)

        vec_doc = next(r for r in fused if r.filepath == "vec.md")
        fts_doc = next(r for r in fused if r.filepath == "fts.md")
        weights = audit_info["vec.md"]["weights"]

        self.assertEqual(weights["original_vec"], 2.5)
        self.assertEqual(weights["original_fts"], 1.5)
        self.assertGreater(vec_doc.score, fts_doc.score)

    def test_image_target_biases_default_rrf_toward_bm25(self):
        searcher = HybridSearcher(
            backend=StubBackend(),
            storage=StubStorage(),
            rrf_k=60,
            content_type="image",
        )
        all_results = {
            "original_fts": [_make_search_result("fts.jpg", 0.9, "fts", "image")],
            "original_vec": [_make_search_result("vec.jpg", 0.9, "vec", "image")],
        }

        fused, audit_info = searcher._reciprocal_rank_fusion(all_results)

        fts_doc = next(r for r in fused if r.filepath == "fts.jpg")
        vec_doc = next(r for r in fused if r.filepath == "vec.jpg")
        weights = audit_info["fts.jpg"]["weights"]

        self.assertEqual(weights["original_fts"], 2.5)
        self.assertEqual(weights["original_vec"], 1.5)
        self.assertAlmostEqual(fts_doc.score, 2.5 / 61, places=5)
        self.assertGreater(fts_doc.score, 1.5 / 61)
        self.assertTrue(audit_info["vec.jpg"]["media_compensation"])


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
        scores, path = searcher._rerank_candidates(candidates, "test query")
        self.assertIn("a.md", scores)
        self.assertIn("b.md", scores)
        for s in scores.values():
            self.assertIsInstance(s, float)

    def test_reranks_only_top_k_candidates(self):
        backend = StubBackend(mode="hybrid")
        backend.rerank = MagicMock(return_value=[0.91, 0.72])
        searcher = HybridSearcher(backend=backend, storage=StubStorage(), rerank_top_k=2)
        candidates = [
            _make_search_result("a.md", 0.95),
            _make_search_result("b.md", 0.85),
            _make_search_result("c.md", 0.75),
            _make_search_result("d.md", 0.65),
        ]

        scores, path = searcher._rerank_candidates(candidates, "query")

        backend.rerank.assert_called_once()
        rerank_docs = backend.rerank.call_args[0][1]
        self.assertEqual([d["filepath"] for d in rerank_docs], ["a.md", "b.md"])
        self.assertEqual(scores["a.md"], 0.91)
        self.assertEqual(scores["b.md"], 0.72)
        self.assertEqual(scores["c.md"], 0.5)
        self.assertEqual(scores["d.md"], 0.5)

    def test_rerank_top_k_zero_skips_reranker(self):
        backend = StubBackend(mode="hybrid")
        backend.rerank = MagicMock()
        searcher = HybridSearcher(backend=backend, storage=StubStorage(), rerank_top_k=0)
        candidates = [_make_search_result("doc.md", 0.9)]

        scores, path = searcher._rerank_candidates(candidates, "query")

        backend.rerank.assert_not_called()
        self.assertEqual(scores["doc.md"], 0.5)

    def test_embed_mode_returns_default_score(self):
        backend = StubBackend(mode="embed")
        searcher = HybridSearcher(backend=backend, storage=StubStorage())
        candidates = [_make_search_result("doc.md", 0.9)]
        scores, path = searcher._rerank_candidates(candidates, "query")
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

    def test_text_only_rerank_blend_uses_conservative_weights(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage(), limit=12)
        rrf = [_make_search_result(f"doc{i}.md", 12 - i, "vec", "text") for i in range(12)]
        rerank_scores = {
            result.filepath: float(index)
            for index, result in enumerate(reversed(rrf), start=1)
        }

        blended = searcher._blend_scores(rrf, rerank_scores, reranker_path="text")

        top_rank = next(r for r in blended if r.filepath == "doc0.md")
        mid_rank = next(r for r in blended if r.filepath == "doc3.md")
        tail_rank = next(r for r in blended if r.filepath == "doc10.md")

        self.assertAlmostEqual(top_rank.audit.blend_weights["rrf"], 0.90)
        self.assertAlmostEqual(mid_rank.audit.blend_weights["rrf"], 0.85)
        self.assertAlmostEqual(tail_rank.audit.blend_weights["rrf"], 0.75)

    def test_memory_rollup_updates_audit_boost_and_final_score(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage(), limit=12)

        sibling_a = _make_search_result("memory_a.md", 0.9, "fts")
        sibling_b = _make_search_result("memory_b.md", 0.6, "vec")
        sibling_a.memory_id = "memory-1"
        sibling_b.memory_id = "memory-1"

        rrf = [sibling_a, sibling_b]
        rerank_scores = {
            "memory_a.md": 0.9,
            "memory_b.md": 0.2,
        }

        blended = searcher._blend_scores(rrf, rerank_scores)

        self.assertEqual(len(blended), 1)
        result = blended[0]
        self.assertEqual(result.memory_hit_count, 2)
        self.assertIsNotNone(result.audit)
        self.assertAlmostEqual(result.audit.memory_rollup_boost, 1.03)
        self.assertAlmostEqual(result.score, result.audit.final_blended_score)

    def test_memory_rollup_keeps_singletons_unboosted(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage(), limit=12)

        solo = _make_search_result("memory_solo.md", 0.9, "fts")
        solo.memory_id = "memory-2"

        blended = searcher._blend_scores([solo], {"memory_solo.md": 0.8})

        self.assertEqual(len(blended), 1)
        result = blended[0]
        self.assertEqual(result.memory_hit_count, 1)
        self.assertIsNotNone(result.audit)
        self.assertAlmostEqual(result.audit.memory_rollup_boost, 1.0)
        self.assertAlmostEqual(result.score, result.audit.final_blended_score)

    def test_memory_rollup_merges_tags_from_sibling_assets(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage(), limit=12)

        root = _make_search_result("memory_root.png", 0.9, "vec", "image", tags=["diagram"])
        child = _make_search_result(
            "memory_child.txt",
            0.8,
            "fts",
            "text",
            tags=["meeting notes", "diagram"],
        )
        root.memory_id = "memory-tags"
        child.memory_id = "memory-tags"

        rolled = searcher._roll_up_memory_hits(searcher._vector_results_to_hybrid([root, child]))

        self.assertEqual(len(rolled), 1)
        self.assertEqual(rolled[0].tags, ["diagram", "meeting notes"])

    def test_memory_rollup_preserves_collection_qualified_filepath(self):
        searcher = HybridSearcher(backend=StubBackend(), storage=StubStorage(), limit=12)

        root = _make_search_result("recallforge://alpha/memories/demo.mp4", 0.2, "vec", "video")
        child = _make_search_result(
            "recallforge://alpha/memories/demo.mp4::transcript:0001",
            0.9,
            "fts",
            "text",
        )
        sibling = _make_search_result(
            "recallforge://alpha/memories/demo.mp4::frame:0001",
            0.6,
            "vec",
            "image",
        )
        for item in (root, child, sibling):
            item.collection = "alpha"
            item.memory_id = "memory-evidence"
            item.memory_root_path = "memories/demo.mp4"
        root.memory_role = "root"
        child.memory_role = "child"
        sibling.memory_role = "child"
        root.body = "Canonical demo video summary."

        blended = searcher._blend_scores(
            [child, sibling, root],
            {
                child.filepath: 0.9,
                sibling.filepath: 0.6,
                root.filepath: 0.2,
            },
        )

        self.assertEqual(len(blended), 1)
        result = blended[0]
        self.assertEqual(result.filepath, "recallforge://alpha/memories/demo.mp4")
        self.assertEqual(result.display_path, "alpha/memories/demo.mp4")
        self.assertEqual(result.memory_root_path, "memories/demo.mp4")
        self.assertEqual(
            result.memory_primary_evidence_path,
            "recallforge://alpha/memories/demo.mp4::transcript:0001",
        )
        self.assertEqual(
            result.memory_supporting_paths,
            ["recallforge://alpha/memories/demo.mp4::frame:0001"],
        )


class TestParallelSearchTaskCapture(unittest.TestCase):
    def test_parallel_search_captures_original_vector(self):
        """After expander removal (REC-108), _run_parallel_searches only
        embeds the original query. Verify the vector is captured correctly."""
        backend = StubBackend(mode="embed")
        storage = StubStorage()
        searcher = HybridSearcher(backend=backend, storage=storage, limit=5)

        captured_vectors = []

        def mock_search_vec(vector, limit: int = 20, **kwargs):
            captured_vectors.append(tuple(vector.tolist() if hasattr(vector, "tolist") else list(vector)))
            return []

        storage.search_vec = mock_search_vec

        query = "base query"
        searcher._run_parallel_searches(query)

        expected_vector = tuple(backend.embed_text(query).tolist())
        self.assertEqual(len(captured_vectors), 1)
        self.assertEqual(captured_vectors[0], expected_vector)


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

    def _make_searcher(self, mode: str = "hybrid") -> HybridSearcher:
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

    def test_search_hybrid_mode(self):
        """hybrid mode: reranking enabled, returns results."""
        searcher = self._make_searcher("hybrid")
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


class TestImageQueryHybridPipeline(unittest.TestCase):
    """Image query path should use the same fusion/rerank pipeline as text."""

    def test_search_image_skips_reranker_by_default(self):
        backend = StubBackend(mode="hybrid")
        backend.rerank = MagicMock(return_value=[0.88, 0.66])
        vec_results = [
            _make_search_result("doc1.md", 0.9, "vec"),
            _make_search_result("doc2.md", 0.8, "vec"),
        ]
        storage = StubStorage(fts_results=[], vec_results=vec_results)
        searcher = HybridSearcher(backend=backend, storage=storage, limit=2)

        results = searcher.search_image("/tmp/query.png")

        self.assertEqual(len(results), 2)
        backend.rerank.assert_not_called()
        self.assertEqual(results[0].source, "original_vec")
        self.assertEqual(results[0].audit.reranker_scoring_path, "media_disabled")

    def test_search_image_runs_rrf_and_reranker_when_opted_in(self):
        backend = StubBackend(mode="hybrid")
        backend.rerank = MagicMock(return_value=[0.88, 0.66])
        vec_results = [
            _make_search_result("doc1.md", 0.9, "vec"),
            _make_search_result("doc2.md", 0.8, "vec"),
        ]
        storage = StubStorage(fts_results=[], vec_results=vec_results)
        with patch.dict(os.environ, {"RECALLFORGE_ENABLE_MEDIA_RERANKING": "1"}):
            searcher = HybridSearcher(backend=backend, storage=storage, limit=2)
            results = searcher.search_image("/tmp/query.png")

        self.assertEqual(len(results), 2)
        backend.rerank.assert_called_once()
        self.assertEqual(results[0].source, "original_vec")
        self.assertTrue(all(isinstance(r, HybridResult) for r in results))

    def test_search_image_embed_mode_skips_reranker(self):
        backend = StubBackend(mode="embed")
        backend.rerank = MagicMock()
        storage = StubStorage(vec_results=[_make_search_result("doc.md", 0.9, "vec")])
        searcher = HybridSearcher(backend=backend, storage=storage, limit=1)

        results = searcher.search_image("/tmp/query.png")

        backend.rerank.assert_not_called()
        self.assertEqual(len(results), 1)

    def test_search_image_uses_caption_for_bm25_probe(self):
        backend = StubBackend(mode="hybrid")
        backend.rerank = MagicMock(return_value=[0.88])
        shared = _make_search_result("doc1.md", 0.9, "vec")
        storage = StubStorage(
            fts_results=[_make_search_result("doc1.md", 0.95, "fts")],
            vec_results=[shared],
        )
        searcher = HybridSearcher(backend=backend, storage=storage, limit=1)

        results = searcher.search_image("/tmp/query.png")

        self.assertEqual(storage.last_fts_query, "caption for query.png")
        self.assertIn(storage.last_fts_content_type, (None, "text"))
        self.assertEqual(len(results), 1)
        self.assertIn("original_fts", results[0].source)

    def test_search_image_can_disable_media_query_probe(self):
        backend = StubBackend(mode="hybrid")
        backend.rerank = MagicMock(return_value=[0.88])
        storage = StubStorage(
            fts_results=[_make_search_result("doc1.md", 0.95, "fts")],
            vec_results=[_make_search_result("doc1.md", 0.9, "vec")],
        )
        storage.search_fts = MagicMock(side_effect=storage.search_fts)
        searcher = HybridSearcher(
            backend=backend,
            storage=storage,
            limit=1,
            enable_media_query_probe=False,
        )

        results = searcher.search_image("/tmp/query.png")

        self.assertEqual(len(results), 1)
        storage.search_fts.assert_not_called()
        self.assertEqual(results[0].source, "original_vec")

    def test_search_image_falls_back_to_caption_probe_when_query_embedding_fails(self):
        backend = StubBackend(mode="hybrid")
        backend.rerank = MagicMock(return_value=[0.88])
        backend.embed_image = MagicMock(side_effect=RuntimeError("image embed failed"))
        storage = StubStorage(
            fts_results=[_make_search_result("doc1.md", 0.95, "fts")],
            vec_results=[],
        )
        searcher = HybridSearcher(backend=backend, storage=storage, limit=1)

        with self.assertLogs("recallforge.search", level="WARNING") as captured:
            results = searcher.search_image("/tmp/query.png")

        self.assertEqual(len(results), 1)
        self.assertEqual(storage.last_fts_query, "caption for query.png")
        self.assertIn("original_fts", results[0].source)
        self.assertTrue(
            any("image query embedding failed" in message for message in captured.output)
        )

    def test_search_image_expands_caption_probe_when_enabled(self):
        backend = StubBackend(
            mode="hybrid",
            generated_text_response="query screenshot notes\napplication error screenshot",
        )
        backend.rerank = MagicMock(return_value=[0.88])
        storage = StubStorage(
            fts_results=[_make_search_result("doc1.md", 0.95, "fts")],
            vec_results=[_make_search_result("doc1.md", 0.9, "vec")],
        )
        storage.search_fts = MagicMock(side_effect=storage.search_fts)
        storage.search_vec = MagicMock(side_effect=storage.search_vec)
        searcher = HybridSearcher(backend=backend, storage=storage, limit=1, expand=True)

        results = searcher.search_image("/tmp/query.png")

        self.assertEqual(len(results), 1)
        queried_fts = [args[0] for args, _ in storage.search_fts.call_args_list]
        self.assertEqual(
            queried_fts,
            [
                "caption for query.png",
                "query screenshot notes",
                "application error screenshot",
            ],
        )
        self.assertEqual(storage.search_vec.call_count, 3)

    def test_search_image_skips_failed_expansion_variant_and_keeps_search_alive(self):
        backend = StubBackend(
            mode="hybrid",
            generated_text_response="query screenshot notes\napplication error screenshot",
        )
        backend.rerank = MagicMock(return_value=[0.88])
        storage = StubStorage(
            fts_results=[_make_search_result("doc1.md", 0.95, "fts")],
            vec_results=[_make_search_result("doc1.md", 0.9, "vec")],
        )
        storage.search_fts = MagicMock(side_effect=storage.search_fts)
        searcher = HybridSearcher(backend=backend, storage=storage, limit=1, expand=True)

        def _variant_search(query: str):
            if query == "query screenshot notes":
                raise RuntimeError("embedding backend failed")
            return {"original_vec": [_make_search_result("doc1.md", 0.9, "vec")]}

        searcher._run_parallel_searches = MagicMock(side_effect=_variant_search)

        with self.assertLogs("recallforge.search", level="WARNING") as captured:
            results = searcher.search_image("/tmp/query.png")

        self.assertEqual(len(results), 1)
        queried_fts = [args[0] for args, _ in storage.search_fts.call_args_list]
        self.assertEqual(
            queried_fts,
            [
                "caption for query.png",
                "application error screenshot",
            ],
        )
        self.assertTrue(
            any("Skipping failed query expansion branch" in message for message in captured.output)
        )

    def test_search_video_uses_caption_for_bm25_probe(self):
        backend = StubBackend(mode="hybrid")
        backend.rerank = MagicMock(return_value=[0.88])
        storage = StubStorage(
            fts_results=[_make_search_result("clip.md", 0.95, "fts")],
            vec_results=[_make_search_result("clip.md", 0.9, "vec")],
        )
        searcher = HybridSearcher(backend=backend, storage=storage, limit=1)
        searcher._caption_video_query = MagicMock(return_value="forest timelapse mountains")

        results = searcher.search_video("/tmp/query.mp4")

        self.assertEqual(storage.last_fts_query, "forest timelapse mountains")
        searcher._caption_video_query.assert_called_once_with("/tmp/query.mp4")
        self.assertEqual(len(results), 1)
        self.assertIn("original_fts", results[0].source)
        backend.rerank.assert_not_called()

    def test_search_video_expands_caption_probe_when_enabled(self):
        backend = StubBackend(
            mode="hybrid",
            generated_text_response='["mountain forest timelapse", "alpine nature video"]',
        )
        backend.rerank = MagicMock(return_value=[0.88])
        storage = StubStorage(
            fts_results=[_make_search_result("clip.md", 0.95, "fts")],
            vec_results=[_make_search_result("clip.md", 0.9, "vec")],
        )
        storage.search_fts = MagicMock(side_effect=storage.search_fts)
        storage.search_vec = MagicMock(side_effect=storage.search_vec)
        searcher = HybridSearcher(backend=backend, storage=storage, limit=1, expand=True)
        searcher._caption_video_query = MagicMock(return_value="forest timelapse mountains")

        results = searcher.search_video("/tmp/query.mp4")

        self.assertEqual(len(results), 1)
        queried_fts = [args[0] for args, _ in storage.search_fts.call_args_list]
        self.assertEqual(
            queried_fts,
            [
                "forest timelapse mountains",
                "mountain forest timelapse",
                "alpine nature video",
            ],
        )
        self.assertEqual(storage.search_vec.call_count, 3)

    def test_search_video_falls_back_to_caption_probe_when_query_embedding_fails(self):
        backend = StubBackend(mode="hybrid")
        backend.embed_video = MagicMock(side_effect=RuntimeError("video embed failed"))
        backend.rerank = MagicMock(return_value=[0.88])
        storage = StubStorage(
            fts_results=[_make_search_result("clip.md", 0.95, "fts")],
            vec_results=[],
        )
        searcher = HybridSearcher(backend=backend, storage=storage, limit=1)
        searcher._caption_video_query = MagicMock(return_value="forest timelapse mountains")

        with self.assertLogs("recallforge.search", level="WARNING") as captured:
            results = searcher.search_video("/tmp/query.mp4")

        self.assertEqual(len(results), 1)
        self.assertEqual(storage.last_fts_query, "forest timelapse mountains")
        searcher._caption_video_query.assert_called_once_with("/tmp/query.mp4")
        self.assertIn("original_fts", results[0].source)
        self.assertTrue(
            any("video query embedding failed" in message for message in captured.output)
        )

    def test_text_query_with_media_candidates_skips_reranker_by_default(self):
        backend = StubBackend(mode="hybrid")
        backend.rerank = MagicMock(return_value=[0.99])
        candidate = _make_search_result(
            "img.png",
            0.9,
            "vec",
            content_type="image",
        )
        searcher = HybridSearcher(backend=backend, storage=StubStorage(), limit=1)

        scores, path = searcher._rerank_candidates([candidate], query="diagram")

        backend.rerank.assert_not_called()
        self.assertEqual(scores, {"img.png": 0.5})
        self.assertEqual(path, "media_disabled")


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


class TestN1LookupOptimization(unittest.TestCase):
    """Test that storage backend prefers text_body over get_content() to avoid N+1 lookups."""

    def test_prefers_text_body_over_get_content(self):
        """Verify _make_search_result uses text_body from row without calling get_content()."""
        from recallforge.storage.lancedb_backend import LanceDBBackend

        # Create a mock LanceDBBackend
        backend = LanceDBBackend.__new__(LanceDBBackend)

        # Create a row that has text_body already populated
        row = {
            "collection": "test",
            "file_path": "doc.md",
            "content_hash": "abc123",
            "content_type": "text",
            "title": "Test Doc",
            "pos": 0,
            "text_body": "This is chunk text from embeddings table",
            "user_id": None,
            "session_id": None,
            "project_id": None,
            "profile": None,
        }

        # Track if get_content is called (it shouldn't be for rows with text_body)
        get_content_called = False
        original_get_content = backend.get_content

        def mock_get_content(hash_str):
            nonlocal get_content_called
            get_content_called = True
            return original_get_content(backend, hash_str)

        # Patch get_content to track calls
        backend.get_content = mock_get_content

        # Call _make_search_result with a row that has text_body
        result = backend._make_search_result(row, 0.9, "fts")

        # Verify the body came from text_body, not get_content
        self.assertEqual(result.body, "This is chunk text from embeddings table")
        # get_content should NOT be called when text_body is available
        self.assertFalse(get_content_called, "get_content() should not be called when text_body is available")


class TestIntentAwareQuerySteering(unittest.TestCase):
    """Test intent-aware query steering in RRF fusion."""

    def test_exact_lookup_boosts_fts_weight(self):
        """exact_lookup intent should boost FTS weight and lower vector weight."""
        backend = StubBackend()
        storage = StubStorage()
        searcher = HybridSearcher(backend=backend, storage=storage, intent="exact_lookup")

        # Create results for FTS and vector
        fts_result = _make_search_result("doc1.md", 0.9, source="fts")
        vec_result = _make_search_result("doc2.md", 0.85, source="vec")

        all_results = {
            "original_fts": [fts_result],
            "original_vec": [vec_result],
        }

        fused, audit_info = searcher._reciprocal_rank_fusion(all_results)

        # With exact_lookup: FTS weight=2.5, vector weight=0.8
        # FTS score contribution: 2.5 / (60 + 0 + 1) = 2.5/61 ≈ 0.041
        # Vec score contribution: 0.8 / (60 + 0 + 1) = 0.8/61 ≈ 0.013
        # FTS doc should have higher combined score
        fts_doc = next((r for r in fused if r.filepath == "doc1.md"), None)
        vec_doc = next((r for r in fused if r.filepath == "doc2.md"), None)

        self.assertIsNotNone(fts_doc)
        self.assertIsNotNone(vec_doc)
        self.assertGreater(fts_doc.score, vec_doc.score,
                           "exact_lookup: FTS result should score higher than vector result")

    def test_semantic_boosts_vector_weight(self):
        """semantic intent should boost vector weight and lower FTS weight."""
        backend = StubBackend()
        storage = StubStorage()
        searcher = HybridSearcher(backend=backend, storage=storage, intent="semantic")

        # Create results for FTS and vector
        fts_result = _make_search_result("doc1.md", 0.9, source="fts")
        vec_result = _make_search_result("doc2.md", 0.85, source="vec")

        all_results = {
            "original_fts": [fts_result],
            "original_vec": [vec_result],
        }

        fused, audit_info = searcher._reciprocal_rank_fusion(all_results)

        # With semantic: FTS weight=0.8, vector weight=2.5
        # FTS score contribution: 0.8 / (60 + 0 + 1) = 0.8/61 ≈ 0.013
        # Vec score contribution: 2.5 / (60 + 0 + 1) = 2.5/61 ≈ 0.041
        # Vector doc should have higher combined score
        fts_doc = next((r for r in fused if r.filepath == "doc1.md"), None)
        vec_doc = next((r for r in fused if r.filepath == "doc2.md"), None)

        self.assertIsNotNone(fts_doc)
        self.assertIsNotNone(vec_doc)
        self.assertGreater(vec_doc.score, fts_doc.score,
                           "semantic: vector result should score higher than FTS result")

    def test_broad_equal_weights(self):
        """broad intent should use equal weights for all sources."""
        backend = StubBackend()
        storage = StubStorage()
        searcher = HybridSearcher(backend=backend, storage=storage, intent="broad")

        # Create results for FTS and vector at same rank
        fts_result = _make_search_result("doc1.md", 0.9, source="fts")
        vec_result = _make_search_result("doc2.md", 0.85, source="vec")

        all_results = {
            "original_fts": [fts_result],
            "original_vec": [vec_result],
        }

        fused, audit_info = searcher._reciprocal_rank_fusion(all_results)

        # With broad: both weights=1.0
        # Both have same rank (0), so scores should be equal
        fts_doc = next((r for r in fused if r.filepath == "doc1.md"), None)
        vec_doc = next((r for r in fused if r.filepath == "doc2.md"), None)

        self.assertIsNotNone(fts_doc)
        self.assertIsNotNone(vec_doc)
        # Both have equal RRF contribution: 1.0 / (60 + 0 + 1)
        self.assertAlmostEqual(fts_doc.score, vec_doc.score, places=5,
                              msg="broad: FTS and vector results should have equal scores")

    def test_none_intent_uses_default_weights(self):
        """None intent (default) should use the existing weight behavior."""
        backend = StubBackend()
        storage = StubStorage()
        searcher = HybridSearcher(backend=backend, storage=storage, intent=None)

        # Create results for FTS and vector
        fts_result = _make_search_result("doc1.md", 0.9, source="fts")
        vec_result = _make_search_result("doc2.md", 0.85, source="vec")

        all_results = {
            "original_fts": [fts_result],
            "original_vec": [vec_result],
        }

        fused, audit_info = searcher._reciprocal_rank_fusion(all_results)

        # Default weights: first 2 lists = 2.0, rest = 1.0
        # With 2 sources, both get weight 2.0
        fts_doc = next((r for r in fused if r.filepath == "doc1.md"), None)
        vec_doc = next((r for r in fused if r.filepath == "doc2.md"), None)

        self.assertIsNotNone(fts_doc)
        self.assertIsNotNone(vec_doc)
        # Both have weight 2.0, same rank → equal scores
        self.assertAlmostEqual(fts_doc.score, vec_doc.score, places=5,
                              msg="None intent: should use default weights")

    def test_intent_weight_applied_to_rrf_calculation(self):
        """Verify intent weights are correctly applied in RRF score calculation."""
        from recallforge.search import INTENT_WEIGHTS

        backend = StubBackend()
        storage = StubStorage()

        # Test each intent
        for intent_name, weights in INTENT_WEIGHTS.items():
            searcher = HybridSearcher(backend=backend, storage=storage, intent=intent_name)

            # Create results
            fts_result = _make_search_result("doc_fts.md", 0.9, source="fts")
            vec_result = _make_search_result("doc_vec.md", 0.85, source="vec")

            all_results = {
                "original_fts": [fts_result],
                "original_vec": [vec_result],
            }

            fused, audit_info = searcher._reciprocal_rank_fusion(all_results)

            # Verify weights are applied correctly
            fts_doc = next((r for r in fused if r.filepath == "doc_fts.md"), None)
            vec_doc = next((r for r in fused if r.filepath == "doc_vec.md"), None)

            self.assertIsNotNone(fts_doc, f"Missing FTS doc for intent={intent_name}")
            self.assertIsNotNone(vec_doc, f"Missing vector doc for intent={intent_name}")

            # Calculate expected scores
            k = 60
            expected_fts = weights["original_fts"] / (k + 0 + 1)
            expected_vec = weights["original_vec"] / (k + 0 + 1)

            self.assertAlmostEqual(fts_doc.score, expected_fts, places=5,
                                   msg=f"FTS score mismatch for intent={intent_name}")
            self.assertAlmostEqual(vec_doc.score, expected_vec, places=5,
                                   msg=f"Vector score mismatch for intent={intent_name}")

    def test_intent_in_full_search_pipeline(self):
        """Intent should affect final search results order."""
        # Setup FTS-heavy results (doc1.md ranks high in FTS, low in vector)
        fts_results = [
            _make_search_result("doc1.md", 0.95, source="fts"),  # High FTS rank
            _make_search_result("doc2.md", 0.60, source="fts"),  # Lower FTS rank
        ]
        vec_results = [
            _make_search_result("doc2.md", 0.95, source="vec"),  # High vector rank
            _make_search_result("doc1.md", 0.60, source="vec"),  # Lower vector rank
        ]

        # exact_lookup: should favor doc1 (higher FTS score)
        backend = StubBackend(mode="embed")  # embed mode to skip reranking
        storage = StubStorage(fts_results=fts_results, vec_results=vec_results)
        searcher = HybridSearcher(backend=backend, storage=storage, limit=2, intent="exact_lookup")
        results = searcher.search("test query")

        # doc1 should rank higher with exact_lookup (boosts FTS)
        doc1_rank = next(i for i, r in enumerate(results) if r.filepath == "doc1.md")
        doc2_rank = next(i for i, r in enumerate(results) if r.filepath == "doc2.md")
        self.assertLess(doc1_rank, doc2_rank,
                        "exact_lookup: doc1 (FTS-heavy) should rank before doc2 (vector-heavy)")

        # semantic: should favor doc2 (higher vector score)
        searcher_semantic = HybridSearcher(backend=backend, storage=storage, limit=2, intent="semantic")
        results_semantic = searcher_semantic.search("test query")

        doc1_rank_sem = next(i for i, r in enumerate(results_semantic) if r.filepath == "doc1.md")
        doc2_rank_sem = next(i for i, r in enumerate(results_semantic) if r.filepath == "doc2.md")
        self.assertLess(doc2_rank_sem, doc1_rank_sem,
                        "semantic: doc2 (vector-heavy) should rank before doc1 (FTS-heavy)")


if __name__ == "__main__":
    unittest.main()
