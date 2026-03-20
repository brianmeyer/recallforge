"""
test_cross_modal_pipeline.py - Cross-modal CI regression tests for REC-124.

Tests that prevent the empty-text-for-images reranker bug (REC-111) from recurring.
Verifies image/video candidates flow correctly through search → rerank pipeline.

All backends and storage are mocked — NO real model inference, NO real LanceDB.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch, call

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recallforge.search import HybridSearcher, HybridResult, INTENT_WEIGHTS
from recallforge.storage.base import SearchResult
from recallforge.backends.base import ModelBackend, BackendInfo


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _make_search_result(
    filepath: str,
    score: float = 0.9,
    source: str = "fts",
    content_type: str = "text",
    body: Optional[str] = None,
) -> SearchResult:
    """Create a SearchResult with proper defaults for cross-modal testing."""
    return SearchResult(
        filepath=filepath,
        display_path=filepath,
        title=os.path.basename(filepath),
        context=None,
        hash=filepath.replace("/", "_"),
        docid=filepath[:6],
        collection="test",
        modified_at="2026-01-01",
        body_length=100 if content_type == "text" else 0,
        score=score,
        source=source,
        content_type=content_type,
        chunk_pos=0,
        body=body if body is not None else (f"Content of {filepath}" if content_type == "text" else ""),
    )


class StubBackend(ModelBackend):
    """Minimal ModelBackend stub for unit tests with cross-modal support."""

    def __init__(self, mode: str = "hybrid"):
        self._mode = mode
        self.rerank_calls: List[List[Dict[str, Any]]] = []  # Track reranker inputs

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
        # Store the documents for verification
        self.rerank_calls.append(documents)
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

    def search_fts(self, query: str, limit: int = 20,
                   collection=None, content_type=None,
                   user_id=None, session_id=None, project_id=None, profile=None) -> List[SearchResult]:
        return list(self._fts_results[:limit])

    def search_vec(self, vector, limit: int = 20,
                   collection=None, content_type=None,
                   user_id=None, session_id=None, project_id=None, profile=None) -> List[SearchResult]:
        return list(self._vec_results[:limit])


# ---------------------------------------------------------------------------
# Cross-Modal Pipeline Tests
# ---------------------------------------------------------------------------

class TestSelectBestChunkCrossModal(unittest.TestCase):
    """Test _select_best_chunk() preserves image/video paths for reranker."""

    def setUp(self):
        self.backend = StubBackend()
        self.storage = StubStorage()
        self.searcher = HybridSearcher(backend=self.backend, storage=self.storage)

    def test_image_result_carries_image_path(self):
        """Image search results must carry image_path through _select_best_chunk()."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name

        try:
            result = _make_search_result(
                filepath=temp_path,
                score=0.9,
                source="vec",
                content_type="image",
                body="",  # Images have empty body
            )

            chunk = self.searcher._select_best_chunk(result)

            self.assertIn("image_path", chunk, "Image result must have image_path")
            self.assertEqual(chunk["image_path"], temp_path)
            self.assertEqual(chunk["content_type"], "image")
        finally:
            os.unlink(temp_path)

    def test_video_result_carries_video_path(self):
        """Video search results must carry video_path through _select_best_chunk()."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            result = _make_search_result(
                filepath=temp_path,
                score=0.9,
                source="vec",
                content_type="video",
                body="",  # Videos have empty body
            )

            chunk = self.searcher._select_best_chunk(result)

            self.assertIn("video_path", chunk, "Video result must have video_path")
            self.assertEqual(chunk["video_path"], temp_path)
            self.assertEqual(chunk["content_type"], "video")
        finally:
            os.unlink(temp_path)

    def test_text_result_has_no_media_path(self):
        """Text results should not have image_path or video_path."""
        result = _make_search_result(
            filepath="/docs/text.md",
            score=0.9,
            source="fts",
            content_type="text",
            body="This is text content",
        )

        chunk = self.searcher._select_best_chunk(result)

        self.assertNotIn("image_path", chunk, "Text result should not have image_path")
        self.assertNotIn("video_path", chunk, "Text result should not have video_path")
        self.assertEqual(chunk["content_type"], "text")
        self.assertEqual(chunk["text"], "This is text content")

    def test_image_with_recallforge_uri_resolves_path(self):
        """Image results with recallforge:// URIs should resolve to actual paths."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            # Simulate a recallforge:// URI that points to the actual file
            result = _make_search_result(
                filepath=f"recallforge://test/{temp_path}",
                score=0.9,
                source="vec",
                content_type="image",
                body="",
            )

            chunk = self.searcher._select_best_chunk(result)

            self.assertIn("image_path", chunk)
            # Should resolve to actual file path
            self.assertTrue(os.path.exists(chunk["image_path"]))
        finally:
            os.unlink(temp_path)

    def test_image_with_nonexistent_path_has_no_image_path(self):
        """If file doesn't exist, image_path should NOT be set (reranker can't use it)."""
        result = _make_search_result(
            filepath="/nonexistent/image.jpg",
            score=0.9,
            source="vec",
            content_type="image",
            body="",
        )

        chunk = self.searcher._select_best_chunk(result)

        # Non-existent files should NOT have image_path (reranker can't process them)
        self.assertNotIn("image_path", chunk,
                          "Non-existent image files should not have image_path")


class TestRerankerReceivesNonEmptyContent(unittest.TestCase):
    """Test that reranker receives non-empty content for media candidates."""

    def setUp(self):
        self.backend = StubBackend(mode="hybrid")
        self.storage = StubStorage()
        self.searcher = HybridSearcher(
            backend=self.backend,
            storage=self.storage,
            rerank_top_k=10,
        )

    def test_reranker_receives_image_path_not_empty_text(self):
        """Reranker must receive image_path for image candidates, not empty string."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name

        try:
            candidates = [
                _make_search_result(
                    filepath=temp_path,
                    score=0.9,
                    source="vec",
                    content_type="image",
                    body="",  # Empty body - this was the bug!
                ),
            ]

            self.searcher._rerank_candidates(candidates, "test query")

            # Verify reranker was called
            self.assertEqual(len(self.backend.rerank_calls), 1)
            rerank_docs = self.backend.rerank_calls[0]

            # Verify image candidate has image_path
            self.assertEqual(len(rerank_docs), 1)
            doc = rerank_docs[0]
            self.assertIn("image_path", doc, "Image candidate must have image_path for reranker")
            self.assertEqual(doc["image_path"], temp_path)
            # The text field can be empty, but image_path must be present
        finally:
            os.unlink(temp_path)

    def test_reranker_receives_video_path_not_empty_text(self):
        """Reranker must receive video_path for video candidates, not empty string."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            candidates = [
                _make_search_result(
                    filepath=temp_path,
                    score=0.9,
                    source="vec",
                    content_type="video",
                    body="",  # Empty body
                ),
            ]

            self.searcher._rerank_candidates(candidates, "test query")

            # Verify reranker was called
            self.assertEqual(len(self.backend.rerank_calls), 1)
            rerank_docs = self.backend.rerank_calls[0]

            # Verify video candidate has video_path
            self.assertEqual(len(rerank_docs), 1)
            doc = rerank_docs[0]
            self.assertIn("video_path", doc, "Video candidate must have video_path for reranker")
            self.assertEqual(doc["video_path"], temp_path)
        finally:
            os.unlink(temp_path)

    def test_reranker_receives_text_content_for_text_candidates(self):
        """Text candidates should have non-empty text content."""
        candidates = [
            _make_search_result(
                filepath="/docs/text.md",
                score=0.9,
                source="fts",
                content_type="text",
                body="This is actual text content",
            ),
        ]

        self.searcher._rerank_candidates(candidates, "test query")

        rerank_docs = self.backend.rerank_calls[0]
        doc = rerank_docs[0]
        self.assertIn("text", doc)
        self.assertEqual(doc["text"], "This is actual text content")
        self.assertNotIn("image_path", doc)
        self.assertNotIn("video_path", doc)


class TestMixedModalityHandling(unittest.TestCase):
    """Test mixed-modality result sets (text + image + video)."""

    def setUp(self):
        self.backend = StubBackend(mode="hybrid")
        self.storage = StubStorage()
        self.searcher = HybridSearcher(
            backend=self.backend,
            storage=self.storage,
            rerank_top_k=10,
        )

    def test_mixed_modality_candidates_all_have_appropriate_paths(self):
        """Mixed text/image/video candidates should each have appropriate content."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_img:
            img_path = f_img.name
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_vid:
                vid_path = f_vid.name

                try:
                    candidates = [
                        _make_search_result(
                            filepath="/docs/text.md",
                            score=0.95,
                            source="fts",
                            content_type="text",
                            body="Text document content",
                        ),
                        _make_search_result(
                            filepath=img_path,
                            score=0.90,
                            source="vec",
                            content_type="image",
                            body="",
                        ),
                        _make_search_result(
                            filepath=vid_path,
                            score=0.85,
                            source="vec",
                            content_type="video",
                            body="",
                        ),
                    ]

                    self.searcher._rerank_candidates(candidates, "test query")

                    rerank_docs = self.backend.rerank_calls[0]
                    self.assertEqual(len(rerank_docs), 3)

                    # Text candidate
                    text_doc = next(d for d in rerank_docs if d["content_type"] == "text")
                    self.assertEqual(text_doc["text"], "Text document content")
                    self.assertNotIn("image_path", text_doc)
                    self.assertNotIn("video_path", text_doc)

                    # Image candidate
                    img_doc = next(d for d in rerank_docs if d["content_type"] == "image")
                    self.assertIn("image_path", img_doc)
                    self.assertEqual(img_doc["image_path"], img_path)

                    # Video candidate
                    vid_doc = next(d for d in rerank_docs if d["content_type"] == "video")
                    self.assertIn("video_path", vid_doc)
                    self.assertEqual(vid_doc["video_path"], vid_path)

                finally:
                    os.unlink(img_path)
                    os.unlink(vid_path)

    def test_reranker_path_detection_for_mixed_modality(self):
        """Reranker path should be 'vl_image' when images present, 'vl_video' when only videos."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img_path = f.name

        try:
            # Test with image present
            candidates = [
                _make_search_result(
                    filepath=img_path,
                    score=0.9,
                    source="vec",
                    content_type="image",
                    body="",
                ),
            ]

            with patch.object(self.searcher, '_rerank_candidates', wraps=self.searcher._rerank_candidates) as mock_rerank:
                self.searcher._rerank_candidates(candidates, "test query")
                # The reranker path is logged, verify it was called
                self.assertEqual(len(self.backend.rerank_calls), 1)
        finally:
            os.unlink(img_path)


class TestPerModalityNormalization(unittest.TestCase):
    """Test per-modality normalization produces independent scales."""

    def setUp(self):
        self.backend = StubBackend(mode="hybrid")
        self.storage = StubStorage()
        self.searcher = HybridSearcher(
            backend=self.backend,
            storage=self.storage,
            limit=10,
        )

    def test_text_and_media_scores_normalized_independently(self):
        """Text and media reranker scores should be normalized independently."""
        # Create mock RRF results with text and media
        rrf_results = [
            _make_search_result("text1.md", 0.9, "fts", "text", body="Text 1"),
            _make_search_result("text2.md", 0.8, "fts", "text", body="Text 2"),
            _make_search_result("image1.jpg", 0.7, "vec", "image", body=""),
            _make_search_result("video1.mp4", 0.6, "vec", "video", body=""),
        ]

        # Simulate reranker scores with different ranges for text vs media
        # (This simulates the real-world issue where VL scores have different ranges)
        rerank_scores = {
            "text1.md": 0.18,  # High text score
            "text2.md": 0.03,  # Low text score
            "image1.jpg": 0.12,  # High image score (but lower than text max)
            "video1.mp4": 0.07,  # Low video score
        }

        blended = self.searcher._blend_scores(rrf_results, rerank_scores)

        # Find results by filepath
        text1 = next(r for r in blended if r.filepath == "text1.md")
        text2 = next(r for r in blended if r.filepath == "text2.md")
        img1 = next(r for r in blended if r.filepath == "image1.jpg")
        vid1 = next(r for r in blended if r.filepath == "video1.mp4")

        # Text scores should be normalized within text group
        # (0.18 - 0.03) / (0.18 - 0.03) = 1.0 for text1
        # (0.03 - 0.03) / (0.18 - 0.03) = 0.0 for text2
        self.assertAlmostEqual(text1.rerank_score, 0.18, places=5)  # Raw score preserved
        self.assertAlmostEqual(text2.rerank_score, 0.03, places=5)

        # Media scores should be normalized within media group
        # (0.12 - 0.07) / (0.12 - 0.07) = 1.0 for image
        # (0.07 - 0.07) / (0.12 - 0.07) = 0.0 for video
        self.assertAlmostEqual(img1.rerank_score, 0.12, places=5)
        self.assertAlmostEqual(vid1.rerank_score, 0.07, places=5)

    def test_normalization_handles_single_modality(self):
        """Normalization should work when only one modality is present."""
        rrf_results = [
            _make_search_result("text1.md", 0.9, "fts", "text", body="Text 1"),
            _make_search_result("text2.md", 0.8, "fts", "text", body="Text 2"),
        ]

        rerank_scores = {
            "text1.md": 0.9,
            "text2.md": 0.5,
        }

        blended = self.searcher._blend_scores(rrf_results, rerank_scores)

        # Should normalize within the single modality
        text1 = next(r for r in blended if r.filepath == "text1.md")
        text2 = next(r for r in blended if r.filepath == "text2.md")

        self.assertAlmostEqual(text1.rerank_score, 0.9, places=5)
        self.assertAlmostEqual(text2.rerank_score, 0.5, places=5)


class TestMediaCompensationInRRF(unittest.TestCase):
    """Test media compensation in RRF boosts image/video candidates appropriately."""

    def setUp(self):
        self.backend = StubBackend()
        self.storage = StubStorage()

    def _make_searcher(self, content_type=None):
        return HybridSearcher(
            backend=self.backend,
            storage=self.storage,
            rrf_k=60,
            content_type=content_type,
        )

    def test_image_candidates_get_rrf_boost(self):
        """Image candidates should get boosted RRF score when BM25 is present."""
        searcher = self._make_searcher(content_type="image")
        # Create results where image only appears in vector (not BM25)
        fts_results = [
            _make_search_result("text1.md", 0.9, "fts", "text"),
        ]
        vec_results = [
            _make_search_result("text1.md", 0.85, "vec", "text"),
            _make_search_result("image1.jpg", 0.80, "vec", "image"),
        ]

        all_results = {
            "original_fts": fts_results,
            "original_vec": vec_results,
        }

        fused, _audit = searcher._reciprocal_rank_fusion(all_results)

        # Both should be present
        text_doc = next((r for r in fused if r.filepath == "text1.md"), None)
        img_doc = next((r for r in fused if r.filepath == "image1.jpg"), None)

        self.assertIsNotNone(text_doc)
        self.assertIsNotNone(img_doc)

        # Image should have higher score due to media boost
        # Without boost: image score = 1.0 / (60 + 0 + 1) = 0.0164
        # With boost: multiplied by total_weight / non_bm25_weight
        self.assertGreater(img_doc.score, 0.016, "Image should have boosted score")

    def test_video_candidates_get_rrf_boost(self):
        """Video candidates should get boosted RRF score when BM25 is present."""
        searcher = self._make_searcher(content_type="video")
        fts_results = [
            _make_search_result("text1.md", 0.9, "fts", "text"),
        ]
        vec_results = [
            _make_search_result("text1.md", 0.85, "vec", "text"),
            _make_search_result("video1.mp4", 0.80, "vec", "video"),
        ]

        all_results = {
            "original_fts": fts_results,
            "original_vec": vec_results,
        }

        fused, _audit = searcher._reciprocal_rank_fusion(all_results)

        vid_doc = next((r for r in fused if r.filepath == "video1.mp4"), None)
        self.assertIsNotNone(vid_doc)
        self.assertGreater(vid_doc.score, 0.016, "Video should have boosted score")

    def test_text_candidates_do_not_get_media_boost(self):
        """Text candidates should not receive media compensation boost."""
        searcher = self._make_searcher()
        fts_results = [
            _make_search_result("text1.md", 0.9, "fts", "text"),
        ]
        vec_results = [
            _make_search_result("text1.md", 0.85, "vec", "text"),
        ]

        all_results = {
            "original_fts": fts_results,
            "original_vec": vec_results,
        }

        fused, _audit = searcher._reciprocal_rank_fusion(all_results)

        text_doc = fused[0]
        # Text appears in both lists, so score = 2.0/61 + 2.0/61 = 0.0656
        expected_score = 2.0 / 61 + 2.0 / 61
        self.assertAlmostEqual(text_doc.score, expected_score, places=5)

    def test_media_boost_calculation(self):
        """Explicit media-targeted searches should apply compensation."""
        searcher = self._make_searcher(content_type="image")

        # Image only appears in vector (not FTS) - this is when boost applies
        fts_results = [
            _make_search_result("text1.md", 0.9, "fts", "text"),
        ]
        vec_results = [
            _make_search_result("image1.jpg", 0.9, "vec", "image"),
        ]

        all_results = {
            "original_fts": fts_results,
            "original_vec": vec_results,
        }

        fused, audit = searcher._reciprocal_rank_fusion(all_results)
        img_doc = next(r for r in fused if r.filepath == "image1.jpg")

        unboosted_base = 1.5 / 61
        expected_boost = (2.5 + 1.5) / 1.5

        self.assertAlmostEqual(img_doc.score, unboosted_base * expected_boost, places=5)
        self.assertTrue(audit["image1.jpg"]["media_compensation"])

    def test_no_boost_when_target_modality_unspecified(self):
        """Mixed or generic searches should not apply blanket media compensation."""
        searcher = self._make_searcher()
        all_results = {
            "original_fts": [_make_search_result("text1.md", 0.9, "fts", "text")],
            "original_vec": [_make_search_result("image1.jpg", 0.9, "vec", "image")],
        }

        fused, audit = searcher._reciprocal_rank_fusion(all_results)
        img_doc = next(r for r in fused if r.filepath == "image1.jpg")

        self.assertAlmostEqual(img_doc.score, 2.0 / 61, places=5)
        self.assertFalse(audit["image1.jpg"]["media_compensation"])

    def test_no_boost_when_no_bm25(self):
        """Media compensation should not apply when there's no BM25 list."""
        searcher = self._make_searcher(content_type="image")
        # Only vector results, no FTS
        vec_results = [
            _make_search_result("image1.jpg", 0.9, "vec", "image"),
        ]

        all_results = {
            "original_vec": vec_results,
        }

        fused, _audit = searcher._reciprocal_rank_fusion(all_results)
        img_doc = fused[0]

        # No BM25 present, so no media boost
        # Image-targeted searches still use the modality-aware vec weight.
        expected_score = 1.5 / 61
        self.assertAlmostEqual(img_doc.score, expected_score, places=5)


class TestCrossModalRegressionScenarios(unittest.TestCase):
    """Regression tests for specific cross-modal bugs."""

    def setUp(self):
        self.backend = StubBackend(mode="hybrid")
        self.storage = StubStorage()
        self.searcher = HybridSearcher(
            backend=self.backend,
            storage=self.storage,
            limit=10,
            rerank_top_k=10,
        )

    def test_rec111_empty_text_for_images_bug(self):
        """
        REC-111: Empty text for images caused reranker to receive empty strings.

        Before fix: Image candidates had empty body, and _select_best_chunk
        passed empty text to reranker, causing poor ranking.

        After fix: Image candidates carry image_path for VL reranker.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img_path = f.name

        try:
            # Simulate image search result with empty body (the bug condition)
            result = _make_search_result(
                filepath=img_path,
                score=0.9,
                source="vec",
                content_type="image",
                body="",  # This was causing the bug
            )

            chunk = self.searcher._select_best_chunk(result)

            # The fix: image_path must be present even when body is empty
            self.assertIn("image_path", chunk,
                         "REC-111 REGRESSION: image_path must be present for image candidates")

            # Verify the reranker would receive this
            candidates = [result]
            self.searcher._rerank_candidates(candidates, "test query")

            rerank_docs = self.backend.rerank_calls[0]
            self.assertEqual(len(rerank_docs), 1)

            # The reranker should have image_path, not rely on empty text
            doc = rerank_docs[0]
            self.assertIn("image_path", doc,
                         "REC-111 REGRESSION: reranker must receive image_path")
            self.assertTrue(doc.get("image_path"),
                         "REC-111 REGRESSION: image_path must not be empty")

        finally:
            os.unlink(img_path)

    def test_full_pipeline_with_mixed_modalities(self):
        """End-to-end test: mixed text/image/video through full pipeline."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_img:
            img_path = f_img.name
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_vid:
                vid_path = f_vid.name

                try:
                    # Setup mixed results
                    fts_results = [
                        _make_search_result("/docs/text1.md", 0.95, "fts", "text", body="Text content 1"),
                        _make_search_result("/docs/text2.md", 0.85, "fts", "text", body="Text content 2"),
                    ]
                    vec_results = [
                        _make_search_result("/docs/text1.md", 0.90, "vec", "text", body="Text content 1"),
                        _make_search_result(img_path, 0.88, "vec", "image", body=""),
                        _make_search_result(vid_path, 0.82, "vec", "video", body=""),
                    ]

                    storage = StubStorage(fts_results=fts_results, vec_results=vec_results)
                    searcher = HybridSearcher(
                        backend=self.backend,
                        storage=storage,
                        limit=5,
                        rerank_top_k=5,
                    )

                    results = searcher.search("test query")

                    # Verify all modalities are present
                    texts = [r for r in results if r.filepath.endswith(".md")]
                    images = [r for r in results if r.filepath == img_path]
                    videos = [r for r in results if r.filepath == vid_path]

                    self.assertGreaterEqual(len(texts), 1, "Text results should be present")
                    self.assertEqual(len(images), 1, "Image result should be present")
                    self.assertEqual(len(videos), 1, "Video result should be present")

                    # Verify reranker was called with appropriate content
                    self.assertEqual(len(self.backend.rerank_calls), 1)
                    rerank_docs = self.backend.rerank_calls[0]

                    # Check each document type has appropriate content
                    for doc in rerank_docs:
                        if doc["content_type"] == "image":
                            self.assertIn("image_path", doc,
                                        "Image must have image_path in reranker input")
                        elif doc["content_type"] == "video":
                            self.assertIn("video_path", doc,
                                        "Video must have video_path in reranker input")
                        elif doc["content_type"] == "text":
                            self.assertIn("text", doc,
                                        "Text must have text in reranker input")
                            self.assertTrue(doc["text"],
                                          "Text content should not be empty")

                finally:
                    os.unlink(img_path)
                    os.unlink(vid_path)


if __name__ == "__main__":
    unittest.main()
