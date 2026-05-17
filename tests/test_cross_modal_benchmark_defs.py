"""
Regression tests for benchmark ground-truth definitions.
"""

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_cross_modal_ablation():
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "benchmarks" / "cross_modal_ablation.py"
    spec = importlib.util.spec_from_file_location("cross_modal_ablation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestCrossModalBenchmarkDefinitions(unittest.TestCase):
    class _FakeBackend:
        def __init__(self):
            self._mode = "embed"

        def embed_text(self, _text):
            return [1.0]

        def embed_image(self, _path):
            return [1.0]

        def embed_video(self, _path):
            return [1.0]

        def get_mode(self):
            return self._mode

        def set_mode(self, mode):
            self._mode = mode

    class _FakeStorage:
        def __init__(self):
            self.last_vec_kwargs = None
            self.last_fts_kwargs = None

        def search_vec(self, **kwargs):
            self.last_vec_kwargs = kwargs
            return []

        def search_fts(self, **kwargs):
            self.last_fts_kwargs = kwargs
            return []

    def test_text_to_video_block_matches_category(self):
        module = _load_cross_modal_ablation()

        self.assertTrue(module.TEXT_TO_VIDEO)
        self.assertTrue(
            all(gt.category == "text_to_video" for gt in module.TEXT_TO_VIDEO)
        )

    def test_text_to_document_block_matches_category(self):
        module = _load_cross_modal_ablation()

        self.assertTrue(module.TEXT_TO_DOCUMENT)
        self.assertTrue(
            all(gt.category == "text_to_document" for gt in module.TEXT_TO_DOCUMENT)
        )

    def test_bm25_skip_helper_skips_image_and_video_query_categories(self):
        module = _load_cross_modal_ablation()

        self.assertEqual(
            module._skip_reason_for_stage("bm25", module.IMAGE_TO_TEXT),
            "BM25 can't process image queries",
        )
        self.assertEqual(
            module._skip_reason_for_stage("bm25", module.IMAGE_TO_VIDEO),
            "BM25 can't process image queries",
        )
        self.assertEqual(
            module._skip_reason_for_stage("bm25", module.VIDEO_TO_TEXT),
            "BM25 can't process video queries",
        )
        self.assertEqual(
            module._skip_reason_for_stage("bm25", module.VIDEO_TO_DOCUMENT),
            "BM25 can't process video queries",
        )

    def test_bm25_skip_helper_does_not_skip_text_query_categories(self):
        module = _load_cross_modal_ablation()

        self.assertIsNone(module._skip_reason_for_stage("bm25", module.TEXT_TO_IMAGE))
        self.assertIsNone(module._skip_reason_for_stage("bm25", module.TEXT_TO_VIDEO))
        self.assertIsNone(module._skip_reason_for_stage("bm25", module.MIXED_MODAL))
        self.assertIsNone(module._skip_reason_for_stage("embed", module.IMAGE_TO_TEXT))

    def test_run_search_applies_text_filter_for_image_to_text_category(self):
        module = _load_cross_modal_ablation()
        backend = self._FakeBackend()
        storage = self._FakeStorage()

        module.run_search(
            backend,
            storage,
            module.IMAGE_TO_TEXT[0],
            collection="benchmark",
            stage_mode="embed",
        )

        self.assertIsNotNone(storage.last_vec_kwargs)
        self.assertEqual(storage.last_vec_kwargs["content_type"], "text")

    def test_run_search_applies_image_filter_for_text_to_image_bm25(self):
        module = _load_cross_modal_ablation()
        backend = self._FakeBackend()
        storage = self._FakeStorage()

        module.run_search(
            backend,
            storage,
            module.TEXT_TO_IMAGE[0],
            collection="benchmark",
            stage_mode="bm25",
        )

        self.assertIsNotNone(storage.last_fts_kwargs)
        self.assertEqual(storage.last_fts_kwargs["content_type"], "image")

    def test_run_search_leaves_mixed_modal_unfiltered(self):
        module = _load_cross_modal_ablation()
        backend = self._FakeBackend()
        storage = self._FakeStorage()

        module.run_search(
            backend,
            storage,
            module.MIXED_MODAL[0],
            collection="benchmark",
            stage_mode="bm25",
        )

        self.assertIsNotNone(storage.last_fts_kwargs)
        self.assertIsNone(storage.last_fts_kwargs["content_type"])

    def test_video_frame_assets_count_as_hits_for_parent_video_ground_truth(self):
        module = _load_cross_modal_ablation()

        gt = module.GroundTruth(
            query="coding tutorial demo",
            relevant_paths=["videos/coding_demo.mp4"],
            category="text_to_video",
        )
        result = {
            "filepath": "recallforge://benchmark/videos/coding_demo.mp4::frame:0001@0.00s"
        }

        hit_1, hit_5, hit_10, ndcg, rr, prec_5, prec_10 = module.evaluate_results(
            [result],
            gt,
            module.CORPUS_DIR,
        )

        self.assertTrue(hit_1)
        self.assertTrue(hit_5)
        self.assertTrue(hit_10)
        self.assertEqual(ndcg, 1.0)
        self.assertEqual(rr, 1.0)
        self.assertEqual(prec_5, 1.0)
        self.assertEqual(prec_10, 1.0)

        detailed = module.evaluate_results_detailed([result], gt, module.CORPUS_DIR)
        self.assertTrue(detailed["memory"].hit_at_1)
        self.assertFalse(detailed["asset"].hit_at_1)

    def test_video_transcript_assets_count_as_hits_for_parent_video_ground_truth(self):
        module = _load_cross_modal_ablation()

        gt = module.GroundTruth(
            query="coding tutorial demo",
            relevant_paths=["videos/coding_demo.mp4"],
            category="text_to_video",
        )
        result = {
            "filepath": "recallforge://benchmark/videos/coding_demo.mp4::transcript:0001@0.00s"
        }

        hit_1, hit_5, hit_10, ndcg, rr, prec_5, prec_10 = module.evaluate_results(
            [result],
            gt,
            module.CORPUS_DIR,
        )

        self.assertTrue(hit_1)
        self.assertTrue(hit_5)
        self.assertTrue(hit_10)
        self.assertEqual(ndcg, 1.0)
        self.assertEqual(rr, 1.0)
        self.assertEqual(prec_5, 1.0)
        self.assertEqual(prec_10, 1.0)

        detailed = module.evaluate_results_detailed([result], gt, module.CORPUS_DIR)
        self.assertTrue(detailed["memory"].hit_at_1)
        self.assertFalse(detailed["asset"].hit_at_1)

    def test_output_payload_tracks_partial_progress(self):
        module = _load_cross_modal_ablation()

        categories = {
            "text_to_text": [module.TEXT_TO_TEXT[0]],
            "text_to_image": [module.TEXT_TO_IMAGE[0]],
        }
        stage_result = module.StageResult(
            stage="Vector-only",
            category="text_to_text",
            total_queries=1,
            hits_at_1=1,
            hits_at_5=1,
            hits_at_10=1,
            ndcg_sum=1.0,
            mrr_sum=1.0,
            precision_at_5_sum=1.0,
            precision_at_10_sum=1.0,
        )
        stage_result.asset_hits_at_1 = 0
        stage_result.asset_hits_at_5 = 0
        stage_result.asset_hits_at_10 = 0
        stage_result.asset_ndcg_sum = 0.0
        stage_result.asset_mrr_sum = 0.0
        stage_result.asset_precision_at_5_sum = 0.0
        stage_result.asset_precision_at_10_sum = 0.0
        stage_result.per_query_results.append(
            {
                "query": module.TEXT_TO_TEXT[0].query,
                "hit_at_1": True,
                "hit_at_5": True,
                "hit_at_10": True,
                "asset_level": {
                    "hit_at_1": False,
                    "hit_at_5": False,
                    "hit_at_10": False,
                },
            }
        )

        payload = module._build_output_payload(
            categories,
            {"Vector-only": {"text_to_text": stage_result}},
            [("Vector-only", "vector")],
            expansion_profile=module._resolve_expansion_profile("caption_only"),
            smoke_profile="safe",
            rss_limit_mb=4096,
            peak_rss_mb=512.4,
            indexed_items=74,
            run_status="partial",
            interrupted=True,
            completed_stages=[],
            current_stage="Vector-only",
            current_category="text_to_image",
            error="Interrupted by user",
        )

        self.assertEqual(payload["version"], "0.3.0")
        self.assertEqual(payload["configuration"]["expansion_profile"], "caption_only")
        self.assertEqual(payload["configuration"]["smoke_profile"], "safe")
        self.assertEqual(payload["configuration"]["rss_limit_mb"], 4096)
        self.assertFalse(payload["configuration"]["expand_enabled"])
        self.assertTrue(payload["configuration"]["media_query_probe_enabled"])
        self.assertEqual(payload["telemetry"]["peak_rss_mb"], 512.4)
        self.assertEqual(payload["run_status"], "partial")
        self.assertTrue(payload["interrupted"])
        self.assertEqual(payload["progress"]["indexed_items"], 74)
        self.assertEqual(payload["progress"]["current_stage"], "Vector-only")
        self.assertEqual(payload["progress"]["current_category"], "text_to_image")
        self.assertIn("text_to_text", payload["stages"]["Vector-only"])
        self.assertNotIn("text_to_image", payload["stages"]["Vector-only"])
        self.assertEqual(
            payload["stages"]["Vector-only"]["text_to_text"]["recall_at_1"],
            1.0,
        )
        self.assertEqual(
            payload["stages"]["Vector-only"]["text_to_text"]["asset_level"]["recall_at_1"],
            0.0,
        )
        self.assertFalse(
            payload["stages"]["Vector-only"]["text_to_text"]["per_query_results"][0]["asset_level"]["hit_at_1"]
        )

    def test_output_payload_preserves_video_query_path(self):
        module = _load_cross_modal_ablation()

        gt = module.VIDEO_TO_TEXT[0]
        stage_result = module.StageResult(
            stage="Vector-only",
            category=gt.category,
            total_queries=1,
            hits_at_1=0,
            hits_at_5=0,
            hits_at_10=0,
        )
        stage_result.per_query_results.append(
            {
                "query": gt.query,
                "query_type": gt.query_type,
                "image_query_path": gt.image_query_path,
                "video_query_path": gt.video_query_path,
                "relevant_paths": gt.relevant_paths,
                "hit_at_1": False,
                "hit_at_5": False,
                "hit_at_10": False,
            }
        )

        payload = module._build_output_payload(
            {gt.category: [gt]},
            {"Vector-only": {gt.category: stage_result}},
            [("Vector-only", "embed")],
            expansion_profile=module._resolve_expansion_profile("caption_only"),
            smoke_profile="safe",
            rss_limit_mb=None,
            peak_rss_mb=None,
            indexed_items=1,
            run_status="complete",
            interrupted=False,
            completed_stages=["Vector-only"],
            current_stage=None,
            current_category=None,
        )

        self.assertEqual(
            payload["stages"]["Vector-only"][gt.category]["per_query_results"][0]["video_query_path"],
            gt.video_query_path,
        )

    def test_resolve_expansion_profile_variants(self):
        module = _load_cross_modal_ablation()

        qwen = module._resolve_expansion_profile("qwen")
        off = module._resolve_expansion_profile("off")

        self.assertTrue(qwen.expand)
        self.assertTrue(qwen.allow_generate_text)
        self.assertFalse(off.expand)
        self.assertFalse(off.enable_media_query_probe)

        with self.assertRaises(ValueError):
            module._resolve_expansion_profile("bogus")

    def test_expansion_backend_proxy_can_disable_generate_text(self):
        module = _load_cross_modal_ablation()

        class _Backend:
            def __init__(self):
                self.calls = []

            def generate_text(self, prompt: str, max_tokens: int = 60) -> str:
                self.calls.append((prompt, max_tokens))
                return "ok"

        backend = _Backend()
        disabled = module._ExpansionBackendProxy(backend, allow_generate_text=False)
        enabled = module._ExpansionBackendProxy(backend, allow_generate_text=True)

        with self.assertRaises(NotImplementedError):
            disabled.generate_text("prompt")

        self.assertEqual(enabled.generate_text("prompt", max_tokens=12), "ok")
        self.assertEqual(backend.calls, [("prompt", 12)])

    def test_resolve_output_path_suffixes_non_default_profiles(self):
        module = _load_cross_modal_ablation()

        default_path = module._resolve_output_path(None, "caption_only")
        qwen_path = module._resolve_output_path(None, "qwen")

        self.assertTrue(default_path.endswith("cross_modal_ablation_results.json"))
        self.assertTrue(qwen_path.endswith("cross_modal_ablation_results_qwen.json"))

    def test_apply_smoke_profile_defaults(self):
        module = _load_cross_modal_ablation()

        stages, max_queries, rss_limit = module._apply_smoke_profile_defaults(
            "safe",
            None,
            None,
            None,
        )
        self.assertEqual(stages, ["rrf"])
        self.assertEqual(max_queries, 1)
        self.assertEqual(rss_limit, 6144)

        stages, max_queries, rss_limit = module._apply_smoke_profile_defaults(
            "safe",
            ["hybrid"],
            2,
            2048,
        )
        self.assertEqual(stages, ["hybrid"])
        self.assertEqual(max_queries, 2)
        self.assertEqual(rss_limit, 2048)

        stages, max_queries, rss_limit = module._apply_smoke_profile_defaults(
            "off",
            None,
            None,
            None,
        )
        self.assertIsNone(stages)
        self.assertIsNone(max_queries)
        self.assertIsNone(rss_limit)


if __name__ == "__main__":
    unittest.main()
