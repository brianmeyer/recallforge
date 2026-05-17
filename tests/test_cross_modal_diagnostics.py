"""
Regression tests for cross-modal benchmark diagnostics.
"""

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_diagnostics_module():
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "benchmarks" / "cross_modal_diagnostics.py"
    spec = importlib.util.spec_from_file_location("cross_modal_diagnostics", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestCrossModalDiagnostics(unittest.TestCase):
    def _synthetic_payload(self):
        return {
            "benchmark": "cross_modal_ablation",
            "version": "0.2.0",
            "generated_at": "2026-05-17T00:00:00+00:00",
            "run_status": "complete",
            "categories": {
                "image_to_document": {"queries": 2},
                "image_to_text": {"queries": 15},
                "mixed_modal": {"queries": 20},
            },
            "stages": {
                "Vector-only": {
                    "image_to_document": {
                        "skipped": False,
                        "total_queries": 2,
                        "recall_at_5": 0.0,
                        "recall_at_10": 0.0,
                        "per_query_results": [
                            {
                                "query": "related document",
                                "query_type": "image",
                                "image_query_path": "images/neural_network_diagram.png",
                                "relevant_paths": ["documents/ai_strategy_report.docx"],
                                "hit_at_5": False,
                                "results": [
                                    {
                                        "filepath": "recallforge://benchmark//repo/tests/uat/corpus/images/neural_network_diagram.png"
                                    }
                                ],
                            },
                            {
                                "query": "related document",
                                "query_type": "image",
                                "image_query_path": "images/floor_plan_blueprint.png",
                                "relevant_paths": ["documents/operations_manual.pdf"],
                                "hit_at_5": False,
                                "results": [],
                            },
                        ],
                    },
                    "image_to_text": {
                        "skipped": False,
                        "total_queries": 15,
                        "recall_at_5": 0.4,
                        "recall_at_10": 0.5,
                        "asset_level": {"recall_at_5": 0.4},
                        "per_query_results": [
                            {
                                "query": "",
                                "query_type": "image",
                                "image_query_path": "images/ocean_beach.png",
                                "relevant_paths": ["text/nature_oceans.md"],
                                "hit_at_5": False,
                                "results": [],
                            }
                        ],
                    },
                    "mixed_modal": {
                        "skipped": False,
                        "total_queries": 20,
                        "recall_at_5": 0.9,
                        "recall_at_10": 1.0,
                        "asset_level": {"recall_at_5": 0.9},
                        "per_query_results": [],
                    },
                },
                "BM25-only": {
                    "image_to_document": {
                        "skipped": True,
                        "total_queries": 2,
                        "skip_reason": "BM25 can't process image queries",
                        "per_query_results": [],
                    },
                    "image_to_text": {
                        "skipped": True,
                        "total_queries": 15,
                        "skip_reason": "BM25 can't process image queries",
                        "per_query_results": [],
                    },
                    "mixed_modal": {
                        "skipped": False,
                        "total_queries": 20,
                        "recall_at_5": 0.8,
                        "recall_at_10": 0.9,
                        "asset_level": {"recall_at_5": 0.8},
                        "per_query_results": [],
                    },
                },
                "Vector + BM25 (RRF)": {
                    "image_to_document": {
                        "skipped": False,
                        "total_queries": 2,
                        "recall_at_5": 0.0,
                        "recall_at_10": 0.5,
                        "per_query_results": [
                            {
                                "query": "related document",
                                "query_type": "image",
                                "image_query_path": "images/neural_network_diagram.png",
                                "relevant_paths": ["documents/ai_strategy_report.docx"],
                                "hit_at_5": False,
                                "results": [
                                    {
                                        "filepath": "recallforge://benchmark//repo/tests/uat/corpus/text/science_neuroscience.md",
                                        "audit": {"rrf_sources": {"original_vec": 3, "original_fts": 1}},
                                    }
                                ],
                            }
                        ],
                    },
                    "image_to_text": {
                        "skipped": False,
                        "total_queries": 15,
                        "recall_at_5": 0.65,
                        "recall_at_10": 0.8,
                        "asset_level": {"recall_at_5": 0.65},
                        "per_query_results": [
                            {
                                "query": "",
                                "query_type": "image",
                                "image_query_path": "images/ocean_beach.png",
                                "relevant_paths": ["text/nature_oceans.md"],
                                "hit_at_5": True,
                                "results": [
                                    {
                                        "filepath": "recallforge://benchmark//repo/tests/uat/corpus/text/nature_oceans.md",
                                        "audit": {"rrf_sources": {"original_vec": 1, "original_fts": 2}},
                                    }
                                ],
                            }
                        ],
                    },
                    "mixed_modal": {
                        "skipped": False,
                        "total_queries": 20,
                        "recall_at_5": 0.95,
                        "recall_at_10": 1.0,
                        "asset_level": {"recall_at_5": 0.95},
                        "per_query_results": [],
                    },
                },
                "Vector + BM25 + Reranker": {
                    "image_to_document": {
                        "skipped": False,
                        "total_queries": 2,
                        "recall_at_5": 0.0,
                        "recall_at_10": 0.5,
                        "per_query_results": [],
                    },
                    "image_to_text": {
                        "skipped": False,
                        "total_queries": 15,
                        "recall_at_5": 0.65,
                        "recall_at_10": 0.8,
                        "asset_level": {"recall_at_5": 0.65},
                        "per_query_results": [],
                    },
                    "mixed_modal": {
                        "skipped": False,
                        "total_queries": 20,
                        "recall_at_5": 0.95,
                        "recall_at_10": 1.0,
                        "asset_level": {"recall_at_5": 0.95},
                        "per_query_results": [],
                    },
                },
            },
        }

    def test_diagnostics_rank_and_classify_weak_categories(self):
        module = _load_diagnostics_module()

        diagnostics = module.build_diagnostics(
            self._synthetic_payload(),
            content_type_filters={
                "image_to_document": None,
                "image_to_text": "text",
                "mixed_modal": None,
            },
            min_queries=20,
            weak_recall_at_5=0.6,
        )

        self.assertEqual(diagnostics["weak_categories"][0]["category"], "image_to_document")
        issues = {
            issue["code"]
            for issue in diagnostics["weak_categories"][0]["issues"]
        }
        self.assertIn("under_sampled_category", issues)
        self.assertIn("bm25_modality_blind", issues)
        self.assertIn("embedding_alignment_gap", issues)
        self.assertIn("document_family_filter_gap", issues)
        self.assertIn("generic_query_artifact", issues)
        self.assertIn("parent_asset_metrics_missing", issues)
        self.assertEqual(
            diagnostics["weak_categories"][0]["audit_source_counts_top5"],
            {"original_fts": 1, "original_vec": 1},
        )

    def test_diagnostics_detect_derived_text_lift(self):
        module = _load_diagnostics_module()

        diagnostics = module.build_diagnostics(
            self._synthetic_payload(),
            content_type_filters={
                "image_to_document": None,
                "image_to_text": "text",
                "mixed_modal": None,
            },
            min_queries=20,
            weak_recall_at_5=0.6,
        )
        image_to_text = next(
            item for item in diagnostics["all_categories"] if item["category"] == "image_to_text"
        )
        issues = {issue["code"] for issue in image_to_text["issues"]}

        self.assertIn("derived_text_probe_lift", issues)
        self.assertNotIn("parent_asset_metrics_missing", issues)

    def test_markdown_report_contains_ranked_fixes(self):
        module = _load_diagnostics_module()
        diagnostics = module.build_diagnostics(
            self._synthetic_payload(),
            content_type_filters={
                "image_to_document": None,
                "image_to_text": "text",
                "mixed_modal": None,
            },
            min_queries=20,
            weak_recall_at_5=0.6,
        )

        markdown = module.render_markdown(diagnostics)

        self.assertIn("# Cross-Modal Retrieval Diagnostics", markdown)
        self.assertIn("`image_to_document`", markdown)
        self.assertIn("Prioritized Fix List", markdown)
        self.assertIn("BEIR", markdown)


if __name__ == "__main__":
    unittest.main()
