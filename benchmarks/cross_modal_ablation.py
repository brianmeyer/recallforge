#!/usr/bin/env python3
"""Cross-modal pipeline ablation benchmark for RecallForge.

Tests REAL cross-modal retrieval using actual images, text docs, and documents
from the UAT corpus. Measures what RecallForge is actually built for:
vision-language retrieval where text queries find images and vice versa.

Stages:
1. Vector-only (embed mode, no reranker)
2. BM25-only (full-text search)
3. Vector + BM25 via RRF
4. Vector + BM25 + Reranker (full hybrid)

Categories:
- text_to_text: text query → text document
- text_to_image: text query → relevant image (CROSS-MODAL)
- image_to_text: image as query → relevant text document (CROSS-MODAL)
- document_search: text query → DOCX/PPTX/PDF sections
- mixed_modal: query that should find BOTH text and image results

Metrics: R@1, R@5, R@10, NDCG@10, MRR, latency p50/p95
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CORPUS_DIR = PROJECT_ROOT / "tests" / "uat" / "corpus"
CORPUS_TEXT = CORPUS_DIR / "text"
CORPUS_IMAGES = CORPUS_DIR / "images"
CORPUS_VIDEOS = CORPUS_DIR / "videos"

# ---------------------------------------------------------------------------
# Ground truth definitions — hand-curated cross-modal pairs
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """A query with its expected relevant documents."""
    query: str
    relevant_paths: List[str]  # paths relative to corpus
    category: str
    query_type: str = "text"  # "text" or "image"
    image_query_path: Optional[str] = None  # for image-as-query


# TEXT → TEXT ground truth
TEXT_TO_TEXT = [
    GroundTruth(
        query="how do vector embeddings work for semantic search",
        relevant_paths=["text/ai_embeddings.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="transformer architecture attention mechanism",
        relevant_paths=["text/ai_transformers.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="autonomous AI agents with episodic memory",
        relevant_paths=["text/ai_agents.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="how to make fresh pasta dough at home",
        relevant_paths=["text/cooking_pasta.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="sourdough starter feeding schedule",
        relevant_paths=["text/cooking_sourdough.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="charcoal grill two zone setup searing",
        relevant_paths=["text/cooking_grilling.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="temperate deciduous forest ecosystem oak maple",
        relevant_paths=["text/nature_forests.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="mountain alpine ecology tectonic plates",
        relevant_paths=["text/nature_mountains.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="marine biology coral reef phytoplankton",
        relevant_paths=["text/nature_oceans.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="gothic cathedral flying buttresses ribbed vaults",
        relevant_paths=["text/architecture_gothic.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="reading architectural blueprints floor plans elevations",
        relevant_paths=["text/architecture_blueprints.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="Le Corbusier modern architecture glass curtain walls",
        relevant_paths=["text/architecture_modern.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="NBA basketball three point shooting analytics",
        relevant_paths=["text/sports_basketball.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="marathon training plan progressive mileage tempo runs",
        relevant_paths=["text/sports_running.md"],
        category="text_to_text",
    ),
    GroundTruth(
        query="soccer formations 4-3-3 wingbacks tactics",
        relevant_paths=["text/sports_soccer.md"],
        category="text_to_text",
    ),
]

# TEXT → IMAGE ground truth (the cross-modal test that matters)
TEXT_TO_IMAGE = [
    GroundTruth(
        query="neural network layers diagram with connections",
        relevant_paths=["images/neural_network_diagram.png"],
        category="text_to_image",
    ),
    GroundTruth(
        query="system architecture diagram with API gateway and database",
        relevant_paths=["images/whiteboard_architecture.png"],
        category="text_to_image",
    ),
    GroundTruth(
        query="AI strategy brainstorming mind map whiteboard",
        relevant_paths=["images/whiteboard_brainstorm.png"],
        category="text_to_image",
    ),
    GroundTruth(
        query="plate of pasta with tomato sauce and basil",
        relevant_paths=["images/food_pasta_dish.png"],
        category="text_to_image",
    ),
    GroundTruth(
        query="forest with tall trees and walking path",
        relevant_paths=["images/forest_landscape.png"],
        category="text_to_image",
    ),
    GroundTruth(
        query="mountain landscape with snow peaks and meadow",
        relevant_paths=["images/mountain_landscape.png"],
        category="text_to_image",
    ),
    GroundTruth(
        query="ocean waves beach sand and shells",
        relevant_paths=["images/ocean_beach.png"],
        category="text_to_image",
    ),
    GroundTruth(
        query="floor plan blueprint with rooms and dimensions",
        relevant_paths=["images/floor_plan_blueprint.png"],
        category="text_to_image",
    ),
    GroundTruth(
        query="code editor showing Python search function",
        relevant_paths=["images/code_editor_screenshot.png"],
        category="text_to_image",
    ),
    GroundTruth(
        query="handwritten meeting notes sprint review action items",
        relevant_paths=["images/handwritten_notes.png"],
        category="text_to_image",
    ),
]

# IMAGE → TEXT ground truth (image as query, find relevant text)
IMAGE_TO_TEXT = [
    GroundTruth(
        query="",  # query is the image itself
        relevant_paths=["text/ai_transformers.md", "text/ai_embeddings.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/neural_network_diagram.png",
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/cooking_pasta.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/food_pasta_dish.png",
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/nature_forests.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/forest_landscape.png",
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/nature_mountains.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/mountain_landscape.png",
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/nature_oceans.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/ocean_beach.png",
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/architecture_blueprints.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/floor_plan_blueprint.png",
    ),
]

# MIXED MODAL: queries that should find BOTH text and image
MIXED_MODAL = [
    GroundTruth(
        query="pasta cooking recipe and food photo",
        relevant_paths=["text/cooking_pasta.md", "images/food_pasta_dish.png"],
        category="mixed_modal",
    ),
    GroundTruth(
        query="forest ecology trees landscape",
        relevant_paths=["text/nature_forests.md", "images/forest_landscape.png"],
        category="mixed_modal",
    ),
    GroundTruth(
        query="mountain environment alpine peaks meadow",
        relevant_paths=["text/nature_mountains.md", "images/mountain_landscape.png"],
        category="mixed_modal",
    ),
    GroundTruth(
        query="ocean marine beach waves",
        relevant_paths=["text/nature_oceans.md", "images/ocean_beach.png"],
        category="mixed_modal",
    ),
    GroundTruth(
        query="neural network deep learning architecture diagram",
        relevant_paths=["text/ai_transformers.md", "images/neural_network_diagram.png"],
        category="mixed_modal",
    ),
    GroundTruth(
        query="floor plan building architecture blueprint design",
        relevant_paths=["text/architecture_blueprints.md", "images/floor_plan_blueprint.png"],
        category="mixed_modal",
    ),
    GroundTruth(
        query="software architecture API gateway backend system design",
        relevant_paths=["text/ai_agents.md", "images/whiteboard_architecture.png"],
        category="mixed_modal",
    ),
]

ALL_GROUND_TRUTH = TEXT_TO_TEXT + TEXT_TO_IMAGE + IMAGE_TO_TEXT + MIXED_MODAL

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Results for one stage across one category."""
    stage: str
    category: str
    total_queries: int = 0
    hits_at_1: int = 0
    hits_at_5: int = 0
    hits_at_10: int = 0
    ndcg_sum: float = 0.0
    mrr_sum: float = 0.0
    latencies_ms: list = field(default_factory=list)
    per_query_results: list = field(default_factory=list)  # Store detailed per-query results with audit trails

    @property
    def recall_at_1(self) -> float:
        return self.hits_at_1 / max(self.total_queries, 1)

    @property
    def recall_at_5(self) -> float:
        return self.hits_at_5 / max(self.total_queries, 1)

    @property
    def recall_at_10(self) -> float:
        return self.hits_at_10 / max(self.total_queries, 1)

    @property
    def ndcg_at_10(self) -> float:
        return self.ndcg_sum / max(self.total_queries, 1)

    @property
    def mrr(self) -> float:
        return self.mrr_sum / max(self.total_queries, 1)

    @property
    def p50_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[len(s) // 2]

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.95)]


def _dcg(relevances: List[float], k: int = 10) -> float:
    """Discounted Cumulative Gain."""
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        total += rel / math.log2(i + 2)
    return total


def _ndcg(relevances: List[float], k: int = 10) -> float:
    """Normalized DCG."""
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_results(
    results: List[Dict[str, Any]],
    gt: GroundTruth,
    corpus_dir: Path,
) -> Tuple[bool, bool, bool, float, float]:
    """Evaluate search results against ground truth.
    
    Returns: (hit@1, hit@5, hit@10, ndcg@10, reciprocal_rank)
    """
    # Normalize GT paths to absolute
    gt_paths_abs = set()
    for p in gt.relevant_paths:
        abs_path = str((corpus_dir / p).resolve())
        gt_paths_abs.add(abs_path)
        # Also match by filename stem for flexible path matching
        gt_paths_abs.add(Path(p).stem)

    def is_relevant(result: Dict) -> bool:
        fp = result.get("filepath", "")
        # Check absolute path match
        if fp in gt_paths_abs:
            return True
        # Check if filepath contains the expected filename
        for gp in gt.relevant_paths:
            stem = Path(gp).stem
            if stem in fp:
                return True
        return False

    # Build relevance vector
    relevances = []
    first_hit_rank = None
    for i, r in enumerate(results[:10]):
        rel = 1.0 if is_relevant(r) else 0.0
        relevances.append(rel)
        if rel > 0 and first_hit_rank is None:
            first_hit_rank = i + 1

    hit_1 = any(is_relevant(r) for r in results[:1])
    hit_5 = any(is_relevant(r) for r in results[:5])
    hit_10 = any(is_relevant(r) for r in results[:10])
    ndcg = _ndcg(relevances, 10)
    rr = 1.0 / first_hit_rank if first_hit_rank else 0.0

    return hit_1, hit_5, hit_10, ndcg, rr


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def ingest_corpus(backend, storage, collection: str, corpus_dir: Path) -> int:
    """Ingest the full UAT corpus into RecallForge."""
    indexed = 0

    # Text files
    for txt_file in sorted((corpus_dir / "text").glob("*.md")):
        text = txt_file.read_text(encoding="utf-8")
        try:
            storage.index_document(
                path=str(txt_file),
                text=text,
                collection=collection,
                model="benchmark",
                embed_func=backend.embed_text,
            )
            indexed += 1
        except Exception as e:
            print(f"  WARN: Failed to index {txt_file.name}: {e}")

    # Images
    for img_file in sorted((corpus_dir / "images").glob("*.png")):
        try:
            storage.index_image(
                path=str(img_file),
                collection=collection,
                embed_func=backend.embed_image,
                model="benchmark",
            )
            indexed += 1
        except Exception as e:
            print(f"  WARN: Failed to index {img_file.name}: {e}")

    # Documents (generated)
    doc_dir = corpus_dir / "documents"
    if doc_dir.exists():
        for doc_file in sorted(doc_dir.iterdir()):
            if doc_file.suffix in (".docx", ".pptx", ".pdf"):
                try:
                    storage.index_document_file(
                        path=str(doc_file),
                        collection=collection,
                        embed_func=backend.embed_text,
                        embed_image_func=backend.embed_image,
                        model="benchmark",
                    )
                    indexed += 1
                except Exception as e:
                    print(f"  WARN: Failed to index {doc_file.name}: {e}")

    return indexed


STAGES = [
    ("Vector-only", "embed"),
    ("BM25-only", "bm25"),
    ("Vector + BM25 (RRF)", "rrf"),
    ("Vector + BM25 + Reranker", "hybrid"),
]


def run_search(
    backend,
    storage,
    gt: GroundTruth,
    collection: str,
    stage_mode: str,
    limit: int = 10,
) -> Tuple[List[Dict], float]:
    """Run a single search query and return results + latency_ms."""
    from recallforge.search import HybridSearcher

    t0 = time.perf_counter()

    if gt.query_type == "image" and gt.image_query_path:
        image_path = str(CORPUS_DIR / gt.image_query_path)

        if stage_mode == "embed":
            # Vector-only baseline for image-as-query
            image_vec = backend.embed_image(image_path)
            results = storage.search_vec(
                vector=image_vec,
                collection=collection,
                limit=limit,
            )
        elif stage_mode in ("rrf", "hybrid"):
            # Route image query through HybridSearcher pipeline
            searcher = HybridSearcher(
                backend=backend,
                storage=storage,
                limit=limit,
                collection=collection,
            )
            old_mode = backend.get_mode()
            backend.set_mode("embed" if stage_mode == "rrf" else "hybrid")
            results = searcher.search_image(image_path)
            backend.set_mode(old_mode)
        elif stage_mode == "bm25":
            # BM25 cannot process image query directly (no query text/captions yet)
            results = []
        else:
            raise ValueError(f"Unknown stage mode: {stage_mode}")
    else:
        # Text query
        if stage_mode == "embed":
            # Vector only
            query_vec = backend.embed_text(gt.query)
            results = storage.search_vec(
                vector=query_vec,
                collection=collection,
                limit=limit,
            )
        elif stage_mode == "bm25":
            # BM25 only
            results = storage.search_fts(
                query=gt.query,
                collection=collection,
                limit=limit,
            )
        elif stage_mode == "rrf":
            # RRF without reranker (embed mode)
            searcher = HybridSearcher(
                backend=backend,
                storage=storage,
                limit=limit,
                collection=collection,
            )
            old_mode = backend.get_mode()
            backend.set_mode("embed")
            results = searcher.search(gt.query)
            backend.set_mode(old_mode)
        elif stage_mode == "hybrid":
            # Full hybrid with reranker
            searcher = HybridSearcher(
                backend=backend,
                storage=storage,
                limit=limit,
                collection=collection,
            )
            old_mode = backend.get_mode()
            backend.set_mode("hybrid")
            results = searcher.search(gt.query)
            backend.set_mode(old_mode)
        else:
            raise ValueError(f"Unknown stage mode: {stage_mode}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Normalize results to dicts and capture audit trail
    result_dicts = []
    for r in results:
        if isinstance(r, dict):
            result_dicts.append(r)
        else:
            result = {
                "filepath": getattr(r, "filepath", getattr(r, "path", "")),
                "title": getattr(r, "title", ""),
                "score": getattr(r, "score", 0.0),
            }
            # Capture audit trail if available (HybridResult has audit)
            audit = getattr(r, "audit", None)
            if audit is not None:
                result["audit"] = {
                    "filepath": audit.filepath,
                    "content_type": audit.content_type,
                    "rrf_sources": audit.rrf_sources,
                    "reranker_raw_score": audit.reranker_raw_score,
                    "reranker_normalized_score": audit.reranker_normalized_score,
                    "reranker_scoring_path": audit.reranker_scoring_path,
                    "blend_weights": audit.blend_weights,
                    "final_blended_score": audit.final_blended_score,
                }
            result_dicts.append(result)

    return result_dicts, elapsed_ms


def run_benchmark(
    backend,
    storage,
    collection: str = "benchmark",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full cross-modal ablation benchmark."""
    
    # Group queries by category
    categories = {}
    for gt in ALL_GROUND_TRUTH:
        categories.setdefault(gt.category, []).append(gt)

    print(f"\nQuery categories: {', '.join(f'{k}({len(v)})' for k, v in categories.items())}")

    # Ingest corpus
    print("\nIndexing corpus...")
    indexed = ingest_corpus(backend, storage, collection, CORPUS_DIR)
    print(f"Indexed {indexed} items.\n")

    # Run all stages × categories
    all_results: Dict[str, Dict[str, StageResult]] = {}

    for stage_name, stage_mode in STAGES:
        all_results[stage_name] = {}

        for cat_name, queries in categories.items():
            # Skip BM25 for image queries (BM25 can't process images)
            if stage_mode == "bm25" and cat_name == "image_to_text":
                sr = StageResult(stage=stage_name, category=cat_name, total_queries=len(queries))
                all_results[stage_name][cat_name] = sr
                print(f"  {stage_name} for {cat_name}: SKIPPED (BM25 can't embed images)")
                continue

            effective_mode = stage_mode

            sr = StageResult(stage=stage_name, category=cat_name, total_queries=len(queries))

            for gt in queries:
                try:
                    results, latency = run_search(
                        backend, storage, gt,
                        collection, effective_mode,
                    )
                    h1, h5, h10, ndcg, rr = evaluate_results(results, gt, CORPUS_DIR)
                    sr.hits_at_1 += int(h1)
                    sr.hits_at_5 += int(h5)
                    sr.hits_at_10 += int(h10)
                    sr.ndcg_sum += ndcg
                    sr.mrr_sum += rr
                    sr.latencies_ms.append(latency)
                    # Store per-query result with audit trail for post-hoc analysis
                    sr.per_query_results.append({
                        "query": gt.query,
                        "query_type": gt.query_type,
                        "image_query_path": gt.image_query_path,
                        "relevant_paths": gt.relevant_paths,
                        "hit_at_1": h1,
                        "hit_at_5": h5,
                        "hit_at_10": h10,
                        "ndcg": ndcg,
                        "mrr": rr,
                        "latency_ms": latency,
                        "results": results,  # Includes audit trail when available
                    })
                except Exception as e:
                    print(f"    ERROR: {gt.query[:50]}... → {e}")
                    sr.latencies_ms.append(0)
                    sr.per_query_results.append({
                        "query": gt.query,
                        "query_type": gt.query_type,
                        "image_query_path": gt.image_query_path,
                        "relevant_paths": gt.relevant_paths,
                        "error": str(e),
                    })

            all_results[stage_name][cat_name] = sr
            print(f"  {stage_name} for {cat_name} ({len(queries)}q)... "
                  f"R@1={sr.recall_at_1:.1%} R@5={sr.recall_at_5:.1%} "
                  f"R@10={sr.recall_at_10:.1%} NDCG@10={sr.ndcg_at_10:.3f} "
                  f"MRR={sr.mrr:.3f}")

    # Print results tables
    print("\n" + "=" * 110)
    for cat_name in categories:
        print(f"\n{'=' * 110}")
        print(f"  {cat_name.upper()} ({len(categories[cat_name])} queries)")
        print(f"{'=' * 110}")
        print(f"{'Stage':<35} {'R@1':>6} {'R@5':>6} {'R@10':>6} "
              f"{'NDCG@10':>8} {'MRR':>6} {'p50':>8} {'p95':>8}")
        print("-" * 110)

        for stage_name, _ in STAGES:
            sr = all_results[stage_name][cat_name]
            if sr.total_queries == 0:
                continue
            print(f"{stage_name:<35} "
                  f"{sr.recall_at_1:>5.1%} {sr.recall_at_5:>5.1%} {sr.recall_at_10:>5.1%} "
                  f"{sr.ndcg_at_10:>7.3f} {sr.mrr:>5.3f} "
                  f"{sr.p50_ms:>7.0f}ms {sr.p95_ms:>7.0f}ms")

    # Reranker uplift summary
    print(f"\n{'=' * 110}")
    print("  RERANKER UPLIFT SUMMARY (R@1 and NDCG@10)")
    print(f"{'=' * 110}")
    print(f"{'Category':<20} {'Vec R@1':>8} {'Rerank R@1':>10} {'Δ R@1':>8} "
          f"{'Vec NDCG':>9} {'Rerank NDCG':>12} {'Δ NDCG':>8}")
    print("-" * 80)

    for cat_name in categories:
        vec = all_results["Vector-only"].get(cat_name)
        rerank = all_results["Vector + BM25 + Reranker"].get(cat_name)
        if vec and rerank and vec.total_queries > 0:
            delta_r1 = rerank.recall_at_1 - vec.recall_at_1
            delta_ndcg = rerank.ndcg_at_10 - vec.ndcg_at_10
            print(f"{cat_name:<20} "
                  f"{vec.recall_at_1:>7.1%} {rerank.recall_at_1:>9.1%} "
                  f"{delta_r1:>+7.1%} "
                  f"{vec.ndcg_at_10:>8.3f} {rerank.ndcg_at_10:>11.3f} "
                  f"{delta_ndcg:>+7.3f}")

    # Save results
    output = {
        "benchmark": "cross_modal_ablation",
        "version": "0.2.0",
        "corpus": {
            "text_docs": len(list((CORPUS_DIR / "text").glob("*.md"))),
            "images": len(list((CORPUS_DIR / "images").glob("*.png"))),
            "total_queries": len(ALL_GROUND_TRUTH),
        },
        "categories": {},
        "stages": {},
    }

    for cat_name, queries in categories.items():
        output["categories"][cat_name] = {"queries": len(queries)}

    for stage_name, _ in STAGES:
        output["stages"][stage_name] = {}
        for cat_name in categories:
            sr = all_results[stage_name][cat_name]
            output["stages"][stage_name][cat_name] = {
                "recall_at_1": round(sr.recall_at_1, 4),
                "recall_at_5": round(sr.recall_at_5, 4),
                "recall_at_10": round(sr.recall_at_10, 4),
                "ndcg_at_10": round(sr.ndcg_at_10, 4),
                "mrr": round(sr.mrr, 4),
                "p50_ms": round(sr.p50_ms, 1),
                "p95_ms": round(sr.p95_ms, 1),
                "total_queries": sr.total_queries,
                "per_query_results": sr.per_query_results,  # Includes audit trails for post-hoc analysis
            }

    save_path = output_path or str(
        PROJECT_ROOT / "benchmarks" / "results" / "cross_modal_ablation_results.json"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {save_path}")

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-modal pipeline ablation benchmark"
    )
    parser.add_argument(
        "--backend", choices=["torch", "mlx", "auto"], default="auto",
        help="Model backend",
    )
    parser.add_argument(
        "--quantization", default="4bit",
        help="MLX quantization level",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--store-path", default=None,
        help="Storage directory (default: temp dir)",
    )
    args = parser.parse_args()

    # Initialize backend
    from recallforge import get_backend, get_storage

    store_path = args.store_path or tempfile.mkdtemp(prefix="recallforge_bench_")
    cleanup = args.store_path is None

    try:
        os.environ["RECALLFORGE_BACKEND"] = args.backend
        if args.quantization:
            os.environ["RECALLFORGE_MLX_QUANTIZE"] = args.quantization

        backend = get_backend()
        storage = get_storage(store_path)
        storage.initialize(store_path)

        info = backend.get_info()
        print(f"Backend: {info.name}")
        print(f"Device: {info.device}")
        print(f"Quantization: {info.quantization}")
        print(f"Store: {store_path}")

        run_benchmark(backend, storage, output_path=args.output)

    finally:
        if cleanup and os.path.exists(store_path):
            shutil.rmtree(store_path, ignore_errors=True)


if __name__ == "__main__":
    main()
