#!/usr/bin/env python3
"""Per-modality pipeline ablation benchmark for RecallForge.

Breaks down pipeline ablation by:
1. Text-only queries (searching for text docs)
2. Image/diagram queries (searching for image docs)  
3. Cross-modal queries (text queries that should find image docs, and vice versa)
4. Edge cases: short queries (1-2 words), long queries (10+ words), typo queries

Reuses the same synthetic corpus from pipeline_ablation.py but evaluates
each query category separately to measure reranker uplift per modality.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline_ablation import (
    RESULTS_DIR,
    TestDocument,
    GroundTruth,
    SyntheticBackend,
    generate_synthetic_corpus,
    index_documents,
    evaluate_ranked_keys,
    to_search_doc_key,
    to_hybrid_doc_key,
    percentile,
    try_get_real_backend,
    slugify,
)
from recallforge.search import HybridSearcher
from recallforge.storage.lancedb_backend import LanceDBBackend
from recallforge.backends.base import ModelBackend


@dataclass
class ModalityResult:
    category: str
    stage_name: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_queries: int
    hits_at_1: int
    hits_at_5: int
    hits_at_10: int


def categorize_queries(
    ground_truth: List[GroundTruth],
    documents: List[TestDocument],
) -> Dict[str, List[GroundTruth]]:
    """Split queries into categories based on what they're searching for."""
    doc_map = {d.doc_id: d for d in documents}

    categories: Dict[str, List[GroundTruth]] = {
        "text_only": [],
        "image_only": [],
        "cross_modal": [],
        "short_query": [],
        "long_query": [],
        "typo_query": [],
    }

    for gt in ground_truth:
        # Determine if targets are text or image
        target_types = set()
        for doc_id in gt.relevant_doc_ids:
            if doc_id in doc_map:
                target_types.add(doc_map[doc_id].content_type)

        # Check query characteristics
        words = gt.query.split()
        is_image_query = any(
            w in gt.query.lower()
            for w in ["diagram", "visual", "image", "picture", "photo"]
        )
        targets_images = "image" in target_types
        targets_text = "text" in target_types

        # Primary categorization
        if targets_images and not targets_text:
            categories["image_only"].append(gt)
        elif targets_text and not targets_images:
            categories["text_only"].append(gt)

        # Cross-modal: text-style query targeting image docs (or vice versa)
        if targets_images and not is_image_query:
            categories["cross_modal"].append(gt)
        elif not targets_images and is_image_query:
            categories["cross_modal"].append(gt)

        # Length-based
        if len(words) <= 2:
            categories["short_query"].append(gt)
        elif len(words) >= 6:
            categories["long_query"].append(gt)

    # Generate typo variants from a sample of queries
    rng = random.Random(42)
    sample = rng.sample(ground_truth, min(20, len(ground_truth)))
    for gt in sample:
        typo_query = inject_typo(gt.query, rng)
        if typo_query != gt.query:
            categories["typo_query"].append(
                GroundTruth(query=typo_query, relevant_doc_ids=gt.relevant_doc_ids)
            )

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def inject_typo(query: str, rng: random.Random) -> str:
    """Inject a realistic typo into a query."""
    words = query.split()
    if len(words) < 2:
        return query
    idx = rng.randint(0, len(words) - 1)
    word = words[idx]
    if len(word) < 3:
        return query
    # Swap two adjacent chars
    pos = rng.randint(0, len(word) - 2)
    chars = list(word)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    words[idx] = "".join(chars)
    return " ".join(words)


def run_stage_for_category(
    storage: LanceDBBackend,
    backend: ModelBackend,
    queries: List[GroundTruth],
    collection: str,
    stage: str,
    stage_name: str,
    category: str,
) -> ModalityResult:
    """Run a single pipeline stage for a single query category."""
    metrics: List[Tuple[float, float, float, float]] = []
    latencies: List[float] = []

    if stage == "vector":
        for gt in queries:
            vec = backend.embed_text(gt.query)
            start = time.perf_counter()
            results = storage.search_vec(vec.tolist(), limit=10, collection=collection)
            latencies.append((time.perf_counter() - start) * 1000.0)
            ranked = [to_search_doc_key(r) for r in results]
            metrics.append(evaluate_ranked_keys(ranked, gt.relevant_doc_ids))
    elif stage == "bm25":
        for gt in queries:
            start = time.perf_counter()
            results = storage.search_fts(gt.query, limit=10, collection=collection)
            latencies.append((time.perf_counter() - start) * 1000.0)
            ranked = [to_search_doc_key(r) for r in results]
            metrics.append(evaluate_ranked_keys(ranked, gt.relevant_doc_ids))
    else:
        mode_map = {
            "rrf": "embed",
            "reranker": "hybrid",
        }
        backend.set_mode(mode_map[stage])
        searcher = HybridSearcher(
            backend=backend, storage=storage, limit=10, collection=collection
        )
        for gt in queries:
            start = time.perf_counter()
            results = searcher.search(gt.query)
            latencies.append((time.perf_counter() - start) * 1000.0)
            ranked = [to_hybrid_doc_key(r) for r in results]
            metrics.append(evaluate_ranked_keys(ranked, gt.relevant_doc_ids))

    total = len(metrics)
    r1_vals = [m[0] for m in metrics]
    r5_vals = [m[1] for m in metrics]
    r10_vals = [m[2] for m in metrics]
    mrr_vals = [m[3] for m in metrics]

    return ModalityResult(
        category=category,
        stage_name=stage_name,
        recall_at_1=float(sum(r1_vals) / total if total else 0.0),
        recall_at_5=float(sum(r5_vals) / total if total else 0.0),
        recall_at_10=float(sum(r10_vals) / total if total else 0.0),
        mrr=float(sum(mrr_vals) / total if total else 0.0),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        total_queries=total,
        hits_at_1=int(sum(r1_vals)),
        hits_at_5=int(sum(r5_vals)),
        hits_at_10=int(sum(r10_vals)),
    )


STAGES = [
    ("vector", "Vector-only"),
    ("bm25", "BM25-only"),
    ("rrf", "Vector + BM25 (RRF)"),
    ("reranker", "Vector + BM25 + Reranker"),
]


def print_category_table(category: str, results: List[ModalityResult]) -> None:
    print(f"\n{'=' * 90}")
    print(f"  {category.upper().replace('_', ' ')} ({results[0].total_queries} queries)")
    print(f"{'=' * 90}")
    header = f"{'Stage':<35s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s} {'p50':>10s} {'p95':>10s}"
    print(header)
    print("-" * 90)
    for r in results:
        print(
            f"{r.stage_name:<35s} {r.recall_at_1:>5.1%} {r.recall_at_5:>5.1%} "
            f"{r.recall_at_10:>5.1%} {r.mrr:>6.4f} {r.latency_p50_ms:>8.0f}ms {r.latency_p95_ms:>8.0f}ms"
        )
    # Show reranker uplift vs vector-only
    vector = next((r for r in results if "Vector-only" in r.stage_name), None)
    reranker = next((r for r in results if "Reranker" in r.stage_name), None)
    if vector and reranker:
        delta_r1 = reranker.recall_at_1 - vector.recall_at_1
        sign = lambda x: f"+{x:.1%}" if x >= 0 else f"{x:.1%}"
        print(f"\n  Reranker uplift vs vector-only: R@1={sign(delta_r1)}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-modality pipeline ablation benchmark"
    )
    parser.add_argument(
        "--backend-mode",
        choices=["synthetic", "real"],
        default="real",
    )
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--num-text-docs", type=int, default=200)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "pipeline_ablation_modality_results.json",
    )
    args = parser.parse_args()

    documents, ground_truth = generate_synthetic_corpus(
        num_text_docs=args.num_text_docs,
        num_images=args.num_images,
        seed=args.seed,
    )

    categories = categorize_queries(ground_truth, documents)
    print(f"Query categories: {', '.join(f'{k}({len(v)})' for k, v in categories.items())}")

    collection = "modality_ablation"
    tempdir = tempfile.mkdtemp(prefix="recallforge-modality-ablation-")
    store_path = Path(tempdir) / "store"
    store_path.mkdir(parents=True, exist_ok=True)

    notes: List[str] = []
    try:
        if args.backend_mode == "real":
            backend = try_get_real_backend(
                backend_name=args.backend,
                quantization=args.quantization,
                max_mode="embed",
            )
            notes.append("Using real RecallForge backend.")
        else:
            backend = SyntheticBackend(mode="hybrid")
            notes.append("Using synthetic backend.")
    except Exception as exc:
        backend = SyntheticBackend(mode="hybrid")
        notes.append(f"Fell back to synthetic: {exc}")

    storage = LanceDBBackend(str(store_path))
    storage.initialize(str(store_path))

    try:
        print("Indexing documents...")
        index_documents(storage, backend, documents, collection)
        storage.ensure_fts_index(force_rebuild=True)
        print(f"Indexed {len(documents)} documents.")

        all_results: Dict[str, List[ModalityResult]] = {}

        for cat_name, cat_queries in categories.items():
            cat_results = []
            for stage_key, stage_name in STAGES:
                print(f"  Running {stage_name} for {cat_name} ({len(cat_queries)} queries)...", end="", flush=True)
                result = run_stage_for_category(
                    storage, backend, cat_queries, collection,
                    stage_key, stage_name, cat_name,
                )
                cat_results.append(result)
                print(f" R@1={result.recall_at_1:.1%}")
            all_results[cat_name] = cat_results
            print_category_table(cat_name, cat_results)

        # Summary: reranker uplift per category
        print(f"\n{'=' * 90}")
        print("  RERANKER UPLIFT SUMMARY")
        print(f"{'=' * 90}")
        print(f"{'Category':<20s} {'Vector R@1':>12s} {'Reranker R@1':>14s} {'Uplift':>8s}")
        print("-" * 60)
        for cat_name, cat_results in all_results.items():
            vector = next((r for r in cat_results if "Vector-only" in r.stage_name), None)
            reranker = next((r for r in cat_results if "Reranker" in r.stage_name), None)
            if vector and reranker:
                delta = reranker.recall_at_1 - vector.recall_at_1
                print(
                    f"{cat_name:<20s} {vector.recall_at_1:>11.1%} {reranker.recall_at_1:>13.1%} "
                    f"{delta:>+7.1%}"
                )

        # Save
        payload = {
            "benchmark": "pipeline_ablation_modality",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": {
                "backend_mode": args.backend_mode,
                "backend": args.backend,
                "quantization": args.quantization,
                "num_text_docs": args.num_text_docs,
                "num_images": args.num_images,
                "seed": args.seed,
            },
            "notes": notes,
            "categories": {
                cat: [asdict(r) for r in results]
                for cat, results in all_results.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved: {args.output}")
        return 0

    finally:
        try:
            storage.close()
        except Exception:
            pass
        shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
