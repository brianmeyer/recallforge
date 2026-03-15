#!/usr/bin/env python3
"""Pipeline ablation benchmark for RecallForge.

This benchmark measures how each stage of the RecallForge search pipeline
improves retrieval quality:

1. Vector-only
2. BM25-only
3. Vector + BM25 (RRF)
4. Vector + BM25 + Reranker
5. Full pipeline (+ query expansion)

It is designed to be runnable in two modes:
- synthetic: fast, deterministic, no heavyweight model downloads
- real: uses the actual RecallForge backend (e.g. MLX) if available

Outputs:
- benchmarks/results/pipeline_ablation_dataset.json
- benchmarks/results/pipeline_ablation_ground_truth.json
- benchmarks/results/pipeline_ablation_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Add src to path for local execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recallforge.search import HybridSearcher
from recallforge.storage.base import SearchResult
from recallforge.backends.base import BackendInfo, ModelBackend
from recallforge.storage.lancedb_backend import LanceDBBackend


RESULTS_DIR = Path("benchmarks/results")
DATASET_JSON = RESULTS_DIR / "pipeline_ablation_dataset.json"
GROUND_TRUTH_JSON = RESULTS_DIR / "pipeline_ablation_ground_truth.json"
RESULTS_JSON = RESULTS_DIR / "pipeline_ablation_results.json"


# =============================================================================
# Synthetic dataset
# =============================================================================

TOPICS: List[Dict[str, Any]] = [
    {
        "title": "Implementing RAG Systems",
        "keywords": ["rag", "retrieval", "augmentation", "vector store", "embedding", "context window"],
        "queries": [
            "how to implement rag systems",
            "retrieval augmented generation guide",
            "vector store and context retrieval",
        ],
    },
    {
        "title": "Vector Database Optimization",
        "keywords": ["vector database", "ann", "hnsw", "indexing", "cosine similarity", "latency"],
        "queries": [
            "vector database optimization",
            "improving ann search latency",
            "cosine similarity index tuning",
        ],
    },
    {
        "title": "Machine Learning Pipeline Design",
        "keywords": ["ml pipeline", "feature engineering", "training", "evaluation", "deployment", "orchestration"],
        "queries": [
            "machine learning pipeline design",
            "ml training and deployment workflow",
            "feature engineering orchestration",
        ],
    },
    {
        "title": "Distributed Systems Architecture",
        "keywords": ["distributed systems", "consensus", "replication", "partitioning", "throughput", "fault tolerance"],
        "queries": [
            "distributed systems architecture",
            "replication and partitioning patterns",
            "fault tolerant service design",
        ],
    },
    {
        "title": "Semantic Search Implementation",
        "keywords": ["semantic search", "embedding model", "reranking", "bm25", "hybrid search", "query expansion"],
        "queries": [
            "semantic search implementation",
            "hybrid search with bm25 and vectors",
            "reranking and query expansion pipeline",
        ],
    },
    {
        "title": "Kubernetes Deployment Strategies",
        "keywords": ["kubernetes", "deployment", "rolling update", "helm", "autoscaling", "service mesh"],
        "queries": [
            "kubernetes deployment strategies",
            "rolling update and autoscaling",
            "helm and service mesh setup",
        ],
    },
    {
        "title": "Observability and Monitoring",
        "keywords": ["observability", "metrics", "tracing", "logging", "alerts", "slo"],
        "queries": [
            "observability and monitoring",
            "distributed tracing and metrics",
            "alerts and slo dashboards",
        ],
    },
    {
        "title": "Cross-Modal Retrieval",
        "keywords": ["cross modal", "image search", "text image", "vision language model", "caption", "retrieval"],
        "queries": [
            "cross modal retrieval",
            "text to image retrieval system",
            "vision language search pipeline",
        ],
    },
    {
        "title": "API Gateway Patterns",
        "keywords": ["api gateway", "rate limiting", "authentication", "routing", "aggregation", "backend for frontend"],
        "queries": [
            "api gateway patterns",
            "rate limiting and routing",
            "backend for frontend architecture",
        ],
    },
    {
        "title": "Feature Store Design",
        "keywords": ["feature store", "offline store", "online store", "training serving skew", "features", "mlops"],
        "queries": [
            "feature store design",
            "online and offline feature store",
            "prevent training serving skew",
        ],
    },
]

TEXT_STYLES = ["markdown", "code", "meeting", "technical"]


@dataclass
class TestDocument:
    doc_id: str
    title: str
    content: str
    content_type: str
    queries: List[str]
    topic: str
    keywords: List[str]


@dataclass
class GroundTruth:
    query: str
    relevant_doc_ids: List[str]


@dataclass
class BenchmarkResult:
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


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_text_content(title: str, keywords: List[str], style: str, idx: int) -> str:
    a, b, c, d, e, f = keywords[:6]
    if style == "markdown":
        return f"""# {title}

This note covers {a}, {b}, and {c} in practical systems.

## Summary
- {a} improves retrieval quality
- {b} affects scalability and latency
- {c} helps production reliability

## Details
Teams often combine {d}, {e}, and {f} to improve outcomes.
This document includes examples, tradeoffs, and implementation details.
"""
    if style == "code":
        return f"""```python
# {title}

def build_pipeline(query, docs):
    # keywords: {a}, {b}, {c}, {d}, {e}, {f}
    candidates = retrieve(query, docs)
    ranked = rerank(candidates)
    return ranked

# implementation notes about {a}, {b}, {c}, and {d}
```
"""
    if style == "meeting":
        return f"""# Meeting Notes: {title}

Attendees: Alice, Bob, Carol

## Agenda
1. Discuss {a}
2. Review {b}
3. Plan {c}

## Decisions
We will invest in {d} and {e} this quarter.
Action item: prototype {f} before the next sprint.
"""
    return f"""# {title}

## Overview
This technical document explains {a}, {b}, and {c}.

## Architecture
The system uses {d}, {e}, and {f} for production deployment.

## Notes
The implementation balances recall, latency, and operational complexity.
"""


def generate_synthetic_corpus(num_text_docs: int = 200, num_images: int = 50, seed: int = 42) -> Tuple[List[TestDocument], List[GroundTruth]]:
    rng = random.Random(seed)
    documents: List[TestDocument] = []
    query_to_docs: Dict[str, List[str]] = {}

    # Text docs
    for i in range(num_text_docs):
        topic = TOPICS[i % len(TOPICS)]
        style = TEXT_STYLES[i % len(TEXT_STYLES)]
        doc_id = f"text_{i:04d}"
        title = f"{topic['title']} {i // len(TOPICS) + 1}"
        content = build_text_content(title, topic["keywords"], style, i)
        queries = list(topic["queries"])
        doc = TestDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            content_type="text",
            queries=queries,
            topic=topic["title"],
            keywords=list(topic["keywords"]),
        )
        documents.append(doc)
        for q in queries:
            query_to_docs.setdefault(q, []).append(doc_id)

    # Image docs represented as caption-rich pseudo documents for benchmarking.
    for i in range(num_images):
        topic = TOPICS[(i + 3) % len(TOPICS)]
        doc_id = f"image_{i:04d}"
        title = f"Diagram {topic['title']} {i // len(TOPICS) + 1}"
        k = topic["keywords"]
        content = (
            f"# {title}\n\n"
            f"Caption: diagram showing {k[0]}, {k[1]}, and {k[2]}.\n\n"
            f"Description: visual architecture for {topic['title'].lower()} using {k[3]}, {k[4]}, and {k[5]}.\n"
        )
        queries = [
            f"diagram about {topic['title'].lower()}",
            f"visual architecture for {k[0]} and {k[1]}",
            f"image showing {k[2]} pipeline",
        ]
        doc = TestDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            content_type="image",
            queries=queries,
            topic=topic["title"],
            keywords=list(k),
        )
        documents.append(doc)
        for q in queries:
            query_to_docs.setdefault(q, []).append(doc_id)

    ground_truth = [GroundTruth(query=q, relevant_doc_ids=doc_ids) for q, doc_ids in sorted(query_to_docs.items())]
    return documents, ground_truth


# =============================================================================
# Lightweight synthetic backend
# =============================================================================

class SyntheticBackend(ModelBackend):
    """Deterministic lightweight backend for pipeline ablation.

    Embeddings are simple hashed bag-of-words vectors. Reranker scores token overlap.
    Expander adds lexical/semantic/hypothetical variants.

    Dimension is kept at 2048 to match RecallForge storage schema.
    """

    def __init__(self, mode: str = "full", dim: int = 2048):
        self._mode = mode
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _embed_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed_tokens(self._tokenize(text))

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed_text(t) for t in texts]).astype(np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        return self.embed_text(image_path)

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        return self.embed_texts(image_paths)

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        q = set(self._tokenize(query))
        scores: List[float] = []
        for doc in documents:
            text = doc.get("text") or doc.get("text_body") or ""
            d = set(self._tokenize(text))
            if not q or not d:
                scores.append(0.0)
                continue
            overlap = len(q & d)
            union = len(q | d)
            scores.append(overlap / union if union else 0.0)
        return scores

    def expand_query(self, query: str) -> Dict[str, str]:
        tokens = self._tokenize(query)
        lex = query + " guide tutorial implementation"
        vec = query + " semantic related concepts architecture"
        hyde = "A relevant document discussing " + " ".join(tokens[:8])
        return {"lex": lex, "vec": vec, "hyde": hyde}

    def warm_up(self) -> None:
        return None

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            name="synthetic",
            device="cpu",
            dtype="float32",
            embedder_loaded=True,
            reranker_loaded=True,
            expander_loaded=True,
            memory_allocated_gb=0.0,
            supports_images=True,
            quantization=None,
        )


# =============================================================================
# Benchmark utilities
# =============================================================================


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, math.ceil((p / 100.0) * len(vals)) - 1))
    return float(vals[idx])


def to_search_doc_key(result: SearchResult) -> str:
    display = result.display_path or result.filepath
    return Path(display).name


def to_hybrid_doc_key(result: Any) -> str:
    display = result.display_path or result.filepath
    return Path(display).name


def evaluate_ranked_keys(ranked_doc_ids: List[str], relevant_doc_ids: List[str]) -> Tuple[float, float, float, float]:
    relevant = set(relevant_doc_ids)
    r1 = 1.0 if ranked_doc_ids[:1] and ranked_doc_ids[0] in relevant else 0.0
    r5 = 1.0 if any(doc in relevant for doc in ranked_doc_ids[:5]) else 0.0
    r10 = 1.0 if any(doc in relevant for doc in ranked_doc_ids[:10]) else 0.0
    mrr = 0.0
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            mrr = 1.0 / idx
            break
    return r1, r5, r10, mrr


def compute_benchmark_result(stage_name: str, metrics: List[Tuple[float, float, float, float]], latencies_ms: List[float]) -> BenchmarkResult:
    total = len(metrics)
    r1_vals = [m[0] for m in metrics]
    r5_vals = [m[1] for m in metrics]
    r10_vals = [m[2] for m in metrics]
    mrr_vals = [m[3] for m in metrics]
    return BenchmarkResult(
        stage_name=stage_name,
        recall_at_1=float(sum(r1_vals) / total if total else 0.0),
        recall_at_5=float(sum(r5_vals) / total if total else 0.0),
        recall_at_10=float(sum(r10_vals) / total if total else 0.0),
        mrr=float(sum(mrr_vals) / total if total else 0.0),
        latency_p50_ms=percentile(latencies_ms, 50),
        latency_p95_ms=percentile(latencies_ms, 95),
        total_queries=total,
        hits_at_1=int(sum(r1_vals)),
        hits_at_5=int(sum(r5_vals)),
        hits_at_10=int(sum(r10_vals)),
    )


def index_documents(storage: LanceDBBackend, backend: ModelBackend, documents: List[TestDocument], collection: str) -> None:
    with storage.bulk_mode():
        for doc in documents:
            storage.upsert_memory(
                path=doc.doc_id,
                text=doc.content,
                collection=collection,
                embed_func=backend.embed_text,
                model="synthetic-embed" if isinstance(backend, SyntheticBackend) else "Qwen3-VL-Embedding-2B",
            )


def run_vector_only(storage: LanceDBBackend, backend: ModelBackend, queries: List[GroundTruth], collection: str) -> BenchmarkResult:
    metrics: List[Tuple[float, float, float, float]] = []
    latencies: List[float] = []
    for gt in queries:
        vec = backend.embed_text(gt.query)
        start = time.perf_counter()
        results = storage.search_vec(vec.tolist(), limit=10, collection=collection)
        latencies.append((time.perf_counter() - start) * 1000.0)
        ranked = [to_search_doc_key(r) for r in results]
        metrics.append(evaluate_ranked_keys(ranked, gt.relevant_doc_ids))
    return compute_benchmark_result("Vector-only", metrics, latencies)


def run_bm25_only(storage: LanceDBBackend, queries: List[GroundTruth], collection: str) -> BenchmarkResult:
    metrics: List[Tuple[float, float, float, float]] = []
    latencies: List[float] = []
    for gt in queries:
        start = time.perf_counter()
        results = storage.search_fts(gt.query, limit=10, collection=collection)
        latencies.append((time.perf_counter() - start) * 1000.0)
        ranked = [to_search_doc_key(r) for r in results]
        metrics.append(evaluate_ranked_keys(ranked, gt.relevant_doc_ids))
    return compute_benchmark_result("BM25-only", metrics, latencies)


def run_hybrid_stage(storage: LanceDBBackend, backend: ModelBackend, queries: List[GroundTruth], collection: str, mode: str, stage_name: str) -> BenchmarkResult:
    backend.set_mode(mode)
    searcher = HybridSearcher(backend=backend, storage=storage, limit=10, collection=collection)
    metrics: List[Tuple[float, float, float, float]] = []
    latencies: List[float] = []
    for gt in queries:
        start = time.perf_counter()
        results = searcher.search(gt.query)
        latencies.append((time.perf_counter() - start) * 1000.0)
        ranked = [to_hybrid_doc_key(r) for r in results]
        metrics.append(evaluate_ranked_keys(ranked, gt.relevant_doc_ids))
    return compute_benchmark_result(stage_name, metrics, latencies)


def format_markdown_table(stage_results: List[BenchmarkResult]) -> str:
    lines = [
        "## Pipeline Ablation Results",
        "",
        "| Stage | R@1 | R@5 | R@10 | MRR | Latency p50 (ms) | Latency p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in stage_results:
        lines.append(
            f"| {r.stage_name} | {r.recall_at_1:.4f} | {r.recall_at_5:.4f} | {r.recall_at_10:.4f} | {r.mrr:.4f} | {r.latency_p50_ms:.2f} | {r.latency_p95_ms:.2f} |"
        )
    return "\n".join(lines)


def print_summary(stage_results: List[BenchmarkResult]) -> None:
    print("\n" + "=" * 80)
    print("PIPELINE ABLATION RESULTS")
    print("=" * 80)
    print("| Stage | R@1 | R@5 | R@10 | MRR | p50 | p95 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in stage_results:
        print(
            f"| {r.stage_name} | {r.recall_at_1:.4f} | {r.recall_at_5:.4f} | {r.recall_at_10:.4f} | {r.mrr:.4f} | {r.latency_p50_ms:.2f}ms | {r.latency_p95_ms:.2f}ms |"
        )


def try_get_real_backend(backend_name: str, quantization: str, max_mode: str = "full") -> ModelBackend:
    os.environ["RECALLFORGE_BACKEND"] = backend_name
    os.environ["RECALLFORGE_MLX_QUANTIZE"] = quantization
    os.environ["RECALLFORGE_MODE"] = max_mode
    from recallforge import get_backend
    backend = get_backend()
    backend.set_mode("embed")
    backend.warm_up()
    return backend


def save_dataset_outputs(documents: List[TestDocument], ground_truth: List[GroundTruth]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_JSON.open("w") as f:
        json.dump([asdict(d) for d in documents], f, indent=2)
    with GROUND_TRUTH_JSON.open("w") as f:
        json.dump([asdict(g) for g in ground_truth], f, indent=2)


def run_benchmark(
    backend_mode: str,
    backend_name: str,
    quantization: str,
    num_text_docs: int,
    num_images: int,
    seed: int,
    output_path: Path,
) -> Dict[str, Any]:
    documents, ground_truth = generate_synthetic_corpus(
        num_text_docs=num_text_docs,
        num_images=num_images,
        seed=seed,
    )
    save_dataset_outputs(documents, ground_truth)

    collection = "pipeline_ablation"
    tempdir = tempfile.mkdtemp(prefix="recallforge-pipeline-ablation-")
    store_path = Path(tempdir) / "store"
    store_path.mkdir(parents=True, exist_ok=True)

    backend: ModelBackend
    effective_backend_name = backend_name
    notes: List[str] = []

    try:
        if backend_mode == "real":
            backend = try_get_real_backend(backend_name=backend_name, quantization=quantization, max_mode="embed")
            notes.append("Using real RecallForge backend.")
        else:
            backend = SyntheticBackend(mode="full")
            effective_backend_name = "synthetic"
            notes.append("Using deterministic synthetic backend for fast ablation benchmarking.")
    except Exception as exc:
        backend = SyntheticBackend(mode="full")
        effective_backend_name = "synthetic-fallback"
        notes.append(f"Fell back to synthetic backend because real backend init failed: {exc}")

    storage = LanceDBBackend(str(store_path))
    storage.initialize(str(store_path))

    try:
        index_documents(storage, backend, documents, collection)
        storage.ensure_fts_index(force_rebuild=True)

        stage_results = [
            run_vector_only(storage, backend, ground_truth, collection),
            run_bm25_only(storage, ground_truth, collection),
            run_hybrid_stage(storage, backend, ground_truth, collection, mode="embed", stage_name="Vector + BM25 (RRF)"),
            run_hybrid_stage(storage, backend, ground_truth, collection, mode="hybrid", stage_name="Vector + BM25 + Reranker"),
            run_hybrid_stage(storage, backend, ground_truth, collection, mode="full", stage_name="Full pipeline"),
        ]

        print_summary(stage_results)
        markdown_table = format_markdown_table(stage_results)

        payload = {
            "benchmark": "pipeline_ablation",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": {
                "backend_mode": backend_mode,
                "backend": effective_backend_name,
                "requested_backend": backend_name,
                "quantization": quantization,
                "num_text_docs": num_text_docs,
                "num_images": num_images,
                "seed": seed,
                "dataset_path": str(DATASET_JSON),
                "ground_truth_path": str(GROUND_TRUTH_JSON),
            },
            "notes": notes,
            "results": {slugify(r.stage_name): asdict(r) for r in stage_results},
            "comparison_table_markdown": markdown_table,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)

        print(f"\nSaved dataset: {DATASET_JSON}")
        print(f"Saved ground truth: {GROUND_TRUTH_JSON}")
        print(f"Saved results: {output_path}")
        return payload
    finally:
        try:
            storage.close()
        except Exception:
            pass
        shutil.rmtree(tempdir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RecallForge pipeline ablation benchmark")
    parser.add_argument("--backend-mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--backend", choices=["auto", "mlx", "torch"], default="mlx")
    parser.add_argument("--quantization", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument("--num-text-docs", type=int, default=200)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=RESULTS_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_benchmark(
        backend_mode=args.backend_mode,
        backend_name=args.backend,
        quantization=args.quantization,
        num_text_docs=args.num_text_docs,
        num_images=args.num_images,
        seed=args.seed,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
