"""
benchmark.py - Benchmark Suite for RecallForge.

Tests all combinations of:
- Backends: torch, mlx (if available)
- Modes: embed, hybrid, full

Metrics:
- Cold load time (first model load)
- Warm latency (query time after models loaded)
- Memory usage (approximate)
- Indexing throughput
- Recall@5, MRR
- Cross-modal accuracy
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    backend: str
    mode: str
    quantization: Optional[str]
    
    # Load times
    embedder_load_time_s: float = 0.0
    reranker_load_time_s: float = 0.0
    expander_load_time_s: float = 0.0
    total_load_time_s: float = 0.0
    
    # Memory
    memory_allocated_gb: float = 0.0
    
    # Query latency (warm)
    query_latency_p50_ms: float = 0.0
    query_latency_p95_ms: float = 0.0
    query_latency_p99_ms: float = 0.0
    
    # Indexing
    indexing_throughput_docs_per_sec: float = 0.0
    
    # Quality metrics
    recall_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    
    # Cross-modal
    text_to_image_recall_at_5: float = 0.0
    image_to_text_recall_at_5: float = 0.0
    
    # Errors
    error: Optional[str] = None


def benchmark_backend(
    backend_name: str,
    mode: str,
    quantization: str = "bf16",
    test_queries: List[Dict] = None,
    test_docs: List[Dict] = None,
    test_images: List[str] = None,
) -> BenchmarkResult:
    """
    Run benchmark for a specific backend and mode.
    
    Args:
        backend_name: 'torch' or 'mlx'
        mode: 'embed', 'hybrid', or 'full'
        quantization: 'bf16' or '4bit' (MLX only)
        test_queries: List of {"query": str, "relevant": [doc_ids]}
        test_docs: List of {"id": str, "text": str}
        test_images: List of image paths for cross-modal tests
    
    Returns:
        BenchmarkResult with all metrics
    """
    result = BenchmarkResult(
        backend=backend_name,
        mode=mode,
        quantization=quantization if backend_name == "mlx" else None,
    )
    
    try:
        # Set environment
        os.environ["RECALLFORGE_BACKEND"] = backend_name
        os.environ["RECALLFORGE_MODE"] = mode
        if backend_name == "mlx":
            os.environ["RECALLFORGE_MLX_QUANTIZE"] = quantization
        
        # Import after setting env
        from recallforge import get_backend, get_storage
        from recallforge.storage.lancedb_backend import LanceDBBackend
        import tempfile
        import shutil
        
        # Create temp storage
        temp_dir = tempfile.mkdtemp(prefix="recallforge-bench-")
        
        try:
            # Get backend
            backend = get_backend()
            
            # Benchmark model loading
            print(f"\n[{backend_name}/{mode}] Loading models...")
            gc.collect()
            
            start = time.time()
            backend._load_embedder()
            result.embedder_load_time_s = time.time() - start
            print(f"  Embedder: {result.embedder_load_time_s:.1f}s")
            
            if backend.needs_reranker():
                start = time.time()
                backend._load_reranker()
                result.reranker_load_time_s = time.time() - start
                print(f"  Reranker: {result.reranker_load_time_s:.1f}s")
            
            result.total_load_time_s = (
                result.embedder_load_time_s +
                result.reranker_load_time_s
            )
            
            # Get memory
            info = backend.get_info()
            result.memory_allocated_gb = info.memory_allocated_gb
            
            # Initialize storage
            storage = get_storage(temp_dir)
            
            # Index test documents
            if test_docs:
                print(f"\n[{backend_name}/{mode}] Indexing {len(test_docs)} documents...")
                start = time.time()
                for doc in test_docs:
                    storage.index_document(
                        path=doc["id"],
                        text=doc["text"],
                        collection="benchmark",
                        model="Qwen3-VL-Embedding-2B",
                        embed_func=backend.embed_text,
                    )
                index_time = time.time() - start
                result.indexing_throughput_docs_per_sec = len(test_docs) / index_time
                print(f"  Throughput: {result.indexing_throughput_docs_per_sec:.1f} docs/s")
            
            # Benchmark query latency
            if test_queries:
                print(f"\n[{backend_name}/{mode}] Benchmarking query latency...")
                from recallforge.search import HybridSearcher
                
                searcher = HybridSearcher(
                    backend=backend,
                    storage=storage,
                    limit=5,
                )
                
                latencies = []
                for q in test_queries:
                    start = time.time()
                    searcher.search(q["query"])
                    latencies.append((time.time() - start) * 1000)  # ms
                
                latencies.sort()
                result.query_latency_p50_ms = latencies[len(latencies) // 2]
                result.query_latency_p95_ms = latencies[int(len(latencies) * 0.95)]
                result.query_latency_p99_ms = latencies[int(len(latencies) * 0.99)]
                print(f"  P50: {result.query_latency_p50_ms:.1f}ms, P95: {result.query_latency_p95_ms:.1f}ms, P99: {result.query_latency_p99_ms:.1f}ms")
                
                # Compute recall@5 and MRR
                print(f"\n[{backend_name}/{mode}] Computing quality metrics...")
                recall_scores = []
                rr_scores = []
                
                for q in test_queries:
                    results = searcher.search(q["query"])
                    relevant = set(q.get("relevant", []))
                    
                    if not relevant:
                        continue
                    
                    # Recall@5
                    retrieved = set(r.filepath.split("/")[-1] for r in results[:5])
                    recall = len(retrieved & relevant) / min(len(relevant), 5)
                    recall_scores.append(recall)
                    
                    # MRR
                    for i, r in enumerate(results):
                        if r.filepath.split("/")[-1] in relevant:
                            rr_scores.append(1.0 / (i + 1))
                            break
                    else:
                        rr_scores.append(0.0)
                
                if recall_scores:
                    result.recall_at_5 = sum(recall_scores) / len(recall_scores)
                if rr_scores:
                    result.mrr = sum(rr_scores) / len(rr_scores)
                print(f"  Recall@5: {result.recall_at_5:.3f}, MRR: {result.mrr:.3f}")
            
            # Cross-modal tests
            if test_images and test_queries:
                print(f"\n[{backend_name}/{mode}] Cross-modal tests...")
                # Index test images
                for img_path in test_images:
                    try:
                        storage.index_image(
                            path=img_path,
                            collection="benchmark_images",
                            embed_func=backend.embed_image,
                        )
                    except Exception as e:
                        print(f"  Warning: Could not index {img_path}: {e}")
                
                # Text-to-image
                text_to_image_recalls = []
                for q in test_queries:
                    if "relevant_images" not in q:
                        continue
                    results = storage.search_vec(
                        vector=backend.embed_text(q["query"]).tolist(),
                        limit=5,
                        collection="benchmark_images",
                        content_type="image",
                    )
                    relevant = set(q["relevant_images"])
                    retrieved = set(r.filepath.split("/")[-1] for r in results[:5])
                    recall = len(retrieved & relevant) / min(len(relevant), 5)
                    text_to_image_recalls.append(recall)
                
                if text_to_image_recalls:
                    result.text_to_image_recall_at_5 = sum(text_to_image_recalls) / len(text_to_image_recalls)
                
                print(f"  Text→Image Recall@5: {result.text_to_image_recall_at_5:.3f}")
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        result.error = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def run_full_benchmark(
    backends: List[str] = None,
    modes: List[str] = None,
    quantizations: List[str] = None,
    output_dir: str = "benchmarks",
):
    """
    Run full benchmark suite.
    
    Args:
        backends: List of backends to test (default: ['torch'])
        modes: List of modes to test (default: ['embed', 'hybrid', 'full'])
        quantizations: List of MLX quantizations (default: ['bf16'])
        output_dir: Directory for results
    """
    backends = backends or ["torch"]
    modes = modes or ["embed", "hybrid"]
    quantizations = quantizations or ["bf16"]
    
    # Create test data
    test_docs = [
        {"id": "ai-overview.md", "text": "Artificial intelligence (AI) is intelligence demonstrated by machines. Modern AI systems use neural networks trained on large datasets. Deep learning has enabled breakthroughs in computer vision, natural language processing, and robotics."},
        {"id": "memory-systems.md", "text": "Memory systems for AI agents include episodic memory for past experiences, semantic memory for facts, and working memory for current context. Graph databases provide structured knowledge storage."},
        {"id": "graph-databases.md", "text": "Graph databases store data as nodes and relationships. They excel at representing complex relationships and querying connected data. Popular graph databases include Neo4j, Amazon Neptune, and ArangoDB."},
        {"id": "neural-networks.md", "text": "Neural networks are computing systems inspired by biological brains. They consist of layers of interconnected nodes that process information. Deep neural networks have multiple hidden layers."},
        {"id": "vector-search.md", "text": "Vector search uses embeddings to find similar items. Approximate nearest neighbor (ANN) algorithms enable fast search over large vector spaces. Popular vector databases include Pinecone, Weaviate, and LanceDB."},
        {"id": "cross-encoder.md", "text": "Cross-encoder models jointly encode query and document for relevance scoring. They are more accurate than bi-encoders but slower. Used for reranking in hybrid search systems."},
        {"id": "query-expansion.md", "text": "Query expansion generates multiple variations of a search query to improve recall. Techniques include synonym expansion, HyDE (hypothetical document embeddings), and learned expansion."},
        {"id": "bm25.md", "text": "BM25 is a probabilistic ranking function for full-text search. It considers term frequency, inverse document frequency, and document length. It's the standard baseline for information retrieval."},
        {"id": "embedding-models.md", "text": "Embedding models convert text or images into dense vector representations. Modern embedding models like BERT, Sentence-BERT, and Qwen provide high-quality semantic embeddings."},
        {"id": "reranking.md", "text": "Reranking refines search results using a more sophisticated relevance model. It typically follows a first-stage retrieval using BM25 or vector search. Cross-encoders are commonly used for reranking."},
    ]
    
    test_queries = [
        {"query": "How do AI agents remember things?", "relevant": ["memory-systems.md", "ai-overview.md"]},
        {"query": "What is vector search?", "relevant": ["vector-search.md", "embedding-models.md"]},
        {"query": "Explain BM25 ranking", "relevant": ["bm25.md"]},
        {"query": "What are graph databases used for?", "relevant": ["graph-databases.md", "memory-systems.md"]},
        {"query": "How does cross-encoder reranking work?", "relevant": ["cross-encoder.md", "reranking.md"]},
        {"query": "Neural network architecture", "relevant": ["neural-networks.md", "ai-overview.md"]},
        {"query": "Query expansion techniques", "relevant": ["query-expansion.md"]},
        {"query": "Embedding models for search", "relevant": ["embedding-models.md", "vector-search.md"]},
    ]
    
    # Check for test images
    images_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Molly Notes/LinkedIn Images/")
    test_images = []
    if os.path.exists(images_dir):
        test_images = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith('.png')
        ][:3]  # Limit to 3 images
    
    # Run benchmarks
    results: List[BenchmarkResult] = []
    
    for backend in backends:
        for mode in modes:
            quants = quantizations if backend == "mlx" else [None]
            for quant in quants:
                print(f"\n{'=' * 60}")
                print(f"Backend: {backend}, Mode: {mode}, Quantization: {quant or 'N/A'}")
                print('=' * 60)
                
                result = benchmark_backend(
                    backend_name=backend,
                    mode=mode,
                    quantization=quant or "bf16",
                    test_queries=test_queries,
                    test_docs=test_docs,
                    test_images=test_images,
                )
                results.append(result)
                
                # Clear memory between runs
                gc.collect()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Generate markdown report
    report_path = os.path.join(output_dir, "RESULTS.md")
    with open(report_path, "w") as f:
        f.write("# RecallForge Benchmark Results\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| Backend | Mode | Quant | Load (s) | Memory (GB) | P50 (ms) | Recall@5 |\n")
        f.write("|---------|------|-------|----------|-------------|----------|----------|\n")
        
        for r in results:
            if r.error:
                f.write(f"| {r.backend} | {r.mode} | {r.quantization or '-'} | ERROR | - | - | - |\n")
            else:
                f.write(f"| {r.backend} | {r.mode} | {r.quantization or '-'} | {r.total_load_time_s:.1f} | {r.memory_allocated_gb:.1f} | {r.query_latency_p50_ms:.0f} | {r.recall_at_5:.3f} |\n")
        
        # Detailed results
        f.write("\n## Detailed Results\n\n")
        for r in results:
            f.write(f"### {r.backend}/{r.mode}" + (f"/{r.quantization}" if r.quantization else "") + "\n\n")
            if r.error:
                f.write(f"**Error:** {r.error}\n\n")
            else:
                f.write("**Load Times:**\n")
                f.write(f"- Embedder: {r.embedder_load_time_s:.1f}s\n")
                if r.reranker_load_time_s > 0:
                    f.write(f"- Reranker: {r.reranker_load_time_s:.1f}s\n")
                if r.expander_load_time_s > 0:
                    f.write(f"- Expander: {r.expander_load_time_s:.1f}s\n")
                f.write(f"- **Total:** {r.total_load_time_s:.1f}s\n\n")
                
                f.write("**Query Latency:**\n")
                f.write(f"- P50: {r.query_latency_p50_ms:.1f}ms\n")
                f.write(f"- P95: {r.query_latency_p95_ms:.1f}ms\n")
                f.write(f"- P99: {r.query_latency_p99_ms:.1f}ms\n\n")
                
                f.write("**Quality Metrics:**\n")
                f.write(f"- Recall@5: {r.recall_at_5:.3f}\n")
                f.write(f"- MRR: {r.mrr:.3f}\n")
                if r.text_to_image_recall_at_5 > 0:
                    f.write(f"- Text→Image Recall@5: {r.text_to_image_recall_at_5:.3f}\n")
                f.write("\n")
    
    print(f"Report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecallForge Benchmark Suite")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["torch"],
        choices=["torch", "mlx", "auto"],
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["embed", "hybrid"],
        choices=["embed", "hybrid"],
        help="Modes to benchmark",
    )
    parser.add_argument(
        "--quantizations",
        nargs="+",
        default=["bf16"],
        choices=["bf16", "4bit"],
        help="MLX quantizations (MLX only)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    results = run_full_benchmark(
        backends=args.backends,
        modes=args.modes,
        quantizations=args.quantizations,
        output_dir=args.output,
    )
    
    # Exit with error if any benchmark failed
    errors = [r for r in results if r.error]
    if errors:
        print(f"\n{len(errors)} benchmark(s) failed")
        sys.exit(1)
    
    print("\nAll benchmarks passed!")
    sys.exit(0)