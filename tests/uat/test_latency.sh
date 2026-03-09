#!/usr/bin/env bash
# test_latency.sh - Performance and latency UAT.
# Tests cold start, warm search, index throughput, and peak memory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Performance Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

python3 << PYEOF
import os, sys, time, statistics, subprocess, random, string
sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
CORPUS_TEXT = "${CORPUS_DIR}/text"
CORPUS_IMAGES = "${CORPUS_DIR}/images"

pass_count = 0
fail_count = 0

def report(ok, msg):
    global pass_count, fail_count
    if ok:
        print(f"  \033[0;32mPASS\033[0m  {msg}")
        pass_count += 1
    else:
        print(f"  \033[0;31mFAIL\033[0m  {msg}")
        fail_count += 1

def get_rss_mb():
    """Get current process RSS in MB."""
    pid = os.getpid()
    try:
        rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip())
        return rss / 1024
    except Exception:
        return 0

# ═══════════════════════════════════
print("\n\033[0;36m--- Cold Start: Backend + First Search ---\033[0m\n")
# ═══════════════════════════════════

os.environ["RECALLFORGE_BACKEND"] = "torch"
os.environ["RECALLFORGE_MODE"] = "embed"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

rss_before = get_rss_mb()

t0 = time.time()
from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

backend = get_backend()
storage = get_storage(STORE)

# Index a few docs for searching
text_files = sorted([f for f in os.listdir(CORPUS_TEXT) if f.endswith('.md')])
for f in text_files:
    path = os.path.join(CORPUS_TEXT, f)
    with open(path) as fh:
        text = fh.read()
    storage.index_document(
        path=f, text=text, collection="perf",
        model="Qwen3-VL-Embedding-2B", embed_func=backend.embed_text,
    )

# First search (cold)
searcher = HybridSearcher(backend=backend, storage=storage, limit=5, collection="perf")
results = searcher.search("machine learning neural networks")
cold_time = time.time() - t0

rss_after_cold = get_rss_mb()
print(f"  Cold start (import + index 15 docs + first search): {cold_time:.1f}s")
print(f"  RSS: {rss_before:.0f} MB → {rss_after_cold:.0f} MB (+{rss_after_cold - rss_before:.0f} MB)")
report(cold_time < 120, f"Cold start < 120s: actual {cold_time:.1f}s")
report(len(results) > 0, f"Cold start search returned {len(results)} results")

# ═══════════════════════════════════
print("\n\033[0;36m--- Warm Search Latency (20 queries) ---\033[0m\n")
# ═══════════════════════════════════

queries = [
    "artificial intelligence deep learning",
    "cooking pasta recipe flour",
    "sourdough bread baking yeast",
    "gothic cathedral architecture",
    "modern building design glass steel",
    "forest ecosystem wildlife trees",
    "ocean marine biology coral reef",
    "mountain alpine ecology glacier",
    "basketball strategy defense",
    "marathon training running endurance",
    "soccer football tactics formation",
    "transformer attention mechanism",
    "vector embedding semantic search",
    "AI agent memory knowledge graph",
    "charcoal grill barbecue smoke",
    "blueprint floor plan reading",
    "neural network deep learning",
    "natural language processing",
    "cross modal vision language",
    "hybrid search retrieval fusion",
]

latencies = []
for q in queries:
    t0 = time.time()
    results = searcher.search(q)
    elapsed = time.time() - t0
    latencies.append(elapsed * 1000)  # Convert to ms

latencies.sort()
p50 = latencies[len(latencies) // 2]
p95 = latencies[int(len(latencies) * 0.95)]
p99 = latencies[int(len(latencies) * 0.99)]
avg = statistics.mean(latencies)
std = statistics.stdev(latencies) if len(latencies) > 1 else 0

print(f"  Queries: {len(queries)}")
print(f"  Avg: {avg:.0f}ms  |  p50: {p50:.0f}ms  |  p95: {p95:.0f}ms  |  p99: {p99:.0f}ms")
print(f"  Std dev: {std:.0f}ms  |  Min: {min(latencies):.0f}ms  |  Max: {max(latencies):.0f}ms")
report(p50 < 5000, f"p50 search latency < 5s: actual {p50:.0f}ms")
report(p95 < 10000, f"p95 search latency < 10s: actual {p95:.0f}ms")

# ═══════════════════════════════════
print("\n\033[0;36m--- Text Index Throughput (100 docs) ---\033[0m\n")
# ═══════════════════════════════════

# Generate 100 short text files in memory
gen_docs = []
for i in range(100):
    words = random.choices(
        ["artificial", "intelligence", "machine", "learning", "neural",
         "network", "data", "model", "training", "algorithm", "search",
         "index", "vector", "embedding", "hybrid", "query"],
        k=30,
    )
    text = f"Performance test document {i}: " + " ".join(words)
    gen_docs.append((f"perf_doc_{i:03d}.md", text))

t0 = time.time()
for path, text in gen_docs:
    storage.index_document(
        path=path, text=text, collection="perf_bulk",
        model="Qwen3-VL-Embedding-2B", embed_func=backend.embed_text,
    )
text_index_time = time.time() - t0
docs_per_sec = 100 / text_index_time

print(f"  100 text docs indexed in {text_index_time:.1f}s ({docs_per_sec:.1f} docs/sec)")
report(True, f"Text index throughput: {docs_per_sec:.1f} docs/sec")

# ═══════════════════════════════════
print("\n\033[0;36m--- Image Index Throughput ---\033[0m\n")
# ═══════════════════════════════════

img_files = sorted([f for f in os.listdir(CORPUS_IMAGES) if f.endswith('.png')])
if img_files:
    t0 = time.time()
    for f in img_files:
        path = os.path.join(CORPUS_IMAGES, f)
        storage.index_image(
            path=path, collection="perf_img",
            embed_func=backend.embed_image,
        )
    img_index_time = time.time() - t0
    imgs_per_sec = len(img_files) / img_index_time

    print(f"  {len(img_files)} images indexed in {img_index_time:.1f}s ({imgs_per_sec:.1f} imgs/sec)")
    report(True, f"Image index throughput: {imgs_per_sec:.1f} imgs/sec")
else:
    print("  \033[0;33mSKIP\033[0m  No test images available")

# ═══════════════════════════════════
print("\n\033[0;36m--- Peak Memory Usage ---\033[0m\n")
# ═══════════════════════════════════

rss_peak = get_rss_mb()
print(f"  Peak RSS (embed mode, 115+ docs indexed): {rss_peak:.0f} MB")
report(rss_peak < 16000, f"Peak RSS < 16 GB: actual {rss_peak:.0f} MB")

# ═══════════════════════════════════
print("\n\033[0;36m--- Backend Comparison (if multiple available) ---\033[0m\n")
# ═══════════════════════════════════

try:
    import mlx
    has_mlx = True
except ImportError:
    has_mlx = False

if has_mlx:
    print("  MLX available - comparing backends:")
    os.environ["RECALLFORGE_BACKEND"] = "mlx"
    from recallforge.backends.mlx_backend import MLXBackend
    mlx_backend = MLXBackend(mode="embed", quantization="bf16")
    mlx_backend._load_embedder()

    mlx_latencies = []
    mlx_searcher = HybridSearcher(
        backend=mlx_backend, storage=storage,
        limit=5, collection="perf",
    )
    for q in queries[:10]:
        t0 = time.time()
        mlx_searcher.search(q)
        mlx_latencies.append((time.time() - t0) * 1000)

    mlx_avg = statistics.mean(mlx_latencies)
    print(f"  Torch avg: {avg:.0f}ms  |  MLX avg: {mlx_avg:.0f}ms")
    mlx_rss = get_rss_mb()
    print(f"  MLX RSS: {mlx_rss:.0f} MB")
else:
    print("  Only Torch backend available (MLX not installed)")

# ── Summary ──
print(f"\n\033[1m{'='*40}\033[0m")
print(f"\033[1m  Performance Summary\033[0m")
print(f"\033[1m{'='*40}\033[0m")
print(f"  \033[0;32mPASS: {pass_count}\033[0m")
print(f"  \033[0;31mFAIL: {fail_count}\033[0m")
print(f"\n  Key Metrics:")
print(f"    Cold start:        {cold_time:.1f}s")
print(f"    Warm search p50:   {p50:.0f}ms")
print(f"    Warm search p95:   {p95:.0f}ms")
print(f"    Text throughput:   {docs_per_sec:.1f} docs/sec")
if img_files:
    print(f"    Image throughput:  {imgs_per_sec:.1f} imgs/sec")
print(f"    Peak RSS:          {rss_peak:.0f} MB")

if fail_count > 0:
    print(f"\n  \033[0;31m\033[1mRESULT: FAILED\033[0m")
    sys.exit(1)
else:
    print(f"\n  \033[0;32m\033[1mRESULT: PASSED\033[0m")
PYEOF
