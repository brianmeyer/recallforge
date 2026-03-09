#!/usr/bin/env bash
# test_tiered_modes.sh - Tiered mode (embed/hybrid/full) UAT.
# Verifies model loading, search behavior, and mode switching.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Tiered Mode Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

python3 << PYEOF
import os, sys, time, json
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

# ── Index corpus once ──
print("\n\033[0;36m--- Indexing test corpus ---\033[0m\n")

os.environ["RECALLFORGE_BACKEND"] = "torch"
os.environ["RECALLFORGE_MODE"] = "embed"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

backend = get_backend()
backend._load_embedder()
storage = get_storage(STORE)

# Index text docs
text_files = sorted([f for f in os.listdir(CORPUS_TEXT) if f.endswith('.md')])
for f in text_files:
    path = os.path.join(CORPUS_TEXT, f)
    with open(path) as fh:
        text = fh.read()
    storage.index_document(
        path=f, text=text, collection="uat",
        model="Qwen3-VL-Embedding-2B", embed_func=backend.embed_text,
    )
    print(f"  Indexed: {f}")

# Index images
img_files = sorted([f for f in os.listdir(CORPUS_IMAGES) if f.endswith('.png')])
for f in img_files[:3]:  # Just 3 images for tiered mode test
    path = os.path.join(CORPUS_IMAGES, f)
    storage.index_image(path=path, collection="uat", embed_func=backend.embed_image)
    print(f"  Indexed: {f}")

total_docs = storage.count_documents()
total_emb = storage.count_embeddings()
report(total_docs > 0, f"Indexed {total_docs} documents, {total_emb} embeddings")

query = "How do AI agents use memory and knowledge graphs?"

# ═══════════════════════════════════
print("\n\033[0;36m--- Embed Mode ---\033[0m\n")
# ═══════════════════════════════════

backend_embed = get_backend()
backend_embed.set_mode("embed")
backend_embed._load_embedder()

report(backend_embed.get_mode() == "embed", "Mode set to 'embed'")
report(not backend_embed.needs_reranker(), "Reranker not needed in embed mode")
report(not backend_embed.needs_expander(), "Expander not needed in embed mode")

info_e = backend_embed.get_info()
report(info_e.embedder_loaded, "Embedder loaded")
report(not info_e.reranker_loaded, "Reranker NOT loaded")
report(not info_e.expander_loaded, "Expander NOT loaded")

searcher = HybridSearcher(backend=backend_embed, storage=storage, limit=5, collection="uat")
t0 = time.time()
results_embed = searcher.search(query)
embed_time = time.time() - t0
report(len(results_embed) > 0, f"Embed mode search returned {len(results_embed)} results in {embed_time:.2f}s")

# All rerank scores should be 0.5 (neutral)
all_neutral = all(r.rerank_score == 0.5 for r in results_embed)
report(all_neutral, "Embed mode: all rerank scores are 0.5 (neutral)")

for r in results_embed:
    print(f"    [{r.score:.3f}] {r.title}")

# ═══════════════════════════════════
print("\n\033[0;36m--- Hybrid Mode ---\033[0m\n")
# ═══════════════════════════════════

os.environ["RECALLFORGE_MODE"] = "hybrid"
from recallforge.backends.torch_backend import TorchBackend
backend_hybrid = TorchBackend(mode="hybrid")
backend_hybrid._load_embedder()
backend_hybrid._load_reranker()

report(backend_hybrid.get_mode() == "hybrid", "Mode set to 'hybrid'")
report(backend_hybrid.needs_reranker(), "Reranker needed in hybrid mode")
report(not backend_hybrid.needs_expander(), "Expander not needed in hybrid mode")

info_h = backend_hybrid.get_info()
report(info_h.embedder_loaded, "Embedder loaded")
report(info_h.reranker_loaded, "Reranker loaded")
report(not info_h.expander_loaded, "Expander NOT loaded")

searcher_h = HybridSearcher(backend=backend_hybrid, storage=storage, limit=5, collection="uat")
t0 = time.time()
results_hybrid = searcher_h.search(query)
hybrid_time = time.time() - t0
report(len(results_hybrid) > 0, f"Hybrid mode search returned {len(results_hybrid)} results in {hybrid_time:.2f}s")

# At least some rerank scores should differ from 0.5
has_varied = any(r.rerank_score != 0.5 for r in results_hybrid)
report(has_varied, "Hybrid mode: reranker produces varied scores")

for r in results_hybrid:
    print(f"    [{r.score:.3f}] {r.title} (rerank={r.rerank_score:.3f})")

# ═══════════════════════════════════
print("\n\033[0;36m--- Full Mode ---\033[0m\n")
# ═══════════════════════════════════

os.environ["RECALLFORGE_MODE"] = "full"
backend_full = TorchBackend(mode="full")
backend_full._load_embedder()
backend_full._load_reranker()
backend_full._load_expander()

report(backend_full.get_mode() == "full", "Mode set to 'full'")
report(backend_full.needs_reranker(), "Reranker needed in full mode")
report(backend_full.needs_expander(), "Expander needed in full mode")

info_f = backend_full.get_info()
report(info_f.embedder_loaded, "Embedder loaded")
report(info_f.reranker_loaded, "Reranker loaded")
report(info_f.expander_loaded, "Expander loaded")

searcher_f = HybridSearcher(backend=backend_full, storage=storage, limit=5, collection="uat")
t0 = time.time()
results_full = searcher_f.search(query)
full_time = time.time() - t0
report(len(results_full) > 0, f"Full mode search returned {len(results_full)} results in {full_time:.2f}s")

for r in results_full:
    print(f"    [{r.score:.3f}] {r.title} (rerank={r.rerank_score:.3f})")

# ═══════════════════════════════════
print("\n\033[0;36m--- Mode Switching ---\033[0m\n")
# ═══════════════════════════════════

backend_switch = TorchBackend(mode="embed")
backend_switch._load_embedder()
report(backend_switch.get_mode() == "embed", "Start in embed mode")

backend_switch.set_mode("full")
report(backend_switch.get_mode() == "full", "Switched to full mode")
report(backend_switch.needs_reranker(), "Reranker needed after switch")
report(backend_switch.needs_expander(), "Expander needed after switch")

# Load remaining models
backend_switch._load_reranker()
backend_switch._load_expander()
info_s = backend_switch.get_info()
report(info_s.reranker_loaded and info_s.expander_loaded, "All models loaded after switch")

searcher_s = HybridSearcher(backend=backend_switch, storage=storage, limit=5, collection="uat")
results_switch = searcher_s.search(query)
report(len(results_switch) > 0, f"Search works after mode switch ({len(results_switch)} results)")

# ═══════════════════════════════════
print("\n\033[0;36m--- Timing Comparison ---\033[0m\n")
# ═══════════════════════════════════

print(f"  Embed mode:  {embed_time:.2f}s")
print(f"  Hybrid mode: {hybrid_time:.2f}s")
print(f"  Full mode:   {full_time:.2f}s")

# ── Summary ──
print(f"\n\033[1m{'='*40}\033[0m")
print(f"\033[1m  Tiered Mode Summary\033[0m")
print(f"\033[1m{'='*40}\033[0m")
print(f"  \033[0;32mPASS: {pass_count}\033[0m")
print(f"  \033[0;31mFAIL: {fail_count}\033[0m")

if fail_count > 0:
    print(f"\n  \033[0;31m\033[1mRESULT: FAILED\033[0m")
    sys.exit(1)
else:
    print(f"\n  \033[0;32m\033[1mRESULT: PASSED\033[0m")
PYEOF
