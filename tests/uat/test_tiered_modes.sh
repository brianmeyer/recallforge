#!/usr/bin/env bash
# test_tiered_modes.sh - Tiered mode (embed/hybrid/full) UAT.
# Verifies model loading, search behavior, and mode switching.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Tiered Mode Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

# ── Helper: cleanup GPU memory between phases ──
_cleanup_gpu() {
    python3 -c "
import gc; gc.collect()
try:
    import torch
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
except: pass
" 2>/dev/null || true
}

run_tiered_modes_for_backend() {
    local backend_name="$1"

    _PASS_COUNT=0
    _FAIL_COUNT=0
    _SKIP_COUNT=0

    cleanup_store

    # ── Phase 1: Index corpus (embed mode only — minimal memory) ──
    subsection "Indexing test corpus (backend=${backend_name})"

    if ! python3 <<PYEOF
import os, sys, time
sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
CORPUS_TEXT = "${CORPUS_DIR}/text"
CORPUS_IMAGES = "${CORPUS_DIR}/images"
BACKEND = "${backend_name}"

os.environ["RECALLFORGE_BACKEND"] = BACKEND
os.environ["RECALLFORGE_MODE"] = "embed"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

from recallforge import get_backend, get_storage

backend = get_backend()
backend._load_embedder()
storage = get_storage(STORE)

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

img_files = sorted([f for f in os.listdir(CORPUS_IMAGES) if f.endswith('.png')])
for f in img_files[:3]:
    path = os.path.join(CORPUS_IMAGES, f)
    storage.index_image(path=path, collection="uat", embed_func=backend.embed_image)
    print(f"  Indexed: {f}")

total_docs = storage.count_documents()
total_emb = storage.count_embeddings()
ok = total_docs > 0
print(f"  {'PASS' if ok else 'FAIL'}  Indexed {total_docs} documents, {total_emb} embeddings")
if not ok:
    sys.exit(1)

# Cleanup: unload backend before exiting
del backend, storage
import gc; gc.collect()
try:
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
except:
    pass
PYEOF
    then
        fail "Corpus indexing"
        print_summary "Tiered Mode Tests (backend=${backend_name})"
        return 1
    fi
    pass "Corpus indexed"
    _cleanup_gpu

    # ── Phase 2: Embed Mode (separate process) ──
    subsection "Embed Mode"

    if python3 <<PYEOF
import os, sys, time
sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
BACKEND = "${backend_name}"

os.environ["RECALLFORGE_BACKEND"] = BACKEND
os.environ["RECALLFORGE_MODE"] = "embed"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

backend = get_backend()
backend.set_mode("embed")
backend._load_embedder()
storage = get_storage(STORE)

query = "How do AI agents use memory and knowledge graphs?"

pass_count = 0
fail_count = 0

def report(ok, msg):
    global pass_count, fail_count
    if ok:
        print(f"  \\033[0;32mPASS\\033[0m  {msg}")
        pass_count += 1
    else:
        print(f"  \\033[0;31mFAIL\\033[0m  {msg}")
        fail_count += 1

report(backend.get_mode() == "embed", "Mode set to 'embed'")
report(not backend.needs_reranker(), "Reranker not needed in embed mode")
report(not backend.needs_expander(), "Expander not needed in embed mode")

info_e = backend.get_info()
report(info_e.embedder_loaded, "Embedder loaded")
report(not info_e.reranker_loaded, "Reranker NOT loaded")
report(not info_e.expander_loaded, "Expander NOT loaded")

searcher = HybridSearcher(backend=backend, storage=storage, limit=5, collection="uat")
t0 = time.time()
results = searcher.search(query)
elapsed = time.time() - t0
report(len(results) > 0, f"Embed mode search returned {len(results)} results in {elapsed:.2f}s")

all_neutral = all(r.rerank_score == 0.5 for r in results)
report(all_neutral, "Embed mode: all rerank scores are 0.5 (neutral)")

for r in results:
    print(f"    [{r.score:.3f}] {r.title}")

# Write timing for later comparison
with open("${UAT_STORE}/.embed_time", "w") as f:
    f.write(f"{elapsed:.2f}")

if fail_count > 0:
    sys.exit(1)
PYEOF
    then
        pass "Embed mode"
    else
        fail "Embed mode"
    fi
    _cleanup_gpu

    # ── Phase 3: Hybrid Mode (separate process — embedder + reranker only) ──
    subsection "Hybrid Mode"

    if python3 <<PYEOF
import os, sys, time
sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
BACKEND = "${backend_name}"

os.environ["RECALLFORGE_BACKEND"] = BACKEND
os.environ["RECALLFORGE_MODE"] = "hybrid"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

backend = get_backend()
backend.set_mode("hybrid")
backend._load_embedder()
backend._load_reranker()
storage = get_storage(STORE)

query = "How do AI agents use memory and knowledge graphs?"

pass_count = 0
fail_count = 0

def report(ok, msg):
    global pass_count, fail_count
    if ok:
        print(f"  \\033[0;32mPASS\\033[0m  {msg}")
        pass_count += 1
    else:
        print(f"  \\033[0;31mFAIL\\033[0m  {msg}")
        fail_count += 1

report(backend.get_mode() == "hybrid", "Mode set to 'hybrid'")
report(backend.needs_reranker(), "Reranker needed in hybrid mode")
report(not backend.needs_expander(), "Expander not needed in hybrid mode")

info_h = backend.get_info()
report(info_h.embedder_loaded, "Embedder loaded")
report(info_h.reranker_loaded, "Reranker loaded")
report(not info_h.expander_loaded, "Expander NOT loaded")

searcher = HybridSearcher(backend=backend, storage=storage, limit=5, collection="uat")
t0 = time.time()
results = searcher.search(query)
elapsed = time.time() - t0
report(len(results) > 0, f"Hybrid mode search returned {len(results)} results in {elapsed:.2f}s")

has_varied = any(r.rerank_score != 0.5 for r in results)
report(has_varied, "Hybrid mode: reranker produces varied scores")

for r in results:
    print(f"    [{r.score:.3f}] {r.title} (rerank={r.rerank_score:.3f})")

with open("${UAT_STORE}/.hybrid_time", "w") as f:
    f.write(f"{elapsed:.2f}")

if fail_count > 0:
    sys.exit(1)
PYEOF
    then
        pass "Hybrid mode"
    else
        fail "Hybrid mode"
    fi
    _cleanup_gpu

    # ── Phase 4: Full Mode (separate process — all 3 models) ──
    subsection "Full Mode"

    if python3 <<PYEOF
import os, sys, time
sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
BACKEND = "${backend_name}"

os.environ["RECALLFORGE_BACKEND"] = BACKEND
os.environ["RECALLFORGE_MODE"] = "full"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

backend = get_backend()
backend.set_mode("full")
backend._load_embedder()
backend._load_reranker()
backend._load_expander()
storage = get_storage(STORE)

query = "How do AI agents use memory and knowledge graphs?"

pass_count = 0
fail_count = 0

def report(ok, msg):
    global pass_count, fail_count
    if ok:
        print(f"  \\033[0;32mPASS\\033[0m  {msg}")
        pass_count += 1
    else:
        print(f"  \\033[0;31mFAIL\\033[0m  {msg}")
        fail_count += 1

report(backend.get_mode() == "full", "Mode set to 'full'")
report(backend.needs_reranker(), "Reranker needed in full mode")
report(backend.needs_expander(), "Expander needed in full mode")

info_f = backend.get_info()
report(info_f.embedder_loaded, "Embedder loaded")
report(info_f.reranker_loaded, "Reranker loaded")
report(info_f.expander_loaded, "Expander loaded")

searcher = HybridSearcher(backend=backend, storage=storage, limit=5, collection="uat")
t0 = time.time()
results = searcher.search(query)
elapsed = time.time() - t0
report(len(results) > 0, f"Full mode search returned {len(results)} results in {elapsed:.2f}s")

for r in results:
    print(f"    [{r.score:.3f}] {r.title} (rerank={r.rerank_score:.3f})")

with open("${UAT_STORE}/.full_time", "w") as f:
    f.write(f"{elapsed:.2f}")

if fail_count > 0:
    sys.exit(1)
PYEOF
    then
        pass "Full mode"
    else
        fail "Full mode"
    fi
    _cleanup_gpu

    # ── Phase 5: Mode Switching (separate process) ──
    subsection "Mode Switching"

    if python3 <<PYEOF
import os, sys
sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
BACKEND = "${backend_name}"

os.environ["RECALLFORGE_BACKEND"] = BACKEND
os.environ["RECALLFORGE_MODE"] = "embed"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

pass_count = 0
fail_count = 0

def report(ok, msg):
    global pass_count, fail_count
    if ok:
        print(f"  \\033[0;32mPASS\\033[0m  {msg}")
        pass_count += 1
    else:
        print(f"  \\033[0;31mFAIL\\033[0m  {msg}")
        fail_count += 1

backend = get_backend()
backend.set_mode("embed")
backend._load_embedder()
report(backend.get_mode() == "embed", "Start in embed mode")

backend.set_mode("full")
report(backend.get_mode() == "full", "Switched to full mode")
report(backend.needs_reranker(), "Reranker needed after switch")
report(backend.needs_expander(), "Expander needed after switch")

backend._load_reranker()
backend._load_expander()
info_s = backend.get_info()
report(info_s.reranker_loaded and info_s.expander_loaded, "All models loaded after switch")

storage = get_storage(STORE)
searcher = HybridSearcher(backend=backend, storage=storage, limit=5, collection="uat")
results = searcher.search("How do AI agents use memory and knowledge graphs?")
report(len(results) > 0, f"Search works after mode switch ({len(results)} results)")

if fail_count > 0:
    sys.exit(1)
PYEOF
    then
        pass "Mode switching"
    else
        fail "Mode switching"
    fi
    _cleanup_gpu

    # ── Timing Comparison ──
    subsection "Timing Comparison"

    EMBED_T=$(cat "${UAT_STORE}/.embed_time" 2>/dev/null || echo "N/A")
    HYBRID_T=$(cat "${UAT_STORE}/.hybrid_time" 2>/dev/null || echo "N/A")
    FULL_T=$(cat "${UAT_STORE}/.full_time" 2>/dev/null || echo "N/A")
    echo "  Embed mode:  ${EMBED_T}s"
    echo "  Hybrid mode: ${HYBRID_T}s"
    echo "  Full mode:   ${FULL_T}s"

    print_summary "Tiered Mode Tests (backend=${backend_name})"
}

mapfile -t BACKEND_CANDIDATES < <(live_backend_candidates || true)
if [[ ${#BACKEND_CANDIDATES[@]} -eq 0 ]]; then
    warn "No usable live backend on this host; skipping tiered-mode live coverage."
    skip "Tiered modes live backend"
    print_summary "Tiered Mode Tests"
    exit 0
fi
info "Backend order: ${BACKEND_CANDIDATES[*]}"

selected_backend=""
for backend_name in "${BACKEND_CANDIDATES[@]}"; do
    subsection "Backend Attempt: ${backend_name}"
    if run_tiered_modes_for_backend "${backend_name}"; then
        selected_backend="${backend_name}"
        info "Tiered mode tests passed with backend '${backend_name}'."
        break
    fi

    warn "Tiered mode tests failed with backend '${backend_name}'."
done

if [[ -z "${selected_backend}" ]]; then
    exit 1
fi

exit 0
