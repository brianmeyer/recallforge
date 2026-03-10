#!/usr/bin/env bash
# test_search_quality.sh - Search accuracy and edge case UAT.
# Tests recall@5, MRR, edge cases, and dedup.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Search Quality Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

python3 << PYEOF
import os, sys, time
sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
CORPUS_TEXT = "${CORPUS_DIR}/text"

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

# ── Index corpus ──
import platform
if platform.machine() == "arm64" and platform.system() == "Darwin":
    os.environ["RECALLFORGE_BACKEND"] = "mlx"
    os.environ.setdefault("RECALLFORGE_MLX_QUANTIZE", "4bit")
else:
    os.environ["RECALLFORGE_BACKEND"] = "torch"
os.environ["RECALLFORGE_MODE"] = "embed"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher
from recallforge.backends.torch_backend import TorchBackend

backend = get_backend()
backend._load_embedder()
storage = get_storage(STORE)

text_files = sorted([f for f in os.listdir(CORPUS_TEXT) if f.endswith('.md')])
for f in text_files:
    path = os.path.join(CORPUS_TEXT, f)
    with open(path) as fh:
        text = fh.read()
    storage.index_document(
        path=f, text=text, collection="quality",
        model="Qwen3-VL-Embedding-2B", embed_func=backend.embed_text,
    )

# Load hybrid for quality tests
backend_h = TorchBackend(mode="hybrid")
backend_h._load_embedder()
backend_h._load_reranker()

# ═══════════════════════════════════
print("\n\033[0;36m--- Recall@5 and MRR Tests ---\033[0m\n")
# ═══════════════════════════════════

# (query, list of acceptable doc filenames in top-5)
test_queries = [
    ("transformer architecture attention mechanism", ["ai_transformers.md"]),
    ("AI agent episodic memory knowledge graph", ["ai_agents.md"]),
    ("vector embedding semantic search cosine", ["ai_embeddings.md"]),
    ("making fresh pasta flour eggs", ["cooking_pasta.md"]),
    ("sourdough starter wild yeast bread", ["cooking_sourdough.md"]),
    ("charcoal grill searing smoke brisket", ["cooking_grilling.md"]),
    ("gothic cathedral flying buttresses rose window", ["architecture_gothic.md"]),
    ("modern architecture Le Corbusier glass steel", ["architecture_modern.md"]),
    ("forest ecosystem deciduous trees wildlife", ["nature_forests.md"]),
    ("marathon training long distance running race", ["sports_running.md"]),
]

modes_to_test = {
    "embed": backend,
    "hybrid": backend_h,
}

for mode_name, mode_backend in modes_to_test.items():
    print(f"\n  Mode: {mode_name}")
    recall_hits = 0
    mrr_sum = 0.0

    for query, expected_docs in test_queries:
        searcher = HybridSearcher(
            backend=mode_backend, storage=storage,
            limit=5, collection="quality", content_type="text",
        )
        results = searcher.search(query)
        result_files = [os.path.basename(r.filepath.split("/")[-1]) for r in results[:5]]

        found = False
        rank = 0
        for i, rf in enumerate(result_files):
            if any(exp in rf for exp in expected_docs):
                found = True
                rank = i + 1
                break

        recall_hits += int(found)
        if rank > 0:
            mrr_sum += 1.0 / rank

        status = "\033[0;32m✓\033[0m" if found else "\033[0;31m✗\033[0m"
        rank_str = f"rank={rank}" if rank > 0 else "not found"
        print(f"    {status} '{query[:45]}' → {rank_str}")

    recall_at_5 = recall_hits / len(test_queries)
    mrr = mrr_sum / len(test_queries)
    print(f"\n    Recall@5: {recall_at_5:.0%}  |  MRR: {mrr:.3f}")
    report(recall_at_5 >= 0.6, f"Recall@5 ({mode_name}) >= 60%: actual {recall_at_5:.0%}")
    report(mrr >= 0.4, f"MRR ({mode_name}) >= 0.4: actual {mrr:.3f}")

# ═══════════════════════════════════
print("\n\033[0;36m--- Edge Cases ---\033[0m\n")
# ═══════════════════════════════════

searcher = HybridSearcher(
    backend=backend, storage=storage,
    limit=5, collection="quality",
)

# Empty query
try:
    results = searcher.search("")
    report(True, f"Empty query returns {len(results)} results (no crash)")
except Exception as e:
    report(False, f"Empty query crashed: {e}")

# Very long query
long_query = "machine learning " * 200
try:
    results = searcher.search(long_query)
    report(True, f"Very long query (3400+ chars) returns {len(results)} results")
except Exception as e:
    report(False, f"Long query crashed: {e}")

# Special characters
special_query = "what about C++ & Python's 'decorators'? @#\$%!"
try:
    results = searcher.search(special_query)
    report(True, f"Special characters query returns {len(results)} results")
except Exception as e:
    report(False, f"Special characters query crashed: {e}")

# Non-English text
non_eng = "机器学习和人工智能的最新进展"
try:
    results = searcher.search(non_eng)
    report(True, f"Non-English (Chinese) query returns {len(results)} results")
except Exception as e:
    report(False, f"Non-English query crashed: {e}")

# Unicode/emoji
emoji_query = "🤖 artificial intelligence 🧠"
try:
    results = searcher.search(emoji_query)
    report(True, f"Emoji query returns {len(results)} results")
except Exception as e:
    report(False, f"Emoji query crashed: {e}")

# ═══════════════════════════════════
print("\n\033[0;36m--- Duplicate Detection ---\033[0m\n")
# ═══════════════════════════════════

# Index same document again
dup_text = "This is a duplicate document about AI and machine learning."
storage.index_document(
    path="dup_test.md", text=dup_text, collection="quality",
    model="Qwen3-VL-Embedding-2B", embed_func=backend.embed_text,
)
storage.index_document(
    path="dup_test.md", text=dup_text, collection="quality",
    model="Qwen3-VL-Embedding-2B", embed_func=backend.embed_text,
)

results = searcher.search("duplicate document AI machine learning")
filepaths = [r.filepath for r in results]
dup_count = sum(1 for fp in filepaths if "dup_test" in fp)
report(dup_count <= 1, f"Same doc indexed twice appears at most once in results (found {dup_count})")

# ── Summary ──
print(f"\n\033[1m{'='*40}\033[0m")
print(f"\033[1m  Search Quality Summary\033[0m")
print(f"\033[1m{'='*40}\033[0m")
print(f"  \033[0;32mPASS: {pass_count}\033[0m")
print(f"  \033[0;31mFAIL: {fail_count}\033[0m")

if fail_count > 0:
    print(f"\n  \033[0;31m\033[1mRESULT: FAILED\033[0m")
    sys.exit(1)
else:
    print(f"\n  \033[0;32m\033[1mRESULT: PASSED\033[0m")
PYEOF
