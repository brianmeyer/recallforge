#!/usr/bin/env bash
# test_cross_modal.sh - Cross-modal search UAT (refactored for memory efficiency).
# Runs each mode in its own subprocess to avoid 30GB OOM on 16GB machines.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Cross-Modal Search Tests"
echo -e "${BOLD}${YELLOW}  ★ THIS IS THE KEY DIFFERENTIATOR TEST ★${NC}"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

BACKEND_CANDIDATES=("torch")
if is_apple_silicon; then
    BACKEND_CANDIDATES=("mlx" "torch")
fi
info "Backend order: ${BACKEND_CANDIDATES[*]}"

run_cross_modal_for_backend() {
    local backend_name="$1"
    local overall_fail=0

    cleanup_store

    # ═══════════════════════════════════
    # Phase 1: Index corpus once (shared)
    # ═══════════════════════════════════
    subsection "Indexing Full Corpus (backend=${backend_name})"

    if ! python3 <<PYEOF
import os, sys
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

# Index text
text_files = sorted([f for f in os.listdir(CORPUS_TEXT) if f.endswith('.md')])
for f in text_files:
    path = os.path.join(CORPUS_TEXT, f)
    with open(path) as fh:
        text = fh.read()
    storage.index_document(
        path=f, text=text, collection="xmodal",
        model="Qwen3-VL-Embedding-2B", embed_func=backend.embed_text,
    )
    print(f"  Text: {f}")

# Index images
img_files = sorted([f for f in os.listdir(CORPUS_IMAGES) if f.endswith('.png')])
for f in img_files:
    path = os.path.join(CORPUS_IMAGES, f)
    storage.index_image(path=path, collection="xmodal", embed_func=backend.embed_image)
    print(f"  Image: {f}")

total_docs = storage.count_documents()
total_emb = storage.count_embeddings()
print(f"\n  Indexed {total_docs} documents, {total_emb} embeddings")
PYEOF
    then
        return 1
    fi

    # ═══════════════════════════════════
    # Phase 2: Test each mode in subprocess
    # ═══════════════════════════════════
    for mode in embed hybrid full; do
        if ! python3 <<PYEOF_MODE
import os, sys
sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
CORPUS_TEXT = "${CORPUS_DIR}/text"
CORPUS_IMAGES = "${CORPUS_DIR}/images"
BACKEND = "${backend_name}"
MODE = "$mode"

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

# Load backend for this mode only
os.environ["RECALLFORGE_BACKEND"] = BACKEND
os.environ["RECALLFORGE_MODE"] = MODE
os.environ["RECALLFORGE_STORE_PATH"] = STORE

from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

backend = get_backend()
backend.set_mode(MODE)
backend._load_embedder()
if MODE in ["hybrid", "full"]:
    backend._load_reranker()
if MODE == "full":
    backend._load_expander()

storage = get_storage(STORE)

def search_text(query, limit=5, content_type=None):
    searcher = HybridSearcher(
        backend=backend, storage=storage,
        limit=limit, collection="xmodal",
        content_type=content_type,
    )
    return searcher.search(query)

def search_by_image(image_path, limit=5, content_type=None):
    vec = backend.embed_image(image_path)
    results = storage.search_vec(
        vec.tolist(), limit=limit,
        collection="xmodal", content_type=content_type,
    )
    return results

def check_results(results, expected_keywords, top_k=5):
    titles = [r.title.lower() if hasattr(r, "title") else str(r).lower() for r in results[:top_k]]
    paths = []
    for r in results[:top_k]:
        fp = r.filepath if hasattr(r, "filepath") else ""
        dp = r.display_path if hasattr(r, "display_path") else ""
        paths.append(f"{fp} {dp}".lower())
    all_text = " ".join(titles) + " " + " ".join(paths)
    found = any(kw.lower() in all_text for kw in expected_keywords)
    return found

print(f"\\n\\033[1m\\033[0;34m═════ Backend: {BACKEND.upper()} | Mode: {MODE.upper()} ═════\\033[0m")

# ═ Text-to-Text ═
print("\\n  \\033[0;36mA. Text-to-Text Search\\033[0m")
t2t_queries = [
    ("machine learning training neural networks", ["ai_transformers", "ai_embeddings", "ai_agents"]),
    ("homemade bread baking sourdough", ["cooking_sourdough", "cooking_pasta"]),
    ("cathedral flying buttresses stained glass", ["architecture_gothic"]),
    ("forest trees wildlife ecosystem", ["nature_forests"]),
    ("basketball three-point shooting defense", ["sports_basketball"]),
]
hits = 0
for query, expected in t2t_queries:
    results = search_text(query, limit=5, content_type="text")
    found = check_results(results, expected)
    hits += int(found)
    status = "\\033[0;32m✓\\033[0m" if found else "\\033[0;31m✗\\033[0m"
    top3 = ", ".join([r.title[:30] for r in results[:3]]) if results else "no results"
    print(f"    {status} '{query[:40]}' -> {top3}")
report(hits >= 3, f"Text-to-Text ({MODE}): {hits}/5")

# ═ Text-to-Image ═
print("\\n  \\033[0;36mB. Text-to-Image Search\\033[0m")
t2i_queries = [
    ("whiteboard diagram system architecture", ["whiteboard_architecture", "whiteboard_brainstorm"]),
    ("handwritten meeting notes", ["handwritten_notes"]),
    ("floor plan blueprint building layout", ["floor_plan_blueprint"]),
    ("food plate pasta dish cooking", ["food_pasta_dish"]),
    ("forest trees green landscape nature", ["forest_landscape", "mountain_landscape"]),
]
hits = 0
for query, expected in t2i_queries:
    results = search_text(query, limit=5, content_type="image")
    found = check_results(results, expected)
    hits += int(found)
    status = "\\033[0;32m✓\\033[0m" if found else "\\033[0;31m✗\\033[0m"
    top3 = ", ".join([r.title[:30] for r in results[:3]]) if results else "no results"
    print(f"    {status} '{query[:40]}' -> {top3}")
report(hits >= 2, f"Text-to-Image ({MODE}): {hits}/5")

# ═ Image-to-Text ═
print("\\n  \\033[0;36mC. Image-to-Text Search\\033[0m")
i2t_queries = [
    (os.path.join(CORPUS_IMAGES, "food_pasta_dish.png"), ["cooking_pasta", "cooking_sourdough", "cooking_grilling"]),
    (os.path.join(CORPUS_IMAGES, "neural_network_diagram.png"), ["ai_transformers", "ai_embeddings", "ai_agents"]),
    (os.path.join(CORPUS_IMAGES, "forest_landscape.png"), ["nature_forests", "nature_mountains"]),
]
hits = 0
for img_path, expected in i2t_queries:
    if os.path.exists(img_path):
        results = search_by_image(img_path, limit=5, content_type="text")
        found = check_results(results, expected)
        hits += int(found)
        img_name = os.path.basename(img_path)
        status = "\\033[0;32m✓\\033[0m" if found else "\\033[0;31m✗\\033[0m"
        top3 = ", ".join([r.title[:30] for r in results[:3]]) if results else "no results"
        print(f"    {status} {img_name} -> {top3}")
report(hits >= 1, f"Image-to-Text ({MODE}): {hits}/3")

# ═ Image-to-Image ═
print("\\n  \\033[0;36mD. Image-to-Image Search\\033[0m")
i2i_queries = [
    (os.path.join(CORPUS_IMAGES, "whiteboard_architecture.png"), ["whiteboard_brainstorm", "whiteboard_architecture"]),
    (os.path.join(CORPUS_IMAGES, "forest_landscape.png"), ["mountain_landscape", "ocean_beach", "forest_landscape"]),
    (os.path.join(CORPUS_IMAGES, "neural_network_diagram.png"), ["code_editor_screenshot", "whiteboard_architecture", "neural_network_diagram"]),
]
hits = 0
for img_path, expected in i2i_queries:
    if os.path.exists(img_path):
        results = search_by_image(img_path, limit=5, content_type="image")
        found = check_results(results, expected)
        hits += int(found)
        img_name = os.path.basename(img_path)
        status = "\\033[0;32m✓\\033[0m" if found else "\\033[0;31m✗\\033[0m"
        top3 = ", ".join([r.title[:30] for r in results[:3]]) if results else "no results"
        print(f"    {status} {img_name} -> {top3}")
report(hits >= 1, f"Image-to-Image ({MODE}): {hits}/3")

# Exit with fail if any test failed
sys.exit(1 if fail_count > 0 else 0)
PYEOF_MODE
        then
            overall_fail=1
        fi
    done

    [[ ${overall_fail} -eq 0 ]]
}

selected_backend=""
for backend_name in "${BACKEND_CANDIDATES[@]}"; do
    if [[ "${backend_name}" == "mlx" ]] && ! has_mlx; then
        warn "MLX backend unavailable on this host; falling back to torch."
        continue
    fi

    subsection "Backend Attempt: ${backend_name}"
    if run_cross_modal_for_backend "${backend_name}"; then
        selected_backend="${backend_name}"
        info "Cross-modal tests passed with backend '${backend_name}'."
        break
    fi

    warn "Cross-modal tests failed with backend '${backend_name}'."
done

if [[ -z "${selected_backend}" ]]; then
    fail "Cross-modal tests (all backend attempts failed)"
    print_summary "Cross-Modal Search Tests"
    exit 1
fi

pass "Cross-modal tests (backend=${selected_backend})"
print_summary "Cross-Modal Search Tests"
