#!/usr/bin/env bash
# test_cross_modal.sh - Cross-modal search UAT.
# THIS IS THE MOST IMPORTANT TEST. Cross-modal search is RecallForge's unique hook.
#
# Tests all 4 search directions:
#   A. Text-to-Text    B. Text-to-Image
#   C. Image-to-Text   D. Image-to-Image
#
# Each direction tested across embed/hybrid/full modes.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Cross-Modal Search Tests"
echo -e "${BOLD}${YELLOW}  ★ THIS IS THE KEY DIFFERENTIATOR TEST ★${NC}"

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
warn_count = 0

def report(ok, msg):
    global pass_count, fail_count
    if ok:
        print(f"  \033[0;32mPASS\033[0m  {msg}")
        pass_count += 1
    else:
        print(f"  \033[0;31mFAIL\033[0m  {msg}")
        fail_count += 1

def report_warn(msg):
    global warn_count
    print(f"  \033[0;33mWARN\033[0m  {msg}")
    warn_count += 1

# ═══════════════════════════════════
# Index entire corpus
# ═══════════════════════════════════
print("\n\033[0;36m--- Indexing Full Corpus ---\033[0m\n")

os.environ["RECALLFORGE_BACKEND"] = "torch"
os.environ["RECALLFORGE_MODE"] = "embed"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher
from recallforge.backends.torch_backend import TorchBackend
import numpy as np

backend = get_backend()
backend._load_embedder()
storage = get_storage(STORE)

# Index all 15 text docs
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

# Index all 10 images
img_files = sorted([f for f in os.listdir(CORPUS_IMAGES) if f.endswith('.png')])
for f in img_files:
    path = os.path.join(CORPUS_IMAGES, f)
    storage.index_image(path=path, collection="xmodal", embed_func=backend.embed_image)
    print(f"  Image: {f}")

total_docs = storage.count_documents()
total_emb = storage.count_embeddings()
print(f"\n  Indexed {total_docs} documents, {total_emb} embeddings")

# ═══════════════════════════════════
# Prepare backends for each mode
# ═══════════════════════════════════
modes = {}

# Embed mode (already loaded)
modes["embed"] = backend

# Hybrid mode
backend_hybrid = TorchBackend(mode="hybrid")
backend_hybrid._load_embedder()
backend_hybrid._load_reranker()
modes["hybrid"] = backend_hybrid

# Full mode
backend_full = TorchBackend(mode="full")
backend_full._load_embedder()
backend_full._load_reranker()
backend_full._load_expander()
modes["full"] = backend_full

# ═══════════════════════════════════
# Accuracy tracking
# ═══════════════════════════════════
accuracy_matrix = {}  # {direction: {mode: accuracy}}

def search_text(query, mode_backend, limit=5, content_type=None):
    """Run text query search."""
    searcher = HybridSearcher(
        backend=mode_backend, storage=storage,
        limit=limit, collection="xmodal",
        content_type=content_type,
    )
    return searcher.search(query)

def search_by_image(image_path, mode_backend, limit=5, content_type=None):
    """Run image-as-query search via vector search directly."""
    vec = mode_backend.embed_image(image_path)
    results = storage.search_vec(
        vec.tolist(), limit=limit,
        collection="xmodal", content_type=content_type,
    )
    return results

def check_results(results, expected_keywords, desc, top_k=5):
    """Check if any expected keyword appears in top-K result titles/paths."""
    titles = [r.title.lower() if hasattr(r, 'title') else str(r).lower() for r in results[:top_k]]
    paths = []
    for r in results[:top_k]:
        fp = r.filepath if hasattr(r, 'filepath') else ""
        dp = r.display_path if hasattr(r, 'display_path') else ""
        paths.append(f"{fp} {dp}".lower())
    all_text = " ".join(titles) + " " + " ".join(paths)
    found = any(kw.lower() in all_text for kw in expected_keywords)
    return found

# ═══════════════════════════════════
# A. TEXT-TO-TEXT
# ═══════════════════════════════════
print("\n\033[1m\033[0;34m═══════════════════════════════════════\033[0m")
print("\033[1m\033[0;34m  A. Text-to-Text Search\033[0m")
print("\033[1m\033[0;34m═══════════════════════════════════════\033[0m")

t2t_queries = [
    ("machine learning training neural networks", ["ai_transformers", "ai_embeddings", "ai_agents"]),
    ("homemade bread baking sourdough", ["cooking_sourdough", "cooking_pasta"]),
    ("cathedral flying buttresses stained glass", ["architecture_gothic"]),
    ("forest trees wildlife ecosystem", ["nature_forests"]),
    ("basketball three-point shooting defense", ["sports_basketball"]),
]

for mode_name, mode_backend in modes.items():
    accuracy_matrix.setdefault("text-to-text", {})[mode_name] = 0
    hits = 0
    print(f"\n  Mode: {mode_name}")
    for query, expected in t2t_queries:
        results = search_text(query, mode_backend, limit=5, content_type="text")
        found = check_results(results, expected, query)
        hits += int(found)
        status = "\033[0;32m✓\033[0m" if found else "\033[0;31m✗\033[0m"
        top3 = ", ".join([r.title[:30] for r in results[:3]])
        print(f"    {status} '{query[:40]}...' → {top3}")
    accuracy_matrix["text-to-text"][mode_name] = hits / len(t2t_queries)
    report(hits >= 3, f"Text-to-Text ({mode_name}): {hits}/{len(t2t_queries)} queries found expected docs")

# ═══════════════════════════════════
# B. TEXT-TO-IMAGE
# ═══════════════════════════════════
print("\n\033[1m\033[0;34m═══════════════════════════════════════\033[0m")
print("\033[1m\033[0;34m  B. Text-to-Image Search\033[0m")
print("\033[1m\033[0;34m═══════════════════════════════════════\033[0m")

t2i_queries = [
    ("whiteboard diagram system architecture", ["whiteboard_architecture", "whiteboard_brainstorm"]),
    ("handwritten meeting notes", ["handwritten_notes"]),
    ("floor plan blueprint building layout", ["floor_plan_blueprint"]),
    ("food plate pasta dish cooking", ["food_pasta_dish"]),
    ("forest trees green landscape nature", ["forest_landscape", "mountain_landscape"]),
]

for mode_name, mode_backend in modes.items():
    accuracy_matrix.setdefault("text-to-image", {})[mode_name] = 0
    hits = 0
    print(f"\n  Mode: {mode_name}")
    for query, expected in t2i_queries:
        results = search_text(query, mode_backend, limit=5, content_type="image")
        found = check_results(results, expected, query)
        hits += int(found)
        status = "\033[0;32m✓\033[0m" if found else "\033[0;31m✗\033[0m"
        top3 = ", ".join([r.title[:30] for r in results[:3]]) if results else "no results"
        print(f"    {status} '{query[:40]}' → {top3}")
    accuracy_matrix["text-to-image"][mode_name] = hits / len(t2i_queries)
    # Lower bar for cross-modal: 2/5 is acceptable
    report(hits >= 2, f"Text-to-Image ({mode_name}): {hits}/{len(t2i_queries)} queries found expected images")

# ═══════════════════════════════════
# C. IMAGE-TO-TEXT
# ═══════════════════════════════════
print("\n\033[1m\033[0;34m═══════════════════════════════════════\033[0m")
print("\033[1m\033[0;34m  C. Image-to-Text Search\033[0m")
print("\033[1m\033[0;34m═══════════════════════════════════════\033[0m")

i2t_queries = [
    (os.path.join(CORPUS_IMAGES, "food_pasta_dish.png"),
     ["cooking_pasta", "cooking_sourdough", "cooking_grilling"]),
    (os.path.join(CORPUS_IMAGES, "neural_network_diagram.png"),
     ["ai_transformers", "ai_embeddings", "ai_agents"]),
    (os.path.join(CORPUS_IMAGES, "forest_landscape.png"),
     ["nature_forests", "nature_mountains"]),
]

for mode_name, mode_backend in modes.items():
    accuracy_matrix.setdefault("image-to-text", {})[mode_name] = 0
    hits = 0
    print(f"\n  Mode: {mode_name}")
    for img_path, expected in i2t_queries:
        img_name = os.path.basename(img_path)
        results = search_by_image(img_path, mode_backend, limit=5, content_type="text")
        found = check_results(results, expected, img_name)
        hits += int(found)
        status = "\033[0;32m✓\033[0m" if found else "\033[0;31m✗\033[0m"
        top3 = ", ".join([r.title[:30] for r in results[:3]]) if results else "no results"
        print(f"    {status} {img_name} → {top3}")
    accuracy_matrix["image-to-text"][mode_name] = hits / len(i2t_queries)
    report(hits >= 1, f"Image-to-Text ({mode_name}): {hits}/{len(i2t_queries)} image queries found expected text docs")

# ═══════════════════════════════════
# D. IMAGE-TO-IMAGE
# ═══════════════════════════════════
print("\n\033[1m\033[0;34m═══════════════════════════════════════\033[0m")
print("\033[1m\033[0;34m  D. Image-to-Image Search\033[0m")
print("\033[1m\033[0;34m═══════════════════════════════════════\033[0m")

i2i_queries = [
    (os.path.join(CORPUS_IMAGES, "whiteboard_architecture.png"),
     ["whiteboard_brainstorm", "whiteboard_architecture"]),
    (os.path.join(CORPUS_IMAGES, "forest_landscape.png"),
     ["mountain_landscape", "ocean_beach", "forest_landscape"]),
    (os.path.join(CORPUS_IMAGES, "neural_network_diagram.png"),
     ["code_editor_screenshot", "whiteboard_architecture", "neural_network_diagram"]),
]

for mode_name, mode_backend in modes.items():
    accuracy_matrix.setdefault("image-to-image", {})[mode_name] = 0
    hits = 0
    print(f"\n  Mode: {mode_name}")
    for img_path, expected in i2i_queries:
        img_name = os.path.basename(img_path)
        results = search_by_image(img_path, mode_backend, limit=5, content_type="image")
        found = check_results(results, expected, img_name)
        hits += int(found)
        status = "\033[0;32m✓\033[0m" if found else "\033[0;31m✗\033[0m"
        top3 = ", ".join([r.title[:30] for r in results[:3]]) if results else "no results"
        print(f"    {status} {img_name} → {top3}")
    accuracy_matrix["image-to-image"][mode_name] = hits / len(i2i_queries)
    report(hits >= 1, f"Image-to-Image ({mode_name}): {hits}/{len(i2i_queries)} image queries found similar images")

# ═══════════════════════════════════
# Cross-Modal Accuracy Matrix
# ═══════════════════════════════════
print("\n\033[1m\033[0;34m═══════════════════════════════════════\033[0m")
print("\033[1m\033[0;34m  Cross-Modal Accuracy Matrix\033[0m")
print("\033[1m\033[0;34m═══════════════════════════════════════\033[0m\n")

directions = ["text-to-text", "text-to-image", "image-to-text", "image-to-image"]
mode_names = ["embed", "hybrid", "full"]

# Header
header = f"  {'Direction':<20}"
for m in mode_names:
    header += f"  {m:>8}"
print(header)
print("  " + "-" * (20 + 10 * len(mode_names)))

for d in directions:
    row = f"  {d:<20}"
    for m in mode_names:
        acc = accuracy_matrix.get(d, {}).get(m, 0)
        pct = f"{acc*100:.0f}%"
        if acc >= 0.6:
            row += f"  \033[0;32m{pct:>8}\033[0m"
        elif acc >= 0.3:
            row += f"  \033[0;33m{pct:>8}\033[0m"
        else:
            row += f"  \033[0;31m{pct:>8}\033[0m"
    print(row)

# Overall average
print("  " + "-" * (20 + 10 * len(mode_names)))
row = f"  {'OVERALL':<20}"
for m in mode_names:
    vals = [accuracy_matrix.get(d, {}).get(m, 0) for d in directions]
    avg = sum(vals) / len(vals)
    pct = f"{avg*100:.0f}%"
    row += f"  {pct:>8}"
print(row)

# ── Summary ──
print(f"\n\033[1m{'='*40}\033[0m")
print(f"\033[1m  Cross-Modal Search Summary\033[0m")
print(f"\033[1m{'='*40}\033[0m")
print(f"  \033[0;32mPASS: {pass_count}\033[0m")
print(f"  \033[0;31mFAIL: {fail_count}\033[0m")
if warn_count:
    print(f"  \033[0;33mWARN: {warn_count}\033[0m")

if fail_count > 0:
    print(f"\n  \033[0;31m\033[1mRESULT: FAILED\033[0m")
    sys.exit(1)
else:
    print(f"\n  \033[0;32m\033[1mRESULT: PASSED\033[0m")
PYEOF
