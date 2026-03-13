#!/usr/bin/env bash
# test_backends.sh - Model backend UAT.
# Tests embedder, reranker, expander on each available backend.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Backend Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images
VIDEO_META=$(python3 "${HELPERS_DIR}/generate_test_video.py" \
    "${UAT_STORE}/backend_sample_video.mp4" \
    "${CORPUS_DIR}/images/food_pasta_dish.png" \
    "${CORPUS_DIR}/images/forest_landscape.png" \
    "${CORPUS_DIR}/images/whiteboard_architecture.png")
export UAT_BACKEND_VIDEO_PATH=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["video_path"])' <<<"$VIDEO_META")
export UAT_BACKEND_VIDEO_FFMPEG=$(python3 -c 'import json,sys; print("1" if json.loads(sys.stdin.read())["ffmpeg_available"] else "0")' <<<"$VIDEO_META")
export UAT_BACKEND_VIDEO_REAL=$(python3 -c 'import json,sys; print("1" if json.loads(sys.stdin.read()).get("real_video_available") else "0")' <<<"$VIDEO_META")

MLX_HEALTHY=0
TORCH_HEALTHY=0

if is_apple_silicon && backend_runtime_healthy mlx; then
    MLX_HEALTHY=1
fi

if backend_runtime_healthy torch; then
    TORCH_HEALTHY=1
fi

BACKEND_ORDER=()
if [[ "${MLX_HEALTHY}" -eq 1 ]]; then
    BACKEND_ORDER+=("mlx")
fi
if [[ "${TORCH_HEALTHY}" -eq 1 ]]; then
    BACKEND_ORDER+=("torch")
fi
if [[ ${#BACKEND_ORDER[@]} -gt 0 ]]; then
    info "Backend order: ${BACKEND_ORDER[*]}"
fi

# Helper: flush GPU memory between backend tests
# Each backend test runs in its own subprocess, but MPS/Metal may hold onto
# memory after the process exits. Give the OS a moment to reclaim.
_cleanup_gpu() {
    python3 -c "
import gc; gc.collect()
try:
    import torch
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
except: pass
" 2>/dev/null || true
    # Give Metal/MPS time to release unified memory
    sleep 3
}

# ──────────────────────────────────────────────
subsection "MLX Backend (4-bit Embed Coverage)"
# ──────────────────────────────────────────────

if [[ "${MLX_HEALTHY}" -eq 1 ]]; then
    info "Testing MLX 4-bit embed pipeline (embed_text, embed_texts, embed_image, embed_video)..."

    python3 << 'PYEOF'
import os, sys, time
sys.path.insert(0, "src")

os.environ["RECALLFORGE_BACKEND"] = "mlx"
os.environ["RECALLFORGE_MLX_QUANTIZE"] = "4bit"
os.environ["RECALLFORGE_MODE"] = "embed"

from recallforge import get_backend

backend = get_backend()

print("Loading MLX embedder (4-bit)...")
t0 = time.time()
backend._load_embedder()
print(f"  Embedder loaded in {time.time()-t0:.1f}s")

vec = backend.embed_text("test text")
assert vec.shape == (2048,), f"FAIL: embed_text shape {vec.shape} != (2048,)"
assert vec.dtype.name == "float32", f"FAIL: embed_text dtype {vec.dtype}"
norm = (vec**2).sum()**0.5
assert 0.9 < norm < 1.1, f"FAIL: embed_text norm {norm}"
print("  PASS  MLX 4-bit embed_text: 2048-dim float32 vector")

texts = ["Hello world", "Goodbye world", "Test embedding"]
vecs = backend.embed_texts(texts)
assert vecs.shape == (len(texts), 2048), f"FAIL: embed_texts shape {vecs.shape}"
print(f"  PASS  MLX 4-bit embed_texts batch: {vecs.shape}")

img_path = os.path.join("tests", "uat", "corpus", "images", "whiteboard_architecture.png")
assert os.path.exists(img_path), f"FAIL: missing test image {img_path}"
vec_img = backend.embed_image(img_path)
assert vec_img.shape == (2048,), f"FAIL: embed_image shape {vec_img.shape}"
print("  PASS  MLX 4-bit embed_image: 2048-dim vector")

video_path = os.environ["UAT_BACKEND_VIDEO_PATH"]
if os.environ.get("UAT_BACKEND_VIDEO_REAL") == "1":
    vec_vid = backend.embed_video(video_path)
    assert vec_vid.shape == (2048,), f"FAIL: embed_video shape {vec_vid.shape}"
    print("  PASS  MLX 4-bit embed_video: 2048-dim vector")
else:
    print("  SKIP  MLX 4-bit embed_video: no real video fixture available")

print("\nMLX 4-bit embed coverage: ALL PASSED")
PYEOF
    MLX_EMBED_RC=$?

    if [[ $MLX_EMBED_RC -eq 0 ]]; then
        pass "MLX 4-bit embed coverage"
    elif ! backend_runtime_healthy mlx; then
        skip "MLX 4-bit embed coverage (runtime became unavailable)"
    else
        fail "MLX 4-bit embed coverage had failures"
    fi
else
    skip "MLX 4-bit embed coverage (runtime unavailable)"
fi

# ──────────────────────────────────────────────
subsection "MLX Backend (bf16 Compatibility)"
# ──────────────────────────────────────────────

if [[ "${MLX_HEALTHY}" -eq 1 ]]; then
    info "Testing MLX bf16 compatibility backend..."
    info "Expected behavior: embedder/reranker on MLX, expander via torch fallback."

    python3 << 'PYEOF'
import os, sys, time
sys.path.insert(0, "src")

os.environ["RECALLFORGE_BACKEND"] = "mlx"
os.environ["RECALLFORGE_MODE"] = "full"
os.environ["RECALLFORGE_MLX_QUANTIZE"] = "bf16"

from recallforge import get_backend

backend = get_backend()

# Embedder
print("Loading MLX embedder (bf16)...")
t0 = time.time()
backend._load_embedder()
print(f"  Loaded in {time.time()-t0:.1f}s")

vec = backend.embed_text("Test embedding with MLX.")
assert vec.shape == (2048,), f"FAIL: shape {vec.shape}"
print(f"  PASS  MLX bf16 embed_text: shape={vec.shape}")

# Reranker
print("Loading MLX reranker (bf16)...")
t0 = time.time()
backend._load_reranker()
print(f"  Loaded in {time.time()-t0:.1f}s")

docs = [{"text": "Test doc one"}, {"text": "Test doc two"}]
scores = backend.rerank("test", docs)
assert len(scores) == 2
print(f"  PASS  MLX bf16 rerank: scores={[f'{s:.3f}' for s in scores]}")

# Expander (torch fallback)
print("Loading expander (torch fallback, expected for bf16 coverage)...")
t0 = time.time()
backend._load_expander()
print(f"  Loaded in {time.time()-t0:.1f}s")

exp = backend.expand_query("search optimization")
assert "lex" in exp and "vec" in exp and "hyde" in exp
print(f"  PASS  MLX expander (torch fallback): keys={list(exp.keys())}")

info = backend.get_info()
print(f"  PASS  MLX info: device={info.device} quant={info.quantization} mem={info.memory_allocated_gb:.1f}GB")

print("\nMLX bf16 compatibility backend: ALL PASSED")
PYEOF
    MLX_RC=$?

    if [[ $MLX_RC -eq 0 ]]; then
        pass "MLX bf16 compatibility backend all tests"
    elif ! backend_runtime_healthy mlx; then
        skip "MLX bf16 compatibility backend (runtime became unavailable)"
    else
        fail "MLX bf16 compatibility backend tests had failures"
    fi

    _cleanup_gpu

    # MLX 4-bit
    if backend_runtime_healthy mlx; then
        info "Testing MLX 4-bit backend..."

        python3 << 'PYEOF'
import os, sys, time
sys.path.insert(0, "src")

os.environ["RECALLFORGE_BACKEND"] = "mlx"
os.environ["RECALLFORGE_MODE"] = "embed"
os.environ["RECALLFORGE_MLX_QUANTIZE"] = "4bit"

from recallforge import get_backend

backend = get_backend()

print("Loading MLX embedder (4-bit)...")
t0 = time.time()
backend._load_embedder()
print(f"  Loaded in {time.time()-t0:.1f}s")

vec = backend.embed_text("Test 4-bit embedding.")
assert vec.shape == (2048,), f"FAIL: shape {vec.shape}"
print(f"  PASS  MLX 4-bit embed_text: shape={vec.shape}")

info = backend.get_info()
assert info.quantization == "4bit"
print(f"  PASS  MLX 4-bit info: quant={info.quantization} mem={info.memory_allocated_gb:.1f}GB")

print("\nMLX 4-bit backend: ALL PASSED")
PYEOF
        MLX4_RC=$?
        if [[ $MLX4_RC -eq 0 ]]; then
            pass "MLX 4-bit backend all tests"
        elif ! backend_runtime_healthy mlx; then
            skip "MLX 4-bit backend (runtime became unavailable)"
        else
            fail "MLX 4-bit backend tests had failures"
        fi
    else
        skip "MLX 4-bit backend (runtime unavailable)"
    fi
else
    skip "MLX backend (runtime unavailable)"
    skip "MLX 4-bit backend (runtime unavailable)"
fi

_cleanup_gpu

# ──────────────────────────────────────────────
subsection "Torch Backend"
# ──────────────────────────────────────────────

if [[ "${TORCH_HEALTHY}" -eq 1 ]]; then
    info "Testing Torch backend (embed_text, embed_image, embed_video, rerank, expand_query)..."

    python3 << 'PYEOF'
import os, sys, time
sys.path.insert(0, "src")

os.environ["RECALLFORGE_BACKEND"] = "torch"
os.environ["RECALLFORGE_MODE"] = "full"

from recallforge import get_backend

backend = get_backend()

# ── Embedder ──
print("Loading embedder...")
t0 = time.time()
backend._load_embedder()
print(f"  Embedder loaded in {time.time()-t0:.1f}s")

vec = backend.embed_text("The quick brown fox jumps over the lazy dog.")
assert vec.shape == (2048,), f"FAIL: embed_text shape {vec.shape} != (2048,)"
assert vec.dtype.name == "float32", f"FAIL: embed_text dtype {vec.dtype}"
norm = (vec**2).sum()**0.5
assert 0.9 < norm < 1.1, f"FAIL: embed_text norm {norm}"
print("  PASS  Torch embed_text: 2048-dim float32 vector")

# Batch embed
vecs = backend.embed_texts(["Hello world", "Goodbye world", "Test embedding"])
assert vecs.shape == (3, 2048), f"FAIL: embed_texts shape {vecs.shape}"
print("  PASS  Torch embed_texts batch: (3, 2048)")

# Image embed (use a generated test image if available)
img_dir = os.path.join("tests", "uat", "corpus", "images")
if os.path.exists(img_dir):
    images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    if images:
        img_path = os.path.join(img_dir, images[0])
        vec_img = backend.embed_image(img_path)
        assert vec_img.shape == (2048,), f"FAIL: embed_image shape {vec_img.shape}"
        print(f"  PASS  Torch embed_image: 2048-dim vector from {images[0]}")
    else:
        print("  SKIP  No test images found")
else:
    print("  SKIP  Image corpus not generated yet")

video_path = os.environ["UAT_BACKEND_VIDEO_PATH"]
if os.environ.get("UAT_BACKEND_VIDEO_REAL") == "1":
    vec_vid = backend.embed_video(video_path)
    assert vec_vid.shape == (2048,), f"FAIL: embed_video shape {vec_vid.shape}"
    print("  PASS  Torch embed_video: 2048-dim vector")
else:
    print("  SKIP  Torch embed_video: no real video fixture available")

# ── Reranker ──
print("Loading reranker...")
t0 = time.time()
backend._load_reranker()
print(f"  Reranker loaded in {time.time()-t0:.1f}s")

docs = [
    {"text": "AI agents use memory to store past experiences."},
    {"text": "The recipe calls for two cups of flour and three eggs."},
    {"text": "Memory enables agents to recall previous conversations."},
]
scores = backend.rerank("How do AI agents use memory?", docs)
assert len(scores) == 3, f"FAIL: rerank returned {len(scores)} scores"
assert all(isinstance(s, float) for s in scores), "FAIL: scores not all float"
print(f"  PASS  Torch rerank: {len(scores)} scores = {[f'{s:.3f}' for s in scores]}")

# Verify memory docs score higher than cooking doc
if scores[0] > scores[1] and scores[2] > scores[1]:
    print("  PASS  Reranker correctly ranks memory docs above cooking")
else:
    print("  WARN  Reranker ranking may be suboptimal (not a hard fail)")

# ── Expander ──
print("Loading expander...")
t0 = time.time()
backend._load_expander()
print(f"  Expander loaded in {time.time()-t0:.1f}s")

expansions = backend.expand_query("machine learning training")
assert "lex" in expansions, "FAIL: missing 'lex' expansion"
assert "vec" in expansions, "FAIL: missing 'vec' expansion"
assert "hyde" in expansions, "FAIL: missing 'hyde' expansion"
print(f"  PASS  Torch expand_query: lex='{expansions['lex'][:50]}...'")

# ── Memory usage ──
import subprocess
pid = os.getpid()
try:
    if sys.platform == "darwin":
        rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip())
    else:
        rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip())
    print(f"  INFO  Torch backend RSS: {rss/1024:.0f} MB (all 3 models loaded)")
except Exception:
    pass

# ── Backend info ──
info = backend.get_info()
assert info.name == "torch"
assert info.embedder_loaded
assert info.reranker_loaded
assert info.expander_loaded
print(f"  PASS  Backend info: name={info.name} device={info.device} mem={info.memory_allocated_gb:.1f}GB")

print("\nTorch backend: ALL PASSED")
PYEOF
    TORCH_RC=$?

    if [[ $TORCH_RC -eq 0 ]]; then
        pass "Torch backend all tests"
    elif ! backend_runtime_healthy torch; then
        skip "Torch backend (runtime/model assets became unavailable)"
    else
        fail "Torch backend tests had failures"
    fi
else
    skip "Torch backend (runtime/model assets unavailable)"
fi

_cleanup_gpu

# ──────────────────────────────────────────────
subsection "Backend Auto-Detection"
# ──────────────────────────────────────────────

python3 << 'PYEOF'
import os, sys, platform
sys.path.insert(0, "src")

os.environ["RECALLFORGE_BACKEND"] = "auto"

from recallforge import get_backend
from recallforge.backends import MLX_AVAILABLE, get_mlx_probe_reason

backend = get_backend()
info = backend.get_info()

probe_reason = get_mlx_probe_reason()

if platform.system() == "Darwin" and platform.machine() == "arm64" and MLX_AVAILABLE:
    expected = "mlx"
else:
    expected = "torch"

assert info.name == expected, f"FAIL: auto-detected '{info.name}' but expected '{expected}'"
if probe_reason:
    print(f"  INFO  MLX probe reason: {probe_reason}")
print(f"  PASS  Auto-detection chose '{info.name}' (expected '{expected}') on {platform.system()}/{platform.machine()}")
PYEOF

if [[ $? -eq 0 ]]; then
    pass "Backend auto-detection"
else
    fail "Backend auto-detection"
fi

print_summary "Backend Tests"
