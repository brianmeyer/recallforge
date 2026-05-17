#!/usr/bin/env bash
# test_video_quality.sh - Video retrieval quality UAT.
# Validates text->video and image->video retrieval across search modes.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Video Quality Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

video_backend_runtime_healthy() {
    local backend="$1"
    PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
        RECALLFORGE_BACKEND="${backend}" \
        RECALLFORGE_MODE="embed" \
        RECALLFORGE_MLX_QUANTIZE="${RECALLFORGE_MLX_QUANTIZE:-4bit}" \
        python3 <<PYEOF >/dev/null 2>&1
from recallforge import get_backend

backend = get_backend()
backend.embed_text("video quality probe")
backend.embed_image("${CORPUS_DIR}/images/whiteboard_architecture.png")
PYEOF
}

SELECTED_BACKEND=""
USE_LIVE_BACKEND=0

if [[ "${UAT_VIDEO_LIVE:-0}" == "1" ]]; then
    while IFS= read -r candidate; do
        if [[ -n "${candidate}" ]] && video_backend_runtime_healthy "${candidate}"; then
            SELECTED_BACKEND="${candidate}"
            break
        fi
    done < <(live_backend_candidates || true)

    if [[ -n "${SELECTED_BACKEND}" ]]; then
        USE_LIVE_BACKEND=1
        info "Using live ${SELECTED_BACKEND} backend for video-quality retrieval."
    else
        info "No usable live video backend on this host; using deterministic video-quality backend."
    fi
else
    info "Using deterministic video-quality backend. Set UAT_VIDEO_LIVE=1 to exercise live model retrieval."
fi

export RECALLFORGE_BACKEND="${SELECTED_BACKEND:-torch}"
if [[ "${SELECTED_BACKEND:-}" == "mlx" ]]; then
    export RECALLFORGE_MLX_QUANTIZE=4bit
fi
export RECALLFORGE_STORE_PATH="${UAT_STORE}"
export UAT_VIDEO_USE_LIVE="${USE_LIVE_BACKEND}"

export UAT_VIDEO_PATH="${CORPUS_DIR}/videos/whiteboard_session.mp4"
export UAT_VIDEO_FFMPEG=0
if command -v ffmpeg >/dev/null 2>&1; then
    export UAT_VIDEO_FFMPEG=1
fi
# Corpus videos are committed real MP4s
export UAT_VIDEO_REAL=1
export UAT_VIDEO_IMAGE="${CORPUS_DIR}/images/whiteboard_architecture.png"

python3 <<'PYEOF'
import os
import re
import sys

sys.path.insert(0, "src")

from recallforge import get_backend, get_storage
from recallforge.backends.base import BackendInfo
from recallforge.search import HybridSearcher

STORE = os.environ["RECALLFORGE_STORE_PATH"]
VIDEO_PATH = os.environ["UAT_VIDEO_PATH"]
FFMPEG_AVAILABLE = os.environ["UAT_VIDEO_FFMPEG"] == "1"
REAL_VIDEO_AVAILABLE = os.environ["UAT_VIDEO_REAL"] == "1"
WHITEBOARD_IMAGE = os.environ["UAT_VIDEO_IMAGE"]
USE_LIVE_BACKEND = os.environ["UAT_VIDEO_USE_LIVE"] == "1"

pass_count = 0
fail_count = 0
skip_count = 0


def report(ok, msg):
    global pass_count, fail_count
    if ok:
        print(f"  \033[0;32mPASS\033[0m  {msg}")
        pass_count += 1
    else:
        print(f"  \033[0;31mFAIL\033[0m  {msg}")
        fail_count += 1


def skip(msg):
    global skip_count
    print(f"  \033[0;33mSKIP\033[0m  {msg}")
    skip_count += 1


class ConceptBackend:
    def __init__(self, mode="embed"):
        self._mode = mode

    def set_mode(self, mode):
        self._mode = mode

    def get_mode(self):
        return self._mode

    def _normalize_seed(self, seed: str) -> str:
        lowered = seed.lower()
        if "whiteboard" in lowered or "frame_0003" in lowered:
            return "whiteboard architecture diagram meeting"
        if "forest" in lowered or "frame_0002" in lowered:
            return "forest landscape green trees"
        if "pasta" in lowered or "frame_0001" in lowered:
            return "pasta dish white plate"
        return lowered

    def _vec(self, seed: str):
        import hashlib

        values = [0.0] * 2048
        tokens = re.findall(r"[a-z0-9]+", self._normalize_seed(seed))
        if not tokens:
            tokens = ["empty"]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).hexdigest()
            values[int(digest[:8], 16) % 2048] += 1.0
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def embed_text(self, text):
        return self._vec(text)

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]

    def embed_image(self, path):
        return self._vec(os.path.basename(path))

    def embed_video(self, path):
        return self._vec(os.path.basename(path))

    def rerank(self, query, documents, **_kwargs):
        query_tokens = set(re.findall(r"[a-z0-9]+", self._normalize_seed(query)))
        scores = []
        for doc in documents:
            doc_tokens = set(re.findall(r"[a-z0-9]+", self._normalize_seed(doc.get("text", "") + " " + doc.get("filepath", ""))))
            overlap = len(query_tokens & doc_tokens)
            scores.append(float(overlap) / float(max(len(query_tokens), 1)))
        return scores

    def expand_query(self, query):
        normalized = self._normalize_seed(query)
        return {"lex": normalized, "vec": normalized, "hyde": normalized}

    def needs_reranker(self):
        return self._mode == "hybrid"

    def needs_expander(self):
        return False

    def get_info(self):
        return BackendInfo(
            name="concept-mock",
            device="cpu",
            dtype="float32",
            embedder_loaded=True,
            reranker_loaded=self.needs_reranker(),
            expander_loaded=self.needs_expander(),
            memory_allocated_gb=0.0,
            quantization="none",
        )


print("\n\033[0;36m--- Index Episodic Video ---\033[0m\n")
os.environ["RECALLFORGE_MODE"] = "embed"
backend = get_backend() if USE_LIVE_BACKEND else ConceptBackend("embed")
storage = get_storage(STORE)
video_summary = storage.index_video(
    path=VIDEO_PATH,
    collection="video_quality",
    embed_text_func=backend.embed_text,
    embed_image_func=backend.embed_image,
    embed_video_func=getattr(backend, "embed_video", None),
    model="Qwen3-VL-Embedding-2B",
)
report(video_summary["indexed_transcripts"] >= 1, "video index created transcript sections")
if FFMPEG_AVAILABLE:
    report(video_summary["indexed_frames"] >= 1, "video index created frame sections")
else:
    report(video_summary["indexed_frames"] == 0, "video index stayed transcript-only without ffmpeg")

print("\n\033[0;36m--- Retrieval Matrix ---\033[0m\n")

expected_transcript = "whiteboard_session.mp4::transcript:"
expected_frame = "whiteboard_session.mp4::frame:"
expected_video = VIDEO_PATH
query_text = "whiteboard meeting root memories child frames transcripts action items"


def has_whiteboard_memory(paths):
    return any(
        expected_video in path
        or expected_transcript in path
        or expected_frame in path
        for path in paths
    )

for mode in ("embed", "hybrid"):
    os.environ["RECALLFORGE_MODE"] = mode
    backend_mode = get_backend() if USE_LIVE_BACKEND else ConceptBackend(mode)
    backend_mode.set_mode(mode)

    print(f"  Mode: {mode}")
    text_searcher = HybridSearcher(
        backend=backend_mode,
        storage=storage,
        limit=5,
        collection="video_quality",
        content_type="text",
    )
    transcript_results = text_searcher.search(query_text)
    transcript_paths = [r.filepath for r in transcript_results[:5]]
    transcript_hit = has_whiteboard_memory(transcript_paths)
    report(transcript_hit, f"text→video memory retrieval ({mode})")

    if REAL_VIDEO_AVAILABLE:
        video_searcher = HybridSearcher(
            backend=backend_mode,
            storage=storage,
            limit=5,
            collection="video_quality",
            content_type="video",
        )
        video_results = video_searcher.search_video(VIDEO_PATH)
        video_paths = [r.filepath for r in video_results[:5]]
        video_hit = any(expected_video in path for path in video_paths)
        report(video_hit, f"video→video retrieval ({mode})")

        video_to_text_searcher = HybridSearcher(
            backend=backend_mode,
            storage=storage,
            limit=5,
            collection="video_quality",
            content_type="text",
        )
        video_text_results = video_to_text_searcher.search_video(VIDEO_PATH)
        video_text_paths = [r.filepath for r in video_text_results[:5]]
        video_text_hit = has_whiteboard_memory(video_text_paths)
        report(video_text_hit, f"video→text memory retrieval ({mode})")
    else:
        skip(f"video→video retrieval ({mode}; no real video fixture available)")
        skip(f"video→text retrieval ({mode}; no real video fixture available)")

    if FFMPEG_AVAILABLE:
        image_searcher = HybridSearcher(
            backend=backend_mode,
            storage=storage,
            limit=5,
            collection="video_quality",
            content_type="image",
        )
        frame_text_results = image_searcher.search(query_text)
        frame_text_paths = [r.filepath for r in frame_text_results[:5]]
        frame_text_hit = has_whiteboard_memory(frame_text_paths)
        report(frame_text_hit, f"text→video frame/memory retrieval ({mode})")

        frame_image_results = image_searcher.search_image(WHITEBOARD_IMAGE)
        frame_image_paths = [r.filepath for r in frame_image_results[:5]]
        frame_image_hit = has_whiteboard_memory(frame_image_paths)
        report(frame_image_hit, f"image→video frame/memory retrieval ({mode})")

        video_to_image_searcher = HybridSearcher(
            backend=backend_mode,
            storage=storage,
            limit=5,
            collection="video_quality",
            content_type="image",
        )
        video_image_results = video_to_image_searcher.search_video(VIDEO_PATH)
        video_image_paths = [r.filepath for r in video_image_results[:5]]
        video_image_hit = has_whiteboard_memory(video_image_paths)
        report(video_image_hit, f"video→image memory retrieval ({mode})")
    else:
        skip(f"video→image retrieval ({mode}; ffmpeg unavailable)")
        skip(f"frame retrieval checks ({mode}; ffmpeg unavailable)")

print(f"\n\033[1m{'='*40}\033[0m")
print(f"\033[1m  Video Quality Summary\033[0m")
print(f"\033[1m{'='*40}\033[0m")
print(f"  \033[0;32mPASS: {pass_count}\033[0m")
print(f"  \033[0;31mFAIL: {fail_count}\033[0m")
print(f"  \033[0;33mSKIP: {skip_count}\033[0m")

if fail_count > 0:
    print(f"\n  \033[0;31m\033[1mRESULT: FAILED\033[0m")
    raise SystemExit(1)

print(f"\n  \033[0;32m\033[1mRESULT: PASSED\033[0m")
PYEOF
