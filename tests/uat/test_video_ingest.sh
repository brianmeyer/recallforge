#!/usr/bin/env bash
# test_video_ingest.sh - Video ingest UAT.
# Validates transcript indexing everywhere and frame indexing when ffmpeg exists.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Video Ingest Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

SELECTED_BACKEND="$(select_live_backend || true)"
if [[ -n "${SELECTED_BACKEND}" ]]; then
    export RECALLFORGE_BACKEND="${SELECTED_BACKEND}"
fi
if [[ "${SELECTED_BACKEND:-}" == "mlx" ]]; then
    export RECALLFORGE_MLX_QUANTIZE=4bit
fi
export RECALLFORGE_MODE=embed
export RECALLFORGE_STORE_PATH="${UAT_STORE}"

subsection "Backend Probe"

if python3 <<PYEOF >/dev/null 2>&1
import sys
sys.path.insert(0, "src")
from recallforge import get_backend

backend = get_backend()
backend.embed_text("video ingest probe")
PYEOF
then
    pass "live backend can embed text for video ingest"
else
    skip "video CLI ingest (no usable live backend on this host)"
    print_summary "Video Ingest Tests"
    exit 0
fi

subsection "Generate Synthetic Video"

VIDEO_META=$(python3 "${HELPERS_DIR}/generate_test_video.py" \
    "${UAT_STORE}/sample_video.mp4" \
    "${CORPUS_DIR}/images/food_pasta_dish.png" \
    "${CORPUS_DIR}/images/forest_landscape.png" \
    "${CORPUS_DIR}/images/whiteboard_architecture.png")

VIDEO_PATH=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["video_path"])' <<<"$VIDEO_META")
FFMPEG_AVAILABLE=$(python3 -c 'import json,sys; print("1" if json.loads(sys.stdin.read())["ffmpeg_available"] else "0")' <<<"$VIDEO_META")

if [[ -f "$VIDEO_PATH" ]]; then
    pass "synthetic video fixture created"
else
    fail "synthetic video fixture created"
fi

subsection "CLI Video Index"

OUTPUT=$(recallforge index "$VIDEO_PATH" --collection video_cli_test --store-path "$UAT_STORE" 2>&1 || true)
if echo "$OUTPUT" | grep -q "Indexing video" && ! echo "$OUTPUT" | grep -qE "Traceback|Error indexing|OSError:"; then
    pass "recallforge index <video>"
else
    fail "recallforge index <video>"
    echo "    Output: $(echo "$OUTPUT" | head -5)"
fi

subsection "Transcript Search"

OUTPUT=$(recallforge search "whiteboard architecture diagram from a meeting" \
    --collection video_cli_test \
    --content-type text \
    --store-path "$UAT_STORE" 2>&1 || true)

if echo "$OUTPUT" | grep -q "sample_video.mp4::transcript:"; then
    pass "video transcript assets searchable via CLI"
else
    fail "video transcript assets searchable via CLI"
    echo "    Output: $(echo "$OUTPUT" | head -8)"
fi

subsection "Derived Asset Inspection"

python3 <<PYEOF
import os
import sys

sys.path.insert(0, "src")

from recallforge import get_storage

store = get_storage("${UAT_STORE}")
rows = store._embeddings_table.search().where("collection = 'video_cli_test'").to_list()

transcript_rows = [r for r in rows if "::transcript:" in r.get("file_path", "")]
frame_rows = [r for r in rows if "::frame:" in r.get("file_path", "")]

print(f"TRANSCRIPTS={len(transcript_rows)}")
print(f"FRAMES={len(frame_rows)}")
PYEOF

TRANSCRIPT_COUNT=$(python3 - <<PYEOF
import os
import sys
sys.path.insert(0, "src")
from recallforge import get_storage
store = get_storage("${UAT_STORE}")
rows = store._embeddings_table.search().where("collection = 'video_cli_test'").to_list()
print(sum(1 for r in rows if "::transcript:" in r.get("file_path", "")))
PYEOF
)

FRAME_COUNT=$(python3 - <<PYEOF
import os
import sys
sys.path.insert(0, "src")
from recallforge import get_storage
store = get_storage("${UAT_STORE}")
rows = store._embeddings_table.search().where("collection = 'video_cli_test'").to_list()
print(sum(1 for r in rows if "::frame:" in r.get("file_path", "")))
PYEOF
)

if [[ "$TRANSCRIPT_COUNT" -ge 1 ]]; then
    pass "video ingest created transcript embeddings ($TRANSCRIPT_COUNT)"
else
    fail "video ingest created transcript embeddings"
fi

if [[ "$FFMPEG_AVAILABLE" == "1" ]]; then
    if [[ "$FRAME_COUNT" -ge 1 ]]; then
        pass "video ingest created frame embeddings ($FRAME_COUNT)"
    else
        fail "video ingest created frame embeddings"
    fi

    OUTPUT=$(recallforge search --image "${CORPUS_DIR}/images/whiteboard_architecture.png" \
        --collection video_cli_test \
        --content-type image \
        --store-path "$UAT_STORE" 2>&1 || true)
    if echo "$OUTPUT" | grep -q "sample_video.mp4::frame:"; then
        pass "video frames retrievable via CLI image query"
    else
        fail "video frames retrievable via CLI image query"
        echo "    Output: $(echo "$OUTPUT" | head -8)"
    fi
else
    skip "video frame extraction (ffmpeg unavailable; transcript-only fallback active)"
fi

print_summary "Video Ingest Tests"
