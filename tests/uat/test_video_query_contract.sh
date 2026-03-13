#!/usr/bin/env bash
# test_video_query_contract.sh - Raw video query contract UAT.
# Makes the lack/presence of native raw-video search explicit in run_all.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Raw Video Query Contract"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

subsection "Generate Synthetic Video"

VIDEO_META=$(python3 "${HELPERS_DIR}/generate_test_video.py" \
    "${UAT_STORE}/sample_video.mp4" \
    "${CORPUS_DIR}/images/food_pasta_dish.png" \
    "${CORPUS_DIR}/images/forest_landscape.png" \
    "${CORPUS_DIR}/images/whiteboard_architecture.png")

VIDEO_PATH=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["video_path"])' <<<"$VIDEO_META")
VIDEO_REAL=$(python3 -c 'import json,sys; print("1" if json.loads(sys.stdin.read()).get("real_video_available") else "0")' <<<"$VIDEO_META")

if [[ -f "$VIDEO_PATH" ]]; then
    pass "synthetic video fixture created"
else
    fail "synthetic video fixture created"
    print_summary "Raw Video Query Contract"
    exit 1
fi

if [[ "$VIDEO_REAL" != "1" ]]; then
    skip "raw video query smoke requires a real video fixture"
    print_summary "Raw Video Query Contract"
    exit 0
fi

subsection "Interface Detection"

CLI_HAS_VIDEO=0
if recallforge search --help 2>&1 | grep -q -- '--video'; then
    CLI_HAS_VIDEO=1
    pass "CLI exposes raw video query flag (--video)"
else
    skip "CLI does not expose raw video query flag (--video)"
fi

SEARCHER_HAS_VIDEO=$(python3 <<'PYEOF'
import sys
sys.path.insert(0, "src")
from recallforge.search import HybridSearcher
print("1" if hasattr(HybridSearcher, "search_video") else "0")
PYEOF
)

if [[ "$SEARCHER_HAS_VIDEO" == "1" ]]; then
    pass "HybridSearcher exposes raw video query API"
else
    skip "HybridSearcher does not expose raw video query API"
fi

if [[ "$CLI_HAS_VIDEO" == "0" && "$SEARCHER_HAS_VIDEO" == "0" ]]; then
    skip "raw video query not implemented; current video support is ingest-only via transcript/frame assets"
    print_summary "Raw Video Query Contract"
    exit 0
fi

if [[ "$CLI_HAS_VIDEO" != "$SEARCHER_HAS_VIDEO" ]]; then
    fail "raw video query surface is partially implemented (CLI/searcher mismatch)"
    print_summary "Raw Video Query Contract"
    exit 1
fi

SELECTED_BACKEND="$(select_live_backend || true)"
if [[ -z "${SELECTED_BACKEND}" ]]; then
    skip "raw video query smoke (no usable live backend on this host)"
    print_summary "Raw Video Query Contract"
    exit 0
fi

export RECALLFORGE_BACKEND="${SELECTED_BACKEND}"
if [[ "${SELECTED_BACKEND}" == "mlx" ]]; then
    export RECALLFORGE_MLX_QUANTIZE=4bit
fi
export RECALLFORGE_MODE=embed

subsection "CLI Raw Video Query Smoke"

OUTPUT=$(recallforge index "$VIDEO_PATH" --collection video_query_contract --store-path "$UAT_STORE" 2>&1 || true)
if echo "$OUTPUT" | grep -q "Indexing video" && ! echo "$OUTPUT" | grep -qE "Traceback|Error indexing|OSError:"; then
    pass "recallforge index <video> for raw video query smoke"
else
    fail "recallforge index <video> for raw video query smoke"
    echo "    Output: $(echo "$OUTPUT" | head -5)"
fi

OUTPUT=$(recallforge search --video "$VIDEO_PATH" --collection video_query_contract --content-type video --store-path "$UAT_STORE" 2>&1 || true)
if echo "$OUTPUT" | grep -qE "sample_video\\.mp4(\\b|')"; then
    pass "recallforge search --video returns video-linked results"
else
    fail "recallforge search --video returns video-linked results"
    echo "    Output: $(echo "$OUTPUT" | head -8)"
fi

python3 <<PYEOF
import os
import sys
sys.path.insert(0, "src")

from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

os.environ["RECALLFORGE_STORE_PATH"] = "${UAT_STORE}"
backend = get_backend()
storage = get_storage("${UAT_STORE}")
searcher = HybridSearcher(
    backend=backend,
    storage=storage,
    limit=5,
    collection="video_query_contract",
    content_type="text",
)
results = searcher.search_video("${VIDEO_PATH}")
paths = [r.filepath for r in results]
if any("sample_video.mp4::transcript:" in path for path in paths):
    print("  \033[0;32mPASS\033[0m  HybridSearcher.search_video returns transcript-linked results")
else:
    print("  \033[0;31mFAIL\033[0m  HybridSearcher.search_video returns transcript-linked results")
    raise SystemExit(1)
PYEOF

print_summary "Raw Video Query Contract"
