#!/usr/bin/env bash
# test_benchmarks.sh - Benchmark smoke coverage for RecallForge.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Benchmark Smoke Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

export RECALLFORGE_BENCH_CACHE="${UAT_STORE}/bench-cache"

subsection "COCO Smoke Benchmark"

if ! python3 -c "import datasets" >/dev/null 2>&1; then
    skip "COCO smoke benchmark (datasets package not installed)"
    print_summary "Benchmark Smoke Tests"
    exit 0
fi

OUTPUT=$(python3 benchmarks/cross_modal_accuracy.py --backend auto --mode embed --dataset coco --limit 10 2>&1) || BENCH_EXIT=$?
BENCH_EXIT=${BENCH_EXIT:-0}

if [[ "$BENCH_EXIT" -ne 0 ]]; then
    if echo "$OUTPUT" | grep -qiE "Could not load .*HuggingFace|Connection|timed out|SSL|Temporary failure|Name or service not known"; then
        skip "COCO smoke benchmark (dataset download unavailable)"
        print_summary "Benchmark Smoke Tests"
        exit 0
    fi

    fail "COCO smoke benchmark"
    echo "    Output: $(echo "$OUTPUT" | head -8)"
    print_summary "Benchmark Smoke Tests"
    exit 1
fi

if [[ -f benchmarks/cross_modal_results.json ]]; then
    pass "benchmark results json written"
else
    fail "benchmark results json written"
fi

if python3 - <<'PYEOF'
import json
from pathlib import Path

payload = json.loads(Path("benchmarks/cross_modal_results.json").read_text())
assert payload.get("dataset_requested") == "coco"
assert payload.get("limit_requested") == 10
assert "text_to_image" in payload
assert "image_to_text" in payload
print("ok")
PYEOF
then
    pass "benchmark results contain COCO recall metrics"
else
    fail "benchmark results contain COCO recall metrics"
fi

print_summary "Benchmark Smoke Tests"
