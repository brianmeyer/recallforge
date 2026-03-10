#!/usr/bin/env bash
# test_cli.sh - CLI end-to-end UAT.
# Tests all recallforge subcommands.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge CLI Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

export RECALLFORGE_BACKEND=torch
export RECALLFORGE_MODE=embed
export RECALLFORGE_STORE_PATH="${UAT_STORE}"

# ──────────────────────────────────────────────
subsection "Help and Version"
# ──────────────────────────────────────────────

run_test_grep "recallforge --help" "Cross-Modal" recallforge --help
run_test_grep "recallforge --version" "RecallForge" recallforge --version
run_test_grep "recallforge index --help" "index" recallforge index --help
run_test_grep "recallforge search --help" "query" recallforge search --help
run_test_grep "recallforge serve --help" "Start MCP" recallforge serve --help
run_test_grep "recallforge status --help" "status" recallforge status --help

# ──────────────────────────────────────────────
subsection "Index Single Text File"
# ──────────────────────────────────────────────

TEXT_FILE="${CORPUS_DIR}/text/ai_transformers.md"
OUTPUT=$(recallforge index "$TEXT_FILE" --collection cli_test --store-path "$UAT_STORE" 2>&1)
if echo "$OUTPUT" | grep -q "Indexed"; then
    pass "recallforge index <text file>"
else
    fail "recallforge index <text file>"
    echo "    Output: $OUTPUT"
fi

# ──────────────────────────────────────────────
subsection "Index Single Image"
# ──────────────────────────────────────────────

IMG_FILE="${CORPUS_DIR}/images/whiteboard_architecture.png"
if [[ -f "$IMG_FILE" ]]; then
    OUTPUT=$(recallforge index "$IMG_FILE" --collection cli_test --store-path "$UAT_STORE" 2>&1)
    if echo "$OUTPUT" | grep -q "Indexed\|Indexing image"; then
        pass "recallforge index <image>"
    else
        fail "recallforge index <image>"
        echo "    Output: $OUTPUT"
    fi
else
    skip "recallforge index <image> (no test images)"
fi

# ──────────────────────────────────────────────
subsection "Index Directory (recursive)"
# ──────────────────────────────────────────────

OUTPUT=$(recallforge index "${CORPUS_DIR}/text" --collection cli_dir_test --store-path "$UAT_STORE" 2>&1)
INDEXED_COUNT=$(echo "$OUTPUT" | grep -c "Indexing")
if [[ $INDEXED_COUNT -ge 5 ]]; then
    pass "recallforge index <directory> indexed $INDEXED_COUNT files"
else
    fail "recallforge index <directory> indexed only $INDEXED_COUNT files"
fi

# ──────────────────────────────────────────────
subsection "Search Commands"
# ──────────────────────────────────────────────

# Default search
OUTPUT=$(recallforge search "transformer architecture" --store-path "$UAT_STORE" --collection cli_test 2>&1)
if echo "$OUTPUT" | grep -q "Results for"; then
    pass "recallforge search (default mode)"
else
    fail "recallforge search (default mode)"
    echo "    Output: $(echo "$OUTPUT" | head -3)"
fi

# Explicit embed mode
OUTPUT=$(recallforge search "transformer architecture" --mode embed --store-path "$UAT_STORE" --collection cli_dir_test 2>&1)
if echo "$OUTPUT" | grep -q "Results for"; then
    pass "recallforge search --mode embed"
else
    fail "recallforge search --mode embed"
fi

# Explicit hybrid mode
OUTPUT=$(recallforge search "transformer architecture" --mode hybrid --store-path "$UAT_STORE" --collection cli_dir_test 2>&1)
if echo "$OUTPUT" | grep -q "Results for"; then
    pass "recallforge search --mode hybrid"
else
    fail "recallforge search --mode hybrid"
fi

# Explicit full mode
OUTPUT=$(recallforge search "transformer architecture" --mode full --store-path "$UAT_STORE" --collection cli_dir_test 2>&1)
if echo "$OUTPUT" | grep -q "Results for"; then
    pass "recallforge search --mode full"
else
    fail "recallforge search --mode full"
fi

# Search with limit
OUTPUT=$(recallforge search "machine learning" --limit 3 --store-path "$UAT_STORE" --collection cli_dir_test 2>&1)
RESULT_COUNT=$(echo "$OUTPUT" | grep -cE "^\d+\.")
if [[ $RESULT_COUNT -le 3 ]]; then
    pass "recallforge search --limit 3 returns <= 3 results"
else
    fail "recallforge search --limit 3 returned $RESULT_COUNT results"
fi

# Search with content-type filter
OUTPUT=$(recallforge search "architecture" --content-type text --store-path "$UAT_STORE" --collection cli_dir_test 2>&1)
if echo "$OUTPUT" | grep -q "Results for"; then
    pass "recallforge search --content-type text"
else
    fail "recallforge search --content-type text"
fi

# ──────────────────────────────────────────────
subsection "Status Command"
# ──────────────────────────────────────────────

OUTPUT=$(recallforge status --store-path "$UAT_STORE" 2>&1)
if echo "$OUTPUT" | grep -q "RecallForge Status"; then
    pass "recallforge status"
else
    fail "recallforge status"
    echo "    Output: $(echo "$OUTPUT" | head -5)"
fi

if echo "$OUTPUT" | grep -q "Backend:"; then
    pass "Status shows backend info"
else
    fail "Status missing backend info"
fi

if echo "$OUTPUT" | grep -q "Embeddings:"; then
    pass "Status shows embedding count"
else
    fail "Status missing embedding count"
fi

# ──────────────────────────────────────────────
subsection "Serve Command (start/verify/stop)"
# ──────────────────────────────────────────────

info "Starting MCP server in background..."
recallforge serve --mode embed --store-path "$UAT_STORE" &
SERVER_PID=$!
sleep 3

if kill -0 $SERVER_PID 2>/dev/null; then
    pass "recallforge serve starts and stays running (PID=$SERVER_PID)"

    # Graceful shutdown
    kill -INT $SERVER_PID
    sleep 2

    if ! kill -0 $SERVER_PID 2>/dev/null; then
        pass "recallforge serve stops on SIGINT"
    else
        fail "recallforge serve did not stop on SIGINT"
        kill -9 $SERVER_PID 2>/dev/null
    fi
else
    fail "recallforge serve failed to start"
fi

print_summary "CLI Tests"
