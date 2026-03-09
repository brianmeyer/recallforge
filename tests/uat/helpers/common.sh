#!/usr/bin/env bash
# common.sh - Shared utilities for RecallForge UAT scripts.

set -euo pipefail

# ──────────────────────────────────────────────
# Colors
# ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ──────────────────────────────────────────────
# Counters
# ──────────────────────────────────────────────
_PASS_COUNT=0
_FAIL_COUNT=0
_SKIP_COUNT=0

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
UAT_DIR="${REPO_ROOT}/tests/uat"
CORPUS_DIR="${UAT_DIR}/corpus"
HELPERS_DIR="${UAT_DIR}/helpers"
# Each test gets its own temp store to avoid collisions
UAT_STORE="${TMPDIR:-/tmp}/recallforge-uat-$$"

# ──────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────

section() {
    echo ""
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════${NC}"
}

subsection() {
    echo ""
    echo -e "${CYAN}--- $1 ---${NC}"
}

pass() {
    echo -e "  ${GREEN}PASS${NC}  $1"
    ((_PASS_COUNT++)) || true
}

fail() {
    echo -e "  ${RED}FAIL${NC}  $1"
    ((_FAIL_COUNT++)) || true
}

skip() {
    echo -e "  ${YELLOW}SKIP${NC}  $1"
    ((_SKIP_COUNT++)) || true
}

info() {
    echo -e "  ${CYAN}INFO${NC}  $1"
}

warn() {
    echo -e "  ${YELLOW}WARN${NC}  $1"
}

# Run a command; pass if exit 0, fail otherwise
# Usage: run_test "description" command arg1 arg2 ...
run_test() {
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        pass "$desc"
        return 0
    else
        fail "$desc"
        return 1
    fi
}

# Run a command; capture output; pass if output contains expected string
# Usage: run_test_grep "description" "expected_string" command arg1 ...
run_test_grep() {
    local desc="$1"
    local expected="$2"
    shift 2
    local output
    output=$("$@" 2>&1) || true
    if echo "$output" | grep -q "$expected"; then
        pass "$desc"
        return 0
    else
        fail "$desc (expected '$expected' not found)"
        echo "    Output: $(echo "$output" | head -3)"
        return 1
    fi
}

# Print summary and return exit code
print_summary() {
    local test_name="${1:-UAT}"
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════${NC}"
    echo -e "${BOLD}  ${test_name} Summary${NC}"
    echo -e "${BOLD}═══════════════════════════════════════${NC}"
    echo -e "  ${GREEN}PASS: ${_PASS_COUNT}${NC}"
    echo -e "  ${RED}FAIL: ${_FAIL_COUNT}${NC}"
    echo -e "  ${YELLOW}SKIP: ${_SKIP_COUNT}${NC}"
    local total=$((_PASS_COUNT + _FAIL_COUNT + _SKIP_COUNT))
    echo -e "  Total: ${total}"
    echo ""

    if [[ $_FAIL_COUNT -gt 0 ]]; then
        echo -e "  ${RED}${BOLD}RESULT: FAILED${NC}"
        return 1
    else
        echo -e "  ${GREEN}${BOLD}RESULT: PASSED${NC}"
        return 0
    fi
}

# Get RSS in MB for a given PID
get_rss_mb() {
    local pid="$1"
    if [[ "$(uname)" == "Darwin" ]]; then
        ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.1f", $1/1024}'
    else
        ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.1f", $1/1024}'
    fi
}

# Cleanup temp store on exit
cleanup_store() {
    if [[ -d "$UAT_STORE" ]]; then
        rm -rf "$UAT_STORE"
    fi
}

# Ensure test images exist (generate if needed)
ensure_test_images() {
    local img_dir="${CORPUS_DIR}/images"
    if [[ ! -f "${img_dir}/whiteboard_architecture.png" ]]; then
        info "Generating test images..."
        python3 "${HELPERS_DIR}/generate_test_images.py"
    fi
}

# Check if we're on Apple Silicon
is_apple_silicon() {
    [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]
}

# Check if CUDA is available
has_cuda() {
    python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

# Check if MLX is importable
has_mlx() {
    python3 -c "import mlx" 2>/dev/null
}

# Elapsed time helper
_timer_start=0
timer_start() {
    _timer_start=$(python3 -c "import time; print(time.time())")
}

timer_elapsed() {
    local now
    now=$(python3 -c "import time; print(time.time())")
    python3 -c "print(f'{$now - $_timer_start:.2f}s')"
}

timer_elapsed_ms() {
    local now
    now=$(python3 -c "import time; print(time.time())")
    python3 -c "print(f'{($now - $_timer_start)*1000:.0f}')"
}
