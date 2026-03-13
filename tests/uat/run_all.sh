#!/usr/bin/env bash
# run_all.sh - Run the complete RecallForge UAT suite.
#
# Usage:
#   ./tests/uat/run_all.sh           # Run all tests
#   ./tests/uat/run_all.sh --quick   # Skip latency test
#
# Results are logged to tests/uat/uat_results.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

LOG_FILE="${UAT_DIR}/uat_results.log"
QUICK_MODE=false
[[ "${1:-}" == "--quick" ]] && QUICK_MODE=true

echo ""
echo -e "${BOLD}${BLUE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║     RecallForge UAT Suite - Full Run          ║${NC}"
echo -e "${BOLD}${BLUE}║     $(date '+%Y-%m-%d %H:%M:%S %Z')                 ║${NC}"
echo -e "${BOLD}${BLUE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# Generate test images first
ensure_test_images

# Test execution order (dependencies first)
TESTS=(
    "test_install.sh:Install"
    "test_storage.sh:Storage"
    "test_backends.sh:Backends"
    "test_tiered_modes.sh:Tiered Modes"
    "test_document_ingest.sh:Document Ingest"
    "test_video_ingest.sh:Video Ingest"
    "test_video_quality.sh:Video Quality"
    "test_cross_modal.sh:Cross-Modal ★"
    "test_search_quality.sh:Search Quality"
    "test_mcp_server.sh:MCP Server"
    "test_cli.sh:CLI"
    "test_benchmarks.sh:Benchmark Smoke"
)

if [[ "$QUICK_MODE" != "true" ]]; then
    TESTS+=("test_latency.sh:Latency")
fi

declare -A RESULTS
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0

# Header for log
{
    echo "RecallForge UAT Results"
    echo "======================"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "Host: $(uname -n) ($(uname -m))"
    echo "Python: $(python3 --version 2>&1)"
    echo ""
} > "$LOG_FILE"

TOTAL_TESTS=${#TESTS[@]}
CURRENT_TEST=0

for test_entry in "${TESTS[@]}"; do
    IFS=':' read -r test_script test_name <<< "$test_entry"
    CURRENT_TEST=$((CURRENT_TEST + 1))

    echo -e "${BOLD}${CYAN}▶ [${CURRENT_TEST}/${TOTAL_TESTS}] Running: ${test_name}${NC}"
    echo "" >> "$LOG_FILE"
    echo "═══════════════════════════════════════" >> "$LOG_FILE"
    echo "  ${test_name}" >> "$LOG_FILE"
    echo "═══════════════════════════════════════" >> "$LOG_FILE"

    START_TIME=$(python3 -c "import time; print(time.time())")

    if bash "${UAT_DIR}/${test_script}" 2>&1 | tee -a "$LOG_FILE"; then
        RESULTS[$test_name]="PASS"
        echo -e "  ${GREEN}✓ ${test_name}: PASSED${NC}"
    else
        RESULTS[$test_name]="FAIL"
        echo -e "  ${RED}✗ ${test_name}: FAILED${NC}"
    fi

    END_TIME=$(python3 -c "import time; print(time.time())")
    ELAPSED=$(python3 -c "print(f'{$END_TIME - $START_TIME:.1f}s')")
    echo -e "  ${CYAN}Duration: ${ELAPSED}${NC}"
    echo "" >> "$LOG_FILE"
done

# ──────────────────────────────────
# Summary
# ──────────────────────────────────

echo ""
echo -e "${BOLD}${BLUE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║            UAT Suite Summary                  ║${NC}"
echo -e "${BOLD}${BLUE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

SUITE_PASS=0
SUITE_FAIL=0

for test_entry in "${TESTS[@]}"; do
    IFS=':' read -r test_script test_name <<< "$test_entry"
    result="${RESULTS[$test_name]:-UNKNOWN}"

    if [[ "$result" == "PASS" ]]; then
        echo -e "  ${GREEN}✓${NC}  ${test_name}"
        SUITE_PASS=$((SUITE_PASS + 1))
    else
        echo -e "  ${RED}✗${NC}  ${test_name}"
        SUITE_FAIL=$((SUITE_FAIL + 1))
    fi
done

echo ""
echo -e "  Passed: ${GREEN}${SUITE_PASS}${NC}  |  Failed: ${RED}${SUITE_FAIL}${NC}  |  Total: $((SUITE_PASS + SUITE_FAIL))"
echo ""

{
    echo ""
    echo "═══════════════════════════════════════"
    echo "  UAT SUITE SUMMARY"
    echo "═══════════════════════════════════════"
    echo "  Passed: ${SUITE_PASS}"
    echo "  Failed: ${SUITE_FAIL}"
    echo "  Total:  $((SUITE_PASS + SUITE_FAIL))"
} >> "$LOG_FILE"

if [[ $SUITE_FAIL -gt 0 ]]; then
    echo -e "  ${RED}${BOLD}SUITE RESULT: FAILED${NC}"
    echo "  SUITE RESULT: FAILED" >> "$LOG_FILE"
    echo ""
    echo -e "  Log: ${LOG_FILE}"
    exit 1
else
    echo -e "  ${GREEN}${BOLD}SUITE RESULT: ALL PASSED${NC}"
    echo "  SUITE RESULT: ALL PASSED" >> "$LOG_FILE"
    echo ""
    echo -e "  Log: ${LOG_FILE}"
    exit 0
fi
