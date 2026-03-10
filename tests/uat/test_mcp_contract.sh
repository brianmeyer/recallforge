#!/usr/bin/env bash
# test_mcp_contract.sh - Fast MCP contract/UAT (mock backend)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UAT_MCP_LIVE=0 bash "${SCRIPT_DIR}/test_mcp_server.sh" "$@"
