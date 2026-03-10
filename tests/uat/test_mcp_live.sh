#!/usr/bin/env bash
# test_mcp_live.sh - Live MCP UAT (real backend/models)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UAT_MCP_LIVE=1 bash "${SCRIPT_DIR}/test_mcp_server.sh" "$@"
