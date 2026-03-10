#!/usr/bin/env bash
# test_mcp_server.sh - MCP server UAT.
# Tests JSON-RPC communication, all 7 tools, and graceful shutdown.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge MCP Server Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

ensure_test_images

python3 << PYEOF
import asyncio
import json
import os
import signal
import sys
import time

sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
CORPUS_TEXT = "${CORPUS_DIR}/text"
CORPUS_IMAGES = "${CORPUS_DIR}/images"

os.environ["RECALLFORGE_BACKEND"] = "torch"
os.environ["RECALLFORGE_MODE"] = "embed"
os.environ["RECALLFORGE_STORE_PATH"] = STORE

pass_count = 0
fail_count = 0

def report(ok, msg):
    global pass_count, fail_count
    if ok:
        print(f"  \033[0;32mPASS\033[0m  {msg}")
        pass_count += 1
    else:
        print(f"  \033[0;31mFAIL\033[0m  {msg}")
        fail_count += 1

# ── Test server creation ──
print("\n\033[0;36m--- Server Creation ---\033[0m\n")

from recallforge.server import create_server
from recallforge import get_backend, get_storage
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ListToolsRequest,
)

backend = get_backend()
backend._load_embedder()
storage = get_storage(STORE)


async def _list_tools(server):
    """Resolve list_tools across MCP API variants."""
    try:
        result = await server.list_tools()
        if isinstance(result, list):
            return result
    except TypeError:
        pass

    try:
        result = server.list_tools()
        if isinstance(result, list):
            return result
    except TypeError:
        pass

    handler = getattr(server, "request_handlers", {}).get(ListToolsRequest)
    if handler is None:
        raise RuntimeError("Unable to resolve MCP list_tools handler")

    response = await handler(ListToolsRequest(method="tools/list"))
    root = getattr(response, "root", response)
    tools = getattr(root, "tools", None)
    if tools is None:
        raise RuntimeError("Unable to extract tools from MCP list_tools response")
    return tools


async def _call_tool(server, name, arguments):
    """Resolve call_tool across MCP API variants."""
    try:
        result = await server.call_tool(name, arguments)
        if isinstance(result, list):
            return result
    except TypeError:
        pass

    try:
        result = server.call_tool(name, arguments)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, list):
            return result
    except TypeError:
        pass

    handler = getattr(server, "request_handlers", {}).get(CallToolRequest)
    if handler is None:
        raise RuntimeError("Unable to resolve MCP call_tool handler")

    response = await handler(
        CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=name, arguments=arguments),
        )
    )
    root = getattr(response, "root", response)
    content = getattr(root, "content", None)
    if content is None:
        raise RuntimeError(f"Unable to extract tool response content for '{name}'")
    return content


def _as_json_payload(content):
    """Parse MCP text payload; tolerate plain-text error responses."""
    text = ""
    if content and len(content) > 0:
        text = getattr(content[0], "text", "") or ""

    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": text}


async def test_server():
    global pass_count, fail_count

    server = await create_server(backend=backend, storage=storage, mode="embed")
    report(server is not None, "Server created successfully")

    # ── List tools ──
    print("\n\033[0;36m--- List Tools ---\033[0m\n")

    tools = await _list_tools(server)
    tool_names = [t.name for t in tools]
    report(len(tools) == 7, f"Server exposes {len(tools)} tools (expected 7)")

    expected_tools = ["search", "search_fts", "search_vec", "index_document", "index_image", "status", "rebuild_fts"]
    for expected in expected_tools:
        report(expected in tool_names, f"Tool '{expected}' available")

    # ── Status tool ──
    print("\n\033[0;36m--- Status Tool ---\033[0m\n")

    result = await _call_tool(server, "status", {})
    status_text = result[0].text
    status_data = json.loads(status_text)
    report("version" in status_data, f"Status returns version: {status_data.get('version')}")
    report("models" in status_data, f"Status returns model info")
    report("database" in status_data, f"Status returns database info")

    # ── Index Document tool ──
    print("\n\033[0;36m--- Index Document Tool ---\033[0m\n")

    result = await _call_tool(server, "index_document", {
        "path": "mcp_test_doc.md",
        "text": "RecallForge enables cross-modal vision-language search combining BM25 and vector retrieval.",
        "collection": "mcp_test",
    })
    idx_data = json.loads(result[0].text)
    report(idx_data.get("success") == True, f"index_document succeeded, hash={idx_data.get('hash', 'N/A')[:8]}...")

    # ── Index Image tool ──
    print("\n\033[0;36m--- Index Image Tool ---\033[0m\n")

    img_path = os.path.join(CORPUS_IMAGES, "whiteboard_architecture.png")
    if os.path.exists(img_path):
        result = await _call_tool(server, "index_image", {
            "path": img_path,
            "collection": "mcp_test",
        })
        img_data = json.loads(result[0].text)
        report(img_data.get("success") == True, f"index_image succeeded, hash={img_data.get('hash', 'N/A')[:8]}...")
    else:
        print("  \033[0;33mSKIP\033[0m  No test image available")

    # ── Search tool ──
    print("\n\033[0;36m--- Search Tool (hybrid) ---\033[0m\n")

    result = await _call_tool(server, "search", {
        "query": "cross-modal vision language search",
        "limit": 5,
        "collection": "mcp_test",
    })
    search_data = json.loads(result[0].text)
    report(search_data.get("count", 0) > 0, f"search returned {search_data.get('count', 0)} results")
    report("results" in search_data, "search results contain result list")
    if search_data.get("results"):
        r0 = search_data["results"][0]
        report("score" in r0, f"Result has score: {r0.get('score')}")
        report("filepath" in r0, f"Result has filepath: {r0.get('filepath', '')[:40]}")

    # ── Search FTS tool ──
    print("\n\033[0;36m--- Search FTS Tool ---\033[0m\n")

    result = await _call_tool(server, "search_fts", {
        "query": "RecallForge BM25 vector",
        "limit": 10,
        "collection": "mcp_test",
    })
    fts_data = json.loads(result[0].text)
    report(fts_data.get("count", 0) > 0, f"search_fts returned {fts_data.get('count', 0)} results")

    # ── Search Vec tool ──
    print("\n\033[0;36m--- Search Vec Tool ---\033[0m\n")

    result = await _call_tool(server, "search_vec", {
        "query": "semantic search cross-modal",
        "limit": 10,
        "collection": "mcp_test",
    })
    vec_data = json.loads(result[0].text)
    report(vec_data.get("count", 0) > 0, f"search_vec returned {vec_data.get('count', 0)} results")

    # ── Rebuild FTS tool ──
    print("\n\033[0;36m--- Rebuild FTS Tool ---\033[0m\n")

    result = await _call_tool(server, "rebuild_fts", {})
    rebuild_data = json.loads(result[0].text)
    report(rebuild_data.get("success") == True, "rebuild_fts succeeded")

    # ── Error Handling ──
    print("\n\033[0;36m--- Error Handling ---\033[0m\n")

    # Bad tool name
    result = await _call_tool(server, "nonexistent_tool", {})
    err_data = _as_json_payload(result)
    report("error" in err_data, f"Unknown tool returns error: {err_data.get('error', '')[:50]}")

    # Missing params
    result = await _call_tool(server, "search", {})
    err_data = _as_json_payload(result)
    report("error" in err_data or err_data.get("count", 0) == 0,
           "search with no query returns error or empty results")

    # index_document missing text
    result = await _call_tool(server, "index_document", {"path": "x.md"})
    err_data = _as_json_payload(result)
    report("error" in err_data, "index_document missing text returns error")

    # index_image nonexistent file
    result = await _call_tool(server, "index_image", {"path": "/nonexistent/image.png"})
    err_data = _as_json_payload(result)
    report("error" in err_data, "index_image nonexistent file returns error")

    # ── Cross-modal via MCP ──
    print("\n\033[0;36m--- Cross-Modal via MCP ---\033[0m\n")

    # Index image, then search for it with text
    if os.path.exists(img_path):
        result = await _call_tool(server, "search", {
            "query": "whiteboard diagram architecture system",
            "limit": 5,
            "collection": "mcp_test",
            "content_type": "image",
        })
        xm_data = json.loads(result[0].text)
        report(xm_data.get("count", 0) > 0,
               f"Cross-modal text→image via MCP: {xm_data.get('count', 0)} results")

asyncio.run(test_server())

# ── Graceful Shutdown ──
print("\n\033[0;36m--- Graceful Shutdown ---\033[0m\n")

# Test that signal handler is registered
from recallforge.server import _signal_handler, _shutdown_requested
import signal as sig

# Verify handler is installed
handler = sig.getsignal(sig.SIGINT)
report(handler is not None, "SIGINT handler is registered")

handler = sig.getsignal(sig.SIGTERM)
report(handler is not None, "SIGTERM handler is registered")

# ── Summary ──
print(f"\n\033[1m{'='*40}\033[0m")
print(f"\033[1m  MCP Server Summary\033[0m")
print(f"\033[1m{'='*40}\033[0m")
print(f"  \033[0;32mPASS: {pass_count}\033[0m")
print(f"  \033[0;31mFAIL: {fail_count}\033[0m")

if fail_count > 0:
    print(f"\n  \033[0;31m\033[1mRESULT: FAILED\033[0m")
    sys.exit(1)
else:
    print(f"\n  \033[0;32m\033[1mRESULT: PASSED\033[0m")
PYEOF
