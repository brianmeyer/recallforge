"""
server.py - MCP Server for RecallForge.

MCP protocol server with stdio transport.
Tools: search, search_fts, search_vec, ingest, index_document, index_image,
memory_add, memory_update, memory_delete, status, rebuild_fts

Calls backend.warm_up() on server start for predictable latency.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Optional, Callable, TypeVar

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from . import __version__, get_backend, get_storage, warmup_backend
from .search import HybridSearcher


# Configure logging
logger = logging.getLogger("recallforge.server")

# Enable trace mode via environment variable
TRACE_ENABLED = os.environ.get("RECALLFORGE_TRACE", "0") == "1"


def trace_log(operation: str, **kwargs) -> None:
    """Structured trace logging for debugging MCP tools."""
    if TRACE_ENABLED:
        logger.debug(f"[TRACE] {operation}: {kwargs}")


# Global server state
_server: Optional[Server] = None
_shutdown_requested = False
_tool_semaphore: Optional[asyncio.Semaphore] = None
_http_start_time: Optional[float] = None

_MAX_TOOL_CONCURRENCY = max(
    1,
    int(os.environ.get("RECALLFORGE_MCP_MAX_CONCURRENCY", "2")),
)

_T = TypeVar("_T")


def _get_tool_semaphore() -> asyncio.Semaphore:
    """Lazily create a shared semaphore for blocking tool work."""
    global _tool_semaphore
    if _tool_semaphore is None:
        _tool_semaphore = asyncio.Semaphore(_MAX_TOOL_CONCURRENCY)
    return _tool_semaphore


async def _run_blocking(func: Callable[..., _T], *args, **kwargs) -> _T:
    """
    Run blocking backend/storage work in a bounded thread pool lane.

    Prevents event-loop stalls while also capping parallel pressure on
    local model/runtime resources.
    """
    async with _get_tool_semaphore():
        return await asyncio.to_thread(func, *args, **kwargs)


def _resolve_query_inputs(arguments: dict) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Validate mutually exclusive text/image/video query inputs."""
    query = arguments.get("query")
    image_path = arguments.get("image_path")
    video_path = arguments.get("video_path")
    query_text = query.strip() if isinstance(query, str) else ""
    image_text = image_path.strip() if isinstance(image_path, str) else ""
    video_text = video_path.strip() if isinstance(video_path, str) else ""

    if sum((bool(query_text), bool(image_text), bool(video_text))) != 1:
        return None, None, None, "Provide exactly one of: query, image_path, or video_path"
    return (query_text or None), (image_text or None), (video_text or None), None


def _error_response(code: str, message: str, details: dict = None) -> list:
    """Return a structured MCP error response as a list[TextContent].

    Args:
        code:    One of INVALID_INPUT | NOT_FOUND | BACKEND_ERROR | INTERNAL_ERROR
        message: Human-readable description.
        details: Optional extra context dict (defaults to empty dict).

    Returns:
        ``[TextContent(type="text", text=<json>)]`` so callers can return it directly.
    """
    payload = {
        "error": True,
        "code": code,
        "message": message,
        "details": details if details is not None else {},
    }
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    print(f"Received signal {signum}, initiating graceful shutdown...", file=sys.stderr)
    _shutdown_requested = True
    # Exit immediately so SIGINT/SIGTERM interrupts synchronous warm-up as well.
    raise SystemExit(128 + signum)


# Register signal handlers
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


async def create_server(
    backend=None,
    storage=None,
    mode: str = None,
) -> Server:
    """Create and configure the MCP server."""
    server = Server("recallforge")
    
    # Get backend and storage if not provided
    if backend is None:
        backend = get_backend()
    if storage is None:
        store_path = os.environ.get("RECALLFORGE_STORE_PATH")
        storage = get_storage(store_path)
    
    # Set mode if specified
    if mode:
        backend.set_mode(mode)

    # Mutable runtime config — safe to change without restart
    _mutable_config: dict = {
        "mode": mode or os.environ.get("RECALLFORGE_MODE", "hybrid"),
        "collection": "default",
        "max_file_size_mb": 100,
        "rerank_top_k": int(os.environ.get("RECALLFORGE_RERANK_TOP_K", "20")),
        "caption_media": True,
    }
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="search",
                description="Full hybrid search combining BM25, vector search, and reranking (hybrid mode). Optional query expansion generates semantic variants for improved cross-modal retrieval.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "image_path": {"type": "string", "description": "Optional image query path (mutually exclusive with query)"},
                        "video_path": {"type": "string", "description": "Optional video query path (mutually exclusive with query/image_path)"},
                        "limit": {"type": "integer", "description": "Maximum results to return", "default": 10},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image", "video"], "description": "Optional content type filter"},
                        "user_id": {"type": "string", "description": "Optional user namespace filter for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                        "intent": {"type": "string", "enum": ["exact_lookup", "semantic", "broad"], "description": "Optional intent for query steering: exact_lookup (boost BM25), semantic (boost vector), broad (equal weights)"},
                        "rerank_top_k": {"type": "integer", "description": "Maximum number of top RRF candidates to rerank", "default": 20, "minimum": 0},
                        "expand": {"type": "boolean", "description": "Enable VL-aware query expansion. When true, generates semantic variants for text queries and visual descriptions for image-seeking queries. Each variant runs as a separate retrieval branch in RRF. Default: false (opt-in)", "default": False},
                    },
                },
            ),
            Tool(
                name="search_fts",
                description="Full-text search (BM25) using Tantivy",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Maximum results to return", "default": 20},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image", "video"], "description": "Optional content type filter"},
                        "user_id": {"type": "string", "description": "Optional user namespace filter for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                    },
                },
            ),
            Tool(
                name="search_vec",
                description="Vector search using embeddings and ANN",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "image_path": {"type": "string", "description": "Optional image query path (mutually exclusive with query)"},
                        "video_path": {"type": "string", "description": "Optional video query path (mutually exclusive with query/image_path)"},
                        "limit": {"type": "integer", "description": "Maximum results to return", "default": 20},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image", "video"], "description": "Optional content type filter"},
                        "user_id": {"type": "string", "description": "Optional user namespace filter for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                    },
                },
            ),
            Tool(
                name="ingest",
                description="Unified ingest for text, image, video, document, file, or folder. Auto-detects modality and routes accordingly.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Raw text content to ingest"},
                        "path": {"type": "string", "description": "Optional memory path for raw text ingest"},
                        "file_path": {"type": "string", "description": "Path to a single file (text, image, video, or office document)"},
                        "folder_path": {"type": "string", "description": "Path to a folder to ingest"},
                        "recursive": {"type": "boolean", "description": "Recursively ingest subfolders", "default": True},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                        "content_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["text", "image", "video", "document"]},
                            "description": "Allowed content types to ingest",
                            "default": ["text", "image", "video", "document"]
                        },
                        "include_globs": {"type": "array", "items": {"type": "string"}, "description": "Include globs relative to folder root"},
                        "exclude_globs": {"type": "array", "items": {"type": "string"}, "description": "Exclude globs relative to folder root"},
                        "max_file_size_mb": {"type": "integer", "description": "Maximum file size in MB (files exceeding this will be skipped)", "default": 100},
                        "caption_media": {"type": "boolean", "description": "Generate captions for image/video content during ingest", "default": True},
                        "user_id": {"type": "string", "description": "Optional user namespace for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace"},
                        "project_id": {"type": "string", "description": "Optional project namespace"},
                        "profile": {"type": "string", "description": "Optional profile namespace"},
                    },
                },
            ),
            Tool(
                name="index_document",
                description="Index a text document for search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path within collection"},
                        "text": {"type": "string", "description": "Document text content"},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                    },
                    "required": ["path", "text"],
                },
            ),
            Tool(
                name="index_image",
                description="Index an image file for cross-modal search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute path to image file"},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="memory_add",
                description="Add a text memory entry (or replace if the same path already exists)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Memory path key within collection"},
                        "text": {"type": "string", "description": "Memory content"},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                        "user_id": {"type": "string", "description": "Optional user namespace for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace"},
                        "project_id": {"type": "string", "description": "Optional project namespace"},
                        "profile": {"type": "string", "description": "Optional profile namespace"},
                        "importance": {"type": "number", "description": "Importance score 0.0-1.0 (optional)", "minimum": 0, "maximum": 1},
                        "ttl_seconds": {"type": "integer", "description": "Time-to-live in seconds, 0 or null = no expiration (optional)"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "List of string tags (optional)"},
                    },
                    "required": ["path", "text"],
                },
            ),
            Tool(
                name="memory_update",
                description="Update a text memory entry at path, replacing old vectors",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Memory path key within collection"},
                        "text": {"type": "string", "description": "Updated memory content"},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                        "user_id": {"type": "string", "description": "Optional user namespace for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace"},
                        "project_id": {"type": "string", "description": "Optional project namespace"},
                        "profile": {"type": "string", "description": "Optional profile namespace"},
                        "importance": {"type": "number", "description": "Importance score 0.0-1.0 (optional)", "minimum": 0, "maximum": 1},
                        "ttl_seconds": {"type": "integer", "description": "Time-to-live in seconds, 0 or null = no expiration (optional)"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "List of string tags (optional)"},
                    },
                    "required": ["path", "text"],
                },
            ),
            Tool(
                name="memory_delete",
                description="Delete/deactivate a memory entry and remove associated embeddings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Memory path key within collection"},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                        "user_id": {"type": "string", "description": "Optional user namespace for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace"},
                        "project_id": {"type": "string", "description": "Optional project namespace"},
                        "profile": {"type": "string", "description": "Optional profile namespace"},
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="status",
                description="Get server status including model loading and database info",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="rebuild_fts",
                description="Rebuild the full-text search (Tantivy) index",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="list_collections",
                description="List unique collection names in the store, with optional namespace filters",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "Optional user namespace filter"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                    },
                },
            ),
            Tool(
                name="list_namespaces",
                description="List unique namespace combinations (user_id, session_id, project_id, profile) in the store",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string", "description": "Optional collection filter"},
                    },
                },
            ),
            Tool(
                name="rename_collection",
                description="Rename a collection atomically (updates all documents and embeddings)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "old_name": {"type": "string", "description": "Current collection name"},
                        "new_name": {"type": "string", "description": "New collection name"},
                    },
                    "required": ["old_name", "new_name"],
                },
            ),
            Tool(
                name="delete_collection",
                description="Delete all data for a collection (documents, embeddings, and orphaned content)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Collection name to delete"},
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="batch",
                description="Execute multiple RecallForge operations in a single call",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "operations": {
                            "type": "array",
                            "description": "List of operations to execute (max 20)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": {"type": "string", "description": "Tool name to invoke"},
                                    "arguments": {"type": "object", "description": "Arguments for the tool"},
                                },
                                "required": ["tool", "arguments"],
                            },
                        },
                    },
                    "required": ["operations"],
                },
            ),
            Tool(
                name="search_batch",
                description="Run multiple search queries in parallel and merge results using weighted RRF fusion. Each query runs independently, then results are deduplicated and scored by fused RRF rank.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "description": "List of queries (strings or objects with query/mode/intent/weight)",
                            "items": {
                                "oneOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string", "description": "Search query text"},
                                            "mode": {"type": "string", "enum": ["hybrid", "fts", "vec"], "description": "Search mode (default: hybrid)"},
                                            "intent": {"type": "string", "enum": ["exact_lookup", "semantic", "broad"], "description": "Optional intent steering"},
                                            "weight": {"type": "number", "description": "Weight for RRF merging (default: 1.0)", "minimum": 0},
                                        },
                                        "required": ["query"],
                                    },
                                ],
                            },
                            "minItems": 1,
                            "maxItems": 20,
                        },
                        "limit": {"type": "integer", "description": "Maximum final results to return", "default": 10},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image", "video"], "description": "Optional content type filter"},
                        "user_id": {"type": "string", "description": "Optional user namespace filter"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                    },
                    "required": ["queries"],
                },
            ),
            Tool(
                name="explain_results",
                description="Explain WHY each search result was returned and ranked. Returns detailed provenance including RRF source ranks, reranker scores, blend weights, and media compensation for each result.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "image_path": {"type": "string", "description": "Optional image query path (mutually exclusive with query)"},
                        "video_path": {"type": "string", "description": "Optional video query path (mutually exclusive with query/image_path)"},
                        "limit": {"type": "integer", "description": "Maximum results to return", "default": 10},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image", "video"], "description": "Optional content type filter"},
                        "user_id": {"type": "string", "description": "Optional user namespace filter for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                        "intent": {"type": "string", "enum": ["exact_lookup", "semantic", "broad"], "description": "Optional intent for query steering: exact_lookup (boost BM25), semantic (boost vector), broad (equal weights)"},
                        "rerank_top_k": {"type": "integer", "description": "Maximum number of top RRF candidates to rerank", "default": 20, "minimum": 0},
                        "expand": {"type": "boolean", "description": "Enable VL-aware query expansion", "default": False},
                    },
                },
            ),
            Tool(
                name="get_config",
                description="Get current server configuration including version, backend, mode, quantization, data directory, default collection, and max file size",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="set_config",
                description="Update safe runtime configuration values. Allows changing mode (embed/hybrid), collection, and model IDs. Does NOT allow changing backend, quantize, or data_dir (those require a server restart).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["embed", "hybrid"],
                            "description": "Search mode: embed (vector only), hybrid (vector + rerank)",
                        },
                        "collection": {
                            "type": "string",
                            "description": "Default collection name used when none is specified",
                        },
                        "max_file_size_mb": {
                            "type": "number",
                            "description": "Maximum file size in MB for ingest operations",
                            "minimum": 1,
                        },
                        "rerank_top_k": {
                            "type": "number",
                            "description": "Maximum number of top RRF candidates to rerank in search (0 disables reranking)",
                            "minimum": 0,
                        },
                        "caption_media": {
                            "type": "boolean",
                            "description": "Enable ingest-time image/video caption generation for BM25 indexing",
                        },
                        "embedder_model": {
                            "type": "string",
                            "description": "HuggingFace model ID for the embedding model (changing unloads cached model)",
                        },
                        "reranker_model": {
                            "type": "string",
                            "description": "HuggingFace model ID for the reranker model (changing unloads cached model)",
                        },
                        "captioner_model": {
                            "type": "string",
                            "description": "HuggingFace model ID for the captioning model (changing unloads cached model)",
                        },
                    },
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Execute a tool."""
        try:
            if name == "batch":
                return await _handle_batch(arguments, backend, storage, _mutable_config)
            return await _dispatch_tool(name, arguments, backend, storage, _mutable_config)
        except Exception as e:
            return _error_response("INTERNAL_ERROR", str(e), {"exception_type": type(e).__name__})
    
    return server


_MAX_BATCH_SIZE = 20


async def _dispatch_tool(
    name: str,
    arguments: dict,
    backend,
    storage,
    mutable_config: Optional[dict] = None,
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Route a single tool call to the appropriate handler."""
    # Apply mutable config defaults: if a handler expects collection/max_file_size_mb
    # and the caller didn't explicitly provide them, use the mutable config values.
    if mutable_config:
        if "collection" not in arguments and "collection" in mutable_config:
            arguments.setdefault("collection", mutable_config["collection"])
        if "max_file_size_mb" not in arguments and "max_file_size_mb" in mutable_config:
            arguments.setdefault("max_file_size_mb", mutable_config["max_file_size_mb"])
        if "rerank_top_k" not in arguments and "rerank_top_k" in mutable_config:
            arguments.setdefault("rerank_top_k", mutable_config["rerank_top_k"])
        if "caption_media" not in arguments and "caption_media" in mutable_config:
            arguments.setdefault("caption_media", mutable_config["caption_media"])
    if name == "search":
        return await _handle_search(arguments, backend, storage)
    elif name == "explain_results":
        return await _handle_explain_results(arguments, backend, storage)
    elif name == "search_fts":
        return await _handle_search_fts(arguments, storage)
    elif name == "search_vec":
        return await _handle_search_vec(arguments, backend, storage)
    elif name == "ingest":
        return await _handle_ingest(arguments, backend, storage)
    elif name == "index_document":
        return await _handle_index_document(arguments, backend, storage)
    elif name == "index_image":
        return await _handle_index_image(arguments, backend, storage)
    elif name == "memory_add":
        return await _handle_memory_add(arguments, backend, storage)
    elif name == "memory_update":
        return await _handle_memory_update(arguments, backend, storage)
    elif name == "memory_delete":
        return await _handle_memory_delete(arguments, storage)
    elif name == "status":
        return await _handle_status(backend, storage)
    elif name == "rebuild_fts":
        return await _handle_rebuild_fts(storage)
    elif name == "list_collections":
        return await _handle_list_collections(arguments, storage)
    elif name == "list_namespaces":
        return await _handle_list_namespaces(arguments, storage)
    elif name == "rename_collection":
        return await _handle_rename_collection(arguments, storage)
    elif name == "delete_collection":
        return await _handle_delete_collection(arguments, storage)
    elif name == "get_config":
        return await _handle_get_config(backend, storage, mutable_config or {})
    elif name == "set_config":
        return await _handle_set_config(arguments, backend, storage, mutable_config if mutable_config is not None else {})
    elif name == "search_batch":
        return await _handle_search_batch(arguments, backend, storage)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _handle_batch(
    arguments: dict,
    backend,
    storage,
    mutable_config: Optional[dict] = None,
) -> list[TextContent]:
    """Execute multiple RecallForge operations in a single call."""
    operations = arguments.get("operations")
    if not isinstance(operations, list):
        return [TextContent(type="text", text=json.dumps({"error": "operations must be a list"}))]

    if len(operations) > _MAX_BATCH_SIZE:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": f"Batch size {len(operations)} exceeds maximum of {_MAX_BATCH_SIZE}"}
                ),
            )
        ]

    batch_results = []
    succeeded = 0
    failed = 0

    for i, op in enumerate(operations):
        tool_name = op.get("tool", "")
        op_args = op.get("arguments", {})

        # Reject nested batch calls
        if tool_name == "batch":
            batch_results.append({
                "index": i,
                "tool": tool_name,
                "status": "error",
                "result": {"error": "Nested batch operations are not allowed"},
            })
            failed += 1
            continue

        try:
            content_list = await _dispatch_tool(tool_name, op_args, backend, storage, mutable_config)
            # Unwrap the first TextContent result into a parsed dict when possible
            if content_list and hasattr(content_list[0], "text"):
                try:
                    result_payload = json.loads(content_list[0].text)
                except (json.JSONDecodeError, AttributeError):
                    result_payload = content_list[0].text
            else:
                result_payload = None
            batch_results.append({
                "index": i,
                "tool": tool_name,
                "status": "success",
                "result": result_payload,
            })
            succeeded += 1
        except Exception as exc:
            batch_results.append({
                "index": i,
                "tool": tool_name,
                "status": "error",
                "result": {"error": str(exc)},
            })
            failed += 1

    output = {
        "batch_results": batch_results,
        "total": len(operations),
        "succeeded": succeeded,
        "failed": failed,
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle hybrid search."""
    query, image_path, video_path, input_error = _resolve_query_inputs(arguments)
    limit = arguments.get("limit", 10)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")
    intent = arguments.get("intent")
    rerank_top_k = arguments.get("rerank_top_k", 20)
    expand = arguments.get("expand", False)

    trace_log("search_start", query=(query or image_path or video_path or "")[:50], limit=limit, collection=collection, content_type=content_type,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile, intent=intent, rerank_top_k=rerank_top_k, expand=expand)

    if input_error:
        return _error_response("INVALID_INPUT", input_error)

    searcher = HybridSearcher(
        backend=backend,
        storage=storage,
        limit=limit,
        collection=collection,
        content_type=content_type,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        intent=intent,
        rerank_top_k=rerank_top_k,
        expand=expand,
    )

    if image_path:
        results = await _run_blocking(searcher.search_image, image_path)
    elif video_path:
        results = await _run_blocking(searcher.search_video, video_path)
    else:
        results = await _run_blocking(searcher.search, query)

    trace_log("search_done", query=(query or image_path or video_path or "")[:50], count=len(results))

    output = {
        "query": query,
        "image_path": image_path,
        "video_path": video_path,
        "mode": backend.get_mode(),
        "count": len(results),
        "results": [
            {
                "filepath": r.filepath,
                "title": r.title,
                "score": round(r.score, 4),
                "rerank_score": round(r.rerank_score, 4),
                "rrf_rank": r.rrf_rank,
                "source": r.source,
                "snippet": (r.body or "")[:500] if r.body else None,
                "user_id": getattr(r, "user_id", None),
                "session_id": getattr(r, "session_id", None),
                "project_id": getattr(r, "project_id", None),
                "profile": getattr(r, "profile", None),
            }
            for r in results
        ],
    }

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_explain_results(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle explain_results - returns detailed scoring provenance for each result."""
    query, image_path, video_path, input_error = _resolve_query_inputs(arguments)
    limit = arguments.get("limit", 10)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")
    intent = arguments.get("intent")
    rerank_top_k = arguments.get("rerank_top_k", 20)
    expand = arguments.get("expand", False)

    trace_log("explain_results_start", query=(query or image_path or video_path or "")[:50], limit=limit, collection=collection, content_type=content_type,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile, intent=intent, rerank_top_k=rerank_top_k, expand=expand)

    if input_error:
        return _error_response("INVALID_INPUT", input_error)

    searcher = HybridSearcher(
        backend=backend,
        storage=storage,
        limit=limit,
        collection=collection,
        content_type=content_type,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        intent=intent,
        rerank_top_k=rerank_top_k,
        expand=expand,
    )

    if image_path:
        results = await _run_blocking(searcher.search_image, image_path)
    elif video_path:
        results = await _run_blocking(searcher.search_video, video_path)
    else:
        results = await _run_blocking(searcher.search, query)

    trace_log("explain_results_done", query=(query or image_path or video_path or "")[:50], count=len(results))

    # Build detailed explanation for each result
    explained_results = []
    for r in results:
        explanation = {
            "filepath": r.filepath,
            "title": r.title,
            "final_score": round(r.score, 4),
            "content_type": r.content_type if hasattr(r, 'content_type') else "text",
            "source": r.source,
        }
        
        if r.audit:
            explanation["provenance"] = {
                "rrf": {
                    "sources": r.audit.rrf_sources,  # {source_name: rank}
                    "rrf_score": round(r.audit.rrf_score, 6),
                    "media_compensation_applied": r.audit.media_compensation_applied,
                },
                "reranker": {
                    "raw_score": round(r.audit.reranker_raw_score, 6),
                    "normalized_score": round(r.audit.reranker_normalized_score, 6),
                    "scoring_path": r.audit.reranker_scoring_path,  # text, vl_image, vl_video, etc.
                },
                "blend": {
                    "weights": r.audit.blend_weights,  # {"rrf": 0.75, "rerank": 0.25}
                    "final_blended_score": round(r.audit.final_blended_score, 6),
                },
            }
        else:
            explanation["provenance"] = {
                "note": "Audit trail not available - result may be from vector-only path"
            }
        
        explanation["rrf_rank"] = r.rrf_rank
        explanation["rerank_score"] = round(r.rerank_score, 4)
        explanation["snippet"] = (r.body or "")[:500] if r.body else None
        
        explained_results.append(explanation)

    output = {
        "query": query,
        "image_path": image_path,
        "video_path": video_path,
        "mode": backend.get_mode(),
        "count": len(results),
        "results": explained_results,
    }

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search_fts(arguments: dict, storage) -> list[TextContent]:
    """Handle FTS search."""
    query = arguments.get("query", "")
    limit = arguments.get("limit", 20)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    trace_log("search_fts_start", query=query[:50], limit=limit, collection=collection, content_type=content_type,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

    if not query:
        return _error_response("INVALID_INPUT", "Query is required")

    results = await _run_blocking(
        storage.search_fts,
        query=query,
        limit=limit,
        collection=collection,
        content_type=content_type,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
    )

    trace_log("search_fts_done", query=query[:50], count=len(results))

    output = {
        "query": query,
        "count": len(results),
        "results": [
            {
                "filepath": r.filepath,
                "title": r.title,
                "score": round(r.score, 4),
                "source": r.source,
                "user_id": r.user_id,
                "session_id": r.session_id,
                "project_id": r.project_id,
                "profile": r.profile,
            }
            for r in results
        ],
    }

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search_vec(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle vector search."""
    query, image_path, video_path, input_error = _resolve_query_inputs(arguments)
    limit = arguments.get("limit", 20)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    trace_log("search_vec_start", query=(query or image_path or video_path or "")[:50], limit=limit, collection=collection, content_type=content_type,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

    if input_error:
        return _error_response("INVALID_INPUT", input_error)

    if image_path:
        vector = await _run_blocking(backend.embed_image, image_path)
    elif video_path:
        embed_video = getattr(backend, "embed_video", None)
        if not callable(embed_video):
            return _error_response("NOT_FOUND", "Backend does not support raw video queries")
        vector = await _run_blocking(embed_video, video_path)
    else:
        vector = await _run_blocking(backend.embed_text, query)

    results = await _run_blocking(
        storage.search_vec,
        vector=vector.tolist() if hasattr(vector, 'tolist') else list(vector),
        limit=limit,
        collection=collection,
        content_type=content_type,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
    )

    trace_log("search_vec_done", query=(query or image_path or video_path or "")[:50], count=len(results))

    output = {
        "query": query,
        "image_path": image_path,
        "video_path": video_path,
        "count": len(results),
        "results": [
            {
                "filepath": r.filepath,
                "title": r.title,
                "score": round(r.score, 4),
                "source": r.source,
                "user_id": r.user_id,
                "session_id": r.session_id,
                "project_id": r.project_id,
                "profile": r.profile,
            }
            for r in results
        ],
    }

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search_batch(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle parallel batch search with RRF merge."""
    from .search import BatchQuery, search_batch

    queries_raw = arguments.get("queries", [])
    limit = arguments.get("limit", 10)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    trace_log("search_batch_start", query_count=len(queries_raw), limit=limit, collection=collection,
              content_type=content_type, user_id=user_id, session_id=session_id,
              project_id=project_id, profile=profile)

    # Validate and normalize queries
    if not isinstance(queries_raw, list) or len(queries_raw) == 0:
        return _error_response("INVALID_INPUT", "queries must be a non-empty array")

    if len(queries_raw) > 20:
        return _error_response("INVALID_INPUT", "queries array must contain at most 20 items")

    # Convert to BatchQuery objects
    queries = []
    for i, q in enumerate(queries_raw):
        if isinstance(q, str):
            queries.append(BatchQuery(query=q))
        elif isinstance(q, dict):
            query_text = q.get("query")
            if not query_text or not isinstance(query_text, str):
                return _error_response("INVALID_INPUT", f"queries[{i}].query must be a non-empty string")
            weight = q.get("weight", 1.0)
            if not isinstance(weight, (int, float)) or weight < 0:
                return _error_response("INVALID_INPUT", f"queries[{i}].weight must be a non-negative number")
            queries.append(BatchQuery(
                query=query_text,
                mode=q.get("mode"),
                intent=q.get("intent"),
                weight=float(weight),
            ))
        else:
            return _error_response("INVALID_INPUT", f"queries[{i}] must be a string or object")

    results = await _run_blocking(
        search_batch,
        queries=queries,
        backend=backend,
        storage=storage,
        limit=limit,
        collection=collection,
        content_type=content_type,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
    )

    trace_log("search_batch_done", query_count=len(queries), count=len(results))

    output = {
        "query_count": len(queries),
        "limit": limit,
        "count": len(results),
        "results": [
            {
                "filepath": r.filepath,
                "title": r.title,
                "score": round(r.score, 4),
                "source": r.source,
                "query_scores": {str(k): round(v, 4) for k, v in r.query_scores.items()},
                "snippet": (r.body or "")[:500] if r.body else None,
                "user_id": getattr(r, "user_id", None),
                "session_id": getattr(r, "session_id", None),
                "project_id": getattr(r, "project_id", None),
                "profile": getattr(r, "profile", None),
            }
            for r in results
        ],
    }

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_ingest(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle unified ingest."""
    text = arguments.get("text")
    path = arguments.get("path")
    file_path = arguments.get("file_path")
    folder_path = arguments.get("folder_path")
    recursive = arguments.get("recursive", True)
    collection = arguments.get("collection", "default")
    content_types = arguments.get("content_types", ["text", "image", "video", "document"])
    include_globs = arguments.get("include_globs")
    exclude_globs = arguments.get("exclude_globs")
    max_file_size_mb = arguments.get("max_file_size_mb", 100)
    caption_media = arguments.get("caption_media", True)
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    trace_log("ingest_start", collection=collection, text=bool(text), file_path=file_path, folder_path=folder_path,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

    output = await _run_blocking(
        storage.ingest,
        collection=collection,
        text=text,
        path=path,
        file_path=file_path,
        folder_path=folder_path,
        recursive=recursive,
        content_types=content_types,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        embed_text_func=backend.embed_text,
        embed_image_func=backend.embed_image,
        embed_video_func=getattr(backend, "embed_video", None),
        model="Qwen3-VL-Embedding-2B",
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        max_file_size_mb=max_file_size_mb,
        caption_media=caption_media,
    )

    trace_log("ingest_done", collection=collection, indexed_text=output.get("indexed_text", 0), indexed_images=output.get("indexed_images", 0))
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_index_document(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle document indexing."""
    path = arguments.get("path", "")
    text = arguments.get("text", "")
    collection = arguments.get("collection", "default")
    
    trace_log("index_document_start", path=path, collection=collection)
    
    if not path or not text:
        return _error_response("INVALID_INPUT", "path and text are required")
    
    content_hash = await _run_blocking(
        storage.index_document,
        path=path,
        text=text,
        collection=collection,
        model="Qwen3-VL-Embedding-2B",
        embed_func=backend.embed_text,
    )
    
    trace_log("index_document_done", path=path, hash=content_hash[:8])
    
    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_index_image(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle image indexing."""
    path = arguments.get("path", "")
    collection = arguments.get("collection", "default")
    
    trace_log("index_image_start", path=path, collection=collection)
    
    if not path:
        return _error_response("INVALID_INPUT", "path is required")

    if not os.path.exists(path):
        return _error_response("NOT_FOUND", f"File not found: {path}", {"path": path})
    
    content_hash = await _run_blocking(
        storage.index_image,
        path=path,
        collection=collection,
        embed_func=backend.embed_image,
    )
    
    trace_log("index_image_done", path=path, hash=content_hash[:8])
    
    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_memory_add(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle memory add."""
    path = arguments.get("path", "")
    text = arguments.get("text", "")
    collection = arguments.get("collection", "default")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")
    importance = arguments.get("importance")
    ttl_seconds = arguments.get("ttl_seconds")
    tags = arguments.get("tags")

    trace_log("memory_add_start", path=path, collection=collection,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile,
              importance=importance, ttl_seconds=ttl_seconds, tags=tags)

    if not path or not text:
        return _error_response("INVALID_INPUT", "path and text are required")

    content_hash = await _run_blocking(
        storage.upsert_memory,
        path=path,
        text=text,
        collection=collection,
        model="Qwen3-VL-Embedding-2B",
        embed_func=backend.embed_text,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        importance=importance,
        ttl_seconds=ttl_seconds,
        tags=tags,
    )

    trace_log("memory_add_done", path=path, hash=content_hash[:8])

    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
        "operation": "add",
        "user_id": user_id,
        "session_id": session_id,
        "project_id": project_id,
        "profile": profile,
        "importance": importance,
        "ttl_seconds": ttl_seconds,
        "tags": tags,
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_memory_update(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle memory update."""
    path = arguments.get("path", "")
    text = arguments.get("text", "")
    collection = arguments.get("collection", "default")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")
    importance = arguments.get("importance")
    ttl_seconds = arguments.get("ttl_seconds")
    tags = arguments.get("tags")

    trace_log("memory_update_start", path=path, collection=collection,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile,
              importance=importance, ttl_seconds=ttl_seconds, tags=tags)

    if not path or not text:
        return _error_response("INVALID_INPUT", "path and text are required")

    content_hash = await _run_blocking(
        storage.upsert_memory,
        path=path,
        text=text,
        collection=collection,
        model="Qwen3-VL-Embedding-2B",
        embed_func=backend.embed_text,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        importance=importance,
        ttl_seconds=ttl_seconds,
        tags=tags,
    )

    trace_log("memory_update_done", path=path, hash=content_hash[:8])

    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
        "operation": "update",
        "user_id": user_id,
        "session_id": session_id,
        "project_id": project_id,
        "profile": profile,
        "importance": importance,
        "ttl_seconds": ttl_seconds,
        "tags": tags,
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_memory_delete(arguments: dict, storage) -> list[TextContent]:
    """Handle memory delete."""
    path = arguments.get("path", "")
    collection = arguments.get("collection", "default")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    trace_log("memory_delete_start", path=path, collection=collection,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

    if not path:
        return _error_response("INVALID_INPUT", "path is required")

    output = await _run_blocking(
        storage.delete_memory,
        path=path,
        collection=collection,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
    )

    trace_log("memory_delete_done", path=path, removed_vectors=output.get("removed_vectors", 0))

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_get_config(backend, storage, mutable_config: dict) -> list[TextContent]:
    """Return current server configuration."""
    info = await _run_blocking(backend.get_info)

    # Resolve data_dir: prefer storage attribute, fall back to env, then default
    raw_data_dir = (
        getattr(storage, "_store_path", None)
        or os.environ.get("RECALLFORGE_STORE_PATH")
        or os.path.join(os.path.expanduser("~"), ".recallforge")
    )
    try:
        from pathlib import Path as _Path
        data_dir = str(_Path(raw_data_dir).expanduser().resolve())
    except Exception:
        data_dir = raw_data_dir

    # Get model IDs from backend (REC-116)
    model_ids = {}
    if hasattr(backend, "get_model_ids"):
        model_ids = await _run_blocking(backend.get_model_ids)

    config = {
        "version": __version__,
        "backend": info.name,
        "mode": mutable_config.get("mode", backend.get_mode()),
        "quantize": info.quantization or "none",
        "data_dir": data_dir,
        "collection": mutable_config.get("collection", "default"),
        "max_file_size_mb": mutable_config.get("max_file_size_mb", 100),
        "rerank_top_k": mutable_config.get("rerank_top_k", int(os.environ.get("RECALLFORGE_RERANK_TOP_K", "20"))),
        "caption_media": mutable_config.get("caption_media", True),
    }
    # Add model IDs if available (REC-116)
    if model_ids:
        config["embedder_model"] = model_ids.get("embedder_model", "")
        config["reranker_model"] = model_ids.get("reranker_model", "")
        config["captioner_model"] = model_ids.get("captioner_model", "")

    return [TextContent(type="text", text=json.dumps(config, indent=2))]


async def _handle_set_config(
    arguments: dict,
    backend,
    storage,
    mutable_config: dict,
) -> list[TextContent]:
    """Validate and apply safe runtime configuration changes."""
    _IMMUTABLE = {"backend", "quantize", "data_dir"}
    _ALLOWED = {"mode", "collection", "max_file_size_mb", "rerank_top_k", "caption_media",
                "embedder_model", "reranker_model", "captioner_model"}

    # Reject attempts to change immutable fields
    attempted_immutable = set(arguments.keys()) & _IMMUTABLE
    if attempted_immutable:
        return _error_response(
            "INVALID_INPUT",
            f"Cannot change at runtime (requires restart): {sorted(attempted_immutable)}",
            {"immutable_fields": sorted(attempted_immutable)},
        )

    # Reject unknown fields
    unknown = set(arguments.keys()) - _ALLOWED - _IMMUTABLE
    if unknown:
        return _error_response(
            "INVALID_INPUT",
            f"Unknown config field(s): {sorted(unknown)}",
            {"unknown_fields": sorted(unknown)},
        )

    # Apply validated changes
    if "mode" in arguments:
        mode_val = arguments["mode"]
        if mode_val not in ("embed", "hybrid"):
            return _error_response(
                "INVALID_INPUT",
                f"Invalid mode {mode_val!r}. Must be one of: embed, hybrid",
                {"allowed_values": ["embed", "hybrid"]},
            )
        mutable_config["mode"] = mode_val
        backend.set_mode(mode_val)

    if "collection" in arguments:
        collection_val = arguments["collection"]
        if not isinstance(collection_val, str) or not collection_val.strip():
            return _error_response("INVALID_INPUT", "collection must be a non-empty string")
        mutable_config["collection"] = collection_val.strip()

    if "max_file_size_mb" in arguments:
        max_mb = arguments["max_file_size_mb"]
        if not isinstance(max_mb, (int, float)) or max_mb < 1:
            return _error_response(
                "INVALID_INPUT",
                "max_file_size_mb must be a number >= 1",
                {"provided": max_mb},
            )
        mutable_config["max_file_size_mb"] = int(max_mb)

    if "rerank_top_k" in arguments:
        rerank_top_k = arguments["rerank_top_k"]
        if not isinstance(rerank_top_k, (int, float)) or rerank_top_k < 0:
            return _error_response(
                "INVALID_INPUT",
                "rerank_top_k must be a number >= 0",
                {"provided": rerank_top_k},
            )
        mutable_config["rerank_top_k"] = int(rerank_top_k)

    if "caption_media" in arguments:
        caption_media = arguments["caption_media"]
        if not isinstance(caption_media, bool):
            return _error_response(
                "INVALID_INPUT",
                "caption_media must be a boolean",
                {"provided": caption_media},
            )
        mutable_config["caption_media"] = caption_media

    # Handle model ID changes (REC-116)
    model_updates = {}
    for model_key in ("embedder_model", "reranker_model", "captioner_model"):
        if model_key in arguments:
            model_val = arguments[model_key]
            if not isinstance(model_val, str) or not model_val.strip():
                return _error_response(
                    "INVALID_INPUT",
                    f"{model_key} must be a non-empty string",
                    {"provided": model_val},
                )
            model_updates[model_key] = model_val.strip()

    if model_updates and hasattr(backend, "set_model_ids"):
        await _run_blocking(
            backend.set_model_ids,
            embedder_model=model_updates.get("embedder_model"),
            reranker_model=model_updates.get("reranker_model"),
            captioner_model=model_updates.get("captioner_model"),
        )

    # Return the updated configuration
    return await _handle_get_config(backend, storage, mutable_config)


async def _handle_status(backend, storage) -> list[TextContent]:
    """Handle status request."""
    # Get model status
    info = await _run_blocking(backend.get_info)
    
    # Get storage info
    embeddings_count, documents_count = await asyncio.gather(
        _run_blocking(storage.count_embeddings),
        _run_blocking(storage.count_documents),
    )
    db_info = {
        "embeddings_count": embeddings_count,
        "documents_count": documents_count,
    }
    
    output = {
        "version": __version__,
        "models": {
            "backend": info.name,
            "device": info.device,
            "dtype": info.dtype,
            "embedder_loaded": info.embedder_loaded,
            "reranker_loaded": info.reranker_loaded,
            "memory_gb": info.memory_allocated_gb,
            "quantization": info.quantization,
            "mode": backend.get_mode(),
        },
        "database": db_info,
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_rebuild_fts(storage) -> list[TextContent]:
    """Handle FTS rebuild."""
    try:
        await _run_blocking(storage.rebuild_fts_index)
        output = {"success": True, "message": "FTS index rebuilt"}
        return [TextContent(type="text", text=json.dumps(output, indent=2))]
    except Exception as e:
        return _error_response("BACKEND_ERROR", str(e), {"exception_type": type(e).__name__})


async def _handle_list_collections(arguments: dict, storage) -> list[TextContent]:
    """Handle list_collections."""
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    trace_log("list_collections_start",
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

    collections = await _run_blocking(
        storage.list_collections,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
    )

    output = {
        "count": len(collections),
        "collections": collections,
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_list_namespaces(arguments: dict, storage) -> list[TextContent]:
    """Handle list_namespaces."""
    collection = arguments.get("collection")

    trace_log("list_namespaces_start", collection=collection)

    namespaces = await _run_blocking(
        storage.list_namespaces,
        collection=collection,
    )

    output = {
        "count": len(namespaces),
        "namespaces": namespaces,
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_rename_collection(arguments: dict, storage) -> list[TextContent]:
    """Handle rename_collection."""
    old_name = arguments.get("old_name", "")
    new_name = arguments.get("new_name", "")

    trace_log("rename_collection_start", old_name=old_name, new_name=new_name)

    if not old_name or not new_name:
        return _error_response("INVALID_INPUT", "old_name and new_name are required")

    if old_name == new_name:
        return _error_response("INVALID_INPUT", "old_name and new_name must be different")

    try:
        result = await _run_blocking(
            storage.rename_collection,
            old_name=old_name,
            new_name=new_name,
        )
        trace_log("rename_collection_done", old_name=old_name, new_name=new_name,
                  embeddings_updated=result.get("embeddings_updated", 0),
                  documents_updated=result.get("documents_updated", 0))
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return _error_response("BACKEND_ERROR", str(e), {"exception_type": type(e).__name__})


async def _handle_delete_collection(arguments: dict, storage) -> list[TextContent]:
    """Handle delete_collection."""
    name = arguments.get("name", "")

    trace_log("delete_collection_start", name=name)

    if not name:
        return _error_response("INVALID_INPUT", "name is required")

    try:
        result = await _run_blocking(
            storage.delete_collection,
            name=name,
        )
        trace_log("delete_collection_done", name=name,
                  embeddings_deleted=result.get("embeddings_deleted", 0),
                  documents_deleted=result.get("documents_deleted", 0),
                  orphans_cleaned=result.get("orphans_cleaned", 0))
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return _error_response("BACKEND_ERROR", str(e), {"exception_type": type(e).__name__})


async def _initialize_runtime(mode: Optional[str] = None):
    """Create backend+storage and preload models once for process lifetime."""
    resolved_mode = mode or os.environ.get("RECALLFORGE_MODE", "hybrid")
    # Ensure env var matches resolved mode so warmup_backend() is consistent
    os.environ["RECALLFORGE_MODE"] = resolved_mode
    store_path = os.environ.get("RECALLFORGE_STORE_PATH")

    print(f"RecallForge v{__version__}", file=sys.stderr)
    print(f"Warming up models (mode={resolved_mode})...", file=sys.stderr)

    try:
        backend = warmup_backend()
        print("All models warmed up and resident.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Model warm-up failed: {e}", file=sys.stderr)
        print("Models will load on first use (slower first query).", file=sys.stderr)
        backend = get_backend()

    storage = get_storage(store_path)
    server = await create_server(backend=backend, storage=storage, mode=resolved_mode)
    return server, backend, storage


async def run_stdio_server(mode: Optional[str] = None) -> None:
    """Run MCP over stdio transport (default, backwards compatible)."""
    global _server

    storage = None
    try:
        server, _, storage = await _initialize_runtime(mode=mode)
        _server = server
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        if storage is not None:
            try:
                await _run_blocking(storage.close)
            except Exception as e:
                print(f"Warning: storage.close() failed during shutdown: {e}", file=sys.stderr)


async def run_http_server(port: int = 7433, host: str = "127.0.0.1", mode: Optional[str] = None) -> None:
    """Run MCP over HTTP/SSE transport with process-persistent models."""
    global _server, _http_start_time

    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Mount, Route
    except ImportError as exc:
        raise RuntimeError(
            "HTTP mode requires mcp SSE + starlette dependencies. "
            "Install optional server extras (e.g., recallforge[server])."
        ) from exc

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "HTTP mode requires uvicorn. Install optional server extras "
            "(e.g., recallforge[server])."
        ) from exc

    server, backend, storage = await _initialize_runtime(mode=mode)
    _server = server
    _http_start_time = time.time()
    sse = SseServerTransport("/messages/")

    async def health(_request):
        info = await _run_blocking(backend.get_info)
        uptime = 0
        if _http_start_time is not None:
            uptime = int(max(0, time.time() - _http_start_time))
        embedder_ok = bool(info.embedder_loaded)
        reranker_ok = bool(getattr(info, "reranker_loaded", False))
        is_hybrid = backend.get_mode() == "hybrid"
        all_loaded = embedder_ok and (not is_hybrid or reranker_ok)
        model_ids = backend.get_model_ids() if hasattr(backend, "get_model_ids") else {}
        return JSONResponse(
            {
                "status": "ok" if all_loaded else "degraded",
                "models_loaded": all_loaded,
                "embedder_loaded": embedder_ok,
                "reranker_loaded": reranker_ok,
                "uptime_seconds": uptime,
                **{f"{k}_model": v for k, v in model_ids.items()},
            },
            status_code=200 if all_loaded else 503,
        )

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())
        return Response()

    async def shutdown():
        if storage is not None:
            try:
                await _run_blocking(storage.close)
            except Exception as e:
                print(f"Warning: storage.close() failed during shutdown: {e}", file=sys.stderr)

    app = Starlette(
        routes=[
            Route("/health", endpoint=health, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        on_shutdown=[shutdown],
    )

    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()


def warmup_models() -> None:
    """Pre-load backend singleton models and exit."""
    backend = warmup_backend()
    info = backend.get_info()
    print(
        json.dumps(
            {
                "status": "ok",
                "backend": info.name,
                "embedder_loaded": info.embedder_loaded,
                "reranker_loaded": info.reranker_loaded,
                "mode": backend.get_mode(),
            },
            indent=2,
        )
    )


def run_server(transport: str = "stdio", port: int = 7433, host: str = "127.0.0.1", mode: Optional[str] = None) -> None:
    """CLI entry point for MCP server transports."""
    if transport == "http":
        asyncio.run(run_http_server(port=port, host=host, mode=mode))
    else:
        asyncio.run(run_stdio_server(mode=mode))


if __name__ == "__main__":
    run_server()
