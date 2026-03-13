"""
server.py - MCP Server for RecallForge.

MCP protocol server with stdio transport.
Tools: search, search_fts, search_vec, ingest, index_document, index_image,
memory_add, memory_update, memory_delete, index_folder, status, rebuild_fts

Calls backend.warm_up() on server start for predictable latency.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Optional, Callable, TypeVar

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from . import __version__, get_backend, get_storage
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
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="search",
                description="Full hybrid search combining BM25, vector search, query expansion (full mode), and reranking (hybrid/full mode)",
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
                name="index_folder",
                description="Index text files in a folder into memory entries",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder_path": {"type": "string", "description": "Absolute or relative folder path"},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                        "recursive": {"type": "boolean", "description": "Recursively index subfolders", "default": True},
                        "include_globs": {"type": "array", "items": {"type": "string"}, "description": "Include globs relative to folder root"},
                        "exclude_globs": {"type": "array", "items": {"type": "string"}, "description": "Exclude globs relative to folder root"},
                        "user_id": {"type": "string", "description": "Optional user namespace for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace"},
                        "project_id": {"type": "string", "description": "Optional project namespace"},
                        "profile": {"type": "string", "description": "Optional profile namespace"},
                    },
                    "required": ["folder_path"],
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
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Execute a tool."""
        try:
            if name == "search":
                return await _handle_search(arguments, backend, storage)
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
            elif name == "index_folder":
                return await _handle_index_folder(arguments, backend, storage)
            elif name == "status":
                return await _handle_status(backend, storage)
            elif name == "rebuild_fts":
                return await _handle_rebuild_fts(storage)
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
    
    return server


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

    trace_log("search_start", query=(query or image_path or video_path or "")[:50], limit=limit, collection=collection, content_type=content_type,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

    if input_error:
        return [TextContent(type="text", text=json.dumps({"error": input_error}))]

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
        return [TextContent(type="text", text=json.dumps({"error": "Query is required"}))]

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
        return [TextContent(type="text", text=json.dumps({"error": input_error}))]

    if image_path:
        vector = await _run_blocking(backend.embed_image, image_path)
    elif video_path:
        embed_video = getattr(backend, "embed_video", None)
        if not callable(embed_video):
            return [TextContent(type="text", text=json.dumps({"error": "Backend does not support raw video queries"}))]
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
        return [TextContent(type="text", text=json.dumps({"error": "path and text are required"}))]
    
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
        return [TextContent(type="text", text=json.dumps({"error": "path is required"}))]
    
    if not os.path.exists(path):
        return [TextContent(type="text", text=json.dumps({"error": f"File not found: {path}"}))]
    
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
        return [TextContent(type="text", text=json.dumps({"error": "path and text are required"}))]

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
        return [TextContent(type="text", text=json.dumps({"error": "path and text are required"}))]

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
        return [TextContent(type="text", text=json.dumps({"error": "path is required"}))]

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


async def _handle_index_folder(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle folder indexing."""
    folder_path = arguments.get("folder_path", "")
    collection = arguments.get("collection", "default")
    recursive = arguments.get("recursive", True)
    include_globs = arguments.get("include_globs")
    exclude_globs = arguments.get("exclude_globs")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    trace_log("index_folder_start", folder_path=folder_path, collection=collection, recursive=recursive,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

    if not folder_path:
        return [TextContent(type="text", text=json.dumps({"error": "folder_path is required"}))]

    output = await _run_blocking(
        storage.index_folder,
        folder_path=folder_path,
        collection=collection,
        recursive=recursive,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        model="Qwen3-VL-Embedding-2B",
        embed_func=backend.embed_text,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
    )

    trace_log("index_folder_done", folder_path=folder_path, indexed=output.get("indexed", 0))

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


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
            "expander_loaded": info.expander_loaded,
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
    except Exception as e:
        output = {"success": False, "error": str(e)}
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def main() -> None:
    """Main entry point for MCP server."""
    global _server
    
    # Get configuration from environment
    mode = os.environ.get("RECALLFORGE_MODE", "full")
    store_path = os.environ.get("RECALLFORGE_STORE_PATH")
    
    backend = None
    storage = None
    try:
        # Initialize backend and storage
        backend = get_backend()
        storage = get_storage(store_path)

        # Warm up models
        print(f"RecallForge v{__version__}", file=sys.stderr)
        print(f"Warming up models (mode={mode})...", file=sys.stderr)
        try:
            backend.warm_up()
            print("All models warmed up and resident.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Model warm-up failed: {e}", file=sys.stderr)
            print("Models will load on first use (slower first query).", file=sys.stderr)

        # Create server
        server = await create_server(backend=backend, storage=storage, mode=mode)
        _server = server

        # Run server
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


def run_server() -> None:
    """Entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    run_server()
