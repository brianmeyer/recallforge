"""
server.py - MCP Server for RecallForge.

MCP protocol server with stdio transport.
Tools: search, search_fts, search_vec, ingest, index_document, index_image,
memory_add, memory_update, memory_delete, index_folder, status, rebuild_fts

Calls backend.warm_up() on server start for predictable latency.
"""

import asyncio
import json
import os
import signal
import sys
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from . import __version__, get_backend, get_storage
from .search import HybridSearcher


# Global server state
_server: Optional[Server] = None
_shutdown_requested = False


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
                        "limit": {"type": "integer", "description": "Maximum results to return", "default": 10},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image"], "description": "Optional content type filter"},
                    },
                    "required": ["query"],
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
                        "content_type": {"type": "string", "enum": ["text", "image"], "description": "Optional content type filter"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="search_vec",
                description="Vector search using embeddings and ANN",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Maximum results to return", "default": 20},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image"], "description": "Optional content type filter"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="ingest",
                description="Unified ingest for text, image, file, or folder. Auto-detects modality and routes accordingly.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Raw text content to ingest"},
                        "path": {"type": "string", "description": "Optional memory path for raw text ingest"},
                        "file_path": {"type": "string", "description": "Path to a single file (text or image)"},
                        "folder_path": {"type": "string", "description": "Path to a folder to ingest"},
                        "recursive": {"type": "boolean", "description": "Recursively ingest subfolders", "default": True},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                        "content_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["text", "image"]},
                            "description": "Allowed content types to ingest",
                            "default": ["text", "image"]
                        },
                        "include_globs": {"type": "array", "items": {"type": "string"}, "description": "Include globs relative to folder root"},
                        "exclude_globs": {"type": "array", "items": {"type": "string"}, "description": "Exclude globs relative to folder root"}
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
                        "exclude_globs": {"type": "array", "items": {"type": "string"}, "description": "Exclude globs relative to folder root"}
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
    query = arguments.get("query", "")
    limit = arguments.get("limit", 10)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    
    if not query:
        return [TextContent(type="text", text=json.dumps({"error": "Query is required"}))]
    
    searcher = HybridSearcher(
        backend=backend,
        storage=storage,
        limit=limit,
        collection=collection,
        content_type=content_type,
    )
    
    results = searcher.search(query)
    
    output = {
        "query": query,
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
    
    if not query:
        return [TextContent(type="text", text=json.dumps({"error": "Query is required"}))]
    
    results = storage.search_fts(
        query=query,
        limit=limit,
        collection=collection,
        content_type=content_type,
    )
    
    output = {
        "query": query,
        "count": len(results),
        "results": [
            {
                "filepath": r.filepath,
                "title": r.title,
                "score": round(r.score, 4),
                "source": r.source,
            }
            for r in results
        ],
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search_vec(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle vector search."""
    query = arguments.get("query", "")
    limit = arguments.get("limit", 20)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    
    if not query:
        return [TextContent(type="text", text=json.dumps({"error": "Query is required"}))]
    
    # Embed query
    vector = backend.embed_text(query)
    
    results = storage.search_vec(
        vector=vector.tolist() if hasattr(vector, 'tolist') else list(vector),
        limit=limit,
        collection=collection,
        content_type=content_type,
    )
    
    output = {
        "query": query,
        "count": len(results),
        "results": [
            {
                "filepath": r.filepath,
                "title": r.title,
                "score": round(r.score, 4),
                "source": r.source,
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
    content_types = arguments.get("content_types", ["text", "image"])
    include_globs = arguments.get("include_globs")
    exclude_globs = arguments.get("exclude_globs")

    output = storage.ingest(
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
        model="Qwen3-VL-Embedding-2B",
    )
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_index_document(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle document indexing."""
    path = arguments.get("path", "")
    text = arguments.get("text", "")
    collection = arguments.get("collection", "default")
    
    if not path or not text:
        return [TextContent(type="text", text=json.dumps({"error": "path and text are required"}))]
    
    content_hash = storage.index_document(
        path=path,
        text=text,
        collection=collection,
        model="Qwen3-VL-Embedding-2B",
        embed_func=backend.embed_text,
    )
    
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
    
    if not path:
        return [TextContent(type="text", text=json.dumps({"error": "path is required"}))]
    
    if not os.path.exists(path):
        return [TextContent(type="text", text=json.dumps({"error": f"File not found: {path}"}))]
    
    content_hash = storage.index_image(
        path=path,
        collection=collection,
        embed_func=backend.embed_image,
    )
    
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

    if not path or not text:
        return [TextContent(type="text", text=json.dumps({"error": "path and text are required"}))]

    content_hash = storage.upsert_memory(
        path=path,
        text=text,
        collection=collection,
        model="Qwen3-VL-Embedding-2B",
        embed_func=backend.embed_text,
    )

    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
        "operation": "add",
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_memory_update(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle memory update."""
    path = arguments.get("path", "")
    text = arguments.get("text", "")
    collection = arguments.get("collection", "default")

    if not path or not text:
        return [TextContent(type="text", text=json.dumps({"error": "path and text are required"}))]

    content_hash = storage.upsert_memory(
        path=path,
        text=text,
        collection=collection,
        model="Qwen3-VL-Embedding-2B",
        embed_func=backend.embed_text,
    )

    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
        "operation": "update",
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_memory_delete(arguments: dict, storage) -> list[TextContent]:
    """Handle memory delete."""
    path = arguments.get("path", "")
    collection = arguments.get("collection", "default")

    if not path:
        return [TextContent(type="text", text=json.dumps({"error": "path is required"}))]

    output = storage.delete_memory(path=path, collection=collection)
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_index_folder(arguments: dict, backend, storage) -> list[TextContent]:
    """Handle folder indexing."""
    folder_path = arguments.get("folder_path", "")
    collection = arguments.get("collection", "default")
    recursive = arguments.get("recursive", True)
    include_globs = arguments.get("include_globs")
    exclude_globs = arguments.get("exclude_globs")

    if not folder_path:
        return [TextContent(type="text", text=json.dumps({"error": "folder_path is required"}))]

    output = storage.index_folder(
        folder_path=folder_path,
        collection=collection,
        recursive=recursive,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        model="Qwen3-VL-Embedding-2B",
        embed_func=backend.embed_text,
    )
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_status(backend, storage) -> list[TextContent]:
    """Handle status request."""
    # Get model status
    info = backend.get_info()
    
    # Get storage info
    db_info = {
        "embeddings_count": storage.count_embeddings(),
        "documents_count": storage.count_documents(),
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
        storage.rebuild_fts_index()
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


def run_server() -> None:
    """Entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    run_server()
