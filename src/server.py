"""
server.py - MCP Server for QMD-VL.

MCP protocol server with stdio transport.
Tools: search (hybrid), search_fts, search_vec, index_document, index_image, status, rebuild_fts

Calls models.warm_up() on server start so first query isn't slow.
Proper error handling and graceful shutdown.
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

from src import db
from src.models import warm_up, status as model_status
from src.store import index_document as store_index_document, insert_image
from src.search import hybrid_query, HybridSearcher
from src.store import search_fts, search_vec


# Global server state
_server: Optional[Server] = None
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    print(f"Received signal {signum}, initiating graceful shutdown...", file=sys.stderr)
    _shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


async def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("qmd-vl")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="search",
                description="Full hybrid search combining BM25, vector search, query expansion, and reranking",
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
                return await _handle_search(arguments)
            elif name == "search_fts":
                return await _handle_search_fts(arguments)
            elif name == "search_vec":
                return await _handle_search_vec(arguments)
            elif name == "index_document":
                return await _handle_index_document(arguments)
            elif name == "index_image":
                return await _handle_index_image(arguments)
            elif name == "status":
                return await _handle_status(arguments)
            elif name == "rebuild_fts":
                return await _handle_rebuild_fts(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
    
    return server


async def _handle_search(arguments: dict) -> list[TextContent]:
    """Handle hybrid search."""
    query = arguments.get("query", "")
    limit = arguments.get("limit", 10)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    
    if not query:
        return [TextContent(type="text", text=json.dumps({"error": "Query is required"}))]
    
    results = hybrid_query(
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
                "rerank_score": round(r.rerank_score, 4),
                "rrf_rank": r.rrf_rank,
                "source": r.source,
                "snippet": (r.body or "")[:500] if r.body else None,
            }
            for r in results
        ],
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search_fts(arguments: dict) -> list[TextContent]:
    """Handle FTS search."""
    query = arguments.get("query", "")
    limit = arguments.get("limit", 20)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    
    if not query:
        return [TextContent(type="text", text=json.dumps({"error": "Query is required"}))]
    
    results = search_fts(
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


async def _handle_search_vec(arguments: dict) -> list[TextContent]:
    """Handle vector search."""
    from src.models import embed_text
    
    query = arguments.get("query", "")
    limit = arguments.get("limit", 20)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    
    if not query:
        return [TextContent(type="text", text=json.dumps({"error": "Query is required"}))]
    
    # Embed query
    vector = embed_text(query)
    
    results = search_vec(
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


async def _handle_index_document(arguments: dict) -> list[TextContent]:
    """Handle document indexing."""
    from src.models import get_registry
    
    path = arguments.get("path", "")
    text = arguments.get("text", "")
    collection = arguments.get("collection", "default")
    
    if not path or not text:
        return [TextContent(type="text", text=json.dumps({"error": "path and text are required"}))]
    
    registry = get_registry()
    
    content_hash = store_index_document(
        path=path,
        text=text,
        collection=collection,
        model="Qwen3-VL-Embedding-2B",
        embed_func=registry.embed_text,
    )
    
    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_index_image(arguments: dict) -> list[TextContent]:
    """Handle image indexing."""
    from src.models import get_registry
    
    path = arguments.get("path", "")
    collection = arguments.get("collection", "default")
    
    if not path:
        return [TextContent(type="text", text=json.dumps({"error": "path is required"}))]
    
    if not os.path.exists(path):
        return [TextContent(type="text", text=json.dumps({"error": f"File not found: {path}"}))]
    
    registry = get_registry()
    
    content_hash = insert_image(
        path=path,
        collection=collection,
        embed_func=registry.embed_image,
    )
    
    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_status(arguments: dict) -> list[TextContent]:
    """Handle status request."""
    # Get model status
    models = model_status()
    
    # Get database info
    db_info = {}
    if db.embeddings_table is not None:
        try:
            db_info["embeddings_count"] = db.embeddings_table.count_rows()
        except:
            db_info["embeddings_count"] = 0
    if db.documents_table is not None:
        try:
            db_info["documents_count"] = db.documents_table.count_rows()
        except:
            db_info["documents_count"] = 0
    
    output = {
        "models": models,
        "database": db_info,
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_rebuild_fts(arguments: dict) -> list[TextContent]:
    """Handle FTS rebuild."""
    try:
        db.rebuild_fts_index()
        output = {"success": True, "message": "FTS index rebuilt"}
    except Exception as e:
        output = {"success": False, "error": str(e)}
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def main() -> None:
    """Main entry point for MCP server."""
    global _server
    
    # Initialize database first (before model warm-up)
    db.initialize_database()
    
    # Warm up models
    print("Warming up QMD-VL models...", file=sys.stderr)
    try:
        warm_up()
        print("All models warmed up and resident.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Model warm-up failed: {e}", file=sys.stderr)
        print("Models will load on first use (slower first query).", file=sys.stderr)
    
    # Create server
    server = await create_server()
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
