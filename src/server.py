"""
server.py - MCP Server for QMD-VL.

Implements Model Context Protocol (MCP) server with stdio transport.
Tools: search, search_fts, search_vec, index_document, index_image, status, rebuild_fts
"""

import asyncio
import json
import os
import signal
import sys
from typing import Any, Dict, List, Optional

from src import db
from src.db import rebuild_fts_index
from src.models import get_registry, warm_up, status as model_status
from src.store import (
    index_document,
    insert_image,
    search_fts,
    search_vec,
)
from src.search import hybrid_query


# =============================================================================
# MCP Protocol Implementation
# =============================================================================

class MCPServer:
    """MCP Server for QMD-VL."""
    
    def __init__(self):
        self.name = "qmd-vl"
        self.version = "0.1.0"
        self.running = False
        self._initialized = False
    
    def _get_tools(self) -> List[Dict[str, Any]]:
        """Return list of available MCP tools."""
        return [
            {
                "name": "search",
                "description": "Hybrid search combining BM25, vector search, query expansion, and reranking. Best for general queries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 10)",
                            "default": 10
                        },
                        "collection": {
                            "type": "string",
                            "description": "Filter by collection"
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["text", "image"],
                            "description": "Filter by content type"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_fts",
                "description": "Full-text search using BM25 via Tantivy. Fast keyword matching.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 20)",
                            "default": 20
                        },
                        "collection": {
                            "type": "string",
                            "description": "Filter by collection"
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["text", "image"],
                            "description": "Filter by content type"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_vec",
                "description": "Vector semantic search. Good for conceptual queries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 20)",
                            "default": 20
                        },
                        "collection": {
                            "type": "string",
                            "description": "Filter by collection"
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["text", "image"],
                            "description": "Filter by content type"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "index_document",
                "description": "Index a text document for search.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path within collection"
                        },
                        "text": {
                            "type": "string",
                            "description": "Document content"
                        },
                        "collection": {
                            "type": "string",
                            "description": "Collection name (default: default)",
                            "default": "default"
                        }
                    },
                    "required": ["path", "text"]
                }
            },
            {
                "name": "index_image",
                "description": "Index an image for search.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to image file"
                        },
                        "collection": {
                            "type": "string",
                            "description": "Collection name (default: default)",
                            "default": "default"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "status",
                "description": "Get server status including model loading and index statistics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "rebuild_fts",
                "description": "Rebuild the full-text search index. Use after bulk updates.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }
    
    async def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {"tools": self._get_tools()}
    
    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "search":
                return await self._tool_search(arguments)
            elif tool_name == "search_fts":
                return await self._tool_search_fts(arguments)
            elif tool_name == "search_vec":
                return await self._tool_search_vec(arguments)
            elif tool_name == "index_document":
                return await self._tool_index_document(arguments)
            elif tool_name == "index_image":
                return await self._tool_index_image(arguments)
            elif tool_name == "status":
                return await self._tool_status(arguments)
            elif tool_name == "rebuild_fts":
                return await self._tool_rebuild_fts(arguments)
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Unknown tool: {tool_name}"
                    }],
                    "isError": True
                }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }],
                "isError": True
            }
    
    async def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hybrid search."""
        query = args.get("query", "")
        limit = args.get("limit", 10)
        collection = args.get("collection")
        content_type = args.get("content_type")
        
        if not query:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: query is required"
                }],
                "isError": True
            }
        
        results = hybrid_query(
            query=query,
            limit=limit,
            collection=collection,
            content_type=content_type,
        )
        
        # Format results
        output = []
        for r in results:
            output.append({
                "filepath": r.filepath,
                "title": r.title,
                "score": round(r.score, 4),
                "rerank_score": round(r.rerank_score, 4),
                "rrf_rank": r.rrf_rank,
                "source": r.source,
                "snippet": (r.body[:300] + "...") if r.body and len(r.body) > 300 else r.body,
            })
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(output, indent=2)
            }]
        }
    
    async def _tool_search_fts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FTS search."""
        query = args.get("query", "")
        limit = args.get("limit", 20)
        collection = args.get("collection")
        content_type = args.get("content_type")
        
        if not query:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: query is required"
                }],
                "isError": True
            }
        
        results = search_fts(
            query=query,
            limit=limit,
            collection=collection,
            content_type=content_type,
        )
        
        output = []
        for r in results:
            output.append({
                "filepath": r.filepath,
                "title": r.title,
                "score": round(r.score, 4),
                "source": r.source,
                "snippet": (r.body[:300] + "...") if r.body and len(r.body) > 300 else r.body,
            })
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(output, indent=2)
            }]
        }
    
    async def _tool_search_vec(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector search."""
        query = args.get("query", "")
        limit = args.get("limit", 20)
        collection = args.get("collection")
        content_type = args.get("content_type")
        
        if not query:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: query is required"
                }],
                "isError": True
            }
        
        # Embed query
        registry = get_registry()
        vector = registry.embed_text(query)
        
        results = search_vec(
            vector=vector.tolist() if hasattr(vector, 'tolist') else list(vector),
            limit=limit,
            collection=collection,
            content_type=content_type,
        )
        
        output = []
        for r in results:
            output.append({
                "filepath": r.filepath,
                "title": r.title,
                "score": round(r.score, 4),
                "source": r.source,
                "snippet": (r.body[:300] + "...") if r.body and len(r.body) > 300 else r.body,
            })
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(output, indent=2)
            }]
        }
    
    async def _tool_index_document(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Index a document."""
        path = args.get("path", "")
        text = args.get("text", "")
        collection = args.get("collection", "default")
        
        if not path or not text:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: path and text are required"
                }],
                "isError": True
            }
        
        registry = get_registry()
        content_hash = index_document(
            path=path,
            text=text,
            collection=collection,
            model="Qwen3-VL-Embedding-2B",
            embed_func=registry.embed_text,
        )
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": True,
                    "path": path,
                    "collection": collection,
                    "hash": content_hash[:12] + "...",
                })
            }]
        }
    
    async def _tool_index_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Index an image."""
        path = args.get("path", "")
        collection = args.get("collection", "default")
        
        if not path:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: path is required"
                }],
                "isError": True
            }
        
        if not os.path.exists(path):
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Image not found: {path}"
                }],
                "isError": True
            }
        
        registry = get_registry()
        content_hash = insert_image(
            path=path,
            collection=collection,
            embed_func=registry.embed_image,
        )
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": True,
                    "path": path,
                    "collection": collection,
                    "hash": content_hash[:12] + "...",
                })
            }]
        }
    
    async def _tool_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get server status."""
        models = model_status()
        
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
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "server": {
                        "name": self.name,
                        "version": self.version,
                        "initialized": self._initialized,
                    },
                    "models": models,
                    "database": db_info,
                }, indent=2)
            }]
        }
    
    async def _tool_rebuild_fts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild FTS index."""
        rebuild_fts_index()
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": True,
                    "message": "FTS index rebuilt",
                })
            }]
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle a single MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        handlers = {
            "initialize": self.handle_initialize,
            "tools/list": self.handle_list_tools,
            "tools/call": self.handle_call_tool,
        }
        
        handler = handlers.get(method)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        
        try:
            result = await handler(params)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
    
    async def run(self):
        """Run the MCP server on stdin/stdout."""
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Main loop: read from stdin, write to stdout
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        
        writer = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin,
            sys.stdout
        )
        
        while self.running:
            try:
                line = await reader.readline()
                if not line:
                    break
                
                request = json.loads(line.decode('utf-8').strip())
                response = await self.handle_request(request)
                
                if response:
                    response_str = json.dumps(response) + "\n"
                    writer.write(response_str.encode('utf-8'))
                    await writer.drain()
            
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {e}"
                    }
                }
                writer.write((json.dumps(error_response) + "\n").encode('utf-8'))
                await writer.drain()
            
            except Exception as e:
                break
        
        writer.close()


def run_server():
    """Entry point for MCP server."""
    # Initialize database
    db.initialize_database()
    
    # Warm up models
    print("QMD-VL MCP Server starting...")
    warm_up()
    print("QMD-VL MCP Server ready")
    
    # Run server
    server = MCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    run_server()