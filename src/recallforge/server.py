"""
server.py - MCP Server for RecallForge.

MCP protocol server with stdio or HTTP/SSE transport.
Tools: search, search_fts, search_vec, explain_results, search_batch, ingest,
index_document, index_image, index_audio, memory_add, memory_update, memory_delete,
memory_add_conversation, memory_get, list_memories, memory_graph_entities,
memory_graph_related, status, rebuild_fts, list_collections,
list_namespaces, rename_collection, delete_collection, batch, get_config,
set_config. Resources expose canonical memories via memory:// URIs.

Calls backend.warm_up() on server start for predictable latency.
HTTP mode exposes /health, /sse, and /messages/.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import unquote
from typing import Awaitable, Callable, Optional, TypeVar

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, Resource, ResourceTemplate, TextContent, Tool

from . import __version__, get_backend, get_storage, warmup_backend
from .audio import is_audio_file, load_audio_transcript_segments
from .conversations import normalize_conversation_turns
from .documents import extract_document_artifacts, is_document_file
from .search import HybridSearcher
from .video import is_video_file


# Configure logging
logger = logging.getLogger("recallforge.server")

# Enable trace mode via environment variable
TRACE_ENABLED = os.environ.get("RECALLFORGE_TRACE", "0") == "1"
_IMAGE_QUERY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic"}


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
_MAX_FILE_QUERY_READ_BYTES = max(
    1,
    int(os.environ.get("RECALLFORGE_MAX_FILE_QUERY_READ_BYTES", "65536")),
)

_T = TypeVar("_T")
_ProgressSend = Callable[[float, Optional[float], Optional[str]], Awaitable[None]]


class _ToolProgressReporter:
    """Best-effort MCP progress notification helper for long-running tools."""

    def __init__(self, send: Optional[_ProgressSend] = None):
        self._send = send

    @property
    def enabled(self) -> bool:
        return self._send is not None

    async def report(
        self,
        progress: float,
        total: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        if self._send is None:
            return
        try:
            await self._send(float(progress), None if total is None else float(total), message)
        except Exception as exc:
            logger.debug("Failed to send MCP progress notification: %s", exc)


def _progress_reporter_for_request(server: Server) -> _ToolProgressReporter:
    """Create a progress reporter for the active MCP request, if requested."""
    try:
        request_context = server.request_context
    except LookupError:
        return _ToolProgressReporter()

    meta = getattr(request_context, "meta", None)
    progress_token = getattr(meta, "progressToken", None)
    session = getattr(request_context, "session", None)
    if progress_token is None or session is None:
        return _ToolProgressReporter()

    send_progress = getattr(session, "send_progress_notification", None)
    if not callable(send_progress):
        return _ToolProgressReporter()

    request_id = str(getattr(request_context, "request_id", "")) or None

    async def _send(progress: float, total: Optional[float], message: Optional[str]) -> None:
        await send_progress(
            progress_token,
            progress,
            total=total,
            message=message,
            related_request_id=request_id,
        )

    return _ToolProgressReporter(_send)


def _schedule_progress_from_thread(
    loop: asyncio.AbstractEventLoop,
    progress: _ToolProgressReporter,
    value: float,
    total: Optional[float],
    message: str,
) -> None:
    """Schedule a progress notification from worker-thread callbacks."""
    if not progress.enabled:
        return
    future = asyncio.run_coroutine_threadsafe(progress.report(value, total, message), loop)

    def _log_failure(done_future):
        try:
            done_future.result()
        except Exception as exc:
            logger.debug("Scheduled MCP progress notification failed: %s", exc)

    future.add_done_callback(_log_failure)


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


def _normalize_query_text(text: str, max_chars: int = 4000) -> str:
    """Collapse whitespace and bound file-derived text queries."""
    compact = " ".join((text or "").split()).strip()
    if len(compact) <= max_chars:
        return compact
    truncated = compact[: max_chars - 3].rsplit(" ", 1)[0].strip()
    return (truncated or compact[: max_chars - 3]).rstrip() + "..."


def _resolve_file_query_input(
    file_path: str,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Convert a generic file query into text, image, or video query input."""
    resolved = Path(file_path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return None, None, None, f"File query not found: {file_path}"

    resolved_str = str(resolved)
    suffix = resolved.suffix.lower()
    if suffix in _IMAGE_QUERY_EXTENSIONS:
        return None, resolved_str, None, None
    if is_video_file(resolved):
        return None, None, resolved_str, None
    if is_audio_file(resolved):
        try:
            segments, sidecar_path = load_audio_transcript_segments(resolved, resolved_str)
        except Exception as exc:
            return None, None, None, f"Failed to extract audio transcript query from {resolved.name}: {exc}"
        text = _normalize_query_text(
            "\n".join(segment.text.strip() for segment in segments if segment.text.strip())
        )
        if not text:
            return None, None, None, (
                f"No transcript sidecar found for {resolved.name}. "
                "Add .srt, .vtt, .txt, or .transcript.json next to the audio file."
            )
        label = f"Transcript from {Path(sidecar_path).name}: " if sidecar_path else ""
        return f"{label}{text}", None, None, None
    if is_document_file(resolved):
        try:
            artifacts = extract_document_artifacts(resolved, resolved_str)
        except Exception as exc:
            return None, None, None, f"Failed to extract document query from {resolved.name}: {exc}"

        merged = _normalize_query_text(
            "\n\n".join(
                section.text.strip()
                for section in artifacts.sections
                if isinstance(section.text, str)
                and section.text.strip()
            )
        )
        if merged:
            return merged, None, None, None
        return (
            None,
            None,
            None,
            f"No extractable document text found in {resolved.name}. Install OCR-capable PDF support for scanned/image-only documents.",
        )

    try:
        file_size = resolved.stat().st_size
        with resolved.open("rb") as handle:
            raw_bytes = handle.read(_MAX_FILE_QUERY_READ_BYTES)
    except Exception as exc:
        return None, None, None, f"Failed to read file query {resolved.name}: {exc}"

    if file_size > _MAX_FILE_QUERY_READ_BYTES:
        logger.info(
            "Capped file query read path=%s file_size=%d read_limit=%d",
            resolved_str,
            file_size,
            _MAX_FILE_QUERY_READ_BYTES,
        )

    if b"\x00" in raw_bytes[:2048]:
        return None, None, None, f"Unsupported binary file query: {resolved.name}"

    text = _normalize_query_text(raw_bytes.decode("utf-8", errors="replace"))
    if not text:
        return None, None, None, f"File query {resolved.name} did not contain searchable text"
    return text, None, None, None


def _resolve_query_inputs(
    arguments: dict,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Validate mutually exclusive text/image/video/file query inputs."""
    query = arguments.get("query")
    image_path = arguments.get("image_path")
    video_path = arguments.get("video_path")
    file_path = arguments.get("file_path")
    query_text = query.strip() if isinstance(query, str) else ""
    image_text = image_path.strip() if isinstance(image_path, str) else ""
    video_text = video_path.strip() if isinstance(video_path, str) else ""
    file_text = file_path.strip() if isinstance(file_path, str) else ""

    if sum((bool(query_text), bool(image_text), bool(video_text), bool(file_text))) != 1:
        return None, None, None, None, "Provide exactly one of: query, image_path, video_path, or file_path"
    return (query_text or None), (image_text or None), (video_text or None), (file_text or None), None


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


def _normalize_memory_uri(uri: str) -> str:
    """Extract a memory id from a memory:// URI."""
    raw = str(uri).strip()
    if not raw.startswith("memory://"):
        raise ValueError(f"Unsupported resource URI: {uri}")
    return unquote(raw[len("memory://"):]).strip("/")


def _list_memories_from_storage(storage, **kwargs) -> list[dict]:
    list_memories = getattr(storage, "list_memories", None)
    if not callable(list_memories):
        return []
    return list_memories(**kwargs)


def _get_memory_from_storage(storage, memory_id: Optional[str] = None, **kwargs) -> Optional[dict]:
    get_memory = getattr(storage, "get_memory", None)
    if not callable(get_memory):
        return None
    if memory_id is None:
        return get_memory(**kwargs)
    return get_memory(memory_id, **kwargs)


def _list_memory_entities_from_storage(storage, **kwargs) -> list[dict]:
    list_memory_entities = getattr(storage, "list_memory_entities", None)
    if not callable(list_memory_entities):
        return []
    return list_memory_entities(**kwargs)


def _find_related_memories_from_storage(storage, **kwargs) -> list[dict]:
    find_related_memories = getattr(storage, "find_related_memories", None)
    if not callable(find_related_memories):
        return []
    return find_related_memories(**kwargs)


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
    server = Server("recallforge", version=__version__)
    
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

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        memories = await _run_blocking(
            _list_memories_from_storage,
            storage,
            collection=None,
            limit=100,
        )
        resources: list[Resource] = []
        for memory in memories:
            memory_id = memory.get("memory_id")
            if not memory_id:
                continue
            title = memory.get("title") or memory.get("path") or memory_id
            resources.append(
                Resource(
                    name=title,
                    uri=f"memory://{memory_id}",
                    title=title,
                    description=f"Canonical memory object for {memory.get('path') or title}",
                    mimeType="application/json",
                )
            )
        return resources

    @server.list_resource_templates()
    async def list_resource_templates() -> list[ResourceTemplate]:
        return [
            ResourceTemplate(
                name="memory",
                uriTemplate="memory://{memory_id}",
                title="Memory Resource",
                description="Read a canonical memory object by stable memory_id.",
                mimeType="application/json",
            )
        ]

    @server.read_resource()
    async def read_resource(uri) -> str:
        memory_id = _normalize_memory_uri(str(uri))
        memory = await _run_blocking(_get_memory_from_storage, storage, memory_id)
        if not memory:
            raise ValueError(f"Memory not found: {memory_id}")
        return json.dumps(memory, indent=2)
    
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
                        "file_path": {"type": "string", "description": "Optional generic file query path (mutually exclusive with query/image_path/video_path). Text files are read directly; image/video/document files auto-route through the matching query path."},
                        "limit": {"type": "integer", "description": "Maximum results to return", "default": 10},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image", "video", "audio"], "description": "Optional content type filter"},
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
                        "content_type": {"type": "string", "enum": ["text", "image", "video", "audio"], "description": "Optional content type filter"},
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
                        "file_path": {"type": "string", "description": "Optional generic file query path (mutually exclusive with query/image_path/video_path)"},
                        "limit": {"type": "integer", "description": "Maximum results to return", "default": 20},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image", "video", "audio"], "description": "Optional content type filter"},
                        "user_id": {"type": "string", "description": "Optional user namespace filter for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                    },
                },
            ),
            Tool(
                name="ingest",
                description="Unified ingest for text, image, video, audio, document, file, or folder. Auto-detects modality and routes accordingly.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Raw text content to ingest"},
                        "path": {"type": "string", "description": "Optional memory path for raw text ingest"},
                        "file_path": {"type": "string", "description": "Path to a single file (text, image, video, audio, or office document)"},
                        "folder_path": {"type": "string", "description": "Path to a folder to ingest"},
                        "recursive": {"type": "boolean", "description": "Recursively ingest subfolders", "default": True},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                        "content_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["text", "image", "video", "audio", "document"]},
                            "description": "Allowed content types to ingest",
                            "default": ["text", "image", "video", "audio", "document"]
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
                name="index_audio",
                description="Index an audio file through transcript sidecars for transcript-first search",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute path to audio file"},
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
                name="memory_add_conversation",
                description="Add or replace a conversation memory with a canonical parent and turn-level child memories",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Conversation memory root path within collection"},
                        "turns": {
                            "type": "array",
                            "description": "Conversation turns or message groups in chronological order",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "description": "Speaker role such as user, assistant, agent, tool, or system"},
                                    "speaker": {"type": "string", "description": "Optional speaker/persona label"},
                                    "content": {"type": "string", "description": "Turn text content"},
                                    "text": {"type": "string", "description": "Alias for content"},
                                    "timestamp": {"type": "string", "description": "Optional timestamp string, ideally ISO 8601"},
                                },
                            },
                        },
                        "title": {"type": "string", "description": "Optional conversation title"},
                        "summary": {"type": "string", "description": "Optional caller-provided summary for the parent memory"},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                        "user_id": {"type": "string", "description": "Optional user namespace for multi-tenant isolation"},
                        "session_id": {"type": "string", "description": "Optional session/thread namespace"},
                        "project_id": {"type": "string", "description": "Optional project namespace"},
                        "profile": {"type": "string", "description": "Optional profile namespace"},
                        "importance": {"type": "number", "description": "Importance score 0.0-1.0 (optional)", "minimum": 0, "maximum": 1},
                        "ttl_seconds": {"type": "integer", "description": "Time-to-live in seconds, 0 or null = no expiration (optional)"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Extra string tags (optional)"},
                    },
                    "required": ["path", "turns"],
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
                name="memory_get",
                description="Fetch a canonical memory object by memory_id or root path",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Stable memory identifier"},
                        "path": {"type": "string", "description": "Optional canonical root path when memory_id is unknown"},
                        "collection": {"type": "string", "description": "Collection name", "default": "default"},
                        "user_id": {"type": "string", "description": "Optional user namespace filter"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                    },
                },
            ),
            Tool(
                name="memory_graph_entities",
                description="List extracted entity mentions for a memory, path, or entity key with source evidence",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Stable memory identifier to inspect"},
                        "path": {"type": "string", "description": "Memory root path or child path to inspect"},
                        "entity": {"type": "string", "description": "Optional entity name/key to navigate across memories"},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "limit": {"type": "integer", "description": "Maximum entity mentions to return", "default": 100},
                        "user_id": {"type": "string", "description": "Optional user namespace filter"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                    },
                },
            ),
            Tool(
                name="memory_graph_related",
                description="Find memories related by shared extracted entities, with supporting evidence",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Stable memory identifier to use as the seed"},
                        "path": {"type": "string", "description": "Memory root path or child path to use as the seed"},
                        "entity": {"type": "string", "description": "Entity name/key to use as the seed"},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "limit": {"type": "integer", "description": "Maximum related memories to return", "default": 20},
                        "user_id": {"type": "string", "description": "Optional user namespace filter"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                    },
                },
            ),
            Tool(
                name="list_memories",
                description="List canonical root memories for a collection or namespace",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "limit": {"type": "integer", "description": "Maximum memories to return", "default": 50},
                        "user_id": {"type": "string", "description": "Optional user namespace filter"},
                        "session_id": {"type": "string", "description": "Optional session namespace filter"},
                        "project_id": {"type": "string", "description": "Optional project namespace filter"},
                        "profile": {"type": "string", "description": "Optional profile namespace filter"},
                    },
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
                        "content_type": {"type": "string", "enum": ["text", "image", "video", "audio"], "description": "Optional content type filter"},
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
                        "file_path": {"type": "string", "description": "Optional generic file query path (mutually exclusive with query/image_path/video_path)"},
                        "limit": {"type": "integer", "description": "Maximum results to return", "default": 10},
                        "collection": {"type": "string", "description": "Optional collection filter"},
                        "content_type": {"type": "string", "enum": ["text", "image", "video", "audio"], "description": "Optional content type filter"},
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
        progress = _progress_reporter_for_request(server)
        try:
            if name == "batch":
                return await _handle_batch(arguments, backend, storage, _mutable_config, progress=progress)
            return await _dispatch_tool(name, arguments, backend, storage, _mutable_config, progress=progress)
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
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Route a single tool call to the appropriate handler."""
    progress = progress or _ToolProgressReporter()
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
        return await _handle_search(arguments, backend, storage, progress=progress)
    elif name == "explain_results":
        return await _handle_explain_results(arguments, backend, storage, progress=progress)
    elif name == "search_fts":
        return await _handle_search_fts(arguments, storage, progress=progress)
    elif name == "search_vec":
        return await _handle_search_vec(arguments, backend, storage, progress=progress)
    elif name == "ingest":
        return await _handle_ingest(arguments, backend, storage, progress=progress)
    elif name == "index_document":
        return await _handle_index_document(arguments, backend, storage, progress=progress)
    elif name == "index_image":
        return await _handle_index_image(arguments, backend, storage, progress=progress)
    elif name == "index_audio":
        return await _handle_index_audio(arguments, backend, storage, progress=progress)
    elif name == "memory_add":
        return await _handle_memory_add(arguments, backend, storage, progress=progress)
    elif name == "memory_add_conversation":
        return await _handle_memory_add_conversation(arguments, backend, storage, progress=progress)
    elif name == "memory_update":
        return await _handle_memory_update(arguments, backend, storage, progress=progress)
    elif name == "memory_delete":
        return await _handle_memory_delete(arguments, storage, progress=progress)
    elif name == "memory_get":
        return await _handle_memory_get(arguments, storage)
    elif name == "memory_graph_entities":
        return await _handle_memory_graph_entities(arguments, storage)
    elif name == "memory_graph_related":
        return await _handle_memory_graph_related(arguments, storage)
    elif name == "list_memories":
        return await _handle_list_memories(arguments, storage)
    elif name == "status":
        return await _handle_status(backend, storage)
    elif name == "rebuild_fts":
        return await _handle_rebuild_fts(storage, progress=progress)
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
        return await _handle_search_batch(arguments, backend, storage, progress=progress)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _handle_batch(
    arguments: dict,
    backend,
    storage,
    mutable_config: Optional[dict] = None,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Execute multiple RecallForge operations in a single call."""
    progress = progress or _ToolProgressReporter()
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
    total = len(operations)
    progress_total = max(total, 1)

    await progress.report(0, progress_total, f"Starting batch with {total} operation(s)")

    for i, op in enumerate(operations):
        tool_name = op.get("tool", "")
        op_args = op.get("arguments", {})
        await progress.report(
            i,
            progress_total,
            f"Starting batch operation {i + 1}/{total}: {tool_name or 'unknown'}",
        )

        # Reject nested batch calls
        if tool_name == "batch":
            batch_results.append({
                "index": i,
                "tool": tool_name,
                "status": "error",
                "result": {"error": "Nested batch operations are not allowed"},
            })
            failed += 1
            await progress.report(
                i + 1,
                progress_total,
                f"Finished batch operation {i + 1}/{total}: {tool_name} (error)",
            )
            continue

        try:
            content_list = await _dispatch_tool(
                tool_name,
                op_args,
                backend,
                storage,
                mutable_config,
            )
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
            await progress.report(
                i + 1,
                progress_total,
                f"Finished batch operation {i + 1}/{total}: {tool_name} (success)",
            )
        except Exception as exc:
            batch_results.append({
                "index": i,
                "tool": tool_name,
                "status": "error",
                "result": {"error": str(exc)},
            })
            failed += 1
            await progress.report(
                i + 1,
                progress_total,
                f"Finished batch operation {i + 1}/{total}: {tool_name} (error)",
            )

    output = {
        "batch_results": batch_results,
        "total": len(operations),
        "succeeded": succeeded,
        "failed": failed,
    }
    await progress.report(progress_total, progress_total, f"Batch complete: {succeeded} succeeded, {failed} failed")
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle hybrid search."""
    progress = progress or _ToolProgressReporter()
    query, image_path, video_path, file_path, input_error = _resolve_query_inputs(arguments)
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

    if input_error:
        return _error_response("INVALID_INPUT", input_error)

    await progress.report(0, 4, "Validating search input")
    if file_path:
        query, image_path, video_path, file_error = await _run_blocking(_resolve_file_query_input, file_path)
        if file_error:
            return _error_response("INVALID_INPUT", file_error, {"file_path": file_path})
    await progress.report(1, 4, "Resolved search input")

    trace_log("search_start", query=(query or image_path or video_path or file_path or "")[:50], limit=limit, collection=collection, content_type=content_type,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile, intent=intent, rerank_top_k=rerank_top_k, expand=expand)

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
        await progress.report(2, 4, "Running image search")
        results = await _run_blocking(searcher.search_image, image_path)
    elif video_path:
        await progress.report(2, 4, "Running video search")
        results = await _run_blocking(searcher.search_video, video_path)
    else:
        await progress.report(2, 4, "Running hybrid search")
        results = await _run_blocking(searcher.search, query)

    trace_log("search_done", query=(query or image_path or video_path or file_path or "")[:50], count=len(results))
    await progress.report(3, 4, f"Search retrieved {len(results)} result(s)")

    output = {
        "query": query,
        "image_path": image_path,
        "video_path": video_path,
        "file_path": file_path,
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
                "memory_id": getattr(r, "memory_id", None),
                "memory_role": getattr(r, "memory_role", "root"),
                "memory_root_path": getattr(r, "memory_root_path", None),
                "memory_hit_count": getattr(r, "memory_hit_count", 1),
                "memory_primary_evidence_path": getattr(r, "memory_primary_evidence_path", None),
                "memory_supporting_paths": getattr(r, "memory_supporting_paths", None),
                "tags": getattr(r, "tags", None),
            }
            for r in results
        ],
    }

    await progress.report(4, 4, "Search response ready")
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_explain_results(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle explain_results - returns detailed scoring provenance for each result."""
    progress = progress or _ToolProgressReporter()
    query, image_path, video_path, file_path, input_error = _resolve_query_inputs(arguments)
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

    if input_error:
        return _error_response("INVALID_INPUT", input_error)

    await progress.report(0, 4, "Validating explanation input")
    if file_path:
        query, image_path, video_path, file_error = await _run_blocking(_resolve_file_query_input, file_path)
        if file_error:
            return _error_response("INVALID_INPUT", file_error, {"file_path": file_path})
    await progress.report(1, 4, "Resolved explanation input")

    trace_log("explain_results_start", query=(query or image_path or video_path or file_path or "")[:50], limit=limit, collection=collection, content_type=content_type,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile, intent=intent, rerank_top_k=rerank_top_k, expand=expand)

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
        await progress.report(2, 4, "Running image search for explanation")
        results = await _run_blocking(searcher.search_image, image_path)
    elif video_path:
        await progress.report(2, 4, "Running video search for explanation")
        results = await _run_blocking(searcher.search_video, video_path)
    else:
        await progress.report(2, 4, "Running hybrid search for explanation")
        results = await _run_blocking(searcher.search, query)

    trace_log("explain_results_done", query=(query or image_path or video_path or file_path or "")[:50], count=len(results))
    await progress.report(3, 4, f"Building explanations for {len(results)} result(s)")

    # Build detailed explanation for each result
    explained_results = []
    for r in results:
        explanation = {
            "filepath": r.filepath,
            "title": r.title,
            "final_score": round(r.score, 4),
            "content_type": r.content_type if hasattr(r, 'content_type') else "text",
            "source": r.source,
            "memory_id": getattr(r, "memory_id", None),
            "memory_role": getattr(r, "memory_role", "root"),
            "memory_root_path": getattr(r, "memory_root_path", None),
            "memory_hit_count": getattr(r, "memory_hit_count", 1),
            "memory_primary_evidence_path": getattr(r, "memory_primary_evidence_path", None),
            "memory_supporting_paths": getattr(r, "memory_supporting_paths", None),
            "tags": getattr(r, "tags", None),
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
                "memory_rollup": {
                    "memory_hit_count": getattr(r, "memory_hit_count", 1),
                    "boost": round(r.audit.memory_rollup_boost, 6),
                    "primary_evidence_path": r.audit.memory_primary_evidence_path,
                    "supporting_paths": list(r.audit.memory_supporting_paths),
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
        "file_path": file_path,
        "mode": backend.get_mode(),
        "count": len(results),
        "results": explained_results,
    }

    await progress.report(4, 4, "Explanation response ready")
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search_fts(
    arguments: dict,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle FTS search."""
    progress = progress or _ToolProgressReporter()
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

    await progress.report(0, 3, "Validating full-text search input")
    await progress.report(1, 3, "Running full-text search")
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
    await progress.report(2, 3, f"Full-text search retrieved {len(results)} result(s)")

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
                "tags": getattr(r, "tags", None),
            }
            for r in results
        ],
    }

    await progress.report(3, 3, "Full-text search response ready")
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search_vec(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle vector search."""
    progress = progress or _ToolProgressReporter()
    query, image_path, video_path, file_path, input_error = _resolve_query_inputs(arguments)
    limit = arguments.get("limit", 20)
    collection = arguments.get("collection")
    content_type = arguments.get("content_type")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    if input_error:
        return _error_response("INVALID_INPUT", input_error)

    await progress.report(0, 4, "Validating vector search input")
    if file_path:
        query, image_path, video_path, file_error = await _run_blocking(_resolve_file_query_input, file_path)
        if file_error:
            return _error_response("INVALID_INPUT", file_error, {"file_path": file_path})
    await progress.report(1, 4, "Embedding vector search input")

    trace_log("search_vec_start", query=(query or image_path or video_path or file_path or "")[:50], limit=limit, collection=collection, content_type=content_type,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

    if image_path:
        vector = await _run_blocking(backend.embed_image, image_path)
    elif video_path:
        embed_video = getattr(backend, "embed_video", None)
        if not callable(embed_video):
            return _error_response("NOT_FOUND", "Backend does not support raw video queries")
        vector = await _run_blocking(embed_video, video_path)
    else:
        vector = await _run_blocking(backend.embed_text, query)

    await progress.report(2, 4, "Running vector search")
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

    trace_log("search_vec_done", query=(query or image_path or video_path or file_path or "")[:50], count=len(results))
    await progress.report(3, 4, f"Vector search retrieved {len(results)} result(s)")

    output = {
        "query": query,
        "image_path": image_path,
        "video_path": video_path,
        "file_path": file_path,
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
                "tags": getattr(r, "tags", None),
            }
            for r in results
        ],
    }

    await progress.report(4, 4, "Vector search response ready")
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_search_batch(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle parallel batch search with RRF merge."""
    from .search import BatchQuery, search_batch

    progress = progress or _ToolProgressReporter()
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

    await progress.report(0, len(queries), f"Starting batch search with {len(queries)} query item(s)")
    loop = asyncio.get_running_loop()

    def _query_progress(completed: int, total: int, result_count: int) -> None:
        _schedule_progress_from_thread(
            loop,
            progress,
            completed,
            total,
            f"Batch search completed query {completed}/{total}; last branch returned {result_count} candidate(s)",
        )

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
        progress_callback=_query_progress if progress.enabled else None,
    )

    trace_log("search_batch_done", query_count=len(queries), count=len(results))
    await progress.report(len(queries), len(queries), f"Batch search merged {len(results)} result(s)")

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
                "memory_id": getattr(r, "memory_id", None),
                "memory_role": getattr(r, "memory_role", "root"),
                "memory_root_path": getattr(r, "memory_root_path", None),
                "memory_hit_count": getattr(r, "memory_hit_count", 1),
                "memory_primary_evidence_path": getattr(r, "memory_primary_evidence_path", None),
                "memory_supporting_paths": getattr(r, "memory_supporting_paths", None),
                "tags": getattr(r, "tags", None),
            }
            for r in results
        ],
    }

    await progress.report(len(queries), len(queries), "Batch search response ready")
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_ingest(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle unified ingest."""
    progress = progress or _ToolProgressReporter()
    text = arguments.get("text")
    path = arguments.get("path")
    file_path = arguments.get("file_path")
    folder_path = arguments.get("folder_path")
    recursive = arguments.get("recursive", True)
    collection = arguments.get("collection", "default")
    content_types = arguments.get("content_types", ["text", "image", "video", "audio", "document"])
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

    await progress.report(0, 2, "Starting ingest")
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
    indexed_total = sum(
        int(output.get(key, 0) or 0)
        for key in (
            "indexed_text",
            "indexed_images",
            "indexed_videos",
            "indexed_audio",
            "indexed_documents",
            "indexed_document_sections",
            "indexed_video_frames",
            "indexed_video_transcripts",
        )
    )
    await progress.report(2, 2, f"Ingest complete; indexed {indexed_total} item(s)")
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_index_document(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle document indexing."""
    progress = progress or _ToolProgressReporter()
    path = arguments.get("path", "")
    text = arguments.get("text", "")
    collection = arguments.get("collection", "default")
    
    trace_log("index_document_start", path=path, collection=collection)
    
    if not path or not text:
        return _error_response("INVALID_INPUT", "path and text are required")

    await progress.report(0, 1, f"Indexing document {path}")
    content_hash = await _run_blocking(
        storage.index_document,
        path=path,
        text=text,
        collection=collection,
        model="Qwen3-VL-Embedding-2B",
        embed_func=backend.embed_text,
    )
    
    trace_log("index_document_done", path=path, hash=content_hash[:8])
    await progress.report(1, 1, f"Indexed document {path}")
    
    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_index_image(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle image indexing."""
    progress = progress or _ToolProgressReporter()
    path = arguments.get("path", "")
    collection = arguments.get("collection", "default")
    
    trace_log("index_image_start", path=path, collection=collection)
    
    if not path:
        return _error_response("INVALID_INPUT", "path is required")

    if not os.path.exists(path):
        return _error_response("NOT_FOUND", f"File not found: {path}", {"path": path})

    await progress.report(0, 1, f"Indexing image {path}")
    content_hash = await _run_blocking(
        storage.index_image,
        path=path,
        collection=collection,
        embed_func=backend.embed_image,
    )
    
    trace_log("index_image_done", path=path, hash=content_hash[:8])
    await progress.report(1, 1, f"Indexed image {path}")
    
    output = {
        "success": True,
        "path": path,
        "collection": collection,
        "hash": content_hash,
    }
    
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_index_audio(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle transcript-first audio indexing."""
    progress = progress or _ToolProgressReporter()
    path = arguments.get("path", "")
    collection = arguments.get("collection", "default")

    trace_log("index_audio_start", path=path, collection=collection)

    if not path:
        return _error_response("INVALID_INPUT", "path is required")

    if not os.path.exists(path):
        return _error_response("NOT_FOUND", f"File not found: {path}", {"path": path})

    await progress.report(0, 1, f"Indexing audio {path}")
    output = await _run_blocking(
        storage.index_audio,
        path=path,
        collection=collection,
        embed_text_func=backend.embed_text,
    )

    trace_log("index_audio_done", path=path, hash=str(output.get("hash", ""))[:8])
    await progress.report(1, 1, f"Indexed audio {path}")
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_memory_add(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle memory add."""
    progress = progress or _ToolProgressReporter()
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

    await progress.report(0, 1, f"Adding memory {path}")
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
    await progress.report(1, 1, f"Added memory {path}")

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


async def _handle_memory_add_conversation(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle conversation memory ingest."""
    progress = progress or _ToolProgressReporter()
    path = arguments.get("path", "")
    turns = arguments.get("turns")
    collection = arguments.get("collection", "default")
    title = arguments.get("title")
    summary = arguments.get("summary")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")
    importance = arguments.get("importance")
    ttl_seconds = arguments.get("ttl_seconds")
    tags = arguments.get("tags")

    trace_log("memory_add_conversation_start", path=path, collection=collection,
              user_id=user_id, session_id=session_id, project_id=project_id, profile=profile,
              importance=importance, ttl_seconds=ttl_seconds, tags=tags)

    if not isinstance(path, str) or not path.strip():
        return _error_response("INVALID_INPUT", "path is required")
    if tags is not None and not isinstance(tags, list):
        return _error_response("INVALID_INPUT", "tags must be an array of strings")

    try:
        normalize_conversation_turns(turns)
    except ValueError as exc:
        return _error_response("INVALID_INPUT", str(exc))

    index_conversation = getattr(storage, "index_conversation", None)
    if not callable(index_conversation):
        return _error_response("BACKEND_ERROR", "Storage backend does not support conversation memories")

    try:
        await progress.report(0, 1, f"Indexing conversation memory {path}")
        output = await _run_blocking(
            index_conversation,
            path=path,
            turns=turns,
            collection=collection,
            model="Qwen3-VL-Embedding-2B",
            embed_func=backend.embed_text,
            title=title,
            summary=summary,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            importance=importance,
            ttl_seconds=ttl_seconds,
            tags=tags,
        )
    except ValueError as exc:
        return _error_response("INVALID_INPUT", str(exc))

    trace_log(
        "memory_add_conversation_done",
        path=path,
        hash=str(output.get("hash", ""))[:8],
        indexed_turns=output.get("indexed_turns", 0),
    )
    await progress.report(1, 1, f"Indexed conversation memory {path}")

    output = dict(output)
    output["operation"] = "add_conversation"
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_memory_update(
    arguments: dict,
    backend,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle memory update."""
    progress = progress or _ToolProgressReporter()
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

    await progress.report(0, 1, f"Updating memory {path}")
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
    await progress.report(1, 1, f"Updated memory {path}")

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


async def _handle_memory_delete(
    arguments: dict,
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle memory delete."""
    progress = progress or _ToolProgressReporter()
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

    await progress.report(0, 1, f"Deleting memory {path}")
    output = await _run_blocking(
        storage.delete_memory,
        path=path,
        collection=collection,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
    )
    await progress.report(1, 1, f"Deleted memory {path}")

    trace_log("memory_delete_done", path=path, removed_vectors=output.get("removed_vectors", 0))

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_list_memories(arguments: dict, storage) -> list[TextContent]:
    """Handle canonical memory listing."""
    collection = arguments.get("collection")
    limit = arguments.get("limit", 50)
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    memories = await _run_blocking(
        _list_memories_from_storage,
        storage,
        collection=collection,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        limit=limit,
    )
    output = {
        "success": True,
        "collection": collection,
        "count": len(memories),
        "memories": memories,
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_memory_get(arguments: dict, storage) -> list[TextContent]:
    """Handle canonical memory fetch by id or root path."""
    memory_id = arguments.get("memory_id")
    path = arguments.get("path")
    collection = arguments.get("collection")
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    if not memory_id and not path:
        return _error_response("INVALID_INPUT", "memory_id or path is required")

    memory = await _run_blocking(
        _get_memory_from_storage,
        storage,
        memory_id,
        collection=collection,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        path=path,
    )
    if not memory:
        details = {"memory_id": memory_id} if memory_id else {"path": path}
        return _error_response("NOT_FOUND", "Memory not found", details)

    return [TextContent(type="text", text=json.dumps(memory, indent=2))]


async def _handle_memory_graph_entities(arguments: dict, storage) -> list[TextContent]:
    """Handle entity mention lookup for the memory graph."""
    memory_id = arguments.get("memory_id")
    path = arguments.get("path")
    entity = arguments.get("entity")
    collection = arguments.get("collection")
    limit = arguments.get("limit", 100)
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    if not memory_id and not path and not entity:
        return _error_response("INVALID_INPUT", "memory_id, path, or entity is required")
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return _error_response("INVALID_INPUT", "limit must be an integer")

    if not callable(getattr(storage, "list_memory_entities", None)):
        return _error_response("BACKEND_ERROR", "Storage backend does not support memory graph entities")

    entities = await _run_blocking(
        _list_memory_entities_from_storage,
        storage,
        memory_id=memory_id,
        path=path,
        entity=entity,
        collection=collection,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        limit=limit,
    )

    output = {
        "success": True,
        "count": len(entities),
        "entities": entities,
        "memory_id": memory_id,
        "path": path,
        "entity": entity,
        "collection": collection,
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _handle_memory_graph_related(arguments: dict, storage) -> list[TextContent]:
    """Handle related-memory lookup through shared graph entities."""
    memory_id = arguments.get("memory_id")
    path = arguments.get("path")
    entity = arguments.get("entity")
    collection = arguments.get("collection")
    limit = arguments.get("limit", 20)
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")
    project_id = arguments.get("project_id")
    profile = arguments.get("profile")

    if not memory_id and not path and not entity:
        return _error_response("INVALID_INPUT", "memory_id, path, or entity is required")
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return _error_response("INVALID_INPUT", "limit must be an integer")

    if not callable(getattr(storage, "find_related_memories", None)):
        return _error_response("BACKEND_ERROR", "Storage backend does not support related memory graph lookup")

    related = await _run_blocking(
        _find_related_memories_from_storage,
        storage,
        memory_id=memory_id,
        path=path,
        entity=entity,
        collection=collection,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        limit=limit,
    )

    output = {
        "success": True,
        "count": len(related),
        "related_memories": related,
        "memory_id": memory_id,
        "path": path,
        "entity": entity,
        "collection": collection,
    }
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


async def _handle_rebuild_fts(
    storage,
    progress: Optional[_ToolProgressReporter] = None,
) -> list[TextContent]:
    """Handle FTS rebuild."""
    progress = progress or _ToolProgressReporter()
    try:
        await progress.report(0, 1, "Rebuilding full-text search index")
        await _run_blocking(storage.rebuild_fts_index)
        output = {"success": True, "message": "FTS index rebuilt"}
        await progress.report(1, 1, "Full-text search index rebuilt")
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
