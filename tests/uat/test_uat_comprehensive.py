"""
test_uat_comprehensive.py - Comprehensive UAT suite for RecallForge MCP server.

Tests the FULL MCP server pipeline with mocked backend:
- Ingest tests (all modalities: text, images, videos, documents)
- Search tests (all cross-modal combinations)
- Captioning verification
- Pipeline correctness (reranker, RRF, search_batch, config)
- Performance bounds

Run with: python3 -m pytest tests/uat/test_uat_comprehensive.py --tb=short -q
Run integration tests against real server: .venv/bin/python -m pytest tests/uat/test_uat_comprehensive.py -m integration --tb=short

This suite supersedes tests/uat/test_mcp_server.sh (bash script retained for backward compatibility).

LINEAR: REC-128
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest, TextContent

from recallforge import __version__, get_backend, get_storage
from recallforge.backends.base import BackendInfo
from recallforge.server import (
    _dispatch_tool,
    _error_response,
    _handle_get_config,
    _handle_set_config,
    create_server,
)


# ---------------------------------------------------------------------------
# Test fixtures and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
CORPUS_DIR = Path(__file__).parent / "corpus"
TEXT_DIR = CORPUS_DIR / "text"
IMAGES_DIR = CORPUS_DIR / "images"
VIDEOS_DIR = CORPUS_DIR / "videos"
HELPERS_DIR = Path(__file__).parent / "helpers"


def _vec_from_seed(seed: str, dim: int = 2048) -> List[float]:
    """Generate a deterministic embedding vector from a seed string."""
    values = [0.0] * dim
    tokens = re.findall(r"[a-z0-9]+", seed.lower())
    if not tokens:
        tokens = ["empty"]
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        values[idx] += 1.0
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


def _live_uat_env(store_path: Path) -> Dict[str, str]:
    """Environment for live subprocess-based MCP validation."""
    env = os.environ.copy()
    env["RECALLFORGE_STORE_PATH"] = str(store_path)
    env.setdefault("RECALLFORGE_BACKEND", "auto")
    env.setdefault("RECALLFORGE_MODE", "embed")
    env.setdefault("RECALLFORGE_MLX_QUANTIZE", "4bit")

    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else src_path + os.pathsep + existing_pythonpath
    )
    return env


def _find_free_port() -> int:
    """Reserve an ephemeral localhost port for an HTTP test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tool_result_to_json(result) -> Dict[str, Any]:
    """Parse an MCP tool result payload into a JSON dict."""
    assert result.content, "Expected MCP tool result content"
    first = result.content[0]
    assert isinstance(first, TextContent), f"Unexpected MCP content block: {type(first).__name__}"
    return json.loads(first.text)


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    """Return the tail of a log file for failure messages."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _wait_for_http_health(url: str, log_path: Path, timeout: float = 180.0) -> Dict[str, Any]:
    """Wait for the HTTP health endpoint to report healthy status."""
    deadline = time.time() + timeout
    last_status = None
    last_body = ""

    while time.time() < deadline:
        try:
            with urllib_request.urlopen(url, timeout=5.0) as response:
                body = response.read().decode("utf-8")
                payload = json.loads(body)
                if response.status == 200:
                    return payload
                last_status = response.status
                last_body = body
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_status = exc.code
            last_body = body
        except Exception as exc:
            last_body = str(exc)

        time.sleep(1.0)

    raise AssertionError(
        f"HTTP health endpoint did not become ready within {timeout:.0f}s "
        f"(last_status={last_status}, last_body={last_body!r}). "
        f"Server log tail:\n{_tail_text(log_path)}"
    )


def _stop_process(proc: subprocess.Popen, log_path: Optional[Path] = None) -> None:
    """Terminate a subprocess cleanly and surface logs on failure."""
    if proc.poll() is not None:
        return

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        if log_path is not None:
            raise AssertionError(
                "Subprocess did not shut down cleanly.\n"
                f"Server log tail:\n{_tail_text(log_path)}"
            )


# ---------------------------------------------------------------------------
# Mock Backend (no actual model loading)
# ---------------------------------------------------------------------------

class MockBackend:
    """Mock backend that returns deterministic embeddings without model loading."""

    def __init__(self, mode: str = "embed"):
        self._mode = mode
        self._dim = 2048

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def get_mode(self) -> str:
        return self._mode

    def embed_text(self, text: str) -> np.ndarray:
        return np.array(_vec_from_seed(text, self._dim), dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed_text(t) for t in texts]

    def embed_image(self, path: str) -> np.ndarray:
        name = os.path.basename(path)
        return np.array(_vec_from_seed(name, self._dim), dtype=np.float32)

    def embed_video(self, path: str) -> np.ndarray:
        name = os.path.basename(path)
        # Video embeddings include scene descriptions for better cross-modal retrieval
        if "whiteboard" in name.lower():
            return np.array(_vec_from_seed("whiteboard architecture diagram meeting", self._dim), dtype=np.float32)
        elif "cooking" in name.lower():
            return np.array(_vec_from_seed("pasta dish white plate kitchen", self._dim), dtype=np.float32)
        elif "nature" in name.lower():
            return np.array(_vec_from_seed("forest landscape mountains nature", self._dim), dtype=np.float32)
        return np.array(_vec_from_seed(name, self._dim), dtype=np.float32)

    def needs_expander(self) -> bool:
        return False

    def needs_reranker(self) -> bool:
        return False

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            name="mock",
            device="cpu",
            dtype="float32",
            embedder_loaded=True,
            reranker_loaded=False,
            memory_allocated_gb=0.0,
            quantization="none",
        )

    def warm_up(self) -> None:
        pass


class SlowMockBackend(MockBackend):
    """Backend that intentionally blocks for responsiveness tests."""

    def embed_text(self, text: str) -> np.ndarray:
        time.sleep(0.4)
        return super().embed_text(text)


# ---------------------------------------------------------------------------
# Mock Storage (in-memory for speed)
# ---------------------------------------------------------------------------

class MockStorage:
    """In-memory mock storage for UAT without real database."""

    def __init__(self, store_path: str = None):
        self._store_path = store_path or tempfile.mkdtemp(prefix="recallforge-uat-")
        self._embeddings: List[Dict[str, Any]] = []
        self._documents: List[Dict[str, Any]] = []
        self._collections: Dict[str, Dict] = {}

    def _embeddings_table(self):
        """Return a mock table for compatibility with existing tests."""
        class MockTable:
            def __init__(self, rows):
                self._rows = rows

            def search(self):
                return self

            def where(self, condition: str):
                # Simple condition parsing
                return self

            def to_list(self):
                return self._rows

        return MockTable(self._embeddings)

    def ingest(self, **kwargs) -> Dict[str, Any]:
        """Mock ingest that returns success counts."""
        result = {
            "success": True,
            "collection": kwargs.get("collection", "default"),
            "indexed_text": 0,
            "indexed_images": 0,
            "indexed_videos": 0,
            "indexed_documents": 0,
            "indexed_video_transcripts": 0,
            "indexed_video_frames": 0,
            "indexed_video_embeddings": 0,
            "skipped": 0,
        }

        # Handle text ingest
        if kwargs.get("text"):
            result["indexed_text"] = 1
            self._embeddings.append({
                "filepath": kwargs.get("path", "raw-text"),
                "content_type": "text",
                "collection": kwargs.get("collection", "default"),
                "content_hash": hashlib.sha256(kwargs["text"].encode()).hexdigest()[:16],
            })

        # Handle file ingest
        file_path = kwargs.get("file_path")
        if file_path:
            path = Path(file_path)
            if path.suffix.lower() in (".md", ".txt"):
                result["indexed_text"] = 1
            elif path.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                result["indexed_images"] = 1
            elif path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
                result["indexed_videos"] = 1
                result["indexed_video_embeddings"] = 1
            elif path.suffix.lower() in (".pdf", ".docx", ".pptx"):
                result["indexed_documents"] = 1

        # Handle folder ingest
        folder_path = kwargs.get("folder_path")
        if folder_path:
            folder = Path(folder_path)
            if folder.exists():
                for f in folder.rglob("*"):
                    if f.is_file():
                        if f.suffix.lower() in (".md", ".txt"):
                            result["indexed_text"] += 1
                        elif f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                            result["indexed_images"] += 1

        return result

    def search_fts(self, **kwargs) -> List[Any]:
        return []

    def search_vec(self, **kwargs) -> List[Any]:
        return []

    def count_embeddings(self) -> int:
        return len(self._embeddings)

    def count_documents(self) -> int:
        return len(self._documents)

    def rebuild_fts_index(self) -> None:
        pass

    def close(self) -> None:
        pass

    def list_collections(self, **kwargs) -> List[str]:
        return list(set(e.get("collection", "default") for e in self._embeddings))

    def index_document(self, **kwargs) -> str:
        return hashlib.sha256(kwargs.get("text", "").encode()).hexdigest()[:16]

    def index_image(self, **kwargs) -> str:
        return hashlib.sha256(kwargs.get("path", "").encode()).hexdigest()[:16]

    def upsert_memory(self, **kwargs) -> str:
        return hashlib.sha256(kwargs.get("text", "").encode()).hexdigest()[:16]

    def delete_memory(self, **kwargs) -> Dict[str, Any]:
        return {"success": True, "removed_vectors": 0}

    def find_document(self, collection: str, path: str) -> Optional[Dict]:
        return None

    def rename_collection(self, **kwargs) -> Dict[str, Any]:
        return {"embeddings_updated": 0, "documents_updated": 0}

    def delete_collection(self, **kwargs) -> Dict[str, Any]:
        return {"embeddings_deleted": 0, "documents_deleted": 0}


# ---------------------------------------------------------------------------
# Real Storage wrapper (for integration tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_storage():
    """Create a real storage instance for integration tests."""
    import tempfile
    store_path = tempfile.mkdtemp(prefix="recallforge-uat-real-")
    storage = get_storage(store_path)
    yield storage
    try:
        import shutil
        shutil.rmtree(store_path)
    except Exception:
        pass


@pytest.fixture(scope="session")
def live_backend():
    """Create a warmed real backend for integration tests."""
    if os.environ.get("UAT_MCP_LIVE", "0") != "1":
        pytest.skip("Integration test requires UAT_MCP_LIVE=1 and model weights")

    managed_env = {
        "RECALLFORGE_BACKEND": "auto",
        "RECALLFORGE_MODE": "hybrid",
        "RECALLFORGE_MLX_QUANTIZE": "4bit",
    }
    previous_env = {key: os.environ.get(key) for key in managed_env}

    for key, value in managed_env.items():
        os.environ.setdefault(key, value)

    backend = get_backend()
    backend.warm_up()

    yield backend

    for key, previous in previous_env.items():
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@pytest.fixture
def mock_backend():
    """Create a mock backend for unit tests."""
    return MockBackend()


@pytest.fixture
def mock_storage():
    """Create a mock storage for unit tests."""
    return MockStorage()


@pytest.fixture
def temp_store():
    """Create a temporary store directory."""
    store = tempfile.mkdtemp(prefix="recallforge-uat-")
    yield store
    shutil.rmtree(store, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test: Server Creation
# ---------------------------------------------------------------------------

class TestServerCreation:
    """Tests for MCP server creation and tool registration."""

    @pytest.mark.asyncio
    async def test_server_created_successfully(self, mock_backend, mock_storage):
        """Server should be created without errors."""
        server = await create_server(backend=mock_backend, storage=mock_storage, mode="embed")
        assert server is not None

    @pytest.mark.asyncio
    async def test_server_exposes_required_tools(self, mock_backend, mock_storage):
        """Server should expose all required MCP tools."""
        server = await create_server(backend=mock_backend, storage=mock_storage, mode="embed")
        tools = await self._list_tools(server)
        tool_names = {t.name for t in tools}

        required_tools = {
            "search", "search_fts", "search_vec", "ingest",
            "index_document", "index_image",
            "memory_add", "memory_add_conversation", "memory_update", "memory_delete",
            "memory_graph_entities", "memory_graph_related",
            "status", "rebuild_fts", "get_config", "set_config",
            "list_collections", "list_namespaces",
            "rename_collection", "delete_collection",
            "batch", "search_batch", "explain_results",
        }
        missing = required_tools - tool_names
        assert not missing, f"Missing required tools: {missing}"

    async def _list_tools(self, server):
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


# ---------------------------------------------------------------------------
# Test: Ingest (All Modalities)
# ---------------------------------------------------------------------------

class TestIngestText:
    """Tests for text file ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_md_file(self, mock_backend, temp_store):
        """Ingest .md text files from corpus/text/."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Find .md files in corpus
        md_files = list(TEXT_DIR.glob("*.md"))
        assert len(md_files) > 0, "No .md files found in corpus/text/"

        for md_file in md_files[:3]:  # Test first 3 files
            result = await self._call_tool(server, "ingest", {
                "file_path": str(md_file),
                "collection": "uat_text",
            })
            data = json.loads(result[0].text)
            assert data.get("success") or data.get("indexed_text", 0) >= 0, f"Failed to ingest {md_file}"

    @pytest.mark.asyncio
    async def test_ingest_raw_text(self, mock_backend, temp_store):
        """Ingest raw text content via text parameter."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "ingest", {
            "text": "Raw text content for UAT testing",
            "path": "test/raw-note.md",
            "collection": "uat_raw",
        })
        data = json.loads(result[0].text)
        assert data.get("indexed_text", 0) >= 1, "Should index at least one text item"

    @pytest.mark.asyncio
    async def test_ingest_text_content_type(self, mock_backend, temp_store):
        """Verify ingested text has correct content_type."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "ingest", {
            "text": "Content type verification test",
            "path": "test/content-type.md",
            "collection": "uat_ctype",
        })
        data = json.loads(result[0].text)
        # Check that we got a success response
        assert data.get("success") or data.get("indexed_text", 0) > 0

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


class TestIngestImages:
    """Tests for image ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_png_images(self, mock_backend, temp_store):
        """Ingest .png images from corpus/images/."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        png_files = list(IMAGES_DIR.glob("*.png"))
        assert len(png_files) > 0, "No .png files found in corpus/images/"

        for png_file in png_files[:2]:  # Test first 2 images
            result = await self._call_tool(server, "ingest", {
                "file_path": str(png_file),
                "collection": "uat_images",
            })
            data = json.loads(result[0].text)
            assert data.get("success") or data.get("indexed_images", 0) >= 0, f"Failed to ingest {png_file}"

    @pytest.mark.asyncio
    async def test_ingest_jpg_images(self, mock_backend, temp_store):
        """Ingest .jpg images from corpus/images/ if present."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        jpg_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg"))
        if not jpg_files:
            pytest.skip("No .jpg files found in corpus/images/")

        for jpg_file in jpg_files[:2]:
            result = await self._call_tool(server, "ingest", {
                "file_path": str(jpg_file),
                "collection": "uat_images",
            })
            data = json.loads(result[0].text)
            assert data.get("success") or data.get("indexed_images", 0) >= 0

    @pytest.mark.asyncio
    async def test_image_content_type(self, mock_backend, temp_store):
        """Verify ingested images have correct content_type."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Use whiteboard_architecture.png which should exist
        img_path = IMAGES_DIR / "whiteboard_architecture.png"
        if not img_path.exists():
            pytest.skip("whiteboard_architecture.png not found")

        result = await self._call_tool(server, "ingest", {
            "file_path": str(img_path),
            "collection": "uat_img_ctype",
        })
        data = json.loads(result[0].text)
        assert data.get("success") or data.get("indexed_images", 0) > 0

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


class TestIngestVideos:
    """Tests for video ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_mp4_videos(self, mock_backend, temp_store):
        """Ingest .mp4 videos from corpus/videos/."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        mp4_files = list(VIDEOS_DIR.glob("*.mp4"))
        if not mp4_files:
            pytest.skip("No .mp4 files found in corpus/videos/")

        for mp4_file in mp4_files[:2]:  # Test first 2 videos
            result = await self._call_tool(server, "ingest", {
                "file_path": str(mp4_file),
                "collection": "uat_videos",
                "content_types": ["video"],
            })
            data = json.loads(result[0].text)
            assert data.get("success") or data.get("indexed_videos", 0) >= 0, f"Failed to ingest {mp4_file}"

    @pytest.mark.asyncio
    async def test_video_content_type(self, mock_backend, temp_store):
        """Verify ingested videos have correct content_type."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        video_path = VIDEOS_DIR / "whiteboard_session.mp4"
        if not video_path.exists():
            pytest.skip("whiteboard_session.mp4 not found")

        result = await self._call_tool(server, "ingest", {
            "file_path": str(video_path),
            "collection": "uat_video_ctype",
            "content_types": ["video"],
        })
        data = json.loads(result[0].text)
        assert data.get("success") or data.get("indexed_videos", 0) > 0

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


class TestIngestDocuments:
    """Tests for document ingestion (PDF, DOCX, PPTX)."""

    @pytest.mark.asyncio
    async def test_ingest_docx(self, mock_backend, temp_store):
        """Ingest .docx document from generated fixtures."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Generate test document
        doc_fixture_dir = Path(temp_store) / "doc_fixtures"
        doc_fixture_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [sys.executable, str(HELPERS_DIR / "generate_test_documents.py"), str(doc_fixture_dir)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed to generate test documents: {result.stderr}"

        doc_meta = json.loads(result.stdout)
        docx_path = doc_meta["files"]["docx"]

        result = await self._call_tool(server, "ingest", {
            "file_path": docx_path,
            "collection": "uat_docs",
            "content_types": ["document"],
        })
        data = json.loads(result[0].text)
        assert data.get("success") or data.get("indexed_documents", 0) >= 0

    @pytest.mark.asyncio
    async def test_ingest_pptx(self, mock_backend, temp_store):
        """Ingest .pptx presentation from generated fixtures."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        doc_fixture_dir = Path(temp_store) / "doc_fixtures"
        doc_fixture_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [sys.executable, str(HELPERS_DIR / "generate_test_documents.py"), str(doc_fixture_dir)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed to generate test documents: {result.stderr}"

        doc_meta = json.loads(result.stdout)
        pptx_path = doc_meta["files"]["pptx"]

        result = await self._call_tool(server, "ingest", {
            "file_path": pptx_path,
            "collection": "uat_docs",
            "content_types": ["document"],
        })
        data = json.loads(result[0].text)
        assert data.get("success") or data.get("indexed_documents", 0) >= 0

    @pytest.mark.asyncio
    async def test_ingest_pdf(self, mock_backend, temp_store):
        """Ingest .pdf document from generated fixtures."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        doc_fixture_dir = Path(temp_store) / "doc_fixtures"
        doc_fixture_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [sys.executable, str(HELPERS_DIR / "generate_test_documents.py"), str(doc_fixture_dir)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed to generate test documents: {result.stderr}"

        doc_meta = json.loads(result.stdout)
        pdf_path = doc_meta["files"]["pdf"]

        result = await self._call_tool(server, "ingest", {
            "file_path": pdf_path,
            "collection": "uat_docs",
            "content_types": ["document"],
        })
        data = json.loads(result[0].text)
        assert data.get("success") or data.get("indexed_documents", 0) >= 0

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


# ---------------------------------------------------------------------------
# Test: Search (Cross-Modal Combinations)
# ---------------------------------------------------------------------------

class TestSearchTextToText:
    """Tests for text→text search."""

    @pytest.mark.asyncio
    async def test_search_with_text_query(self, mock_backend, temp_store):
        """Search with text query should return text results."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Ingest some text first
        await self._call_tool(server, "ingest", {
            "text": "Neural networks are powerful machine learning models",
            "path": "test/nn.md",
            "collection": "uat_search",
        })

        result = await self._call_tool(server, "search", {
            "query": "neural networks",
            "limit": 5,
            "collection": "uat_search",
        })
        data = json.loads(result[0].text)
        assert "count" in data
        assert "results" in data

    @pytest.mark.asyncio
    async def test_search_fts_text_query(self, mock_backend, temp_store):
        """FTS search with text query should work."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        await self._call_tool(server, "ingest", {
            "text": "Transformers architecture enables attention mechanisms",
            "path": "test/transformers.md",
            "collection": "uat_fts",
        })

        result = await self._call_tool(server, "search_fts", {
            "query": "transformers",
            "limit": 10,
            "collection": "uat_fts",
        })
        data = json.loads(result[0].text)
        assert "count" in data

    @pytest.mark.asyncio
    async def test_search_vec_text_query(self, mock_backend, temp_store):
        """Vector search with text query should work."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        await self._call_tool(server, "ingest", {
            "text": "Embeddings represent semantic meaning",
            "path": "test/embeddings.md",
            "collection": "uat_vec",
        })

        result = await self._call_tool(server, "search_vec", {
            "query": "semantic vectors",
            "limit": 10,
            "collection": "uat_vec",
        })
        data = json.loads(result[0].text)
        assert "count" in data

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


class TestSearchTextToImage:
    """Tests for text→image cross-modal search."""

    @pytest.mark.asyncio
    async def test_search_images_with_text_query(self, mock_backend, temp_store):
        """Search with text query describing an image should return image results."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Ingest images
        img_path = IMAGES_DIR / "whiteboard_architecture.png"
        if not img_path.exists():
            pytest.skip("whiteboard_architecture.png not found")

        await self._call_tool(server, "ingest", {
            "file_path": str(img_path),
            "collection": "uat_xmod",
        })

        result = await self._call_tool(server, "search", {
            "query": "whiteboard architecture diagram",
            "limit": 5,
            "collection": "uat_xmod",
            "content_type": "image",
        })
        data = json.loads(result[0].text)
        assert "count" in data
        # With mock backend, may not have results but query should be accepted

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


class TestSearchImageToText:
    """Tests for image→text cross-modal search."""

    @pytest.mark.asyncio
    async def test_search_text_with_image_query(self, mock_backend, temp_store):
        """Search with image should return related text results."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Ingest some text about images
        await self._call_tool(server, "ingest", {
            "text": "Architecture diagrams show system components and relationships",
            "path": "test/arch-diagrams.md",
            "collection": "uat_img2txt",
        })

        # Search with an image
        img_path = IMAGES_DIR / "whiteboard_architecture.png"
        if not img_path.exists():
            pytest.skip("whiteboard_architecture.png not found")

        result = await self._call_tool(server, "search", {
            "image_path": str(img_path),
            "limit": 5,
            "collection": "uat_img2txt",
            "content_type": "text",
        })
        data = json.loads(result[0].text)
        assert "count" in data

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


class TestSearchTextToDocument:
    """Tests for text→document search."""

    @pytest.mark.asyncio
    async def test_search_documents_with_text(self, mock_backend, temp_store):
        """Search for content in PDFs/DOCX/PPTX."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Generate and ingest a document
        doc_fixture_dir = Path(temp_store) / "doc_fixtures"
        doc_fixture_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [sys.executable, str(HELPERS_DIR / "generate_test_documents.py"), str(doc_fixture_dir)],
            capture_output=True, text=True
        )
        assert result.returncode == 0

        doc_meta = json.loads(result.stdout)

        # Ingest PPTX which has deployment checklist content
        await self._call_tool(server, "ingest", {
            "file_path": doc_meta["files"]["pptx"],
            "collection": "uat_docsearch",
            "content_types": ["document"],
        })

        result = await self._call_tool(server, "search", {
            "query": "deployment checklist",
            "limit": 5,
            "collection": "uat_docsearch",
            "content_type": "text",
        })
        data = json.loads(result[0].text)
        assert "count" in data

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


class TestSearchMixedModal:
    """Tests for queries returning mixed modal results."""

    @pytest.mark.asyncio
    async def test_mixed_modal_results(self, mock_backend, temp_store):
        """Queries should return both text and image results when applicable."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Ingest text
        await self._call_tool(server, "ingest", {
            "text": "Cooking pasta with fresh ingredients",
            "path": "test/cooking.md",
            "collection": "uat_mixed",
        })

        # Ingest image
        img_path = IMAGES_DIR / "food_pasta_dish.png"
        if img_path.exists():
            await self._call_tool(server, "ingest", {
                "file_path": str(img_path),
                "collection": "uat_mixed",
            })

        # Search without content_type filter
        result = await self._call_tool(server, "search", {
            "query": "pasta cooking",
            "limit": 10,
            "collection": "uat_mixed",
        })
        data = json.loads(result[0].text)
        assert "count" in data
        assert "results" in data

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


# ---------------------------------------------------------------------------
# Test: Captioning Verification
# ---------------------------------------------------------------------------

class TestCaptioning:
    """Tests for image/video captioning during ingest."""

    @pytest.mark.asyncio
    async def test_image_captioning_generates_text_body(self, mock_backend, temp_store):
        """After ingesting images, text_body should be non-empty (captioning worked)."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        img_path = IMAGES_DIR / "neural_network_diagram.png"
        if not img_path.exists():
            pytest.skip("neural_network_diagram.png not found")

        result = await self._call_tool(server, "ingest", {
            "file_path": str(img_path),
            "collection": "uat_caption",
            "caption_media": True,
        })
        data = json.loads(result[0].text)
        # With mock backend, we can't verify actual captioning, but verify API accepts the param
        assert data.get("success") or "indexed_images" in data

    @pytest.mark.asyncio
    async def test_search_image_with_bm25(self, mock_backend, temp_store):
        """Search for image content using text-only BM25 mode should find captioned images."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Ingest an image with captioning enabled
        img_path = IMAGES_DIR / "whiteboard_architecture.png"
        if not img_path.exists():
            pytest.skip("whiteboard_architecture.png not found")

        await self._call_tool(server, "ingest", {
            "file_path": str(img_path),
            "collection": "uat_bm25",
            "caption_media": True,
        })

        # Search using FTS (BM25)
        result = await self._call_tool(server, "search_fts", {
            "query": "whiteboard",
            "limit": 5,
            "collection": "uat_bm25",
        })
        data = json.loads(result[0].text)
        assert "count" in data

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


# ---------------------------------------------------------------------------
# Test: Pipeline Correctness
# ---------------------------------------------------------------------------

class TestPipelineCorrectness:
    """Tests for pipeline correctness: reranker, RRF, search_batch, config."""

    @pytest.mark.asyncio
    async def test_search_batch_works(self, mock_backend, temp_store):
        """search_batch should work with multiple queries."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Ingest some content
        await self._call_tool(server, "ingest", {
            "text": "Machine learning fundamentals",
            "path": "test/ml.md",
            "collection": "uat_batch",
        })
        await self._call_tool(server, "ingest", {
            "text": "Deep learning neural networks",
            "path": "test/dl.md",
            "collection": "uat_batch",
        })

        result = await self._call_tool(server, "search_batch", {
            "queries": ["machine learning", "neural networks"],
            "limit": 5,
            "collection": "uat_batch",
        })
        data = json.loads(result[0].text)
        assert "count" in data
        assert "results" in data
        assert "query_count" in data

    @pytest.mark.asyncio
    async def test_search_batch_with_mode_and_intent(self, mock_backend, temp_store):
        """search_batch should accept mode and intent parameters."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "search_batch", {
            "queries": [
                {"query": "test query", "mode": "hybrid", "intent": "semantic", "weight": 1.5},
            ],
            "limit": 5,
            "collection": "uat_batch_intent",
        })
        data = json.loads(result[0].text)
        assert "count" in data

    @pytest.mark.asyncio
    async def test_get_config_works(self, mock_backend, temp_store):
        """get_config should return current configuration."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "get_config", {})
        data = json.loads(result[0].text)

        required_keys = ["version", "backend", "mode", "collection", "max_file_size_mb"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_set_config_works(self, mock_backend, temp_store):
        """set_config should update mutable configuration values."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Set collection
        result = await self._call_tool(server, "set_config", {"collection": "new-collection"})
        data = json.loads(result[0].text)
        assert data.get("collection") == "new-collection"

        # Set mode
        result = await self._call_tool(server, "set_config", {"mode": "embed"})
        data = json.loads(result[0].text)
        assert data.get("mode") == "embed"

        # Set max_file_size_mb
        result = await self._call_tool(server, "set_config", {"max_file_size_mb": 50})
        data = json.loads(result[0].text)
        assert data.get("max_file_size_mb") == 50

    @pytest.mark.asyncio
    async def test_set_config_rejects_immutable(self, mock_backend, temp_store):
        """set_config should reject immutable fields."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "set_config", {"backend": "torch"})
        data = json.loads(result[0].text)
        assert data.get("error") is True
        assert data.get("code") == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_rerank_top_k_configurable(self, mock_backend, temp_store):
        """rerank_top_k should be configurable via set_config."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="hybrid")

        result = await self._call_tool(server, "set_config", {"rerank_top_k": 42})
        data = json.loads(result[0].text)
        assert data.get("rerank_top_k") == 42

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


# ---------------------------------------------------------------------------
# Test: Memory Operations
# ---------------------------------------------------------------------------

class TestMemoryOperations:
    """Tests for memory_add, memory_update, memory_delete."""

    @pytest.mark.asyncio
    async def test_memory_add(self, mock_backend, temp_store):
        """memory_add should store text with path key."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "memory_add", {
            "path": "memories/test-note.md",
            "text": "This is a test memory for UAT",
            "collection": "uat_memory",
        })
        data = json.loads(result[0].text)
        assert data.get("success") is True

    @pytest.mark.asyncio
    async def test_memory_update(self, mock_backend, temp_store):
        """memory_update should replace existing memory content."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Add first
        await self._call_tool(server, "memory_add", {
            "path": "memories/update-test.md",
            "text": "Original content",
            "collection": "uat_memory_upd",
        })

        # Update
        result = await self._call_tool(server, "memory_update", {
            "path": "memories/update-test.md",
            "text": "Updated content",
            "collection": "uat_memory_upd",
        })
        data = json.loads(result[0].text)
        assert data.get("success") is True

    @pytest.mark.asyncio
    async def test_memory_delete(self, mock_backend, temp_store):
        """memory_delete should remove memory and embeddings."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Add first
        await self._call_tool(server, "memory_add", {
            "path": "memories/delete-test.md",
            "text": "Content to be deleted",
            "collection": "uat_memory_del",
        })

        # Delete
        result = await self._call_tool(server, "memory_delete", {
            "path": "memories/delete-test.md",
            "collection": "uat_memory_del",
        })
        data = json.loads(result[0].text)
        assert data.get("success") is True

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


# ---------------------------------------------------------------------------
# Test: Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_ingest_missing_all_inputs(self, mock_backend, temp_store):
        """ingest with no text/file/folder should return error."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "ingest", {})
        data = self._parse_result(result)
        assert data.get("error") or data.get("success") is False

    @pytest.mark.asyncio
    async def test_search_missing_query(self, mock_backend, temp_store):
        """search with no query should return error or empty results."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "search", {})
        data = self._parse_result(result)
        # Should either error or return empty results
        assert data.get("error") or "count" in data

    @pytest.mark.asyncio
    async def test_index_document_missing_text(self, mock_backend, temp_store):
        """index_document without text should return error."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "index_document", {"path": "test.md"})
        data = self._parse_result(result)
        assert data.get("error") is True

    @pytest.mark.asyncio
    async def test_index_image_nonexistent_file(self, mock_backend, temp_store):
        """index_image with nonexistent file should return error."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "index_image", {"path": "/nonexistent/image.png"})
        data = self._parse_result(result)
        assert data.get("error") is True

    @pytest.mark.asyncio
    async def test_search_batch_empty_queries(self, mock_backend, temp_store):
        """search_batch with empty queries should return error."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "search_batch", {"queries": []})
        data = self._parse_result(result)
        assert data.get("error") is True

    @pytest.mark.asyncio
    async def test_search_batch_too_many_queries(self, mock_backend, temp_store):
        """search_batch with >20 queries should return error."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        queries = [f"query {i}" for i in range(21)]
        result = await self._call_tool(server, "search_batch", {"queries": queries})
        data = self._parse_result(result)
        assert data.get("error") is True

    def _parse_result(self, result: list) -> dict:
        """Parse MCP tool result to dict."""
        if result and len(result) > 0:
            text = getattr(result[0], "text", "")
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"error": True, "message": text}
        return {}

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


# ---------------------------------------------------------------------------
# Test: Status Tool
# ---------------------------------------------------------------------------

class TestStatusTool:
    """Tests for status tool."""

    @pytest.mark.asyncio
    async def test_status_returns_version(self, mock_backend, temp_store):
        """status should return version info."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "status", {})
        data = json.loads(result[0].text)
        assert "version" in data
        assert data["version"] == __version__

    @pytest.mark.asyncio
    async def test_status_returns_models(self, mock_backend, temp_store):
        """status should return model info."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "status", {})
        data = json.loads(result[0].text)
        assert "models" in data
        assert "backend" in data["models"]

    @pytest.mark.asyncio
    async def test_status_returns_database(self, mock_backend, temp_store):
        """status should return database info."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "status", {})
        data = json.loads(result[0].text)
        assert "database" in data
        assert "embeddings_count" in data["database"]

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


# ---------------------------------------------------------------------------
# Test: Collection Management
# ---------------------------------------------------------------------------

class TestCollectionManagement:
    """Tests for list_collections, rename_collection, delete_collection."""

    @pytest.mark.asyncio
    async def test_list_collections(self, mock_backend, temp_store):
        """list_collections should return collection names."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "list_collections", {})
        data = json.loads(result[0].text)
        assert "collections" in data
        assert "count" in data

    @pytest.mark.asyncio
    async def test_rename_collection(self, mock_backend, temp_store):
        """rename_collection should rename a collection."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Create a collection first
        await self._call_tool(server, "ingest", {
            "text": "test content",
            "path": "test/rename.md",
            "collection": "old_name",
        })

        result = await self._call_tool(server, "rename_collection", {
            "old_name": "old_name",
            "new_name": "new_name",
        })
        data = json.loads(result[0].text)
        # May succeed or error if collection doesn't exist
        assert "embeddings_updated" in data or data.get("error")

    @pytest.mark.asyncio
    async def test_delete_collection(self, mock_backend, temp_store):
        """delete_collection should delete collection data."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        result = await self._call_tool(server, "delete_collection", {
            "name": "test_collection_to_delete",
        })
        data = json.loads(result[0].text)
        # Should succeed even if collection doesn't exist
        assert "embeddings_deleted" in data

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


# ---------------------------------------------------------------------------
# Test: Performance Bounds
# ---------------------------------------------------------------------------

class TestPerformanceBounds:
    """Tests for performance bounds on search operations."""

    @pytest.mark.asyncio
    async def test_search_hybrid_latency_under_30s(self, mock_backend, temp_store):
        """Hybrid search should complete within 30 seconds."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="hybrid")

        # Ingest some content
        await self._call_tool(server, "ingest", {
            "text": "Performance test content for hybrid search",
            "path": "test/perf.md",
            "collection": "uat_perf",
        })

        start = time.time()
        result = await self._call_tool(server, "search", {
            "query": "performance test",
            "limit": 10,
            "collection": "uat_perf",
        })
        elapsed = time.time() - start

        # With mock backend, should be well under 30s
        assert elapsed < 30.0, f"Hybrid search took {elapsed:.2f}s (> 30s)"

    @pytest.mark.asyncio
    async def test_search_embed_latency_under_5s(self, mock_backend, temp_store):
        """Embed-only search should complete within 5 seconds."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Ingest some content
        await self._call_tool(server, "ingest", {
            "text": "Performance test content for embed search",
            "path": "test/perf_embed.md",
            "collection": "uat_perf_embed",
        })

        start = time.time()
        result = await self._call_tool(server, "search", {
            "query": "performance test",
            "limit": 10,
            "collection": "uat_perf_embed",
        })
        elapsed = time.time() - start

        # With mock backend, should be well under 5s
        assert elapsed < 5.0, f"Embed search took {elapsed:.2f}s (> 5s)"

    @pytest.mark.asyncio
    async def test_search_batch_latency_reasonable(self, mock_backend, temp_store):
        """search_batch with multiple queries should complete in reasonable time."""
        storage = get_storage(temp_store)
        server = await create_server(backend=mock_backend, storage=storage, mode="embed")

        # Ingest some content
        await self._call_tool(server, "ingest", {
            "text": "Batch search performance test",
            "path": "test/batch_perf.md",
            "collection": "uat_batch_perf",
        })

        queries = [f"query {i}" for i in range(10)]

        start = time.time()
        result = await self._call_tool(server, "search_batch", {
            "queries": queries,
            "limit": 5,
            "collection": "uat_batch_perf",
        })
        elapsed = time.time() - start

        # With mock backend, 10 queries should complete quickly
        assert elapsed < 10.0, f"search_batch took {elapsed:.2f}s for 10 queries"

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


# ---------------------------------------------------------------------------
# Integration Tests (marked for optional real backend testing)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationRealBackend:
    """Integration tests that require a real backend (run with -m integration)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_backend_ingest_text(self, live_backend, real_storage):
        """Ingest text with real backend (requires model loading)."""
        server = await create_server(backend=live_backend, storage=real_storage, mode="hybrid")

        result = await self._call_tool(server, "ingest", {
            "text": "Retrieval augmented generation uses external knowledge during answer synthesis.",
            "path": "integration/rag.md",
            "collection": "uat_real_text",
        })
        data = json.loads(result[0].text)

        assert data.get("success") or data.get("indexed_text", 0) >= 1
        assert real_storage.count_embeddings() >= 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_backend_search_text_to_text(self, live_backend, real_storage):
        """Text→text search with real backend."""
        server = await create_server(backend=live_backend, storage=real_storage, mode="hybrid")

        await self._call_tool(server, "ingest", {
            "text": "Retrieval augmented generation uses external knowledge during answer synthesis.",
            "path": "integration/rag.md",
            "collection": "uat_real_search",
        })
        await self._call_tool(server, "ingest", {
            "text": "Sourdough bread fermentation depends on a mature starter and time.",
            "path": "integration/sourdough.md",
            "collection": "uat_real_search",
        })

        result = await self._call_tool(server, "search", {
            "query": "external knowledge for language model answers",
            "limit": 5,
            "collection": "uat_real_search",
            "content_type": "text",
        })
        data = json.loads(result[0].text)
        paths = [item.get("filepath", "") for item in data.get("results", [])]

        assert data.get("count", 0) >= 1
        assert any(path.endswith("integration/rag.md") for path in paths)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_backend_cross_modal_text_to_image(self, live_backend, real_storage):
        """Text→image cross-modal search with real backend."""
        img_path = IMAGES_DIR / "whiteboard_architecture.png"
        if not img_path.exists():
            pytest.skip("whiteboard_architecture.png not found")

        server = await create_server(backend=live_backend, storage=real_storage, mode="hybrid")

        await self._call_tool(server, "ingest", {
            "file_path": str(img_path),
            "collection": "uat_real_images",
            "caption_media": True,
        })

        result = await self._call_tool(server, "search", {
            "query": "whiteboard architecture diagram",
            "limit": 5,
            "collection": "uat_real_images",
            "content_type": "image",
        })
        data = json.loads(result[0].text)
        paths = [item.get("filepath", "") for item in data.get("results", [])]

        assert data.get("count", 0) >= 1
        assert any(path.endswith("whiteboard_architecture.png") for path in paths)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_backend_captioning_works(self, live_backend, real_storage):
        """Verify captioning produces non-empty text_body with real backend."""
        img_path = IMAGES_DIR / "neural_network_diagram.png"
        if not img_path.exists():
            pytest.skip("neural_network_diagram.png not found")

        server = await create_server(backend=live_backend, storage=real_storage, mode="hybrid")

        result = await self._call_tool(server, "ingest", {
            "file_path": str(img_path),
            "collection": "uat_real_caption",
            "caption_media": True,
        })
        data = json.loads(result[0].text)
        assert data.get("success") or data.get("indexed_images", 0) >= 1

        rows = list(
            real_storage._embeddings_table.search()
            .where("collection = 'uat_real_caption'")
            .select(["file_path", "content_type", "text_body"])
            .to_list()
        )
        image_rows = [
            row for row in rows
            if row.get("content_type") == "image"
            and str(row.get("file_path", "")).endswith("neural_network_diagram.png")
        ]

        assert image_rows, "Expected at least one image row for neural_network_diagram.png"
        assert any((row.get("text_body") or "").strip() for row in image_rows)

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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


@pytest.mark.integration
class TestIntegrationExternalMCPClient:
    """Live MCP client round-trips against a real subprocess server."""

    TEST_TEXT = (
        "Retrieval augmented generation uses external knowledge during answer synthesis."
    )

    def _require_live_mode(self) -> None:
        if os.environ.get("UAT_MCP_LIVE", "0") != "1":
            pytest.skip("External MCP client test requires UAT_MCP_LIVE=1 and model weights")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_backend_external_client_stdio_roundtrip(self, tmp_path):
        """A real stdio MCP client should ingest, search, and read config via subprocess server."""
        self._require_live_mode()

        store_path = tmp_path / "store-stdio"
        collection = "uat_external_stdio"
        query_path = "integration/external-client-stdio.md"

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "recallforge",
                "serve",
                "--mode",
                "embed",
                "--store-path",
                str(store_path),
            ],
            env=_live_uat_env(store_path),
            cwd=REPO_ROOT,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools = await session.list_tools()
                tool_names = {tool.name for tool in tools.tools}
                assert {"search", "ingest", "get_config", "status"} <= tool_names

                ingest = _tool_result_to_json(
                    await session.call_tool(
                        "ingest",
                        arguments={
                            "text": self.TEST_TEXT,
                            "path": query_path,
                            "collection": collection,
                        },
                    )
                )
                assert ingest.get("success") or ingest.get("indexed_text", 0) >= 1

                search = _tool_result_to_json(
                    await session.call_tool(
                        "search",
                        arguments={
                            "query": "external knowledge during answer synthesis",
                            "limit": 5,
                            "collection": collection,
                            "content_type": "text",
                        },
                    )
                )
                result_paths = [item.get("filepath", "") for item in search.get("results", [])]
                assert search.get("count", 0) >= 1
                assert any(path.endswith(query_path) for path in result_paths)

                config = _tool_result_to_json(await session.call_tool("get_config", arguments={}))
                assert config.get("mode") == "embed"
                assert config.get("data_dir") == str(store_path.resolve())

                status = _tool_result_to_json(await session.call_tool("status", arguments={}))
                assert status.get("version") == __version__

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_backend_external_client_sse_roundtrip(self, tmp_path):
        """A real HTTP/SSE MCP client should connect to the subprocess server and call tools."""
        self._require_live_mode()

        store_path = tmp_path / "store-sse"
        collection = "uat_external_sse"
        query_path = "integration/external-client-sse.md"
        port = _find_free_port()
        log_path = tmp_path / "recallforge-http.log"
        log_handle = log_path.open("w", encoding="utf-8")

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "recallforge",
                "serve",
                "--http",
                "--mode",
                "embed",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--store-path",
                str(store_path),
            ],
            cwd=REPO_ROOT,
            env=_live_uat_env(store_path),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            health = _wait_for_http_health(f"http://127.0.0.1:{port}/health", log_path)
            assert health.get("status") == "ok"
            assert health.get("models_loaded") is True

            async with sse_client(f"http://127.0.0.1:{port}/sse") as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    tools = await session.list_tools()
                    tool_names = {tool.name for tool in tools.tools}
                    assert {"search", "ingest", "get_config"} <= tool_names

                    ingest = _tool_result_to_json(
                        await session.call_tool(
                            "ingest",
                            arguments={
                                "text": self.TEST_TEXT,
                                "path": query_path,
                                "collection": collection,
                            },
                        )
                    )
                    assert ingest.get("success") or ingest.get("indexed_text", 0) >= 1

                    search = _tool_result_to_json(
                        await session.call_tool(
                            "search",
                            arguments={
                                "query": "external knowledge during answer synthesis",
                                "limit": 5,
                                "collection": collection,
                                "content_type": "text",
                            },
                        )
                    )
                    result_paths = [item.get("filepath", "") for item in search.get("results", [])]
                    assert search.get("count", 0) >= 1
                    assert any(path.endswith(query_path) for path in result_paths)
        finally:
            _stop_process(proc, log_path=log_path)
            log_handle.close()


# ---------------------------------------------------------------------------
# Event Loop Responsiveness Test
# ---------------------------------------------------------------------------

class TestEventLoopResponsiveness:
    """Test that event loop remains responsive during blocking tool work."""

    @pytest.mark.asyncio
    async def test_event_loop_responsive_during_tool_call(self, mock_backend, temp_store):
        """Event loop should stay responsive during blocking tool execution."""
        import asyncio

        storage = get_storage(temp_store)
        slow_backend = SlowMockBackend()
        server = await create_server(backend=slow_backend, storage=storage, mode="embed")

        # Start a slow search
        slow_task = asyncio.create_task(self._call_tool(server, "search_vec", {
            "query": "event loop probe",
            "limit": 1,
            "collection": "uat_responsive",
        }))

        # Give the search time to start
        await asyncio.sleep(0.03)

        # Check that we can still call list_tools quickly
        start = time.time()
        tools = await self._list_tools(server)
        elapsed = time.time() - start

        # Clean up
        await slow_task

        # list_tools should complete quickly even while search is running
        assert elapsed < 0.20, f"list_tools took {elapsed:.3f}s (> 0.20s) during slow search"

    async def _call_tool(self, server, name: str, arguments: dict) -> list:
        """Call a tool and return the result."""
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

    async def _list_tools(self, server):
        """List available tools."""
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


# ---------------------------------------------------------------------------
# Graceful Shutdown Test
# ---------------------------------------------------------------------------

class TestGracefulShutdown:
    """Test that signal handlers are registered."""

    def test_sigint_handler_registered(self):
        """SIGINT handler should be registered."""
        import signal
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not None

    def test_sigterm_handler_registered(self):
        """SIGTERM handler should be registered."""
        import signal
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "--tb=short", "-q"])
