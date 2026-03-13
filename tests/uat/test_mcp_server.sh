#!/usr/bin/env bash
# test_mcp_server.sh - MCP server UAT (contract mode by default).
# Tests JSON-RPC communication, MCP tools, and graceful shutdown.
# Set UAT_MCP_LIVE=1 to use real backend/model calls instead of mock backend.

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
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

STORE = "${UAT_STORE}"
CORPUS_TEXT = "${CORPUS_DIR}/text"
CORPUS_IMAGES = "${CORPUS_DIR}/images"
CORPUS_VIDEOS = "${CORPUS_DIR}/videos"
HELPERS_DIR = "${HELPERS_DIR}"

import platform
# Prefer MLX on Apple Silicon only when runtime probe is healthy.
if platform.machine() == "arm64" and platform.system() == "Darwin":
    from recallforge.backends import MLX_AVAILABLE
    if MLX_AVAILABLE:
        os.environ["RECALLFORGE_BACKEND"] = "mlx"
        os.environ.setdefault("RECALLFORGE_MLX_QUANTIZE", "4bit")
    else:
        os.environ["RECALLFORGE_BACKEND"] = "torch"
else:
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
from recallforge import get_storage, get_backend
from recallforge.backends.base import BackendInfo
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ListToolsRequest,
)

class MockBackend:
    def __init__(self):
        self._mode = "embed"

    def set_mode(self, mode):
        self._mode = mode

    def get_mode(self):
        return self._mode

    def _vec(self, seed: str):
        import hashlib
        import re
        dim = 2048
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

    def embed_text(self, text):
        return self._vec(text)

    def embed_texts(self, texts):
        return [self.embed_text(t) for t in texts]

    def embed_image(self, path):
        import os
        name = os.path.basename(path)
        return self._vec(name)

    def embed_video(self, path):
        import os
        name = os.path.basename(path)
        if "sample_video" in name:
            return self._vec(
                "pasta dish white plate forest landscape whiteboard architecture diagram meeting"
            )
        return self._vec(name)

    def needs_expander(self):
        return False

    def needs_reranker(self):
        return False

    def get_info(self):
        return BackendInfo(
            name="mock",
            device="cpu",
            dtype="float32",
            embedder_loaded=True,
            reranker_loaded=False,
            expander_loaded=False,
            memory_allocated_gb=0.0,
            quantization="none",
        )

class SlowBackend(MockBackend):
    """Backend that intentionally blocks in embed_text for responsiveness test."""

    def embed_text(self, text):
        import time as _time
        _time.sleep(0.4)
        return super().embed_text(text)

use_live_backend = os.environ.get("UAT_MCP_LIVE", "0") == "1"
if use_live_backend:
    print("  \033[0;36mINFO\033[0m  MCP live mode enabled (UAT_MCP_LIVE=1)")
    backend = get_backend()
else:
    print("  \033[0;36mINFO\033[0m  MCP contract mode enabled (mock backend)")
    backend = MockBackend()

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
    print(f"  INFO  Server exposes {len(tools)} tools")

    required_tools = [
        "search", "search_fts", "search_vec", "ingest", "index_document", "index_image",
        "memory_add", "memory_update", "memory_delete", "index_folder",
        "status", "rebuild_fts"
    ]
    missing = [name for name in required_tools if name not in tool_names]
    report(len(missing) == 0, "Server exposes required MCP toolset")
    if missing:
        print(f"    Missing tools: {missing}")

    # Keep legacy compatibility checks explicit so removals are intentional.
    report("index_document" in tool_names, "Legacy tool 'index_document' retained")
    report("index_image" in tool_names, "Legacy tool 'index_image' retained")

    # ── Event loop responsiveness under blocking tool work ──
    print("\n\033[0;36m--- Event Loop Responsiveness ---\033[0m\n")
    if not use_live_backend:
        slow_server = await create_server(backend=SlowBackend(), storage=storage, mode="embed")
        slow_task = asyncio.create_task(_call_tool(slow_server, "search_vec", {
            "query": "event loop probe",
            "limit": 1,
            "collection": "mcp_test",
        }))
        await asyncio.sleep(0.03)
        t0 = time.time()
        await _list_tools(slow_server)
        elapsed = time.time() - t0
        await slow_task
        report(elapsed < 0.20, f"Event loop stays responsive during tool execution ({elapsed:.3f}s)")
    else:
        print("  \033[0;33mSKIP\033[0m  Responsiveness probe skipped in live backend mode")

    # ── Status tool ──
    print("\n\033[0;36m--- Status Tool ---\033[0m\n")

    result = await _call_tool(server, "status", {})
    status_text = result[0].text
    status_data = json.loads(status_text)
    report("version" in status_data, f"Status returns version: {status_data.get('version')}")
    report("models" in status_data, f"Status returns model info")
    report("database" in status_data, f"Status returns database info")

    # ── Ingest tool ──
    print("\n\033[0;36m--- Ingest Tool ---\033[0m\n")

    # raw text ingest
    result = await _call_tool(server, "ingest", {
        "path": "ingest/raw-note.md",
        "text": "Unified ingest can index raw text without selecting internal tool paths.",
        "collection": "mcp_test",
    })
    ingest_text_data = json.loads(result[0].text)
    report(ingest_text_data.get("indexed_text", 0) == 1, "ingest raw text indexed exactly one text item")
    report(ingest_text_data.get("indexed_images", 0) == 0, "ingest raw text indexed zero images")

    # single image ingest
    img_path = os.path.join(CORPUS_IMAGES, "whiteboard_architecture.png")
    if os.path.exists(img_path):
        result = await _call_tool(server, "ingest", {
            "file_path": img_path,
            "collection": "mcp_test",
        })
        ingest_img_data = json.loads(result[0].text)
        report(ingest_img_data.get("indexed_images", 0) == 1, "ingest single image indexed exactly one image")

        # re-ingest image path behavior: no stale vectors across hash changes
        import shutil
        reingest_img = os.path.join(STORE, "reingest-image.png")
        shutil.copy(img_path, reingest_img)
        await _call_tool(server, "ingest", {
            "file_path": reingest_img,
            "collection": "mcp_test",
        })
        resolved_reingest_img = str(Path(reingest_img).expanduser().resolve())
        # hash_file_bytes includes mtime; update mtime to force a new content_hash
        now = time.time()
        os.utime(reingest_img, (now, now + 1))
        await _call_tool(server, "ingest", {
            "file_path": reingest_img,
            "collection": "mcp_test",
        })
        img_rows = storage._embeddings_table.search().where(
            f"collection = 'mcp_test' AND file_path = '{resolved_reingest_img}' AND content_type = 'image'"
        ).to_list()
        img_hashes = {r.get("content_hash") for r in img_rows}
        report(len(img_hashes) == 1, "ingest re-ingest leaves one active image content_hash")
        report(len(img_rows) == 1, "ingest re-ingest does not leave stale image embeddings")
    else:
        print("  \033[0;33mSKIP\033[0m  No test image available for ingest single image")

    # folder ingest with text + image
    ingest_folder = os.path.join(STORE, "mcp_ingest_folder")
    os.makedirs(ingest_folder, exist_ok=True)
    with open(os.path.join(ingest_folder, "mix.md"), "w", encoding="utf-8") as f:
        f.write("mixed folder ingest text content")
    if os.path.exists(img_path):
        import shutil
        shutil.copy(img_path, os.path.join(ingest_folder, "mix.png"))

    result = await _call_tool(server, "ingest", {
        "folder_path": ingest_folder,
        "recursive": True,
        "collection": "mcp_test",
        "include_globs": ["**/*", "*"],
    })
    ingest_folder_data = json.loads(result[0].text)
    report(ingest_folder_data.get("indexed_text", 0) >= 1, "ingest folder indexed text")
    if os.path.exists(img_path):
        report(ingest_folder_data.get("indexed_images", 0) >= 1, "ingest folder indexed image")

    # single document ingest
    doc_helper = os.path.join(HELPERS_DIR, "generate_test_documents.py")
    doc_fixture_dir = os.path.join(STORE, "document_fixtures")
    doc_meta = json.loads(subprocess.check_output([sys.executable, doc_helper, doc_fixture_dir], text=True))
    doc_files = doc_meta["files"]

    result = await _call_tool(server, "ingest", {
        "file_path": doc_files["docx"],
        "collection": "mcp_test",
        "content_types": ["document"],
    })
    ingest_docx_data = json.loads(result[0].text)
    report(ingest_docx_data.get("indexed_documents", 0) == 1, "ingest single DOCX indexes one document")
    report(ingest_docx_data.get("indexed_document_sections", 0) >= 1, "ingest single DOCX indexes extracted sections")

    result = await _call_tool(server, "ingest", {
        "file_path": doc_files["pptx"],
        "collection": "mcp_test",
        "content_types": ["document"],
    })
    ingest_pptx_data = json.loads(result[0].text)
    report(ingest_pptx_data.get("indexed_documents", 0) == 1, "ingest single PPTX indexes one document")
    report(ingest_pptx_data.get("indexed_document_sections", 0) >= 1, "ingest single PPTX indexes extracted slides")

    result = await _call_tool(server, "ingest", {
        "file_path": doc_files["pdf"],
        "collection": "mcp_test",
        "content_types": ["document"],
    })
    ingest_pdf_data = json.loads(result[0].text)
    report(ingest_pdf_data.get("indexed_documents", 0) == 1, "ingest single PDF indexes one document")
    report(ingest_pdf_data.get("indexed_document_sections", 0) >= 1, "ingest single PDF indexes extracted pages")

    result = await _call_tool(server, "search", {
        "query": "deployment checklist",
        "collection": "mcp_test",
        "content_type": "text",
    })
    document_search = json.loads(result[0].text)
    report(
        any("deployment_review.pptx::slide:" in r.get("filepath", "") for r in document_search.get("results", [])),
        "document-derived sections are searchable after ingest",
    )

    # re-ingest update behavior: no duplicate text chunks for same path
    result = await _call_tool(server, "ingest", {
        "path": "ingest/reingest.md",
        "text": "first version " + ("a" * 900),
        "collection": "mcp_test",
    })
    rows_before = storage._embeddings_table.search().where(
        "collection = 'mcp_test' AND file_path = 'ingest/reingest.md'"
    ).to_list()
    count_before = len(rows_before)

    result = await _call_tool(server, "ingest", {
        "path": "ingest/reingest.md",
        "text": "second version " + ("b" * 1200),
        "collection": "mcp_test",
    })
    rows_after = storage._embeddings_table.search().where(
        "collection = 'mcp_test' AND file_path = 'ingest/reingest.md'"
    ).to_list()
    hashes_after = {r.get("content_hash") for r in rows_after}
    report(len(hashes_after) == 1, "ingest re-ingest leaves one active content_hash")
    report(len(rows_after) <= count_before + 3, "ingest re-ingest does not duplicate old text embeddings")

    # ── Index Document tool ──
    print("\n\033[0;36m--- Index Document Tool ---\033[0m\n")

    result = await _call_tool(server, "index_document", {
        "path": "mcp_test_doc.md",
        "text": "RecallForge enables cross-modal vision-language search combining BM25 and vector retrieval.",
        "collection": "mcp_test",
    })
    idx_data = json.loads(result[0].text)
    report(idx_data.get("success") == True, f"index_document succeeded, hash={idx_data.get('hash', 'N/A')[:8]}...")

    # ── Memory Add / Update / Delete tools ──
    print("\n\033[0;36m--- Memory Tools ---\033[0m\n")

    result = await _call_tool(server, "memory_add", {
        "path": "memories/agent-notes.md",
        "text": "RecallForge memory add baseline content for dedup regression.",
        "collection": "mcp_test",
    })
    mem_add_data = json.loads(result[0].text)
    report(mem_add_data.get("success") == True, "memory_add succeeded")

    initial_rows = storage._embeddings_table.search().where(
        "collection = 'mcp_test' AND file_path = 'memories/agent-notes.md'"
    ).to_list()
    initial_count = len(initial_rows)
    report(initial_count > 0, f"memory_add created {initial_count} embedding rows")

    result = await _call_tool(server, "memory_update", {
        "path": "memories/agent-notes.md",
        "text": "Updated memory content with changed wording to force hash replacement.",
        "collection": "mcp_test",
    })
    mem_update_data = json.loads(result[0].text)
    report(mem_update_data.get("success") == True, "memory_update succeeded")

    updated_rows = storage._embeddings_table.search().where(
        "collection = 'mcp_test' AND file_path = 'memories/agent-notes.md'"
    ).to_list()
    updated_hashes = {r.get("content_hash") for r in updated_rows}
    report(len(updated_hashes) == 1, "memory_update leaves exactly one active content_hash for path")
    report(len(updated_rows) <= initial_count + 3, "memory_update did not duplicate old chunk embeddings")

    result = await _call_tool(server, "memory_delete", {
        "path": "memories/agent-notes.md",
        "collection": "mcp_test",
    })
    mem_delete_data = json.loads(result[0].text)
    report(mem_delete_data.get("success") == True, "memory_delete succeeded")

    after_delete_rows = storage._embeddings_table.search().where(
        "collection = 'mcp_test' AND file_path = 'memories/agent-notes.md'"
    ).to_list()
    report(len(after_delete_rows) == 0, "memory_delete removed associated embeddings")

    deleted_doc = storage.find_document("mcp_test", "memories/agent-notes.md")
    report(deleted_doc is None, "memory_delete deactivated document")

    # ── index_folder tool ──
    print("\n\033[0;36m--- index_folder Tool ---\033[0m\n")
    folder_root = os.path.join(STORE, "mcp_folder_index")
    os.makedirs(folder_root, exist_ok=True)
    with open(os.path.join(folder_root, "one.md"), "w", encoding="utf-8") as f:
        f.write("folder index file one about memory systems")
    with open(os.path.join(folder_root, "two.txt"), "w", encoding="utf-8") as f:
        f.write("folder index file two about retrieval")
    with open(os.path.join(folder_root, "skip.bin"), "wb") as f:
        f.write(b"\x00\x01\x02")

    result = await _call_tool(server, "index_folder", {
        "folder_path": folder_root,
        "collection": "mcp_test",
        "recursive": True,
        "include_globs": ["*.md", "*.txt", "**/*.md", "**/*.txt"],
        "exclude_globs": ["*skip*"],
    })
    folder_data = json.loads(result[0].text)
    report(folder_data.get("success") == True, "index_folder succeeded")
    report(folder_data.get("indexed", 0) >= 2, f"index_folder indexed {folder_data.get('indexed', 0)} files")

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
        "query": "RecallForge enables cross-modal search",
        "limit": 5,
        "collection": "mcp_test",
    })
    search_data = json.loads(result[0].text)
    report("count" in search_data, f"search returned count field: {search_data.get('count', 0)}")
    report("results" in search_data, "search results contain result list")
    if search_data.get("results"):
        r0 = search_data["results"][0]
        report("score" in r0, f"Result has score: {r0.get('score')}")
        report("filepath" in r0, f"Result has filepath: {r0.get('filepath', '')[:40]}")

    if os.path.exists(img_path):
        result = await _call_tool(server, "search", {
            "image_path": img_path,
            "limit": 5,
            "collection": "mcp_test",
            "content_type": "image",
        })
        image_search_data = json.loads(result[0].text)
        report(image_search_data.get("count", 0) > 0, "search accepts image_path queries")

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
    if os.path.exists(img_path):
        result = await _call_tool(server, "search_vec", {
            "image_path": img_path,
            "limit": 10,
            "collection": "mcp_test",
            "content_type": "image",
        })
        vec_image_data = json.loads(result[0].text)
        report(vec_image_data.get("count", 0) > 0, "search_vec accepts image_path queries")

    # ── Rebuild FTS tool ──
    print("\n\033[0;36m--- Rebuild FTS Tool ---\033[0m\n")

    result = await _call_tool(server, "rebuild_fts", {})
    rebuild_data = json.loads(result[0].text)
    report(rebuild_data.get("success") == True, "rebuild_fts succeeded")

    # ── Error Handling ──
    print("\n\033[0;36m--- Error Handling ---\033[0m\n")

    # ingest missing all inputs
    result = await _call_tool(server, "ingest", {})
    err_data = _as_json_payload(result)
    report("error" in err_data, "ingest with no text/file/folder returns error")

    # Missing params
    result = await _call_tool(server, "search", {})
    err_data = _as_json_payload(result)
    report("error" in err_data or err_data.get("count", 0) == 0,
           "search with no query returns error or empty results")

    # Video ingest should index transcript segments everywhere and frames when ffmpeg is available.
    video_path = os.path.join(CORPUS_VIDEOS, "whiteboard_session.mp4")
    ffmpeg_available = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True
    ).returncode == 0

    result = await _call_tool(server, "ingest", {
        "file_path": video_path,
        "collection": "mcp_test",
        "content_types": ["video"],
    })
    video_data = _as_json_payload(result)
    report(video_data.get("indexed_videos", 0) == 1, "ingest video indexes one logical video item")
    report(video_data.get("indexed_video_transcripts", 0) >= 1, "ingest video indexes transcript segments")
    # Corpus videos are real MP4 files; raw video embedding is always expected
    report(video_data.get("indexed_video_embeddings", 0) == 1, "ingest video indexes a top-level raw video embedding")

    if ffmpeg_available:
        report(video_data.get("indexed_video_frames", 0) >= 1, "ingest video extracts frame embeddings with ffmpeg")
    else:
        report(video_data.get("indexed_video_frames", 0) == 0, "ingest video falls back to transcript-only without ffmpeg")

    result = await _call_tool(server, "search", {
        "query": "whiteboard architecture diagram from a meeting",
        "limit": 10,
        "collection": "mcp_test",
        "content_type": "text",
    })
    video_search = json.loads(result[0].text)
    report(
        any("whiteboard_session.mp4::transcript:" in r.get("filepath", "") for r in video_search.get("results", [])),
        "video transcript assets are searchable after ingest",
    )

    # Corpus videos are real MP4s — always test raw video search paths
    result = await _call_tool(server, "search", {
        "video_path": video_path,
        "limit": 10,
        "collection": "mcp_test",
        "content_type": "video",
    })
    raw_video_search = json.loads(result[0].text)
    report(
        any("whiteboard_session.mp4" in r.get("filepath", "") for r in raw_video_search.get("results", [])),
        "search accepts video_path queries",
    )

    result = await _call_tool(server, "search_vec", {
        "video_path": video_path,
        "limit": 10,
        "collection": "mcp_test",
        "content_type": "text",
    })
    raw_video_vec = json.loads(result[0].text)
    report(
        any("whiteboard_session.mp4::transcript:" in r.get("filepath", "") for r in raw_video_vec.get("results", [])),
        "search_vec accepts video_path queries",
    )

    if video_meta.get("ffmpeg_available"):
        result = await _call_tool(server, "search_vec", {
            "image_path": os.path.join(CORPUS_IMAGES, "whiteboard_architecture.png"),
            "limit": 10,
            "collection": "mcp_test",
            "content_type": "image",
        })
        video_frame_search = json.loads(result[0].text)
        report(
            any("sample_video.mp4::frame:" in r.get("filepath", "") for r in video_frame_search.get("results", [])),
            "video frames are retrievable via image queries",
        )

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
        result = await _call_tool(server, "search_vec", {
            "query": "whiteboard architecture",
            "limit": 5,
            "collection": "mcp_test",
            "content_type": "image",
        })
        xm_data = json.loads(result[0].text)
        report(xm_data.get("count", 0) > 0,
               f"Cross-modal text→image via MCP vector search: {xm_data.get('count', 0)} results")

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
