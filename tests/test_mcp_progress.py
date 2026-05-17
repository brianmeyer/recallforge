"""
test_mcp_progress.py - Progress notification coverage for MCP tool handlers.

These tests use mocked backends/storage and a fake progress sink. They verify
that handlers emit protocol-ready progress updates without needing a live MCP
HTTP/SSE client.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
from mcp.shared.memory import create_connected_server_and_client_session

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recallforge.server import (
    _ToolProgressReporter,
    create_server,
    _handle_batch,
    _handle_ingest,
    _handle_search_batch,
)
from recallforge.storage.base import SearchResult


class _FakeInfo:
    name = "stub-backend"
    device = "cpu"
    dtype = "float32"
    embedder_loaded = True
    reranker_loaded = False
    memory_allocated_gb = 0.0
    quantization = None


class ProgressRecorder:
    def __init__(self):
        self.events = []

    @property
    def reporter(self):
        return _ToolProgressReporter(self.send)

    async def send(self, progress, total, message):
        self.events.append(
            {
                "progress": progress,
                "total": total,
                "message": message,
            }
        )


def _make_backend():
    backend = MagicMock()
    backend.get_mode.return_value = "embed"
    backend.embed_text.return_value = np.ones(8, dtype=np.float32)
    backend.embed_image.return_value = np.ones(8, dtype=np.float32)
    backend.get_info.return_value = _FakeInfo()
    backend.needs_reranker.return_value = False
    return backend


def _result(path: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        filepath=path,
        display_path=path,
        title=path,
        context=None,
        hash=f"hash-{path}",
        docid=f"doc-{path}",
        collection="default",
        modified_at="now",
        body_length=20,
        score=score,
        source="fts",
        body=f"Body for {path}",
    )


def _make_storage():
    storage = MagicMock()
    storage.count_embeddings.return_value = 0
    storage.count_documents.return_value = 0
    storage.delete_memory.return_value = {"success": True, "removed_vectors": 1}
    storage.ingest.return_value = {
        "success": True,
        "indexed_text": 1,
        "indexed_images": 0,
        "indexed_videos": 0,
        "indexed_audio": 0,
        "indexed_documents": 0,
    }
    storage.search_fts.side_effect = [
        [_result("alpha.md", 0.9)],
        [_result("beta.md", 0.8)],
    ]
    return storage


class TestMcpProgress(unittest.IsolatedAsyncioTestCase):
    async def test_batch_emits_operation_progress(self):
        backend = _make_backend()
        storage = _make_storage()
        recorder = ProgressRecorder()

        result = await _handle_batch(
            {
                "operations": [
                    {"tool": "status", "arguments": {}},
                    {"tool": "memory_delete", "arguments": {"path": "notes/demo.md"}},
                ]
            },
            backend,
            storage,
            progress=recorder.reporter,
        )

        data = json.loads(result[0].text)
        self.assertEqual(data["succeeded"], 2)
        messages = [event["message"] for event in recorder.events]
        self.assertIn("Starting batch with 2 operation(s)", messages)
        self.assertTrue(any("Finished batch operation 1/2: status (success)" == msg for msg in messages))
        self.assertTrue(any("Finished batch operation 2/2: memory_delete (success)" == msg for msg in messages))
        self.assertEqual(recorder.events[-1]["progress"], recorder.events[-1]["total"])

    async def test_ingest_emits_start_and_completion_progress(self):
        backend = _make_backend()
        storage = _make_storage()
        recorder = ProgressRecorder()

        result = await _handle_ingest(
            {"text": "hello", "path": "notes/hello.md", "collection": "default"},
            backend,
            storage,
            progress=recorder.reporter,
        )

        data = json.loads(result[0].text)
        self.assertTrue(data["success"])
        self.assertEqual([event["progress"] for event in recorder.events], [0.0, 2.0])
        self.assertIn("Ingest complete; indexed 1 item(s)", recorder.events[-1]["message"])

    async def test_search_batch_emits_per_query_partial_progress(self):
        backend = _make_backend()
        storage = _make_storage()
        recorder = ProgressRecorder()

        result = await _handle_search_batch(
            {
                "queries": [
                    {"query": "alpha", "mode": "fts"},
                    {"query": "beta", "mode": "fts"},
                ],
                "limit": 5,
            },
            backend,
            storage,
            progress=recorder.reporter,
        )
        await asyncio.sleep(0.05)

        data = json.loads(result[0].text)
        self.assertEqual(data["query_count"], 2)
        partial_messages = [
            event["message"]
            for event in recorder.events
            if "Batch search completed query" in event["message"]
        ]
        self.assertEqual(len(partial_messages), 2)
        self.assertTrue(any("last branch returned 1 candidate(s)" in msg for msg in partial_messages))
        self.assertEqual(recorder.events[-1]["progress"], 2.0)
        self.assertEqual(recorder.events[-1]["total"], 2.0)

    async def test_client_session_receives_progress_notifications(self):
        backend = _make_backend()
        storage = _make_storage()
        server = await create_server(backend=backend, storage=storage, mode="embed")
        events = []

        async def on_progress(progress, total, message):
            events.append((progress, total, message))

        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(
                "ingest",
                arguments={"text": "hello", "path": "notes/hello.md", "collection": "default"},
                progress_callback=on_progress,
            )

        data = json.loads(result.content[0].text)
        self.assertTrue(data["success"])
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0][0], 0.0)
        self.assertIn("Starting ingest", events[0][2])
        self.assertEqual(events[-1][0], events[-1][1])
        self.assertIn("Ingest complete", events[-1][2])


if __name__ == "__main__":
    unittest.main()
