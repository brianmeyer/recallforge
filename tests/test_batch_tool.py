"""
test_batch_tool.py - Unit tests for the batch MCP tool (REC-27).

All backends/storage are mocked — no real inference, no real DB.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

# NOTE: pyproject.toml sets pythonpath=["src"], so the local src/ is on path.
# Explicit insert kept for safety when running the file directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mcp.types import TextContent, ListToolsRequest

from recallforge.server import (
    _dispatch_tool,
    _handle_batch,
    _MAX_BATCH_SIZE,
    create_server,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeInfo:
    """Lightweight stand-in for BackendInfo so json.dumps works."""
    name = "stub-backend"
    device = "cpu"
    dtype = "float32"
    embedder_loaded = True
    reranker_loaded = False
    expander_loaded = False
    memory_allocated_gb = 0.0
    quantization = None


def _make_backend():
    b = MagicMock()
    b.get_mode.return_value = "fast"
    b.embed_text.return_value = np.ones(128, dtype=np.float32)
    b.embed_image.return_value = np.ones(128, dtype=np.float32)
    b.get_info.return_value = _FakeInfo()
    return b


def _make_storage():
    s = MagicMock()
    s.count_embeddings.return_value = 0
    s.count_documents.return_value = 0
    return s


# ---------------------------------------------------------------------------
# _dispatch_tool
# ---------------------------------------------------------------------------

class TestDispatchTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = _make_backend()
        self.storage = _make_storage()

    async def test_unknown_tool_raises(self):
        with self.assertRaises(ValueError) as ctx:
            await _dispatch_tool("nonexistent_tool", {}, self.backend, self.storage)
        self.assertIn("Unknown tool", str(ctx.exception))

    async def test_status_dispatched(self):
        result = await _dispatch_tool("status", {}, self.backend, self.storage)
        self.assertEqual(len(result), 1)
        data = json.loads(result[0].text)
        self.assertIn("version", data)

    async def test_rebuild_fts_dispatched(self):
        self.storage.rebuild_fts_index.return_value = None
        result = await _dispatch_tool("rebuild_fts", {}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("success"))

    async def test_memory_delete_dispatched(self):
        self.storage.delete_memory.return_value = {"removed_vectors": 1}
        result = await _dispatch_tool(
            "memory_delete", {"path": "test/key"}, self.backend, self.storage
        )
        data = json.loads(result[0].text)
        self.assertEqual(data.get("removed_vectors"), 1)


# ---------------------------------------------------------------------------
# _handle_batch
# ---------------------------------------------------------------------------

class TestHandleBatch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = _make_backend()
        self.storage = _make_storage()

    # --- input validation ---

    async def test_missing_operations_key(self):
        result = await _handle_batch({}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertIn("error", data)

    async def test_operations_not_list(self):
        result = await _handle_batch(
            {"operations": "not-a-list"}, self.backend, self.storage
        )
        data = json.loads(result[0].text)
        self.assertIn("error", data)

    async def test_exceeds_max_batch_size(self):
        ops = [{"tool": "status", "arguments": {}}] * (_MAX_BATCH_SIZE + 1)
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertIn("error", data)
        self.assertIn(str(_MAX_BATCH_SIZE), data["error"])

    async def test_exactly_max_batch_size_accepted(self):
        """_MAX_BATCH_SIZE operations should be accepted without error."""
        self.storage.rebuild_fts_index.return_value = None
        ops = [{"tool": "rebuild_fts", "arguments": {}}] * _MAX_BATCH_SIZE
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertNotIn("error", data)
        self.assertEqual(data["total"], _MAX_BATCH_SIZE)

    # --- nested batch rejection ---

    async def test_nested_batch_rejected(self):
        ops = [
            {"tool": "batch", "arguments": {"operations": []}},
        ]
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["failed"], 1)
        self.assertEqual(data["succeeded"], 0)
        item = data["batch_results"][0]
        self.assertEqual(item["status"], "error")
        self.assertIn("Nested batch", item["result"]["error"])

    # --- happy-path results ---

    async def test_single_success(self):
        self.storage.rebuild_fts_index.return_value = None
        ops = [{"tool": "rebuild_fts", "arguments": {}}]
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["succeeded"], 1)
        self.assertEqual(data["failed"], 0)
        item = data["batch_results"][0]
        self.assertEqual(item["index"], 0)
        self.assertEqual(item["tool"], "rebuild_fts")
        self.assertEqual(item["status"], "success")
        self.assertTrue(item["result"]["success"])

    async def test_multiple_successes(self):
        self.storage.rebuild_fts_index.return_value = None
        ops = [
            {"tool": "rebuild_fts", "arguments": {}},
            {"tool": "rebuild_fts", "arguments": {}},
        ]
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["succeeded"], 2)
        self.assertEqual(data["failed"], 0)
        self.assertEqual(data["batch_results"][0]["index"], 0)
        self.assertEqual(data["batch_results"][1]["index"], 1)

    async def test_error_in_one_does_not_stop_others(self):
        self.storage.rebuild_fts_index.return_value = None
        ops = [
            {"tool": "rebuild_fts", "arguments": {}},
            {"tool": "unknown_xyz", "arguments": {}},  # will error
            {"tool": "rebuild_fts", "arguments": {}},
        ]
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["succeeded"], 2)
        self.assertEqual(data["failed"], 1)
        self.assertEqual(data["batch_results"][0]["status"], "success")
        self.assertEqual(data["batch_results"][1]["status"], "error")
        self.assertEqual(data["batch_results"][2]["status"], "success")

    async def test_result_index_matches_position(self):
        self.storage.rebuild_fts_index.return_value = None
        ops = [{"tool": "rebuild_fts", "arguments": {}}] * 5
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        for i, item in enumerate(data["batch_results"]):
            self.assertEqual(item["index"], i)

    async def test_empty_operations(self):
        result = await _handle_batch({"operations": []}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["succeeded"], 0)
        self.assertEqual(data["failed"], 0)
        self.assertEqual(data["batch_results"], [])

    async def test_memory_add_in_batch(self):
        self.storage.upsert_memory.return_value = "abc123"
        ops = [
            {
                "tool": "memory_add",
                "arguments": {"path": "test/mem", "text": "hello world"},
            }
        ]
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertEqual(data["succeeded"], 1)
        item = data["batch_results"][0]
        self.assertEqual(item["status"], "success")
        self.assertEqual(item["result"]["path"], "test/mem")

    async def test_mixed_success_and_nested_batch(self):
        self.storage.rebuild_fts_index.return_value = None
        ops = [
            {"tool": "rebuild_fts", "arguments": {}},
            {"tool": "batch", "arguments": {"operations": []}},
        ]
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertEqual(data["succeeded"], 1)
        self.assertEqual(data["failed"], 1)

    async def test_status_in_batch(self):
        ops = [{"tool": "status", "arguments": {}}]
        result = await _handle_batch({"operations": ops}, self.backend, self.storage)
        data = json.loads(result[0].text)
        self.assertEqual(data["succeeded"], 1)
        item = data["batch_results"][0]
        self.assertEqual(item["status"], "success")
        self.assertIn("version", item["result"])


# ---------------------------------------------------------------------------
# create_server integration: batch appears in list_tools
# ---------------------------------------------------------------------------

class TestBatchInServer(unittest.IsolatedAsyncioTestCase):
    async def _get_tool_names(self):
        backend = _make_backend()
        storage = _make_storage()
        server = await create_server(backend=backend, storage=storage)
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list", params=None))
        return [t.name for t in result.root.tools]

    async def test_batch_tool_registered(self):
        names = await self._get_tool_names()
        self.assertIn("batch", names)

    async def test_all_existing_tools_still_present(self):
        names = await self._get_tool_names()
        expected = {
            "search", "search_fts", "search_vec", "ingest",
            "index_document", "index_image",
            "memory_add", "memory_update", "memory_delete",
            "index_folder", "status", "rebuild_fts", "batch",
        }
        self.assertTrue(expected.issubset(set(names)), f"Missing tools: {expected - set(names)}")

    async def test_batch_tool_schema(self):
        backend = _make_backend()
        storage = _make_storage()
        server = await create_server(backend=backend, storage=storage)
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list", params=None))
        batch_tool = next(t for t in result.root.tools if t.name == "batch")
        schema = batch_tool.inputSchema
        self.assertEqual(schema["type"], "object")
        self.assertIn("operations", schema["properties"])
        self.assertIn("operations", schema["required"])
        # operations must be an array
        self.assertEqual(schema["properties"]["operations"]["type"], "array")


if __name__ == "__main__":
    unittest.main()
