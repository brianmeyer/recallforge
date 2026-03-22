"""
test_config_tools.py - Unit tests for get_config and set_config MCP tools (REC-40).

All backends/storage are mocked — no real inference, no real DB.
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mcp.types import TextContent, ListToolsRequest

from recallforge.server import (
    _dispatch_tool,
    _handle_get_config,
    _handle_list_memories,
    _handle_memory_get,
    _handle_set_config,
    create_server,
)
from recallforge import __version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeInfo:
    name = "stub-mlx"
    device = "mps"
    dtype = "float16"
    embedder_loaded = True
    reranker_loaded = False
    memory_allocated_gb = 1.0
    quantization = "4bit"


def _make_backend(mode="hybrid"):
    b = MagicMock()
    b.get_mode.return_value = mode
    b.embed_text.return_value = np.ones(128, dtype=np.float32)
    b.embed_image.return_value = np.ones(128, dtype=np.float32)
    b.get_info.return_value = _FakeInfo()
    b.set_mode = MagicMock()
    # REC-116: Model ID methods
    b.get_model_ids.return_value = {
        "embedder_model": "arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit",
        "reranker_model": "arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit",
        "captioner_model": "mlx-community/Qwen3.5-0.8B-4bit",
    }
    b.set_model_ids = MagicMock(return_value={
        "embedder_model": "arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit",
        "reranker_model": "arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit",
        "captioner_model": "mlx-community/Qwen3.5-0.8B-4bit",
    })
    return b


def _make_storage(store_path="/tmp/test-store"):
    s = MagicMock()
    s._store_path = store_path
    s.count_embeddings.return_value = 0
    s.count_documents.return_value = 0
    s.list_memories.return_value = [
        {
            "memory_id": "mem-123",
            "collection": "default",
            "path": "notes/demo.md",
            "title": "demo",
            "content_type": "text",
            "updated_at": 123,
        }
    ]
    s.get_memory.return_value = {
        "memory_id": "mem-123",
        "collection": "default",
        "path": "notes/demo.md",
        "title": "demo",
        "content_type": "text",
        "children": [],
        "snippets": [],
    }
    return s


def _mutable_config(**overrides):
    cfg = {"mode": "hybrid", "collection": "default", "max_file_size_mb": 100, "rerank_top_k": 20}
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# _handle_get_config
# ---------------------------------------------------------------------------

class TestHandleGetConfig(unittest.IsolatedAsyncioTestCase):

    async def test_returns_all_required_keys(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_get_config(backend, storage, cfg)
        self.assertEqual(len(result), 1)
        data = json.loads(result[0].text)
        for key in ("version", "backend", "mode", "quantize", "data_dir", "collection", "max_file_size_mb", "rerank_top_k"):
            self.assertIn(key, data, f"Missing key: {key}")

    async def test_version_matches_package(self):
        backend = _make_backend()
        storage = _make_storage()
        result = await _handle_get_config(backend, storage, _mutable_config())
        data = json.loads(result[0].text)
        self.assertEqual(data["version"], __version__)

    async def test_backend_from_info(self):
        backend = _make_backend()
        storage = _make_storage()
        result = await _handle_get_config(backend, storage, _mutable_config())
        data = json.loads(result[0].text)
        self.assertEqual(data["backend"], "stub-mlx")

    async def test_mode_from_mutable_config(self):
        backend = _make_backend(mode="hybrid")
        storage = _make_storage()
        cfg = _mutable_config(mode="embed")
        result = await _handle_get_config(backend, storage, cfg)
        data = json.loads(result[0].text)
        # mutable_config takes precedence
        self.assertEqual(data["mode"], "embed")

    async def test_quantize_from_info(self):
        backend = _make_backend()
        storage = _make_storage()
        result = await _handle_get_config(backend, storage, _mutable_config())
        data = json.loads(result[0].text)
        self.assertEqual(data["quantize"], "4bit")

    async def test_quantize_none_becomes_none_string(self):
        backend = _make_backend()
        backend.get_info.return_value.quantization = None
        storage = _make_storage()
        result = await _handle_get_config(backend, storage, _mutable_config())
        data = json.loads(result[0].text)
        self.assertEqual(data["quantize"], "none")

    async def test_data_dir_from_storage(self):
        backend = _make_backend()
        storage = _make_storage(store_path="/custom/path")
        result = await _handle_get_config(backend, storage, _mutable_config())
        data = json.loads(result[0].text)
        self.assertIn("/custom/path", data["data_dir"])

    async def test_collection_from_mutable_config(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config(collection="my-project")
        result = await _handle_get_config(backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["collection"], "my-project")

    async def test_max_file_size_mb_from_mutable_config(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config(max_file_size_mb=50)
        result = await _handle_get_config(backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["max_file_size_mb"], 50)

    async def test_no_required_parameters(self):
        """get_config works with empty mutable_config (uses defaults)."""
        backend = _make_backend()
        storage = _make_storage()
        result = await _handle_get_config(backend, storage, {})
        data = json.loads(result[0].text)
        self.assertEqual(data["collection"], "default")
        self.assertEqual(data["max_file_size_mb"], 100)
        self.assertEqual(data["rerank_top_k"], 20)


# ---------------------------------------------------------------------------
# _handle_set_config
# ---------------------------------------------------------------------------

class TestHandleSetConfig(unittest.IsolatedAsyncioTestCase):

    async def test_set_mode_valid(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"mode": "embed"}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["mode"], "embed")
        backend.set_mode.assert_called_once_with("embed")
        self.assertEqual(cfg["mode"], "embed")

    async def test_set_mode_hybrid(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"mode": "hybrid"}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["mode"], "hybrid")

    async def test_set_mode_full_rejected(self):
        """'full' mode was removed — should be rejected as invalid."""
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"mode": "full"}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))
        self.assertEqual(data["code"], "INVALID_INPUT")

    async def test_set_mode_invalid(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"mode": "turbo"}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))
        self.assertEqual(data["code"], "INVALID_INPUT")
        backend.set_mode.assert_not_called()

    async def test_set_collection(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"collection": "research"}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["collection"], "research")
        self.assertEqual(cfg["collection"], "research")

    async def test_set_collection_empty_string_rejected(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"collection": "   "}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))
        self.assertEqual(data["code"], "INVALID_INPUT")

    async def test_set_max_file_size_mb(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"max_file_size_mb": 250}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["max_file_size_mb"], 250)
        self.assertEqual(cfg["max_file_size_mb"], 250)

    async def test_set_max_file_size_mb_float_truncated(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"max_file_size_mb": 99.9}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["max_file_size_mb"], 99)

    async def test_set_max_file_size_mb_zero_rejected(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"max_file_size_mb": 0}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))
        self.assertEqual(data["code"], "INVALID_INPUT")

    async def test_set_max_file_size_mb_negative_rejected(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"max_file_size_mb": -5}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))

    async def test_set_rerank_top_k(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"rerank_top_k": 42}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["rerank_top_k"], 42)
        self.assertEqual(cfg["rerank_top_k"], 42)

    async def test_set_rerank_top_k_negative_rejected(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"rerank_top_k": -1}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))
        self.assertEqual(data["code"], "INVALID_INPUT")

    async def test_immutable_backend_rejected(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"backend": "torch"}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))
        self.assertEqual(data["code"], "INVALID_INPUT")
        self.assertIn("backend", data["details"]["immutable_fields"])

    async def test_immutable_quantize_rejected(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"quantize": "bf16"}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))
        self.assertIn("quantize", data["details"]["immutable_fields"])

    async def test_immutable_data_dir_rejected(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"data_dir": "/tmp/new"}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))
        self.assertIn("data_dir", data["details"]["immutable_fields"])

    async def test_unknown_field_rejected(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"turbo_mode": True}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertTrue(data.get("error"))
        self.assertEqual(data["code"], "INVALID_INPUT")

    async def test_multiple_fields_at_once(self):
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config(
            {"mode": "embed", "collection": "archive", "max_file_size_mb": 200},
            backend, storage, cfg
        )
        data = json.loads(result[0].text)
        self.assertEqual(data["mode"], "embed")
        self.assertEqual(data["collection"], "archive")
        self.assertEqual(data["max_file_size_mb"], 200)

    async def test_returns_updated_config_shape(self):
        """set_config response has same shape as get_config."""
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config()
        result = await _handle_set_config({"mode": "hybrid"}, backend, storage, cfg)
        data = json.loads(result[0].text)
        for key in ("version", "backend", "mode", "quantize", "data_dir", "collection", "max_file_size_mb", "rerank_top_k"):
            self.assertIn(key, data, f"Missing key in set_config response: {key}")

    async def test_empty_arguments_returns_current_config(self):
        """Calling set_config with no changes returns current config unchanged."""
        backend = _make_backend()
        storage = _make_storage()
        cfg = _mutable_config(mode="hybrid", collection="notes", max_file_size_mb=75)
        result = await _handle_set_config({}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["mode"], "hybrid")
        self.assertEqual(data["collection"], "notes")
        self.assertEqual(data["max_file_size_mb"], 75)
        backend.set_mode.assert_not_called()


# ---------------------------------------------------------------------------
# _dispatch_tool routing
# ---------------------------------------------------------------------------

class TestDispatchConfigTools(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.backend = _make_backend()
        self.storage = _make_storage()

    async def test_get_config_dispatched(self):
        cfg = _mutable_config()
        result = await _dispatch_tool("get_config", {}, self.backend, self.storage, cfg)
        data = json.loads(result[0].text)
        self.assertIn("version", data)
        self.assertIn("mode", data)

    async def test_set_config_dispatched(self):
        cfg = _mutable_config()
        result = await _dispatch_tool(
            "set_config", {"mode": "embed"}, self.backend, self.storage, cfg
        )
        data = json.loads(result[0].text)
        self.assertEqual(data["mode"], "embed")

    async def test_get_config_no_mutable_config_uses_defaults(self):
        """When mutable_config is None, _dispatch_tool uses empty dict (graceful)."""
        result = await _dispatch_tool("get_config", {}, self.backend, self.storage, None)
        data = json.loads(result[0].text)
        self.assertIn("version", data)
        self.assertEqual(data["collection"], "default")


# ---------------------------------------------------------------------------
# create_server: new tools appear in list_tools
# ---------------------------------------------------------------------------

class TestConfigToolsInServer(unittest.IsolatedAsyncioTestCase):

    async def _get_tool_names(self):
        backend = _make_backend()
        storage = _make_storage()
        server = await create_server(backend=backend, storage=storage)
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list", params=None))
        return [t.name for t in result.root.tools]

    async def test_get_config_registered(self):
        names = await self._get_tool_names()
        self.assertIn("get_config", names)

    async def test_set_config_registered(self):
        names = await self._get_tool_names()
        self.assertIn("set_config", names)

    async def test_explain_results_registered(self):
        names = await self._get_tool_names()
        self.assertIn("explain_results", names)

    async def test_memory_tools_registered(self):
        names = await self._get_tool_names()
        self.assertIn("memory_get", names)
        self.assertIn("list_memories", names)

    async def test_all_original_tools_still_present(self):
        names = await self._get_tool_names()
        expected = {
            "search", "search_fts", "search_vec", "ingest",
            "index_document", "index_image",
            "memory_add", "memory_update", "memory_delete",
            "status", "rebuild_fts", "batch",
            "list_collections", "list_namespaces",
            "rename_collection", "delete_collection",
            "search_batch", "get_config", "set_config", "explain_results",
        }
        missing = expected - set(names)
        self.assertFalse(missing, f"Missing tools: {missing}")


class TestMemoryTools(unittest.IsolatedAsyncioTestCase):

    async def test_list_memories_returns_json(self):
        result = await _handle_list_memories({}, _make_storage())
        data = json.loads(result[0].text)
        self.assertTrue(data["success"])
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["memories"][0]["memory_id"], "mem-123")

    async def test_memory_get_by_id_returns_json(self):
        result = await _handle_memory_get({"memory_id": "mem-123"}, _make_storage())
        data = json.loads(result[0].text)
        self.assertEqual(data["memory_id"], "mem-123")

    async def test_memory_get_by_path_uses_storage_path_lookup(self):
        storage = _make_storage()
        result = await _handle_memory_get({"path": "notes/demo.md"}, storage)
        data = json.loads(result[0].text)
        self.assertEqual(data["path"], "notes/demo.md")
        storage.list_memories.assert_not_called()
        storage.get_memory.assert_called_once()
        args, kwargs = storage.get_memory.call_args
        self.assertEqual(args, ())
        self.assertEqual(kwargs["path"], "notes/demo.md")
        self.assertEqual(kwargs["collection"], None)

    async def test_get_config_schema(self):
        backend = _make_backend()
        storage = _make_storage()
        server = await create_server(backend=backend, storage=storage)
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list", params=None))
        tool = next(t for t in result.root.tools if t.name == "get_config")
        self.assertEqual(tool.inputSchema["type"], "object")
        # No required parameters
        self.assertNotIn("required", tool.inputSchema)

    async def test_set_config_schema(self):
        backend = _make_backend()
        storage = _make_storage()
        server = await create_server(backend=backend, storage=storage)
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list", params=None))
        tool = next(t for t in result.root.tools if t.name == "set_config")
        schema = tool.inputSchema
        self.assertIn("mode", schema["properties"])
        self.assertIn("collection", schema["properties"])
        self.assertIn("max_file_size_mb", schema["properties"])
        self.assertIn("rerank_top_k", schema["properties"])
        # mode is an enum
        self.assertIn("enum", schema["properties"]["mode"])
        self.assertCountEqual(schema["properties"]["mode"]["enum"], ["embed", "hybrid"])

    async def test_explain_results_schema(self):
        backend = _make_backend()
        storage = _make_storage()
        server = await create_server(backend=backend, storage=storage)
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list", params=None))
        tool = next(t for t in result.root.tools if t.name == "explain_results")
        schema = tool.inputSchema
        self.assertEqual(schema["type"], "object")
        self.assertIn("query", schema["properties"])
        self.assertIn("image_path", schema["properties"])
        self.assertIn("video_path", schema["properties"])
        self.assertIn("rerank_top_k", schema["properties"])

    async def test_get_config_via_server_call(self):
        """Integration: get_config round-trip through create_server closure."""
        backend = _make_backend()
        storage = _make_storage()
        server = await create_server(backend=backend, storage=storage, mode="embed")
        handler = server.request_handlers[ListToolsRequest]
        # Verify tool is registered; actual call_tool requires MCP plumbing,
        # so we just confirm dispatch works directly.
        from recallforge.server import _dispatch_tool
        cfg = {"mode": "embed", "collection": "default", "max_file_size_mb": 100, "rerank_top_k": 20}
        result = await _dispatch_tool("get_config", {}, backend, storage, cfg)
        data = json.loads(result[0].text)
        self.assertEqual(data["mode"], "embed")
        self.assertEqual(data["version"], __version__)


if __name__ == "__main__":
    unittest.main()
