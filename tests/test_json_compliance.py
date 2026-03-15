"""JSON compliance tests for MCP tool responses (REC-88)."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mcp.types import TextContent

from recallforge.server import (
    _handle_batch,
    _handle_get_config,
    _handle_index_document,
    _handle_index_image,
    _handle_ingest,
    _handle_list_collections,
    _handle_list_namespaces,
    _handle_memory_add,
    _handle_memory_delete,
    _handle_memory_update,
    _handle_rebuild_fts,
    _handle_search,
    _handle_search_fts,
    _handle_search_vec,
    _handle_set_config,
    _handle_status,
)


class _DummyBackend:
    def __init__(self):
        self._mode = "hybrid"

    def set_mode(self, mode):
        self._mode = mode

    def get_mode(self):
        return self._mode

    def embed_text(self, _text):
        return [0.1, 0.2, 0.3]

    def embed_image(self, _path):
        return [0.1, 0.2, 0.3]

    def embed_video(self, _path):
        return [0.1, 0.2, 0.3]

    def get_info(self):
        return SimpleNamespace(
            name="dummy",
            device="cpu",
            dtype="float32",
            embedder_loaded=True,
            reranker_loaded=False,
            memory_allocated_gb=0.0,
            quantization=None,
        )


class _DummyStorage:
    def __init__(self):
        self._store_path = "/tmp/recallforge-test"

    def search_fts(self, **_kwargs):
        return []

    def search_vec(self, **_kwargs):
        return []

    def ingest(self, **_kwargs):
        return {"success": True, "indexed_text": 0, "indexed_images": 0}

    def index_document(self, **_kwargs):
        return "hash-doc"

    def index_image(self, **_kwargs):
        return "hash-img"

    def upsert_memory(self, **_kwargs):
        return "hash-mem"

    def delete_memory(self, **_kwargs):
        return {"success": True, "removed_vectors": 1}

    def index_folder(self, **_kwargs):
        return {"success": True, "indexed": 0}

    def count_embeddings(self):
        return 0

    def count_documents(self):
        return 0

    def rebuild_fts_index(self):
        return None

    def list_collections(self, **_kwargs):
        return ["default"]

    def list_namespaces(self, **_kwargs):
        return [{"user_id": None, "session_id": None, "project_id": None, "profile": None}]


class _FakeSearchResult:
    def __init__(self, filepath="/tmp/a.txt"):
        self.filepath = filepath
        self.title = "title"
        self.score = 0.9
        self.rerank_score = 0.8
        self.rrf_rank = 1
        self.source = "unit-test"
        self.body = "hello"
        self.user_id = None
        self.session_id = None
        self.project_id = None
        self.profile = None


class _FakeHybridSearcher:
    def __init__(self, **_kwargs):
        pass

    def search(self, _query):
        return [_FakeSearchResult()]

    def search_image(self, _path):
        return [_FakeSearchResult("/tmp/img.png")]

    def search_video(self, _path):
        return [_FakeSearchResult("/tmp/vid.mp4")]


class TestMCPJsonCompliance(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = _DummyBackend()
        self.storage = _DummyStorage()
        self.mutable_config = {"mode": "hybrid", "collection": "default", "max_file_size_mb": 100}

    def _assert_textcontent_json(self, response):
        self.assertIsInstance(response, list)
        self.assertGreater(len(response), 0)
        for item in response:
            self.assertIsInstance(item, TextContent)
            json.loads(item.text)

    async def test_all_tool_handlers_valid_and_invalid_calls_return_json(self):
        with patch("recallforge.server.HybridSearcher", _FakeHybridSearcher):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_path = tmp.name

            try:
                cases = [
                    (
                        lambda: _handle_search({"query": "hello"}, self.backend, self.storage),
                        lambda: _handle_search({"query": "", "image_path": "", "video_path": ""}, self.backend, self.storage),
                    ),
                    (
                        lambda: _handle_search_fts({"query": "hello"}, self.storage),
                        lambda: _handle_search_fts({"query": ""}, self.storage),
                    ),
                    (
                        lambda: _handle_search_vec({"query": "hello"}, self.backend, self.storage),
                        lambda: _handle_search_vec({"query": "", "image_path": "", "video_path": ""}, self.backend, self.storage),
                    ),
                    (
                        lambda: _handle_ingest({"text": "hello", "path": "mem/1"}, self.backend, self.storage),
                        lambda: _handle_ingest({"text": "", "path": ""}, self.backend, self.storage),
                    ),
                    (
                        lambda: _handle_index_document({"path": "doc/1", "text": "x"}, self.backend, self.storage),
                        lambda: _handle_index_document({"path": "", "text": ""}, self.backend, self.storage),
                    ),
                    (
                        lambda: _handle_index_image({"path": image_path}, self.backend, self.storage),
                        lambda: _handle_index_image({"path": "/definitely/missing.png"}, self.backend, self.storage),
                    ),
                    (
                        lambda: _handle_memory_add({"path": "mem/1", "text": "x"}, self.backend, self.storage),
                        lambda: _handle_memory_add({"path": "", "text": ""}, self.backend, self.storage),
                    ),
                    (
                        lambda: _handle_memory_update({"path": "mem/1", "text": "x"}, self.backend, self.storage),
                        lambda: _handle_memory_update({"path": "", "text": ""}, self.backend, self.storage),
                    ),
                    (
                        lambda: _handle_memory_delete({"path": "mem/1"}, self.storage),
                        lambda: _handle_memory_delete({"path": ""}, self.storage),
                    ),
                    (
                        lambda: _handle_status(self.backend, self.storage),
                        lambda: _handle_status(self.backend, self.storage),
                    ),
                    (
                        lambda: _handle_rebuild_fts(self.storage),
                        lambda: _handle_rebuild_fts(self.storage),
                    ),
                    (
                        lambda: _handle_list_collections({}, self.storage),
                        lambda: _handle_list_collections({"user_id": 123}, self.storage),
                    ),
                    (
                        lambda: _handle_list_namespaces({}, self.storage),
                        lambda: _handle_list_namespaces({"collection": 123}, self.storage),
                    ),
                    (
                        lambda: _handle_get_config(self.backend, self.storage, self.mutable_config),
                        lambda: _handle_get_config(self.backend, self.storage, self.mutable_config),
                    ),
                    (
                        lambda: _handle_set_config({"mode": "embed"}, self.backend, self.storage, self.mutable_config),
                        lambda: _handle_set_config({"backend": "forbidden"}, self.backend, self.storage, self.mutable_config),
                    ),
                    (
                        lambda: _handle_batch({"operations": [{"tool": "status", "arguments": {}}]}, self.backend, self.storage, self.mutable_config),
                        lambda: _handle_batch({"operations": "not-a-list"}, self.backend, self.storage, self.mutable_config),
                    ),
                ]

                for valid_call, invalid_call in cases:
                    valid_response = await valid_call()
                    self._assert_textcontent_json(valid_response)

                    invalid_response = await invalid_call()
                    self._assert_textcontent_json(invalid_response)
            finally:
                if os.path.exists(image_path):
                    os.unlink(image_path)

    async def test_batch_error_entries_and_edge_cases_are_strict_json(self):
        operations = [
            {"tool": "status", "arguments": {}},
            {"tool": "memory_add", "arguments": {"path": "", "text": ""}},
            {"tool": "unknown_tool", "arguments": {}},
            {"tool": "batch", "arguments": {"operations": []}},
        ]

        response = await _handle_batch(
            {"operations": operations},
            self.backend,
            self.storage,
            self.mutable_config,
        )
        self._assert_textcontent_json(response)

        payload = json.loads(response[0].text)
        self.assertEqual(payload["total"], len(operations))

        for item in payload["batch_results"]:
            self.assertIn("status", item)
            self.assertIn("result", item)
            if item["status"] == "error":
                self.assertIsInstance(item["result"], dict)
                self.assertIn("error", item["result"])

    async def test_batch_handles_empty_and_type_errors_as_json(self):
        empty_response = await _handle_batch(
            {"operations": []},
            self.backend,
            self.storage,
            self.mutable_config,
        )
        self._assert_textcontent_json(empty_response)

        missing_operations = await _handle_batch(
            {},
            self.backend,
            self.storage,
            self.mutable_config,
        )
        self._assert_textcontent_json(missing_operations)

        wrong_type = await _handle_batch(
            {"operations": {}},
            self.backend,
            self.storage,
            self.mutable_config,
        )
        self._assert_textcontent_json(wrong_type)
