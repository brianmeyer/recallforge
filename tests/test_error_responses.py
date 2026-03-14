"""
test_error_responses.py — REC-28

Tests that MCP tool handlers return structured error responses conforming to:
  {"error": true, "code": "<CODE>", "message": "...", "details": {...}}
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recallforge.server import _error_response


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse(response) -> dict:
    """Parse the first TextContent item from a tool response into a dict."""
    assert len(response) == 1, f"Expected 1 item, got {len(response)}"
    return json.loads(response[0].text)


# ---------------------------------------------------------------------------
# _error_response helper unit tests
# ---------------------------------------------------------------------------

class TestErrorResponseHelper:
    """Unit tests for the _error_response() helper itself."""

    def test_required_keys_present(self):
        result = _parse(_error_response("INVALID_INPUT", "something bad"))
        assert "error" in result
        assert "code" in result
        assert "message" in result
        assert "details" in result

    def test_error_flag_is_true(self):
        result = _parse(_error_response("INVALID_INPUT", "something bad"))
        assert result["error"] is True

    def test_code_preserved(self):
        for code in ("INVALID_INPUT", "NOT_FOUND", "BACKEND_ERROR", "INTERNAL_ERROR"):
            result = _parse(_error_response(code, "msg"))
            assert result["code"] == code

    def test_message_preserved(self):
        msg = "Provide exactly one of: query, image_path, or video_path"
        result = _parse(_error_response("INVALID_INPUT", msg))
        assert result["message"] == msg

    def test_details_defaults_to_empty_dict(self):
        result = _parse(_error_response("NOT_FOUND", "missing"))
        assert result["details"] == {}

    def test_details_passed_through(self):
        details = {"path": "/some/file.png", "extra": 42}
        result = _parse(_error_response("NOT_FOUND", "missing", details))
        assert result["details"] == details

    def test_returns_list_of_one_text_content(self):
        from mcp.types import TextContent
        resp = _error_response("INTERNAL_ERROR", "boom")
        assert isinstance(resp, list)
        assert len(resp) == 1
        assert isinstance(resp[0], TextContent)
        assert resp[0].type == "text"

    def test_output_is_valid_json(self):
        resp = _error_response("BACKEND_ERROR", "rerank failed")
        # Should not raise
        parsed = json.loads(resp[0].text)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# _handle_search — INVALID_INPUT
# ---------------------------------------------------------------------------

class TestHandleSearchErrors:
    """_handle_search should return INVALID_INPUT when query inputs are bad."""

    @pytest.fixture
    def mocks(self):
        backend = MagicMock()
        storage = MagicMock()
        return backend, storage

    @pytest.mark.asyncio
    async def test_no_query_inputs_returns_invalid_input(self, mocks):
        from recallforge.server import _handle_search
        backend, storage = mocks
        result = _parse(await _handle_search({}, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_multiple_query_inputs_returns_invalid_input(self, mocks):
        from recallforge.server import _handle_search
        backend, storage = mocks
        args = {"query": "hello", "image_path": "/img.png"}
        result = _parse(await _handle_search(args, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_message_mentions_options(self, mocks):
        from recallforge.server import _handle_search
        backend, storage = mocks
        result = _parse(await _handle_search({}, backend, storage))
        assert "query" in result["message"] or "image_path" in result["message"]


# ---------------------------------------------------------------------------
# _handle_search_fts — INVALID_INPUT
# ---------------------------------------------------------------------------

class TestHandleSearchFtsErrors:
    @pytest.fixture
    def storage(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_empty_query_returns_invalid_input(self, storage):
        from recallforge.server import _handle_search_fts
        result = _parse(await _handle_search_fts({}, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_blank_query_returns_invalid_input(self, storage):
        from recallforge.server import _handle_search_fts
        result = _parse(await _handle_search_fts({"query": ""}, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_message_mentions_query(self, storage):
        from recallforge.server import _handle_search_fts
        result = _parse(await _handle_search_fts({}, storage))
        assert "query" in result["message"].lower() or "required" in result["message"].lower()


# ---------------------------------------------------------------------------
# _handle_search_vec — INVALID_INPUT + NOT_FOUND
# ---------------------------------------------------------------------------

class TestHandleSearchVecErrors:
    @pytest.fixture
    def mocks(self):
        backend = MagicMock()
        storage = MagicMock()
        return backend, storage

    @pytest.mark.asyncio
    async def test_no_inputs_returns_invalid_input(self, mocks):
        from recallforge.server import _handle_search_vec
        backend, storage = mocks
        result = _parse(await _handle_search_vec({}, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_conflicting_inputs_returns_invalid_input(self, mocks):
        from recallforge.server import _handle_search_vec
        backend, storage = mocks
        args = {"query": "hello", "video_path": "/v.mp4"}
        result = _parse(await _handle_search_vec(args, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_video_unsupported_backend_returns_not_found(self, mocks):
        from recallforge.server import _handle_search_vec
        backend, storage = mocks
        # Remove embed_video so the backend doesn't support it
        del backend.embed_video
        args = {"video_path": "/some/video.mp4"}
        result = _parse(await _handle_search_vec(args, backend, storage))
        assert result["error"] is True
        assert result["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# _handle_index_image — INVALID_INPUT + NOT_FOUND
# ---------------------------------------------------------------------------

class TestHandleIndexImageErrors:
    @pytest.fixture
    def mocks(self):
        return MagicMock(), MagicMock()

    @pytest.mark.asyncio
    async def test_missing_path_returns_invalid_input(self, mocks):
        from recallforge.server import _handle_index_image
        backend, storage = mocks
        result = _parse(await _handle_index_image({}, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_not_found(self, mocks, tmp_path):
        from recallforge.server import _handle_index_image
        backend, storage = mocks
        missing = str(tmp_path / "does_not_exist.png")
        result = _parse(await _handle_index_image({"path": missing}, backend, storage))
        assert result["error"] is True
        assert result["code"] == "NOT_FOUND"
        assert missing in result["details"].get("path", "") or missing in result["message"]


# ---------------------------------------------------------------------------
# _handle_index_document — INVALID_INPUT
# ---------------------------------------------------------------------------

class TestHandleIndexDocumentErrors:
    @pytest.fixture
    def mocks(self):
        return MagicMock(), MagicMock()

    @pytest.mark.asyncio
    async def test_missing_path_and_text_returns_invalid_input(self, mocks):
        from recallforge.server import _handle_index_document
        backend, storage = mocks
        result = _parse(await _handle_index_document({}, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_missing_text_returns_invalid_input(self, mocks):
        from recallforge.server import _handle_index_document
        backend, storage = mocks
        result = _parse(await _handle_index_document({"path": "doc.txt"}, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_missing_path_returns_invalid_input(self, mocks):
        from recallforge.server import _handle_index_document
        backend, storage = mocks
        result = _parse(await _handle_index_document({"text": "hello"}, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"


# ---------------------------------------------------------------------------
# _handle_memory_add / update / delete — INVALID_INPUT
# ---------------------------------------------------------------------------

class TestHandleMemoryErrors:
    @pytest.fixture
    def mocks(self):
        return MagicMock(), MagicMock()

    @pytest.mark.asyncio
    async def test_memory_add_missing_fields(self, mocks):
        from recallforge.server import _handle_memory_add
        backend, storage = mocks
        result = _parse(await _handle_memory_add({}, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_memory_update_missing_fields(self, mocks):
        from recallforge.server import _handle_memory_update
        backend, storage = mocks
        result = _parse(await _handle_memory_update({}, backend, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"

    @pytest.mark.asyncio
    async def test_memory_delete_missing_path(self, mocks):
        from recallforge.server import _handle_memory_delete
        backend, storage = mocks
        result = _parse(await _handle_memory_delete({}, storage))
        assert result["error"] is True
        assert result["code"] == "INVALID_INPUT"


# ---------------------------------------------------------------------------
# _handle_rebuild_fts — BACKEND_ERROR
# ---------------------------------------------------------------------------

class TestHandleRebuildFtsErrors:
    @pytest.mark.asyncio
    async def test_storage_exception_returns_backend_error(self):
        from recallforge.server import _handle_rebuild_fts
        storage = MagicMock()
        storage.rebuild_fts_index.side_effect = RuntimeError("index corrupted")
        result = _parse(await _handle_rebuild_fts(storage))
        assert result["error"] is True
        assert result["code"] == "BACKEND_ERROR"
        assert "index corrupted" in result["message"]
