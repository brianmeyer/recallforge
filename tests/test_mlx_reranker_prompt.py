"""
Regression tests for MLX reranker prompt/media alignment.

These tests stay fully mocked so they run on non-Apple CI without loading MLX.
"""

import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import recallforge.backends.mlx_backend as mlx_backend


class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def numpy(self):
        return self._value


class _RecordingProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": _FakeTensor(np.array([[1, 2, 3]], dtype=np.int32)),
            "pixel_values": _FakeTensor(np.ones((1, 1, 1), dtype=np.float32)),
            "image_grid_thw": _FakeTensor(np.array([[1, 1, 1]], dtype=np.int32)),
        }


class _TokenizerOnlyChatTemplate:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "TOKENIZER_TEMPLATE"


class _ProcessorWithoutChatTemplate:
    def __init__(self):
        self.tokenizer = _TokenizerOnlyChatTemplate()

    def apply_chat_template(self, *_args, **_kwargs):
        raise ValueError("Cannot use apply_chat_template because this processor does not have a chat template.")


class _ProcessorWithChatTemplate:
    def __init__(self):
        self.calls = []
        self.tokenizer = _TokenizerOnlyChatTemplate()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "PROCESSOR_TEMPLATE"


class TestMLXRerankerPromptPreparation(unittest.TestCase):
    def _make_backend(self):
        backend = object.__new__(mlx_backend.MLXBackend)
        backend._reranker_processor = _RecordingProcessor()
        return backend

    def test_build_reranker_messages_uses_file_uris_in_query_then_document_order(self):
        backend = self._make_backend()

        messages = backend._build_reranker_messages(
            query="",
            document="doc body",
            instruction="rank this",
            image_path="docs/example.png",
            query_image_path="queries/example.png",
        )

        content = messages[1]["content"]
        image_blocks = [block for block in content if block.get("type") == "image"]

        self.assertEqual(
            [block["image"] for block in image_blocks],
            [
                f"file://{os.path.abspath('queries/example.png')}",
                f"file://{os.path.abspath('docs/example.png')}",
            ],
        )

    def test_query_side_images_use_process_vision_info_for_processor_inputs(self):
        backend = self._make_backend()
        messages = backend._build_reranker_messages(
            query="caption fallback",
            document="doc body",
            instruction="rank this",
            image_path="docs/example.png",
            query_image_path="queries/example.png",
        )

        expected_image_uris = [
            f"file://{os.path.abspath('queries/example.png')}",
            f"file://{os.path.abspath('docs/example.png')}",
        ]

        def fake_process_vision_info(conversations, return_video_kwargs=True):
            self.assertTrue(return_video_kwargs)
            self.assertEqual(len(conversations), 1)
            user_messages = conversations[0]
            self.assertEqual([m["role"] for m in user_messages], ["user"])
            content = user_messages[0]["content"]
            image_blocks = [block for block in content if block.get("type") == "image"]
            self.assertEqual([block["image"] for block in image_blocks], expected_image_uris)
            return ["query_pixels", "doc_pixels"], None, {}

        fake_module = types.SimpleNamespace(process_vision_info=fake_process_vision_info)

        with patch.dict(sys.modules, {"qwen_vl_utils": fake_module}):
            inputs = backend._build_reranker_processor_inputs("PROMPT", messages)

        self.assertIn("input_ids", inputs)
        self.assertEqual(len(backend._reranker_processor.calls), 1)
        call = backend._reranker_processor.calls[0]
        self.assertEqual(call["text"], ["PROMPT"])
        self.assertEqual(call["images"], ["query_pixels", "doc_pixels"])
        self.assertEqual(call["return_tensors"], "pt")
        self.assertTrue(call["padding"])

    def test_text_only_messages_skip_vision_preprocessing(self):
        backend = self._make_backend()
        messages = backend._build_reranker_messages(
            query="plain text query",
            document="doc body",
            instruction="rank this",
        )

        inputs = backend._build_reranker_processor_inputs("PROMPT", messages)

        self.assertIn("input_ids", inputs)
        self.assertEqual(len(backend._reranker_processor.calls), 1)
        self.assertEqual(
            backend._reranker_processor.calls[0],
            {"text": "PROMPT", "return_tensors": "np"},
        )

    def test_apply_chat_template_falls_back_to_tokenizer_template(self):
        backend = object.__new__(mlx_backend.MLXBackend)
        processor = _ProcessorWithoutChatTemplate()

        rendered = backend._apply_chat_template(
            processor,
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "TOKENIZER_TEMPLATE")
        self.assertEqual(len(processor.tokenizer.calls), 1)

    def test_apply_chat_template_prefers_processor_when_available(self):
        backend = object.__new__(mlx_backend.MLXBackend)
        processor = _ProcessorWithChatTemplate()

        rendered = backend._apply_chat_template(
            processor,
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "PROCESSOR_TEMPLATE")
        self.assertEqual(len(processor.calls), 1)
        self.assertEqual(len(processor.tokenizer.calls), 0)

    def test_resolve_heavy_op_concurrency_uses_env_and_falls_back(self):
        backend = object.__new__(mlx_backend.MLXBackend)
        backend._DEFAULT_HEAVY_OP_CONCURRENCY = 1

        with patch.dict(os.environ, {"RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY": "3"}):
            self.assertEqual(backend._resolve_heavy_op_concurrency(), 3)

        with patch.dict(os.environ, {"RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY": "0"}):
            self.assertEqual(backend._resolve_heavy_op_concurrency(), 1)

    def test_heavy_op_gate_is_reentrant_within_same_thread(self):
        backend = object.__new__(mlx_backend.MLXBackend)
        backend._DEFAULT_HEAVY_OP_CONCURRENCY = 1

        mlx_backend._HEAVY_OP_GATE = None
        mlx_backend._HEAVY_OP_GATE_LIMIT = None

        with patch.dict(os.environ, {"RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY": "1"}):
            with backend._hold_heavy_op("outer"):
                with backend._hold_heavy_op("inner"):
                    gate = backend._get_heavy_op_gate()
                    self.assertEqual(getattr(gate._thread_state, "depth", 0), 2)

            gate = backend._get_heavy_op_gate()
            self.assertEqual(getattr(gate._thread_state, "depth", 0), 0)

    def test_embed_videos_falls_back_to_frame_embeddings_when_native_path_fails(self):
        backend = object.__new__(mlx_backend.MLXBackend)
        backend._validate_video_paths = lambda paths: paths
        backend._load_embedder = lambda: None
        backend._get_embedder_num_layers = lambda: 2

        def _raise_native(_path, _num_layers):
            raise mlx_backend.MLXEmbeddingError("native failed")

        backend._embed_video_native = _raise_native
        backend._embed_video_via_frames = lambda _path: np.array([0.6, 0.8], dtype=np.float32)

        embeddings = backend.embed_videos(["clip.mp4"])

        np.testing.assert_allclose(
            embeddings,
            np.array([[0.6, 0.8]], dtype=np.float32),
        )

    def test_embed_video_via_frames_averages_and_normalizes_frame_embeddings(self):
        backend = object.__new__(mlx_backend.MLXBackend)
        backend._VIDEO_MAX_FRAMES = 128
        backend.embed_images = lambda _paths: np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        frames = [
            SimpleNamespace(image_path="frame1.png"),
            SimpleNamespace(image_path="frame2.png"),
        ]

        with patch("recallforge.video.extract_video_frames", return_value=(frames, None)):
            embedding = backend._embed_video_via_frames("clip.mp4")

        np.testing.assert_allclose(
            embedding,
            np.array([0.70710677, 0.70710677], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_resolve_heavy_op_concurrency_defaults_to_one(self):
        backend = object.__new__(mlx_backend.MLXBackend)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY", None)
            self.assertEqual(backend._resolve_heavy_op_concurrency(), 1)

    def test_resolve_heavy_op_concurrency_invalid_value_falls_back_to_one(self):
        backend = object.__new__(mlx_backend.MLXBackend)

        with patch.dict(os.environ, {"RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY": "nope"}):
            self.assertEqual(backend._resolve_heavy_op_concurrency(), 1)

    def test_heavy_op_gate_is_reentrant_on_same_thread(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            lock_path = tmp.name
        try:
            gate = mlx_backend._HeavyOpGate(limit=1, lock_path=lock_path)
            with gate.hold("outer"):
                with gate.hold("inner"):
                    self.assertEqual(getattr(gate._thread_state, "depth", 0), 2)
            self.assertFalse(hasattr(gate._thread_state, "depth"))
        finally:
            os.unlink(lock_path)

    def test_embed_videos_uses_heavy_op_guard(self):
        backend = object.__new__(mlx_backend.MLXBackend)
        calls = []

        @contextmanager
        def fake_hold(name):
            calls.append(name)
            yield

        backend._hold_heavy_op = fake_hold
        backend._validate_video_paths = lambda paths: paths
        backend._load_embedder = lambda: None
        backend._get_embedder_num_layers = lambda: 2
        backend._embed_video_native = lambda _path, _num_layers: np.array([1.0, 0.0], dtype=np.float32)

        embeddings = backend.embed_videos(["clip.mp4"])

        self.assertEqual(calls, ["embed_videos"])
        np.testing.assert_allclose(
            embeddings,
            np.array([[1.0, 0.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
