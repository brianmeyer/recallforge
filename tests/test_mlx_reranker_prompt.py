"""
Regression tests for MLX reranker prompt/media alignment.

These tests stay fully mocked so they run on non-Apple CI without loading MLX.
"""

import os
import sys
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
