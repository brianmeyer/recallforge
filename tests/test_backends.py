"""
test_backends.py - Unit tests for ModelBackend ABC and implementations.

Tests the backend interface contract, tiered modes, and BackendInfo dataclass.
Uses mock/stub implementations — NO real model inference.
"""

import os
import sys
import threading
import time
import unittest
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recallforge.backends.base import ModelBackend, BackendInfo


# ---------------------------------------------------------------------------
# Minimal concrete backend for testing the ABC contract
# ---------------------------------------------------------------------------

class StubBackend(ModelBackend):
    """A minimal concrete ModelBackend for unit testing."""

    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(2048, dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        return [np.ones(2048, dtype=np.float32) for _ in texts]

    def embed_image(self, image_path: str) -> np.ndarray:
        return np.ones(2048, dtype=np.float32)

    def embed_images(self, image_paths: List[str]) -> List[np.ndarray]:
        return [np.ones(2048, dtype=np.float32) for _ in image_paths]

    def embed_video(self, video_path: str) -> np.ndarray:
        return np.ones(2048, dtype=np.float32)

    def embed_videos(self, video_paths: List[str]) -> List[np.ndarray]:
        return [np.ones(2048, dtype=np.float32) for _ in video_paths]

    def rerank(self, query: str, documents: List[Dict[str, Any]], **kwargs) -> List[float]:
        return [0.9 - i * 0.1 for i in range(len(documents))]

    def warm_up(self) -> None:
        pass

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            name="stub",
            device="cpu",
            dtype="float32",
            embedder_loaded=True,
            reranker_loaded=True,
            memory_allocated_gb=0.0,
            supports_images=True,
            quantization=None,
        )


class TestBackendABC(unittest.TestCase):
    """Tests that the ModelBackend ABC contract is properly enforced."""

    def test_cannot_instantiate_abstract_backend(self):
        """ModelBackend cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            ModelBackend()

    def test_stub_implements_abc(self):
        """StubBackend satisfies the ABC."""
        backend = StubBackend()
        self.assertIsInstance(backend, ModelBackend)

    def test_embed_text_returns_ndarray(self):
        backend = StubBackend()
        result = backend.embed_text("hello world")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2048,))

    def test_embed_texts_batch_shape(self):
        backend = StubBackend()
        texts = ["one", "two", "three"]
        result = backend.embed_texts(texts)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_embed_image_returns_ndarray(self):
        backend = StubBackend()
        result = backend.embed_image("/fake/image.png")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2048,))

    def test_rerank_returns_scores_list(self):
        backend = StubBackend()
        docs = [
            {"text": "first doc", "filepath": "/a.txt"},
            {"text": "second doc", "filepath": "/b.txt"},
        ]
        scores = backend.rerank("test query", docs)
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertIsInstance(s, float)

    def test_get_info_returns_backend_info(self):
        backend = StubBackend()
        info = backend.get_info()
        self.assertIsInstance(info, BackendInfo)
        self.assertEqual(info.name, "stub")


class TestTieredModes(unittest.TestCase):
    """Tests for tiered mode support (embed / hybrid)."""

    def test_default_mode_is_hybrid(self):
        backend = StubBackend()
        self.assertEqual(backend.get_mode(), "hybrid")

    def test_set_mode_embed(self):
        backend = StubBackend()
        backend.set_mode("embed")
        self.assertEqual(backend.get_mode(), "embed")

    def test_set_mode_hybrid(self):
        backend = StubBackend()
        backend.set_mode("hybrid")
        self.assertEqual(backend.get_mode(), "hybrid")

    def test_set_mode_full_raises(self):
        """'full' mode was removed — should raise ValueError."""
        backend = StubBackend()
        with self.assertRaises(ValueError):
            backend.set_mode("full")

    def test_set_invalid_mode_raises(self):
        backend = StubBackend()
        with self.assertRaises(ValueError):
            backend.set_mode("turbo")

    def test_needs_reranker_embed_mode(self):
        backend = StubBackend()
        backend.set_mode("embed")
        self.assertFalse(backend.needs_reranker())

    def test_needs_reranker_hybrid_mode(self):
        backend = StubBackend()
        backend.set_mode("hybrid")
        self.assertTrue(backend.needs_reranker())


class TestMLXCaptionerLifecycle(unittest.TestCase):
    """Tests for MLX captioner memory lifecycle helpers without loading MLX."""

    def _make_uninitialized_backend(self):
        from recallforge.backends.mlx_backend import MLXBackend

        backend = object.__new__(MLXBackend)
        backend._model_lock = threading.RLock()
        backend._captioner_model = object()
        backend._captioner_processor = object()
        backend._captioner_idle_timer = None
        backend._captioner_idle_seconds = 0.05
        return backend

    def test_captioner_idle_timer_unloads_model(self):
        backend = self._make_uninitialized_backend()

        backend._schedule_captioner_idle_unload()
        self.assertIsNotNone(backend._captioner_idle_timer)

        time.sleep(0.15)
        self.assertIsNone(backend._captioner_model)
        self.assertIsNone(backend._captioner_processor)
        self.assertIsNone(backend._captioner_idle_timer)

    def test_manual_unload_cancels_captioner_idle_timer(self):
        backend = self._make_uninitialized_backend()
        backend._schedule_captioner_idle_unload()

        backend._unload_captioner()

        self.assertIsNone(backend._captioner_model)
        self.assertIsNone(backend._captioner_processor)
        self.assertIsNone(backend._captioner_idle_timer)


class TestBackendInfo(unittest.TestCase):
    """Tests for BackendInfo dataclass."""

    def test_backend_info_defaults(self):
        info = BackendInfo(name="test", device="cpu", dtype="float32")
        self.assertFalse(info.embedder_loaded)
        self.assertFalse(info.reranker_loaded)
        self.assertEqual(info.memory_allocated_gb, 0.0)
        self.assertTrue(info.supports_images)
        self.assertIsNone(info.quantization)

    def test_backend_info_full(self):
        info = BackendInfo(
            name="torch",
            device="cuda",
            dtype="float16",
            embedder_loaded=True,
            reranker_loaded=True,
            memory_allocated_gb=8.0,
            supports_images=True,
            quantization=None,
        )
        self.assertTrue(info.embedder_loaded)
        self.assertTrue(info.reranker_loaded)
        self.assertAlmostEqual(info.memory_allocated_gb, 8.0)

    def test_backend_info_mlx_quantized(self):
        info = BackendInfo(
            name="mlx",
            device="apple_silicon",
            dtype="bfloat16",
            quantization="4bit",
        )
        self.assertEqual(info.quantization, "4bit")


if __name__ == "__main__":
    unittest.main()
