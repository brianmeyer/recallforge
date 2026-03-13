"""
test_backends.py - Unit tests for ModelBackend ABC and implementations.

Tests the backend interface contract, tiered modes, and BackendInfo dataclass.
Uses mock/stub implementations — NO real model inference.
"""

import os
import sys
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

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.ones((len(texts), 2048), dtype=np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        return np.ones(2048, dtype=np.float32)

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        return np.ones((len(image_paths), 2048), dtype=np.float32)

    def embed_video(self, video_path: str) -> np.ndarray:
        return np.ones(2048, dtype=np.float32)

    def embed_videos(self, video_paths: List[str]) -> np.ndarray:
        return np.ones((len(video_paths), 2048), dtype=np.float32)

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        return [0.9 - i * 0.1 for i in range(len(documents))]

    def expand_query(self, query: str) -> Dict[str, str]:
        return {"lex": query + " keywords", "vec": query + " semantic", "hyde": "hypothetical " + query}

    def warm_up(self) -> None:
        pass

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            name="stub",
            device="cpu",
            dtype="float32",
            embedder_loaded=True,
            reranker_loaded=True,
            expander_loaded=True,
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
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 2048))

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

    def test_expand_query_returns_dict_with_keys(self):
        backend = StubBackend()
        result = backend.expand_query("search query")
        self.assertIn("lex", result)
        self.assertIn("vec", result)
        self.assertIn("hyde", result)

    def test_get_info_returns_backend_info(self):
        backend = StubBackend()
        info = backend.get_info()
        self.assertIsInstance(info, BackendInfo)
        self.assertEqual(info.name, "stub")


class TestTieredModes(unittest.TestCase):
    """Tests for tiered mode support (embed / hybrid / full)."""

    def test_default_mode_is_full(self):
        backend = StubBackend()
        self.assertEqual(backend.get_mode(), "full")

    def test_set_mode_embed(self):
        backend = StubBackend()
        backend.set_mode("embed")
        self.assertEqual(backend.get_mode(), "embed")

    def test_set_mode_hybrid(self):
        backend = StubBackend()
        backend.set_mode("hybrid")
        self.assertEqual(backend.get_mode(), "hybrid")

    def test_set_mode_full(self):
        backend = StubBackend()
        backend.set_mode("full")
        self.assertEqual(backend.get_mode(), "full")

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

    def test_needs_reranker_full_mode(self):
        backend = StubBackend()
        backend.set_mode("full")
        self.assertTrue(backend.needs_reranker())

    def test_needs_expander_embed_mode(self):
        backend = StubBackend()
        backend.set_mode("embed")
        self.assertFalse(backend.needs_expander())

    def test_needs_expander_hybrid_mode(self):
        backend = StubBackend()
        backend.set_mode("hybrid")
        self.assertFalse(backend.needs_expander())

    def test_needs_expander_full_mode(self):
        backend = StubBackend()
        backend.set_mode("full")
        self.assertTrue(backend.needs_expander())


class TestBackendInfo(unittest.TestCase):
    """Tests for BackendInfo dataclass."""

    def test_backend_info_defaults(self):
        info = BackendInfo(name="test", device="cpu", dtype="float32")
        self.assertFalse(info.embedder_loaded)
        self.assertFalse(info.reranker_loaded)
        self.assertFalse(info.expander_loaded)
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
            expander_loaded=True,
            memory_allocated_gb=12.5,
            supports_images=True,
            quantization=None,
        )
        self.assertTrue(info.embedder_loaded)
        self.assertTrue(info.reranker_loaded)
        self.assertTrue(info.expander_loaded)
        self.assertAlmostEqual(info.memory_allocated_gb, 12.5)

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
