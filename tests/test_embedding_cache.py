"""
test_embedding_cache.py - Tests for EmbeddingCache and its integration into HybridSearcher.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from recallforge.cache import EmbeddingCache


# ---------------------------------------------------------------------------
# EmbeddingCache unit tests
# ---------------------------------------------------------------------------

class TestEmbeddingCache:
    def _cache(self, maxsize=4):
        return EmbeddingCache(maxsize=maxsize)

    def test_cache_miss_returns_none(self):
        c = self._cache()
        assert c.get("nonexistent") is None

    def test_put_and_get_returns_same_vector(self):
        c = self._cache()
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        c.put("k1", vec)
        result = c.get("k1")
        assert result is not None
        np.testing.assert_array_equal(result, vec)

    def test_cache_hit_returns_same_object(self):
        """Cache should return the exact same ndarray object (no copy)."""
        c = self._cache()
        vec = np.array([0.5, 0.5], dtype=np.float32)
        c.put("k", vec)
        assert c.get("k") is vec

    def test_lru_eviction_removes_oldest(self):
        """When maxsize is reached, the least-recently inserted key is evicted."""
        c = EmbeddingCache(maxsize=3)
        for i in range(3):
            c.put(f"k{i}", np.array([float(i)]))
        # Cache is full with k0, k1, k2.  Adding k3 should evict k0.
        c.put("k3", np.array([3.0]))
        assert c.get("k0") is None
        assert c.get("k1") is not None
        assert c.get("k2") is not None
        assert c.get("k3") is not None

    def test_put_existing_key_refreshes_order(self):
        """Re-inserting an existing key should move it to the end (MRU position)."""
        c = EmbeddingCache(maxsize=3)
        for i in range(3):
            c.put(f"k{i}", np.array([float(i)]))
        # Refresh k0 (move to MRU end)
        c.put("k0", np.array([99.0]))
        # Adding a 4th item should now evict k1 (oldest after refresh)
        c.put("k3", np.array([3.0]))
        assert c.get("k1") is None
        assert c.get("k0") is not None
        assert c.get("k2") is not None
        assert c.get("k3") is not None

    def test_stats_report_size_and_maxsize(self):
        c = EmbeddingCache(maxsize=10)
        assert c.stats == {"size": 0, "maxsize": 10}
        c.put("a", np.array([1.0]))
        c.put("b", np.array([2.0]))
        assert c.stats == {"size": 2, "maxsize": 10}

    def test_make_key_deterministic(self):
        c = self._cache()
        k1 = c.make_key("text", "hello world")
        k2 = c.make_key("text", "hello world")
        assert k1 == k2

    def test_make_key_differs_by_type(self):
        c = self._cache()
        k_text = c.make_key("text", "data")
        k_image = c.make_key("image", "data")
        assert k_text != k_image

    def test_make_key_differs_by_data(self):
        c = self._cache()
        assert c.make_key("text", "a") != c.make_key("text", "b")

    def test_stats_after_eviction(self):
        c = EmbeddingCache(maxsize=2)
        c.put("a", np.array([1.0]))
        c.put("b", np.array([2.0]))
        c.put("c", np.array([3.0]))  # evicts "a"
        assert c.stats["size"] == 2


# ---------------------------------------------------------------------------
# HybridSearcher integration tests — cache wiring
# ---------------------------------------------------------------------------

def _make_searcher(cache=None, maxsize=256):
    """Build a HybridSearcher with fully mocked backend + storage."""
    from recallforge.search import HybridSearcher

    backend = MagicMock()
    backend.embed_text.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    backend.embed_image.return_value = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    backend.embed_texts.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    backend.needs_reranker.return_value = False
    backend.needs_expander.return_value = False

    storage = MagicMock()
    storage.search_fts.return_value = []
    storage.search_vec.return_value = []

    if cache is None:
        cache = EmbeddingCache(maxsize=maxsize)

    searcher = HybridSearcher(backend=backend, storage=storage, cache=cache)
    return searcher, backend, storage, cache


class TestHybridSearcherCacheIntegration:
    def test_cache_miss_calls_backend(self):
        searcher, backend, storage, cache = _make_searcher()
        searcher._vector_search("hello")
        backend.embed_text.assert_called_once_with("hello")

    def test_cache_hit_skips_backend(self):
        searcher, backend, storage, cache = _make_searcher()
        # First call — populates cache
        searcher._vector_search("hello")
        assert backend.embed_text.call_count == 1
        # Second call — should hit cache
        searcher._vector_search("hello")
        assert backend.embed_text.call_count == 1  # NOT called again

    def test_cache_populated_after_miss(self):
        searcher, backend, storage, cache = _make_searcher()
        searcher._vector_search("hello")
        key = cache.make_key("text", "hello")
        assert cache.get(key) is not None
        np.testing.assert_array_equal(cache.get(key), backend.embed_text.return_value)

    def test_image_cache_miss_calls_backend(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        searcher, backend, storage, cache = _make_searcher()
        searcher.search_image(str(img))
        backend.embed_image.assert_called_once_with(str(img))

    def test_image_cache_hit_skips_backend(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        searcher, backend, storage, cache = _make_searcher()
        searcher.search_image(str(img))
        assert backend.embed_image.call_count == 1
        searcher.search_image(str(img))
        assert backend.embed_image.call_count == 1  # hit

    def test_shared_cache_across_searchers(self):
        """Two searchers sharing the same cache instance share embeddings."""
        shared = EmbeddingCache()
        s1, b1, _, _ = _make_searcher(cache=shared)
        s2, b2, _, _ = _make_searcher(cache=shared)
        s1._vector_search("shared query")
        assert b1.embed_text.call_count == 1
        # s2 should hit cache from s1
        s2._vector_search("shared query")
        assert b2.embed_text.call_count == 0

    def test_default_cache_created_when_none_given(self):
        from recallforge.search import HybridSearcher
        backend = MagicMock()
        backend.needs_reranker.return_value = False
        backend.needs_expander.return_value = False
        storage = MagicMock()
        storage.search_fts.return_value = []
        storage.search_vec.return_value = []
        s = HybridSearcher(backend=backend, storage=storage)
        assert isinstance(s.cache, EmbeddingCache)
