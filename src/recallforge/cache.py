"""
cache.py - LRU Embedding Cache for RecallForge.

Avoids redundant embed_text / embed_image calls for repeated queries.
The cache is deterministic: same input → same vector, so caching is safe.
"""

from hashlib import sha256
import numpy as np


class EmbeddingCache:
    """Simple LRU cache backed by a dict + insertion-order list."""

    def __init__(self, maxsize: int = 256):
        self._maxsize = maxsize
        self._cache: dict[str, np.ndarray] = {}
        self._order: list[str] = []

    def get(self, key: str) -> "np.ndarray | None":
        return self._cache.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._maxsize:
            evict = self._order.pop(0)
            del self._cache[evict]
        self._cache[key] = vector
        self._order.append(key)

    def make_key(self, input_type: str, input_data: str) -> str:
        return sha256(f"{input_type}:{input_data}".encode()).hexdigest()

    @property
    def stats(self) -> dict:
        return {"size": len(self._cache), "maxsize": self._maxsize}
