"""
cache.py - LRU query cache for RecallForge.

Avoids redundant query embeddings and generated expansion calls for repeated
queries. Keys can include model identity and storage index version so cached
retrieval inputs never cross model or index boundaries.
"""

import json
from hashlib import sha256
from typing import Any


class EmbeddingCache:
    """Simple LRU cache backed by a dict + insertion-order list."""

    def __init__(self, maxsize: int = 256):
        self._maxsize = maxsize
        self._cache: dict[str, Any] = {}
        self._order: list[str] = []
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any:
        if key not in self._cache:
            self._misses += 1
            return None
        self._hits += 1
        self._order.remove(key)
        self._order.append(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._maxsize:
            evict = self._order.pop(0)
            del self._cache[evict]
        self._cache[key] = value
        self._order.append(key)

    def make_key(
        self,
        input_type: str,
        input_data: str,
        *,
        model: str | None = None,
        index_version: str | int | None = None,
        namespace: str | None = None,
    ) -> str:
        if model is None and index_version is None and namespace is None:
            return sha256(f"{input_type}:{input_data}".encode()).hexdigest()

        payload = {
            "type": input_type,
            "data": input_data,
            "model": model or "",
            "index_version": "" if index_version is None else str(index_version),
            "namespace": namespace or "",
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def stats(self) -> dict:
        return {"size": len(self._cache), "maxsize": self._maxsize}

    @property
    def metrics(self) -> dict:
        return {
            "size": len(self._cache),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
        }
