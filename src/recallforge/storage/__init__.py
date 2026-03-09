"""
RecallForge Storage Backend System.

Multi-backend storage supporting:
- lancedb: LanceDB (default, vector + FTS)
- Future: chromadb, qdrant
"""

from .base import StorageBackend
from .lancedb_backend import LanceDBBackend

__all__ = [
    "StorageBackend",
    "LanceDBBackend",
]