"""
db.py - LanceDB connection layer for QMD-VL

Replaces SQLite with LanceDB + Apache Arrow for vector storage.
Single store directory contains all four tables: embeddings, documents,
content, and cache.
"""

import os
from pathlib import Path
from typing import Optional
import lancedb
from lancedb.table import Table
import pyarrow as pa
from pyarrow import schema
import hashlib
import time


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_INDEX_DIR = Path.home() / ".qmd"


def get_lance_store_path(index_path: Optional[str] = None) -> str:
    """Resolve the LanceDB store directory path."""
    base = index_path or str(DEFAULT_INDEX_DIR)
    return str(Path(base) / "store.lance")


# ---------------------------------------------------------------------------
# Connection (singleton)
# ---------------------------------------------------------------------------

_conn: Optional[lancedb.DBConnection] = None


def open_database(store_path: str) -> lancedb.DBConnection:
    """Open (or reuse) a LanceDB connection."""
    global _conn
    if _conn is not None:
        return _conn
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    _conn = lancedb.connect(store_path)
    return _conn


def close_database() -> None:
    """Release the singleton connection."""
    global _conn
    _conn = None


# ---------------------------------------------------------------------------
# Arrow Schemas
# ---------------------------------------------------------------------------

EMBED_DIM = 2048  # Qwen3-VL-Embedding-2B dimension


def build_embeddings_schema(dim: int = EMBED_DIM):
    """Schema for the unified embeddings table."""
    # Use fixed-size list for vector column (LanceDB can detect this)
    return schema([
        ("hash_seq", pa.string()),      # PK: "{content_hash}_{seq}"
        ("content_hash", pa.string()),
        ("collection", pa.string()),
        ("file_path", pa.string()),
        ("content_type", pa.string()),   # 'text' | 'image'
        ("title", pa.string()),
        ("text_body", pa.string()),        # BM25-indexed via Tantivy
        ("seq", pa.int32()),
        ("pos", pa.int32()),
        ("model", pa.string()),
        ("embedded_at", pa.int64()),
        ("vector", pa.list_(pa.float32(), list_size=2048)),  # Fixed-size list for LanceDB vector detection
    ])


def build_documents_schema():
    """Schema for the document registry table."""
    return schema([
        ("id", pa.string()),
        ("collection", pa.string()),
        ("file_path", pa.string()),
        ("title", pa.string()),
        ("content_hash", pa.string()),
        ("content_type", pa.string()),
        ("active", pa.int8()),  # 0 | 1
        ("created_at", pa.int64()),
        ("updated_at", pa.int64()),
    ])


def build_content_schema():
    """Schema for the full document body / content table."""
    return schema([
        ("hash", pa.string()),
        ("doc", pa.string()),  # full text or base64 image
        ("content_type", pa.string()),
        ("created_at", pa.int64()),
    ])


def build_cache_schema():
    """Schema for the LLM response cache."""
    return schema([
        ("key", pa.string()),
        ("value", pa.string()),  # JSON string
        ("created_at", pa.int64()),
    ])


# ---------------------------------------------------------------------------
# Table names
# ---------------------------------------------------------------------------

TableName = str  # 'embeddings' | 'documents' | 'content' | 'cache'


# ---------------------------------------------------------------------------
# Module-level table references
# ---------------------------------------------------------------------------

embeddings_table: Optional[Table] = None
documents_table: Optional[Table] = None
content_table: Optional[Table] = None
cache_table: Optional[Table] = None


def escape_sql(s: str) -> str:
    """Escape single quotes for LanceDB filter strings."""
    return s.replace("'", "''")


def initialize_database(store_path: Optional[str] = None) -> None:
    """Initialize (or re-initialize) all LanceDB tables."""
    global embeddings_table, documents_table, content_table, cache_table

    lance_dir = get_lance_store_path(store_path)
    db = open_database(lance_dir)

    existing = db.table_names()

    if "embeddings" in existing:
        embeddings_table = db.open_table("embeddings")
    else:
        embeddings_table = db.create_table("embeddings", schema=build_embeddings_schema())

    if "documents" in existing:
        documents_table = db.open_table("documents")
    else:
        documents_table = db.create_table("documents", schema=build_documents_schema())

    if "content" in existing:
        content_table = db.open_table("content")
    else:
        content_table = db.create_table("content", schema=build_content_schema())

    if "cache" in existing:
        cache_table = db.open_table("cache")
    else:
        cache_table = db.create_table("cache", schema=build_cache_schema())


async def ensure_indices() -> None:
    """Create FTS index on embeddings.text_body if needed."""
    global embeddings_table
    if embeddings_table is None:
        return

    try:
        row_count = embeddings_table.count_rows()
        if row_count == 0:
            return
        
        # Create FTS index on text_body column
        embeddings_table.create_fts_index("text_body", replace=True)
    except Exception:
        # Index may already exist or table may be empty
        pass


async def rebuild_fts_index() -> None:
    """Rebuild the Tantivy FTS index."""
    global embeddings_table
    if embeddings_table is None:
        return

    row_count = embeddings_table.count_rows()
    if row_count == 0:
        return

    try:
        # Just create/replace the index
        embeddings_table.create_fts_index("text_body", replace=True)
    except Exception:
        pass


async def has_vector_index() -> bool:
    """Check if embeddings table has vectors."""
    global embeddings_table
    if embeddings_table is None:
        return False
    try:
        count = embeddings_table.count_rows()
        return count > 0
    except Exception:
        return False