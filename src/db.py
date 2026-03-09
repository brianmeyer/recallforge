"""
db.py - LanceDB connection layer for QMD-VL

LanceDB + Apache Arrow storage for embeddings, documents, content, and cache.
"""

import os
from typing import Optional
import pyarrow as pa
import lancedb

# Default store directory
DEFAULT_INDEX_DIR = os.path.join(os.path.expanduser("~"), ".qmd")


def get_lance_store_path(index_path: Optional[str] = None) -> str:
    """Resolve the LanceDB store directory path."""
    base = index_path or DEFAULT_INDEX_DIR
    return os.path.join(base, "store.lance")


# Singleton connection
_conn = None


def open_database(store_path: str):
    """Open (or reuse) a LanceDB connection."""
    global _conn
    if _conn is not None:
        return _conn
    _conn = lancedb.connect(store_path)
    return _conn


def close_database() -> None:
    """Release the singleton connection."""
    global _conn
    _conn = None


# Embedding dimension (Qwen3-VL-Embedding-2B outputs 2048-dim vectors)
EMBED_DIM = 2048


def build_embeddings_schema(dim: int = EMBED_DIM) -> pa.Schema:
    """Schema for the unified embeddings table."""
    return pa.schema([
        pa.field("hash_seq", pa.string(), nullable=False),      # PK: "{content_hash}_{seq}"
        pa.field("content_hash", pa.string(), nullable=False),
        pa.field("collection", pa.string(), nullable=False),
        pa.field("file_path", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=False),  # 'text' | 'image'
        pa.field("title", pa.string(), nullable=True),
        pa.field("text_body", pa.string(), nullable=True),      # BM25-indexed via Tantivy
        pa.field("seq", pa.int32(), nullable=False),
        pa.field("pos", pa.int32(), nullable=False),
        pa.field("model", pa.string(), nullable=True),
        pa.field("embedded_at", pa.int64(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), list_size=dim), nullable=False),
    ])


def build_documents_schema() -> pa.Schema:
    """Schema for the document registry table."""
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("collection", pa.string(), nullable=False),
        pa.field("file_path", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=True),
        pa.field("content_hash", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=False),
        pa.field("active", pa.int8(), nullable=False),  # 0 | 1
        pa.field("created_at", pa.int64(), nullable=False),
        pa.field("updated_at", pa.int64(), nullable=False),
    ])


def build_content_schema() -> pa.Schema:
    """Schema for the full document body / content table."""
    return pa.schema([
        pa.field("hash", pa.string(), nullable=False),
        pa.field("doc", pa.string(), nullable=False),  # full text or base64 image
        pa.field("content_type", pa.string(), nullable=False),
        pa.field("created_at", pa.int64(), nullable=False),
    ])


def build_cache_schema() -> pa.Schema:
    """Schema for the LLM response cache (query expansion + reranker)."""
    return pa.schema([
        pa.field("key", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),  # JSON string
        pa.field("created_at", pa.int64(), nullable=False),
    ])


# Module-level table references
embeddings_table = None
documents_table = None
content_table = None
cache_table = None


def initialize_database(store_path: Optional[str] = None) -> None:
    """Initialize all LanceDB tables."""
    global embeddings_table, documents_table, content_table, cache_table
    
    lance_dir = get_lance_store_path(store_path)
    db = open_database(lance_dir)
    
    # Use list_tables() (table_names() is deprecated)
    try:
        existing = db.list_tables()
    except AttributeError:
        existing = db.table_names()  # Fallback for older versions
    
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
    
    _ensure_indices()


def _ensure_indices() -> None:
    """Create scalar indices for faster lookups."""
    try:
        doc_indices = documents_table.list_indices() if documents_table else []
        doc_names = {i.name for i in doc_indices}
        if "file_path_scalar" not in doc_names and documents_table:
            documents_table.create_index("file_path")
        if "content_hash_scalar" not in doc_names and documents_table:
            documents_table.create_index("content_hash")
    except Exception:
        pass  # Table may be empty
    
    try:
        content_indices = content_table.list_indices() if content_table else []
        if content_table and not any(i.name == "hash_scalar" for i in content_indices):
            content_table.create_index("hash")
    except Exception:
        pass
    
    try:
        cache_indices = cache_table.list_indices() if cache_table else []
        if cache_table and not any(i.name == "key_scalar" for i in cache_indices):
            cache_table.create_index("key")
    except Exception:
        pass


def ensure_fts_index(force_rebuild: bool = False) -> None:
    """Ensure the Tantivy full-text index exists on embeddings.text_body."""
    if embeddings_table is None:
        return
    
    if embeddings_table.count_rows() == 0:
        return
    
    if force_rebuild:
        try:
            embeddings_table.create_fts_index("text_body", replace=True)
        except Exception:
            pass
        return
    
    indices = embeddings_table.list_indices()
    has_fts = any(
        "text_body" in (i.columns or []) and "FTS" in str(i.index_type or i.type or "").upper()
        for i in indices
    )
    
    if has_fts:
        return
    
    embeddings_table.create_fts_index("text_body", replace=True)


def rebuild_fts_index() -> None:
    """Rebuild the Tantivy FTS index."""
    if embeddings_table is None:
        return
    
    row_count = embeddings_table.count_rows()
    if row_count == 0:
        return
    
    # Drop existing FTS index and recreate
    indices = embeddings_table.list_indices()
    for idx in indices:
        if "text_body" in (idx.columns or []) and "FTS" in str(idx.index_type or idx.type or "").upper():
            embeddings_table.drop_index(idx.name)
            break
    
    ensure_fts_index()


def escape_sql(s: str) -> str:
    """Escape single quotes for LanceDB filter strings."""
    return s.replace("'", "''")