"""
lancedb_backend.py - LanceDB Storage Backend for RecallForge.

LanceDB + Apache Arrow storage for embeddings, documents, content, and cache.
Provides vector search and full-text search (Tantivy).
"""

import fnmatch
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pyarrow as pa
import lancedb

from .base import StorageBackend, Document
from .chunking import (
    BREAK_PATTERNS,
    BreakPoint,
    CHUNK_OVERLAP_CHARS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_CHARS,
    CHUNK_SIZE_TOKENS,
    CHUNK_WINDOW_CHARS,
    CodeFenceRegion,
    chunk_document,
    find_best_cutoff,
    find_code_fences,
    is_inside_code_fence,
    scan_break_points,
)
from .lancedb_shared import (
    DEFAULT_INDEX_DIR,
    _SQL_METACHARACTERS,
    _safe_filter,
    _validate_identifier,
    escape_sql,
    extract_title,
    get_docid,
    hash_content,
    hash_file_bytes,
    logger,
    trace_log,
)
from .fts_manager import FTSManager
from .search_ops import SearchOps
from .indexing_ops import IndexingOps


# =============================================================================
# File Size Limits
# =============================================================================

DEFAULT_MAX_FILE_SIZE_MB = 100


# =============================================================================
# LanceDB Backend Implementation
# =============================================================================

class LanceDBBackend(StorageBackend):
    """
    LanceDB-based storage backend.
    
    Provides:
    - Vector storage and ANN search via LanceDB
    - Full-text search via Tantivy (LanceDB FTS)
    - Document and content storage
    - Result caching
    """
    
    EMBED_DIM = 2048  # Qwen3-VL-Embedding-2B dimension
    
    # FTS rebuild debounce configuration
    FTS_REBUILD_MIN_INTERVAL = 2.0  # Minimum seconds between rebuilds
    FTS_REBUILD_PENDING_THRESHOLD = 10  # Rebuild after this many pending writes
    BULK_FLUSH_DOCS_THRESHOLD = max(1, int(os.environ.get("RECALLFORGE_BULK_FLUSH_DOCS", "75")))
    BULK_FLUSH_EMBEDDINGS_THRESHOLD = max(1, int(os.environ.get("RECALLFORGE_BULK_FLUSH_EMBEDDINGS", "600")))
    
    def __init__(self, store_path: Optional[str] = None):
        """
        Initialize LanceDB backend.
        
        Args:
            store_path: Optional path to storage directory (default: ~/.recallforge)
        """
        self._store_path = store_path
        self._conn = None
        self._embeddings_table = None
        self._documents_table = None
        self._content_table = None
        self._cache_table = None
        
        # FTS rebuild debouncing state
        self._fts_rebuild_pending = 0
        self._fts_last_rebuild = 0.0
        self._fts_needs_rebuild = False
        self._bulk_mode = False  # When True, defer all FTS rebuilds until bulk ends
        self._pending_documents: Dict[str, Dict[str, Any]] = {}
        self._pending_content: Dict[str, Dict[str, Any]] = {}
        self._pending_embeddings: Dict[str, Dict[str, Any]] = {}
        self._pending_embedding_deletes: set[str] = set()
        self._bm25_fallback_max_rows = max(
            100,
            int(os.environ.get("RECALLFORGE_BM25_FALLBACK_MAX_ROWS", "5000")),
        )
        
        # Initialize composed service objects (must be done here for tests that use __new__)
        self._fts = FTSManager(self)
        self._search = SearchOps(self)
        self._indexer = IndexingOps(self)
    
    def initialize(self, store_path: Optional[str] = None) -> None:
        """Initialize the LanceDB database."""
        if store_path:
            self._store_path = store_path
        
        base = self._store_path or DEFAULT_INDEX_DIR
        lance_dir = os.path.join(base, "store.lance")
        
        self._conn = lancedb.connect(lance_dir)
        
        # Get existing tables
        try:
            existing = self._conn.list_tables()
        except AttributeError:
            existing = self._conn.table_names()
        
        # Create or open tables
        if "embeddings" in existing:
            self._embeddings_table = self._conn.open_table("embeddings")
        else:
            try:
                self._embeddings_table = self._conn.create_table(
                    "embeddings",
                    schema=self._build_embeddings_schema()
                )
            except ValueError as e:
                # Table may have been created by another process; try to open it
                if "already exists" in str(e):
                    self._embeddings_table = self._conn.open_table("embeddings")
                else:
                    raise
        
        if "documents" in existing:
            self._documents_table = self._conn.open_table("documents")
        else:
            try:
                self._documents_table = self._conn.create_table(
                    "documents",
                    schema=self._build_documents_schema()
                )
            except ValueError as e:
                if "already exists" in str(e):
                    self._documents_table = self._conn.open_table("documents")
                else:
                    raise
        
        if "content" in existing:
            self._content_table = self._conn.open_table("content")
        else:
            try:
                self._content_table = self._conn.create_table(
                    "content",
                    schema=self._build_content_schema()
                )
            except ValueError as e:
                if "already exists" in str(e):
                    self._content_table = self._conn.open_table("content")
                else:
                    raise
        
        if "cache" in existing:
            self._cache_table = self._conn.open_table("cache")
        else:
            try:
                self._cache_table = self._conn.create_table(
                    "cache",
                    schema=self._build_cache_schema()
                )
            except ValueError as e:
                if "already exists" in str(e):
                    self._cache_table = self._conn.open_table("cache")
                else:
                    raise
        
        self._ensure_indices()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """
        Migrate existing tables to match the current target schema.

        Older stores may lack columns that were added in recent versions,
        most notably the namespace columns ``user_id``, ``session_id``,
        ``project_id``, and ``profile`` (REC-34), as well as Phase-2 metadata
        columns ``importance``, ``ttl_seconds``, ``tags``, and ``expires_at``.

        Writing rows that include fields absent from the on-disk schema raises
        ``Field '<name>' not found``.  This method detects the gap and backfills
        every missing column (initialised to NULL) so the rest of the code can
        assume a consistent schema.

        The migration is driven by the authoritative target schemas returned by
        ``_build_embeddings_schema()`` and ``_build_documents_schema()``, so it
        will automatically cover any new columns added to those schemas in the
        future.

        Safe to call on up-to-date stores — it is a no-op when all columns
        already exist.
        """
        tables_and_schemas = [
            (self._embeddings_table, "embeddings", self._build_embeddings_schema()),
            (self._documents_table,  "documents",  self._build_documents_schema()),
        ]

        for table, table_name, target_schema in tables_and_schemas:
            if table is None:
                continue
            try:
                current_cols = {field.name for field in table.schema}

                # Collect pa.Field objects for every column in the target schema
                # that is absent from the on-disk table.
                new_fields: List[pa.Field] = [
                    field
                    for field in target_schema
                    if field.name not in current_cols
                ]

                if not new_fields:
                    continue

                missing_names = [f.name for f in new_fields]
                logger.info(
                    "_migrate_schema: table '%s' is missing columns %s — adding them now",
                    table_name,
                    missing_names,
                )

                # add_columns accepts a list of pa.Field objects and initialises
                # each new column to NULL (all new fields are nullable=True).
                table.add_columns(new_fields)

                logger.info(
                    "_migrate_schema: successfully added %s to '%s'",
                    missing_names,
                    table_name,
                )
            except Exception as exc:
                logger.error(
                    "_migrate_schema: failed to migrate table '%s': %s",
                    table_name,
                    exc,
                )

    def close(self) -> None:
        """Close the database connection."""
        # Flush pending buffered writes before rebuilding FTS.
        self._flush_pending_writes(force=True)

        # Flush any pending FTS rebuild before closing
        if self._fts_needs_rebuild or self._fts_rebuild_pending > 0:
            self._fts.do_fts_rebuild()
        
        self._conn = None
        self._embeddings_table = None
        self._documents_table = None
        self._content_table = None
        self._cache_table = None
    
    # =========================================================================
    # Schema Definitions
    # =========================================================================

    def _build_embeddings_schema(self) -> pa.Schema:
        """Schema for embeddings table."""
        return pa.schema([
            pa.field("hash_seq", pa.string(), nullable=False),
            pa.field("content_hash", pa.string(), nullable=False),
            pa.field("collection", pa.string(), nullable=False),
            pa.field("file_path", pa.string(), nullable=False),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=True),
            pa.field("text_body", pa.string(), nullable=True),
            pa.field("seq", pa.int32(), nullable=False),
            pa.field("pos", pa.int32(), nullable=False),
            pa.field("model", pa.string(), nullable=True),
            pa.field("embedded_at", pa.int64(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), list_size=self.EMBED_DIM), nullable=False),
            # Namespace fields for multi-tenant isolation
            pa.field("user_id", pa.string(), nullable=True),
            pa.field("session_id", pa.string(), nullable=True),
            pa.field("project_id", pa.string(), nullable=True),
            pa.field("profile", pa.string(), nullable=True),
            # Metadata fields for Phase 2
            pa.field("importance", pa.float32(), nullable=True),
            pa.field("ttl_seconds", pa.int32(), nullable=True),
            pa.field("tags", pa.string(), nullable=True),  # JSON-encoded list of strings
            pa.field("expires_at", pa.int64(), nullable=True),  # Timestamp in ms when entry expires
        ])
    
    def _build_documents_schema(self) -> pa.Schema:
        """Schema for documents table."""
        return pa.schema([
            pa.field("id", pa.string(), nullable=False),
            pa.field("collection", pa.string(), nullable=False),
            pa.field("file_path", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=True),
            pa.field("content_hash", pa.string(), nullable=False),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("active", pa.int8(), nullable=False),
            pa.field("created_at", pa.int64(), nullable=False),
            pa.field("updated_at", pa.int64(), nullable=False),
            # Namespace fields for multi-tenant isolation
            pa.field("user_id", pa.string(), nullable=True),
            pa.field("session_id", pa.string(), nullable=True),
            pa.field("project_id", pa.string(), nullable=True),
            pa.field("profile", pa.string(), nullable=True),
        ])
    
    def _build_content_schema(self) -> pa.Schema:
        """Schema for content table."""
        return pa.schema([
            pa.field("hash", pa.string(), nullable=False),
            pa.field("doc", pa.string(), nullable=False),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("created_at", pa.int64(), nullable=False),
        ])
    
    def _build_cache_schema(self) -> pa.Schema:
        """Schema for cache table."""
        return pa.schema([
            pa.field("key", pa.string(), nullable=False),
            pa.field("value", pa.string(), nullable=False),
            pa.field("created_at", pa.int64(), nullable=False),
        ])
    
    def _has_scalar_index(self, table, column: str) -> bool:
        """Return True if a scalar index already exists for a column."""
        try:
            for idx in table.list_indices():
                columns = getattr(idx, "columns", None) or []
                if column in columns:
                    return True
        except Exception:
            return False
        return False

    def _create_scalar_index_safe(self, table, column: str, name: str, label: str) -> None:
        """Create scalar index using current LanceDB API, if supported."""
        if table is None or self._has_scalar_index(table, column):
            return
        create_scalar = getattr(table, "create_scalar_index", None)
        if not callable(create_scalar):
            logger.debug(f"_ensure_indices: create_scalar_index unavailable; skipping {label}.{column}")
            return
        create_scalar(column, name=name, replace=False)

    def _ensure_indices(self) -> None:
        """Create scalar indices for faster lookups."""
        try:
            self._create_scalar_index_safe(
                self._documents_table, "file_path", "file_path_scalar", "documents"
            )
            self._create_scalar_index_safe(
                self._documents_table, "content_hash", "content_hash_scalar", "documents"
            )
        except Exception as e:
            logger.warning(f"_ensure_indices: failed to create document indices: {e}")

        try:
            self._create_scalar_index_safe(
                self._content_table, "hash", "hash_scalar", "content"
            )
        except Exception as e:
            logger.warning(f"_ensure_indices: failed to create content index: {e}")

        try:
            self._create_scalar_index_safe(
                self._cache_table, "key", "key_scalar", "cache"
            )
        except Exception as e:
            logger.warning(f"_ensure_indices: failed to create cache index: {e}")
    
    def _ensure_bulk_buffers(self) -> None:
        """Initialize bulk-write buffers for tests that construct via __new__."""
        if not hasattr(self, "_pending_documents") or self._pending_documents is None:
            self._pending_documents = {}
        if not hasattr(self, "_pending_content") or self._pending_content is None:
            self._pending_content = {}
        if not hasattr(self, "_pending_embeddings") or self._pending_embeddings is None:
            self._pending_embeddings = {}
        if not hasattr(self, "_pending_embedding_deletes") or self._pending_embedding_deletes is None:
            self._pending_embedding_deletes = set()

    def _flush_pending_writes(self, force: bool = False) -> None:
        """Flush buffered bulk writes to LanceDB in large Arrow batches."""
        self._ensure_bulk_buffers()

        should_flush = force
        if not should_flush:
            should_flush = (
                len(self._pending_documents) >= self.BULK_FLUSH_DOCS_THRESHOLD
                or len(self._pending_content) >= self.BULK_FLUSH_DOCS_THRESHOLD
                or len(self._pending_embeddings) >= self.BULK_FLUSH_EMBEDDINGS_THRESHOLD
            )
        if not should_flush:
            return

        if self._pending_embedding_deletes:
            expr = " OR ".join(
                _safe_filter("hash_seq", hash_seq)
                for hash_seq in sorted(self._pending_embedding_deletes)
            )
            self._embeddings_table.delete(expr)
            self._pending_embedding_deletes.clear()

        if self._pending_content:
            rows = list(self._pending_content.values())
            self._content_table.add(pa.Table.from_pylist(rows, schema=self._build_content_schema()))
            self._pending_content.clear()

        if self._pending_documents:
            rows = list(self._pending_documents.values())
            if rows:
                delete_expr = " OR ".join(_safe_filter("id", row["id"]) for row in rows)
                self._documents_table.delete(delete_expr)
                self._documents_table.add(pa.Table.from_pylist(rows, schema=self._build_documents_schema()))
            self._pending_documents.clear()

        if self._pending_embeddings:
            rows = list(self._pending_embeddings.values())
            self._embeddings_table.add(pa.Table.from_pylist(rows, schema=self._build_embeddings_schema()))
            self._pending_embeddings.clear()

    # =========================================================================
    # Document Operations
    # =========================================================================
    
    def insert_document(
        self,
        collection: str,
        file_path: str,
        title: str,
        content_hash: str,
        content_type: str = "text",
        created_at: Optional[int] = None,
        modified_at: Optional[int] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None
    ) -> str:
        """Insert or update a document."""
        now = int(time.time() * 1000)
        created_ts = created_at or now
        modified_ts = modified_at or now

        # Build namespace filter for finding existing
        ns_filter = f"{_safe_filter('collection', collection)} AND {_safe_filter('file_path', file_path)}"
        if user_id is not None:
            ns_filter += f" AND {_safe_filter('user_id', user_id)}"
        if session_id is not None:
            ns_filter += f" AND {_safe_filter('session_id', session_id)}"
        if project_id is not None:
            ns_filter += f" AND {_safe_filter('project_id', project_id)}"
        if profile is not None:
            ns_filter += f" AND {_safe_filter('profile', profile)}"

        self._ensure_bulk_buffers()

        # Check for existing (including staged bulk rows)
        existing_row = None
        if self._bulk_mode:
            for row in self._pending_documents.values():
                if (
                    row["collection"] == collection
                    and row["file_path"] == file_path
                    and row.get("user_id") == user_id
                    and row.get("session_id") == session_id
                    and row.get("project_id") == project_id
                    and row.get("profile") == profile
                ):
                    existing_row = row
                    break

        if existing_row is None:
            try:
                existing = list(self._documents_table.search()
                    .where(ns_filter)
                    .limit(1)
                    .to_list())
                if len(existing) > 0:
                    existing_row = existing[0]
            except Exception as e:
                logger.warning(f"insert_document: failed to check existing doc {collection}/{file_path}: {e}")

        doc_id = existing_row["id"] if existing_row else str(uuid.uuid4())
        created_value = existing_row.get("created_at", created_ts) if existing_row else created_ts
        row = {
            "id": doc_id,
            "collection": collection,
            "file_path": file_path,
            "title": title,
            "content_hash": content_hash,
            "content_type": content_type,
            "active": 1,
            "created_at": created_value,
            "updated_at": modified_ts,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }

        if self._bulk_mode:
            self._pending_documents[doc_id] = row
            self._flush_pending_writes(force=False)
        else:
            self._documents_table.delete(_safe_filter("id", doc_id))
            self._documents_table.add(pa.Table.from_pylist([row], schema=self._build_documents_schema()))

        return doc_id
    
    def find_document(self, collection: str, file_path: str) -> Optional[Document]:
        """Find a document by collection and path."""
        try:
            rows = list(self._documents_table.search()
                .where(f"{_safe_filter('collection', collection)} AND {_safe_filter('file_path', file_path)} AND active = 1")
                .limit(1)
                .to_list())
            
            if len(rows) == 0:
                return None
            
            r = rows[0]
            return Document(
                id=r["id"],
                collection=r["collection"],
                file_path=r["file_path"],
                title=r["title"] or "",
                content_hash=r["content_hash"],
                content_type=r["content_type"],
                active=bool(r["active"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
        except Exception as e:
            logger.warning(f"find_document: failed to find {collection}/{file_path}: {e}")
            return None
    
    def deactivate_document(self, collection: str, file_path: str) -> None:
        """Mark a document as inactive."""
        self._documents_table.update(
            where=f"{_safe_filter('collection', collection)} AND {_safe_filter('file_path', file_path)} AND active = 1",
            values={"active": 0, "updated_at": int(time.time() * 1000)}
        )
    
    # =========================================================================
    # Content Operations
    # =========================================================================
    
    def insert_content(self, hash_str: str, content: str, content_type: str = "text") -> None:
        """Store content by hash."""
        self._ensure_bulk_buffers()

        if hash_str in self._pending_content:
            return

        try:
            existing = list(self._content_table.search()
                .where(_safe_filter("hash", hash_str))
                .limit(1)
                .to_list())
            if len(existing) > 0:
                return
        except Exception as e:
            logger.warning(f"insert_content: failed to check existing hash {hash_str[:8]}: {e}")

        row = {
            "hash": hash_str,
            "doc": content,
            "content_type": content_type,
            "created_at": int(time.time() * 1000),
        }
        if self._bulk_mode:
            self._pending_content[hash_str] = row
            self._flush_pending_writes(force=False)
        else:
            self._content_table.add(pa.Table.from_pylist([row], schema=self._build_content_schema()))
    
    def get_content(self, hash_str: str) -> Optional[str]:
        """Retrieve content by hash."""
        try:
            rows = list(self._content_table.search()
                .where(_safe_filter("hash", hash_str))
                .limit(1)
                .to_list())
            if len(rows) == 0:
                return None
            return rows[0]["doc"]
        except Exception as e:
            logger.warning(f"get_content: failed to get hash {hash_str[:8]}: {e}")
            return None
    
    # =========================================================================
    # Embedding Operations
    # =========================================================================
    
    def insert_embedding(
        self,
        content_hash: str,
        seq: int,
        pos: int,
        vector: List[float],
        model: str,
        collection: str = "",
        file_path: str = "",
        title: str = "",
        text_body: str = "",
        content_type: str = "text",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        importance: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Insert an embedding with optional metadata."""
        hash_seq = f"{content_hash}_{seq}"
        now = int(time.time() * 1000)

        # Calculate expiration timestamp if TTL is set
        expires_at = None
        if ttl_seconds is not None and ttl_seconds > 0:
            expires_at = now + (ttl_seconds * 1000)

        # Encode tags as JSON if provided
        tags_json = None
        if tags is not None:
            import json
            tags_json = json.dumps(tags)

        self._ensure_bulk_buffers()

        if self._bulk_mode:
            self._pending_embedding_deletes.add(hash_seq)
        else:
            # Delete existing
            try:
                self._embeddings_table.delete(_safe_filter("hash_seq", hash_seq))
            except Exception as e:
                logger.debug(f"insert_embedding: no existing embedding to delete for {hash_seq}: {e}")

        trace_log("insert_embedding", hash_seq=hash_seq, collection=collection, file_path=file_path, seq=seq,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile,
                  importance=importance, ttl_seconds=ttl_seconds, tags=tags)

        row = {
            "hash_seq": hash_seq,
            "content_hash": content_hash,
            "collection": collection,
            "file_path": file_path,
            "content_type": content_type,
            "title": title,
            "text_body": text_body,
            "seq": seq,
            "pos": pos,
            "model": model,
            "embedded_at": now,
            "vector": vector,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
            "importance": importance,
            "ttl_seconds": ttl_seconds,
            "tags": tags_json,
            "expires_at": expires_at,
        }

        if self._bulk_mode:
            self._pending_embeddings[hash_seq] = row
            self._flush_pending_writes(force=False)
        else:
            self._embeddings_table.add(pa.Table.from_pylist([row], schema=self._build_embeddings_schema()))
    
    def has_vectors(self) -> bool:
        """Check if index has any vectors."""
        try:
            count = self._embeddings_table.count_rows()
            return count > 0
        except Exception as e:
            logger.warning(f"has_vectors: failed to count rows: {e}")
            return False
    
    # =========================================================================
    # Search Operations - Delegated to SearchOps
    # =========================================================================
    
    def search_fts(
        self,
        query: str,
        limit: int = 20,
        collection: Optional[str] = None,
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None
    ) -> List[Any]:
        """Full-text search using LanceDB Tantivy."""
        return self._search.search_fts(
            query=query,
            limit=limit,
            collection=collection,
            content_type=content_type,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile
        )
    
    def search_vec(
        self,
        vector: List[float],
        limit: int = 20,
        collection: Optional[str] = None,
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None
    ) -> List[Any]:
        """Vector similarity search."""
        return self._search.search_vec(
            vector=vector,
            limit=limit,
            collection=collection,
            content_type=content_type,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile
        )
    
    def list_collections(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> List[str]:
        """Return sorted list of unique collection names, with optional namespace filters."""
        return self._search.list_collections(
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile
        )
    
    def list_namespaces(
        self,
        collection: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Return unique namespace combinations (user_id, session_id, project_id, profile)."""
        return self._search.list_namespaces(collection=collection)

    def rename_collection(
        self,
        old_name: str,
        new_name: str,
    ) -> Dict[str, Any]:
        """
        Rename a collection atomically.

        Updates all rows in both embeddings and documents tables.
        Returns a summary of the operation.
        """
        if not old_name or not new_name:
            raise ValueError("old_name and new_name are required")
        if old_name == new_name:
            return {"success": True, "old_name": old_name, "new_name": new_name, "embeddings_updated": 0, "documents_updated": 0}

        # Flush any pending writes first
        self._flush_pending_writes(force=True)

        # Update embeddings table
        embeddings_filter = _safe_filter("collection", old_name)
        embeddings_updated = 0
        try:
            # Fetch all rows for this collection
            all_rows = list(self._embeddings_table.search()
                .where(embeddings_filter)
                .limit(100_000)
                .to_list())

            embeddings_updated = len(all_rows)

            if all_rows:
                # Delete all rows for this collection
                self._embeddings_table.delete(embeddings_filter)

                # Modify and re-insert
                for row in all_rows:
                    row["collection"] = new_name

                # Re-insert
                if all_rows:
                    self._embeddings_table.add(pa.Table.from_pylist(all_rows, schema=self._build_embeddings_schema()))

        except Exception as e:
            logger.error(f"rename_collection: failed to update embeddings: {e}")
            raise

        # Update documents table
        documents_filter = _safe_filter("collection", old_name)
        documents_updated = 0
        try:
            all_doc_rows = list(self._documents_table.search()
                .where(documents_filter)
                .limit(100_000)
                .to_list())

            documents_updated = len(all_doc_rows)

            if all_doc_rows:
                # Delete all rows for this collection
                self._documents_table.delete(documents_filter)

                # Modify and re-insert
                for row in all_doc_rows:
                    row["collection"] = new_name

                # Re-insert
                if all_doc_rows:
                    self._documents_table.add(pa.Table.from_pylist(all_doc_rows, schema=self._build_documents_schema()))

        except Exception as e:
            logger.error(f"rename_collection: failed to update documents: {e}")
            raise

        # Schedule FTS rebuild since we modified embeddings
        self._schedule_fts_rebuild()

        return {
            "success": True,
            "old_name": old_name,
            "new_name": new_name,
            "embeddings_updated": embeddings_updated,
            "documents_updated": documents_updated,
        }

    def delete_collection(
        self,
        name: str,
    ) -> Dict[str, Any]:
        """
        Delete all data for a collection.

        Removes all documents and embeddings for the collection,
        and cleans up orphaned content entries.
        """
        if not name:
            raise ValueError("name is required")

        # Flush any pending writes first
        self._flush_pending_writes(force=True)

        # Track content hashes to potentially clean up
        content_hashes_to_check: set = set()

        # Delete from embeddings table
        embeddings_deleted = 0
        try:
            embeddings_filter = _safe_filter("collection", name)

            # Get content hashes before deletion for orphan cleanup
            rows = list(self._embeddings_table.search()
                .where(embeddings_filter)
                .select(["content_hash"])
                .limit(100_000)
                .to_list())
            content_hashes_to_check.update(r["content_hash"] for r in rows)
            embeddings_deleted = len(rows)

            # Delete all embeddings for this collection
            self._embeddings_table.delete(embeddings_filter)
        except Exception as e:
            logger.error(f"delete_collection: failed to delete embeddings: {e}")
            raise

        # Delete from documents table
        documents_deleted = 0
        try:
            documents_filter = _safe_filter("collection", name)

            # Get content hashes before deletion for orphan cleanup
            rows = list(self._documents_table.search()
                .where(documents_filter)
                .select(["content_hash"])
                .limit(100_000)
                .to_list())
            content_hashes_to_check.update(r["content_hash"] for r in rows)
            documents_deleted = len(rows)

            # Delete all documents for this collection
            self._documents_table.delete(documents_filter)
        except Exception as e:
            logger.error(f"delete_collection: failed to delete documents: {e}")
            raise

        # Clean up orphaned content entries
        orphans_cleaned = 0
        if content_hashes_to_check:
            for content_hash in content_hashes_to_check:
                # Check if this content hash is still referenced anywhere
                try:
                    embedding_refs = list(self._embeddings_table.search()
                        .where(_safe_filter("content_hash", content_hash))
                        .limit(1)
                        .to_list())
                    document_refs = list(self._documents_table.search()
                        .where(_safe_filter("content_hash", content_hash))
                        .limit(1)
                        .to_list())

                    if not embedding_refs and not document_refs:
                        # Orphaned - delete from content table
                        self._content_table.delete(_safe_filter("hash", content_hash))
                        orphans_cleaned += 1
                except Exception as e:
                    logger.warning(f"delete_collection: failed to check orphan {content_hash[:8]}: {e}")

        # Schedule FTS rebuild
        self._schedule_fts_rebuild()

        return {
            "success": True,
            "name": name,
            "embeddings_deleted": embeddings_deleted,
            "documents_deleted": documents_deleted,
            "orphans_cleaned": orphans_cleaned,
        }

    def _make_search_result(self, row: Dict[str, Any], score: float, source: str) -> Any:
        """Convert LanceDB row to SearchResult."""
        # Lazy initialization for tests that use __new__
        if not hasattr(self, '_search') or self._search is None:
            from .search_ops import SearchOps
            self._search = SearchOps(self)
        return self._search._make_search_result(row, score, source)
    
    # =========================================================================
    # FTS Operations - Delegated to FTSManager
    # =========================================================================
    
    def bulk_mode(self):
        """Context manager to defer FTS rebuilds during bulk operations."""
        return self._fts.bulk_mode()

    def _schedule_fts_rebuild(self) -> None:
        """Schedule a debounced FTS rebuild."""
        self._fts.schedule_fts_rebuild()

    def _do_fts_rebuild(self) -> None:
        """Execute the actual FTS rebuild with rate limiting."""
        self._fts.do_fts_rebuild()
    
    def ensure_fts_index(self, force_rebuild: bool = False) -> None:
        """Ensure the FTS index exists."""
        self._fts.ensure_fts_index(force_rebuild=force_rebuild)
    
    def rebuild_fts_index(self) -> None:
        """Rebuild the FTS index."""
        self._fts.rebuild_fts_index()
    
    # =========================================================================
    # Indexing Operations - Delegated to IndexingOps
    # =========================================================================
    
    def upsert_memory(
        self,
        path: str,
        text: str,
        collection: str,
        embed_func,
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        importance: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        _skip_delete: bool = False,
    ) -> str:
        """Create or update a text memory, replacing old vectors for this path."""
        return self._indexer.upsert_memory(
            path=path,
            text=text,
            collection=collection,
            embed_func=embed_func,
            model=model,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            importance=importance,
            ttl_seconds=ttl_seconds,
            tags=tags,
            _skip_delete=_skip_delete,
        )
    
    def delete_memory(
        self,
        path: str,
        collection: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deactivate a memory and remove all associated vectors."""
        return self._indexer.delete_memory(
            path=path,
            collection=collection,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )
    
    def delete_path(
        self,
        path: str,
        collection: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        include_children: bool = False,
    ) -> Dict[str, Any]:
        """Delete a logical path and optionally all derived child assets."""
        return self._indexer.delete_path(
            path=path,
            collection=collection,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            include_children=include_children,
        )
    
    def index_folder(
        self,
        folder_path: str,
        collection: str,
        recursive: bool,
        include_globs: Optional[List[str]],
        exclude_globs: Optional[List[str]],
        embed_func,
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> Dict[str, Any]:
        """Index text files from a folder and return summary counts."""
        return self._indexer.index_folder(
            folder_path=folder_path,
            collection=collection,
            recursive=recursive,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            embed_func=embed_func,
            model=model,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            max_file_size_mb=max_file_size_mb,
        )
    
    def index_document(
        self,
        path: str,
        text: str,
        collection: str,
        model: str,
        embed_func,
        content_type: str = "text"
    ) -> str:
        """Full document indexing pipeline for text content."""
        return self._indexer.index_document(
            path=path,
            text=text,
            collection=collection,
            model=model,
            embed_func=embed_func,
            content_type=content_type,
        )

    def index_document_file(
        self,
        path: str,
        collection: str,
        embed_func,
        model: str = "Qwen3-VL-Embedding-2B",
        stored_path: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract and index a document file into structured text assets."""
        return self._indexer.index_document_file(
            path=path,
            collection=collection,
            embed_func=embed_func,
            model=model,
            stored_path=stored_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )
    
    def index_image(
        self,
        path: str,
        collection: str,
        embed_func,
        model: str = "Qwen3-VL-Embedding-2B",
        stored_path: Optional[str] = None,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> str:
        """Index an image file."""
        return self._indexer.index_image(
            path=path,
            collection=collection,
            embed_func=embed_func,
            model=model,
            stored_path=stored_path,
            title=title,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )

    def index_video(
        self,
        path: str,
        collection: str,
        embed_text_func,
        embed_image_func,
        embed_video_func=None,
        model: str = "Qwen3-VL-Embedding-2B",
        stored_path: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        frame_interval_seconds: float = 5.0,
        max_frames: int = 8,
    ) -> Dict[str, Any]:
        """Index a video into a top-level video embedding plus derived assets."""
        return self._indexer.index_video(
            path=path,
            collection=collection,
            embed_text_func=embed_text_func,
            embed_image_func=embed_image_func,
            embed_video_func=embed_video_func,
            model=model,
            stored_path=stored_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            frame_interval_seconds=frame_interval_seconds,
            max_frames=max_frames,
        )

    def ingest(
        self,
        collection: str,
        text: Optional[str],
        path: Optional[str],
        file_path: Optional[str],
        folder_path: Optional[str],
        recursive: bool,
        content_types: List[str],
        include_globs: Optional[List[str]],
        exclude_globs: Optional[List[str]],
        embed_text_func,
        embed_image_func,
        embed_video_func,
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> Dict[str, Any]:
        """Unified multimodal ingest for text/image/file/folder inputs."""
        return self._indexer.ingest(
            collection=collection,
            text=text,
            path=path,
            file_path=file_path,
            folder_path=folder_path,
            recursive=recursive,
            content_types=content_types,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            embed_text_func=embed_text_func,
            embed_image_func=embed_image_func,
            embed_video_func=embed_video_func,
            model=model,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            max_file_size_mb=max_file_size_mb,
        )
    
    # =========================================================================
    # Cache Operations
    # =========================================================================
    
    def get_cached(self, key: str) -> Optional[str]:
        """Get a cached value."""
        if self._cache_table is None:
            return None
        
        try:
            rows = list(self._cache_table.search()
                .where(_safe_filter("key", key))
                .limit(1)
                .to_list())
            if len(rows) == 0:
                return None
            return rows[0]["value"]
        except Exception:
            return None
    
    def set_cached(self, key: str, value: str) -> None:
        """Set a cached value."""
        if self._cache_table is None:
            return
        
        self._cache_table.merge_insert("key").when_matched_update_all().when_not_matched_insert_all().execute([{
            "key": key,
            "value": value,
            "created_at": int(time.time() * 1000),
        }])
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def count_embeddings(self) -> int:
        """Count total embeddings."""
        try:
            return self._embeddings_table.count_rows()
        except Exception:
            return 0
    
    def count_documents(self) -> int:
        """Count total documents."""
        try:
            return self._documents_table.count_rows()
        except Exception:
            return 0
