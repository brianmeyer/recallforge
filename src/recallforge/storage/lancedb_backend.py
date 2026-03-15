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
import struct
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pyarrow as pa
import lancedb

from .base import StorageBackend, SearchResult, Document
from ..documents import extract_document_artifacts, is_document_file
from ..video import extract_video_artifacts, is_video_file


# =============================================================================
# Structured Logging
# =============================================================================

logger = logging.getLogger("recallforge.storage")

# Enable trace mode via environment variable
TRACE_ENABLED = os.environ.get("RECALLFORGE_TRACE", "0") == "1"


def trace_log(operation: str, **kwargs) -> None:
    """Structured trace logging for debugging."""
    if TRACE_ENABLED:
        logger.debug(f"[TRACE] {operation}: {kwargs}")


# Default store directory
DEFAULT_INDEX_DIR = os.path.join(os.path.expanduser("~"), ".recallforge")


# =============================================================================
# Chunking Configuration
# =============================================================================

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4
CHUNK_WINDOW_CHARS = 200

# =============================================================================
# File Size Limits
# =============================================================================

DEFAULT_MAX_FILE_SIZE_MB = 100


# =============================================================================
# Helper Functions
# =============================================================================

# SQL metacharacters that indicate injection attempts (denylist approach).
# We block dangerous SQL characters rather than allowlisting safe ones,
# because file paths legitimately contain (), [], commas, #, &, etc.
_SQL_METACHARACTERS = frozenset("'\";\\\n\r\x00\x1a")
_SQL_COMMENT_PATTERNS = ("--", "/*", "*/")


def _validate_identifier(value: str, field_name: str = "value") -> str:
    """
    Validate a value is safe for SQL filter interpolation.

    Uses a denylist approach: blocks SQL metacharacters and comment patterns
    while allowing the full range of characters found in real-world file paths
    (parentheses, brackets, commas, hashes, ampersands, etc.).

    Args:
        value: The string value to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value if safe

    Raises:
        ValueError: If the value contains SQL metacharacters or invalid patterns
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")

    if not value.strip():
        raise ValueError(f"{field_name} is empty or whitespace-only")

    # Check for SQL comment patterns
    for pattern in _SQL_COMMENT_PATTERNS:
        if pattern in value:
            raise ValueError(f"{field_name} contains forbidden SQL pattern: {pattern}")

    # Check for SQL metacharacters
    if any(c in _SQL_METACHARACTERS for c in value):
        raise ValueError(f"{field_name} contains forbidden SQL metacharacters")

    return value


def _safe_filter(field: str, value: str) -> str:
    """
    Build a safe SQL filter clause for a field-value pair.

    Args:
        field: The field/column name (must be a valid identifier)
        value: The value to filter by

    Returns:
        A safe SQL filter string like "field = 'validated_value'"

    Raises:
        ValueError: If the field or value contains invalid characters
    """
    # Validate field name (stricter - no spaces allowed)
    if not re.match(r"^[\w_]+$", field):
        raise ValueError(f"Invalid field name: {field}")

    # Validate and escape the value
    validated = _validate_identifier(value, field)
    escaped = validated.replace("'", "''")
    return f"{field} = '{escaped}'"


def escape_sql(s: str) -> str:
    """
    DEPRECATED: Escape single quotes for SQL filters.

    This function is deprecated and will be removed in a future version.
    Use _safe_filter() or _validate_identifier() instead.

    Args:
        s: The string to escape

    Returns:
        The validated and escaped string

    Raises:
        ValueError: If the input contains SQL injection patterns
    """
    return _validate_identifier(s, "value").replace("'", "''")


def hash_content(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_file_bytes(file_path: str) -> str:
    """
    Compute SHA-256 hash of file contents + mtime for cache invalidation.
    
    This ensures that:
    1. Same content at same path produces same hash
    2. Modified file (different mtime) triggers re-indexing
    
    Args:
        file_path: Absolute path to file
        
    Returns:
        SHA-256 hash string (hex)
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {file_path}")
    
    # Hash the file contents
    h = hashlib.sha256()
    
    # Read file in chunks for memory efficiency
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except IOError as e:
        logger.error(f"hash_file_bytes: failed to read {file_path}: {e}")
        raise
    
    # Include mtime for cache invalidation
    try:
        mtime = path.stat().st_mtime
        h.update(struct.pack(">d", mtime))
    except OSError as e:
        logger.warning(f"hash_file_bytes: failed to get mtime for {file_path}: {e}")
    
    return h.hexdigest()


def extract_title(content: str, filename: str) -> str:
    """Extract title from content or filename."""
    match = re.match(r"^##?\s+(.+)$", content, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        if title not in ("📝 Notes", "Notes"):
            return title
        match = re.search(r"\n##\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    title_prop = re.search(r"^#\+TITLE:\s*(.+)$", content, re.MULTILINE)
    if title_prop:
        return title_prop.group(1).strip()
    
    return os.path.splitext(os.path.basename(filename))[0]


def get_docid(hash_str: str) -> str:
    """Generate short docid from hash."""
    return hash_str[:6]


# =============================================================================
# Chunking
# =============================================================================

@dataclass
class BreakPoint:
    pos: int
    score: int
    type: str


@dataclass
class CodeFenceRegion:
    start: int
    end: int


BREAK_PATTERNS = [
    (r"\n#{1}(?!#)", 100, "h1"),
    (r"\n#{2}(?!#)", 90, "h2"),
    (r"\n#{3}(?!#)", 80, "h3"),
    (r"\n#{4}(?!#)", 70, "h4"),
    (r"\n#{5}(?!#)", 60, "h5"),
    (r"\n#{6}(?!#)", 50, "h6"),
    (r"\n```", 80, "codeblock"),
    (r"\n(?:---|\*\*\*|___)\s*\n", 60, "hr"),
    (r"\n\n+", 20, "blank"),
    (r"\n[-*]\s", 5, "list"),
    (r"\n\d+\.\s", 5, "numlist"),
    (r"\n", 1, "newline"),
]


def scan_break_points(text: str) -> List[BreakPoint]:
    """Find all potential break points in text."""
    seen: Dict[int, BreakPoint] = {}
    
    for pattern, score, btype in BREAK_PATTERNS:
        for match in re.finditer(pattern, text):
            pos = match.start()
            existing = seen.get(pos)
            if existing is None or score > existing.score:
                seen[pos] = BreakPoint(pos, score, btype)
    
    return sorted(seen.values(), key=lambda b: b.pos)


def find_code_fences(text: str) -> List[CodeFenceRegion]:
    """Find all code fence regions in text."""
    regions: List[CodeFenceRegion] = []
    in_fence = False
    fence_start = 0
    
    for match in re.finditer(r"\n```", text):
        if not in_fence:
            fence_start = match.start()
            in_fence = True
        else:
            regions.append(CodeFenceRegion(fence_start, match.end()))
            in_fence = False
    
    if in_fence:
        regions.append(CodeFenceRegion(fence_start, len(text)))
    
    return regions


def is_inside_code_fence(pos: int, fences: List[CodeFenceRegion]) -> bool:
    """Check if position is inside a code fence."""
    return any(f.start < pos < f.end for f in fences)


def find_best_cutoff(
    break_points: List[BreakPoint],
    target_pos: int,
    window_chars: int = CHUNK_WINDOW_CHARS,
    decay_factor: float = 0.7,
    code_fences: List[CodeFenceRegion] = None
) -> int:
    """Find the best break point near target position."""
    if code_fences is None:
        code_fences = []
    
    window_start = target_pos - window_chars
    best_score = -1
    best_pos = target_pos
    
    for bp in break_points:
        if bp.pos < window_start:
            continue
        if bp.pos > target_pos:
            break
        if is_inside_code_fence(bp.pos, code_fences):
            continue
        
        distance = target_pos - bp.pos
        normalized_dist = distance / window_chars
        multiplier = 1.0 - (normalized_dist * normalized_dist) * decay_factor
        final_score = bp.score * multiplier
        
        if final_score > best_score:
            best_score = final_score
            best_pos = bp.pos
    
    return best_pos


def chunk_document(
    content: str,
    max_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    window_chars: int = CHUNK_WINDOW_CHARS
) -> List[Dict[str, Any]]:
    """Split document into overlapping chunks at natural break points."""
    if len(content) <= max_chars:
        return [{"text": content, "pos": 0}]
    
    break_points = scan_break_points(content)
    code_fences = find_code_fences(content)
    chunks: List[Dict[str, Any]] = []
    char_pos = 0
    
    while char_pos < len(content):
        target_end = min(char_pos + max_chars, len(content))
        end_pos = target_end
        
        if end_pos < len(content):
            best = find_best_cutoff(break_points, target_end, window_chars, 0.7, code_fences)
            if best > char_pos and best <= target_end:
                end_pos = best
        
        if end_pos <= char_pos:
            end_pos = min(char_pos + max_chars, len(content))
        
        chunks.append({"text": content[char_pos:end_pos], "pos": char_pos})
        
        if end_pos >= len(content):
            break
        
        char_pos = end_pos - overlap_chars
        last_chunk_pos = chunks[-1]["pos"]
        if char_pos <= last_chunk_pos:
            char_pos = end_pos
    
    return chunks


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
        self._bm25_fallback_max_rows = max(
            100,
            int(os.environ.get("RECALLFORGE_BM25_FALLBACK_MAX_ROWS", "5000")),
        )
    
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
        # Flush any pending FTS rebuild before closing
        if self._fts_needs_rebuild or self._fts_rebuild_pending > 0:
            self._do_fts_rebuild()
        
        self._conn = None
        self._embeddings_table = None
        self._documents_table = None
        self._content_table = None
        self._cache_table = None
    
    def _schedule_fts_rebuild(self) -> None:
        """
        Schedule a debounced FTS rebuild.

        Instead of rebuilding on every write, we track pending writes
        and rebuild only when threshold is hit or on explicit request.

        In bulk mode, all rebuilds are deferred until bulk mode ends.
        """
        self._fts_rebuild_pending += 1
        self._fts_needs_rebuild = True

        trace_log("schedule_fts_rebuild", pending=self._fts_rebuild_pending, bulk_mode=self._bulk_mode)

        # In bulk mode, defer all rebuilds until bulk ends
        if self._bulk_mode:
            return

        # Immediate rebuild if threshold exceeded
        if self._fts_rebuild_pending >= self.FTS_REBUILD_PENDING_THRESHOLD:
            self._do_fts_rebuild()
    
    def _do_fts_rebuild(self) -> None:
        """
        Execute the actual FTS rebuild with rate limiting.
        """
        now = time.time()
        elapsed = now - self._fts_last_rebuild

        # Rate limit: don't rebuild more than once per min interval
        if elapsed < self.FTS_REBUILD_MIN_INTERVAL and self._fts_rebuild_pending < self.FTS_REBUILD_PENDING_THRESHOLD:
            trace_log("fts_rebuild_skipped", reason="rate_limited", elapsed=elapsed)
            return

        trace_log("fts_rebuild_executing", pending=self._fts_rebuild_pending)

        self._fts_last_rebuild = now
        self._fts_rebuild_pending = 0
        self._fts_needs_rebuild = False

        try:
            self.ensure_fts_index(force_rebuild=True)
        except Exception as e:
            logger.error(f"FTS rebuild failed: {e}")
            # Don't re-raise; FTS rebuild failure is non-fatal

    def bulk_mode(self):
        """
        Context manager to defer FTS rebuilds during bulk operations.

        Usage:
            with storage.bulk_mode():
                for doc in documents:
                    storage.upsert_memory(...)
            # FTS rebuild happens once at context exit

        Returns:
            Context manager that defers FTS rebuilds.
        """
        return _BulkModeContext(self)

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

        # Check for existing
        try:
            existing = list(self._documents_table.search()
                .where(ns_filter)
                .limit(1)
                .to_list())

            if len(existing) > 0:
                doc_id = existing[0]["id"]
                self._documents_table.update(
                    where=_safe_filter("id", doc_id),
                    values={
                        "title": title,
                        "content_hash": content_hash,
                        "content_type": content_type,
                        "active": 1,
                        "updated_at": modified_ts,
                        "user_id": user_id,
                        "session_id": session_id,
                        "project_id": project_id,
                        "profile": profile,
                    }
                )
                return doc_id
        except Exception as e:
            logger.warning(f"insert_document: failed to check existing doc {collection}/{file_path}: {e}")

        # Insert new
        doc_id = str(uuid.uuid4())
        self._documents_table.add([{
            "id": doc_id,
            "collection": collection,
            "file_path": file_path,
            "title": title,
            "content_hash": content_hash,
            "content_type": content_type,
            "active": 1,
            "created_at": created_ts,
            "updated_at": modified_ts,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }])

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
        try:
            existing = list(self._content_table.search()
                .where(_safe_filter("hash", hash_str))
                .limit(1)
                .to_list())
            if len(existing) > 0:
                return
        except Exception as e:
            logger.warning(f"insert_content: failed to check existing hash {hash_str[:8]}: {e}")
        
        self._content_table.add([{
            "hash": hash_str,
            "doc": content,
            "content_type": content_type,
            "created_at": int(time.time() * 1000),
        }])
    
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

        # Delete existing
        try:
            self._embeddings_table.delete(_safe_filter("hash_seq", hash_seq))
        except Exception as e:
            logger.debug(f"insert_embedding: no existing embedding to delete for {hash_seq}: {e}")

        trace_log("insert_embedding", hash_seq=hash_seq, collection=collection, file_path=file_path, seq=seq,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile,
                  importance=importance, ttl_seconds=ttl_seconds, tags=tags)

        self._embeddings_table.add([{
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
        }])
    
    def has_vectors(self) -> bool:
        """Check if index has any vectors."""
        try:
            count = self._embeddings_table.count_rows()
            return count > 0
        except Exception as e:
            logger.warning(f"has_vectors: failed to count rows: {e}")
            return False
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    def ensure_fts_index(self, force_rebuild: bool = False) -> None:
        """Ensure the FTS index exists."""
        if self._embeddings_table is None:
            return
        
        try:
            row_count = self._embeddings_table.count_rows()
        except Exception as e:
            logger.error(f"ensure_fts_index: failed to count rows: {e}")
            return
        
        if row_count == 0:
            return
        
        if force_rebuild:
            try:
                self._embeddings_table.create_fts_index("text_body", replace=True)
                trace_log("fts_index_created", force=True, rows=row_count)
            except Exception as e:
                logger.error(f"ensure_fts_index: failed to create FTS index: {e}")
            return
        
        try:
            indices = self._embeddings_table.list_indices()
        except Exception as e:
            logger.warning(f"ensure_fts_index: failed to list indices: {e}")
            return
        
        has_fts = any(
            "text_body" in (i.columns or []) and "FTS" in str(i.index_type or i.type or "").upper()
            for i in indices
        )
        
        if not has_fts:
            try:
                self._embeddings_table.create_fts_index("text_body", replace=True)
                trace_log("fts_index_created", force=False, rows=row_count)
            except Exception as e:
                logger.error(f"ensure_fts_index: failed to create FTS index: {e}")
    
    def rebuild_fts_index(self) -> None:
        """Rebuild the FTS index."""
        if self._embeddings_table is None:
            return
        
        row_count = self._embeddings_table.count_rows()
        if row_count == 0:
            return
        
        indices = self._embeddings_table.list_indices()
        for idx in indices:
            if "text_body" in (idx.columns or []) and "FTS" in str(idx.index_type or idx.type or "").upper():
                self._embeddings_table.drop_index(idx.name)
                break
        
        self.ensure_fts_index()
    
    def _bm25_fallback(
        self,
        query: str,
        limit: int = 20,
        collection: Optional[str] = None,
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> List[SearchResult]:
        """In-memory BM25 fallback when FTS index fails."""
        try:
            filter_parts = [self._get_ttl_filter()]
            if collection:
                filter_parts.append(_safe_filter("collection", collection))
            if content_type:
                filter_parts.append(_safe_filter("content_type", content_type))
            if user_id is not None:
                filter_parts.append(_safe_filter("user_id", user_id))
            if session_id is not None:
                filter_parts.append(_safe_filter("session_id", session_id))
            if project_id is not None:
                filter_parts.append(_safe_filter("project_id", project_id))
            if profile is not None:
                filter_parts.append(_safe_filter("profile", profile))
            filter_clause = " AND ".join(filter_parts)

            # Keep fallback bounded to avoid OOM on large corpora.
            row_limit = min(self._bm25_fallback_max_rows, max(limit * 50, 200))
            builder = (
                self._embeddings_table.search()
                .where(filter_clause)
                .select(["collection", "file_path", "content_hash", "content_type", "title",
                         "text_body", "embedded_at", "modified_at", "user_id", "session_id",
                         "project_id", "profile", "expires_at"])
                .limit(row_limit)
            )
            rows = builder.to_pandas()
        except Exception:
            return []

        if rows.empty:
            return []

        query_terms = re.findall(r'\w+', query.lower())
        if not query_terms:
            return []

        N = len(rows)
        avgdl = rows["text_body"].str.len().mean() or 1
        k1, b = 1.5, 0.75

        doc_freqs: Dict[str, int] = defaultdict(int)
        for text in rows["text_body"]:
            seen_terms = set(re.findall(r'\w+', (text or "").lower()))
            for t in seen_terms:
                doc_freqs[t] += 1

        results: List[SearchResult] = []
        for _, row in rows.iterrows():
            text = row.get("text_body") or ""
            text_lower = text.lower()
            doc_len = len(text)
            score = 0.0
            for term in query_terms:
                df_t = doc_freqs.get(term, 0)
                if df_t == 0:
                    continue
                idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
                tf = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                tf_comp = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
                score += idf * tf_comp
            if score > 0:
                results.append(self._make_search_result(dict(row), score, "fts"))

        results.sort(key=lambda x: x.score, reverse=True)
        if results:
            max_s = results[0].score
            for r in results:
                r.score = r.score / max_s if max_s > 0 else 0
        return results[:limit]
    
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
    ) -> List[SearchResult]:
        """Full-text search using LanceDB Tantivy."""
        if self._embeddings_table is None:
            return []

        trimmed = query.strip()
        if not trimmed:
            return []

        trace_log("search_fts_start", query=trimmed[:50], limit=limit, collection=collection, content_type=content_type,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        self.ensure_fts_index()

        # Build filter including TTL and namespace fields
        filter_parts = [self._get_ttl_filter()]

        if collection:
            filter_parts.append(_safe_filter("collection", collection))
        if content_type:
            filter_parts.append(_safe_filter("content_type", content_type))
        if user_id is not None:
            filter_parts.append(_safe_filter("user_id", user_id))
        if session_id is not None:
            filter_parts.append(_safe_filter("session_id", session_id))
        if project_id is not None:
            filter_parts.append(_safe_filter("project_id", project_id))
        if profile is not None:
            filter_parts.append(_safe_filter("profile", profile))

        filter_clause = " AND ".join(filter_parts) if filter_parts else None

        # Run FTS search
        try:
            builder = self._embeddings_table.search(trimmed, query_type="fts").limit(limit * 2)
            if filter_clause:
                builder = builder.where(filter_clause)
            results = builder.to_list()
        except Exception as e:
            logger.warning(f"search_fts: FTS index failed, using BM25 fallback: {e}")
            return self._bm25_fallback(trimmed, limit, collection, content_type, user_id, session_id, project_id, profile)

        # Empty FTS results are normal (no matches), not an error.
        # Do NOT run full-table BM25 fallback - only use fallback on true FTS errors.
        if not results:
            trace_log("search_fts_empty", query=trimmed[:50])
            return []

        # Normalize scores
        max_score = max(r.get("_score", 0) for r in results) or 1

        # Dedupe by filepath
        seen: Dict[str, SearchResult] = {}
        for r in results:
            filepath = f"recallforge://{r['collection']}/{r['file_path']}"
            score = r.get("_score", 0) / max_score

            if filepath in seen:
                if score > seen[filepath].score:
                    seen[filepath] = self._make_search_result(r, score, "fts")
            else:
                seen[filepath] = self._make_search_result(r, score, "fts")

        final_results = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:limit]
        trace_log("search_fts_done", count=len(final_results), query=trimmed[:50])
        return final_results
    
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
    ) -> List[SearchResult]:
        """Vector similarity search."""
        if self._embeddings_table is None:
            return []

        if not self.has_vectors():
            return []

        trace_log("search_vec_start", limit=limit, collection=collection, content_type=content_type,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        # Build filter including TTL and namespace fields
        filter_parts = [self._get_ttl_filter()]

        if collection:
            filter_parts.append(_safe_filter("collection", collection))
        if content_type:
            filter_parts.append(_safe_filter("content_type", content_type))
        if user_id is not None:
            filter_parts.append(_safe_filter("user_id", user_id))
        if session_id is not None:
            filter_parts.append(_safe_filter("session_id", session_id))
        if project_id is not None:
            filter_parts.append(_safe_filter("project_id", project_id))
        if profile is not None:
            filter_parts.append(_safe_filter("profile", profile))

        filter_clause = " AND ".join(filter_parts) if filter_parts else None

        # Run vector search
        builder = self._embeddings_table.search(vector, query_type="vector").metric("cosine").limit(limit * 2)
        if filter_clause:
            builder = builder.where(filter_clause)

        results = builder.to_list()

        if not results:
            return []

        # Dedupe by filepath
        seen: Dict[str, SearchResult] = {}
        for r in results:
            filepath = f"recallforge://{r['collection']}/{r['file_path']}"
            distance = r.get("_distance", 1.0)
            score = 1.0 - distance / 2.0

            if filepath in seen:
                if score > seen[filepath].score:
                    seen[filepath] = self._make_search_result(r, score, "vec")
            else:
                seen[filepath] = self._make_search_result(r, score, "vec")

        final_results = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:limit]
        trace_log("search_vec_done", count=len(final_results))
        return final_results
    
    def _make_search_result(self, row: Dict[str, Any], score: float, source: str) -> SearchResult:
        """Convert LanceDB row to SearchResult.
        
        PERFORMANCE OPTIMIZATION: Prefer text_body from embeddings row over get_content() lookup.
        - text_body is already available in the row (from embeddings table query)
        - Only call get_content() as fallback when text_body is empty/None
        This avoids N+1 lookups to content table for every search result.
        """
        collection = row.get("collection", "")
        file_path = row.get("file_path", "")
        content_hash = row.get("content_hash", "")
        content_type = row.get("content_type", "text")

        # P0 OPTIMIZATION: Prefer text_body (already in row) over get_content() lookup
        body = row.get("text_body") or ""
        if not body:
            # Fallback only when text_body is empty - lazy load for final output
            body = self.get_content(content_hash) or ""

        return SearchResult(
            filepath=f"recallforge://{collection}/{file_path}",
            display_path=f"{collection}/{file_path}",
            title=row.get("title", file_path) or "",
            context=None,
            hash=content_hash,
            docid=get_docid(content_hash),
            collection=collection,
            modified_at="",
            body_length=len(body),
            score=score,
            source=source,
            content_type=content_type,
            chunk_pos=row.get("pos", 0) or 0,
            body=body,
            user_id=row.get("user_id"),
            session_id=row.get("session_id"),
            project_id=row.get("project_id"),
            profile=row.get("profile"),
        )
    
    def _get_ttl_filter(self) -> str:
        """Generate filter clause to exclude expired entries.
        
        Returns SQL WHERE clause fragment that filters out expired entries:
        - expires_at IS NULL (no TTL set)
        - expires_at > current_time (not yet expired)
        """
        now_ms = int(time.time() * 1000)
        # Exclude entries where expires_at is set and less than now
        return f"(expires_at IS NULL OR expires_at > {now_ms})"
    
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

    def list_collections(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> List[str]:
        """Return sorted list of unique collection names, with optional namespace filters."""
        if self._embeddings_table is None:
            return []

        try:
            filter_parts: List[str] = []
            if user_id is not None:
                filter_parts.append(_safe_filter("user_id", user_id))
            if session_id is not None:
                filter_parts.append(_safe_filter("session_id", session_id))
            if project_id is not None:
                filter_parts.append(_safe_filter("project_id", project_id))
            if profile is not None:
                filter_parts.append(_safe_filter("profile", profile))

            builder = self._embeddings_table.search().select(["collection"])
            if filter_parts:
                builder = builder.where(" AND ".join(filter_parts))

            rows = builder.limit(100_000).to_list()
            seen: set = set()
            for row in rows:
                val = row.get("collection")
                if val:
                    seen.add(val)
            return sorted(seen)
        except Exception as e:
            logger.warning(f"list_collections: failed: {e}")
            return []

    def list_namespaces(
        self,
        collection: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Return unique namespace combinations (user_id, session_id, project_id, profile)."""
        if self._embeddings_table is None:
            return []

        try:
            filter_parts: List[str] = []
            if collection is not None:
                filter_parts.append(_safe_filter("collection", collection))

            builder = self._embeddings_table.search().select(
                ["user_id", "session_id", "project_id", "profile"]
            )
            if filter_parts:
                builder = builder.where(" AND ".join(filter_parts))

            rows = builder.limit(100_000).to_list()
            seen: set = set()
            for row in rows:
                key = (
                    row.get("user_id") or "",
                    row.get("session_id") or "",
                    row.get("project_id") or "",
                    row.get("profile") or "",
                )
                seen.add(key)

            result = []
            for user_id, session_id, project_id, profile in sorted(seen):
                ns: Dict[str, str] = {}
                if user_id:
                    ns["user_id"] = user_id
                if session_id:
                    ns["session_id"] = session_id
                if project_id:
                    ns["project_id"] = project_id
                if profile:
                    ns["profile"] = profile
                result.append(ns)
            return result
        except Exception as e:
            logger.warning(f"list_namespaces: failed: {e}")
            return []
    
    # =========================================================================
    # High-Level Indexing
    # =========================================================================
    
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
        if content_type != "text":
            raise ValueError("index_document supports only text content")
        return self.upsert_memory(
            path=path,
            text=text,
            collection=collection,
            embed_func=embed_func,
            model=model,
        )
    
    def _embed_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        embed_func,
    ) -> List[List[float]]:
        """Embed chunks with batch support and safe fallbacks.

        Supports:
        - embed_func.embed_texts(texts)
        - embed_func(texts)
        - embed_func.embed_text(text) / embed_func(text) per item fallback
        """
        texts = [chunk["text"] for chunk in chunks]

        if not texts:
            return []

        if hasattr(embed_func, "embed_texts"):
            try:
                vectors = embed_func.embed_texts(texts)
                if hasattr(vectors, "tolist"):
                    vectors = vectors.tolist()
                vectors = list(vectors)
                if len(vectors) == len(texts):
                    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
            except Exception as e:
                logger.debug(f"batch embed via embed_texts failed, falling back: {e}")

        try:
            vectors = embed_func(texts)
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            if isinstance(vectors, (list, tuple)) and len(vectors) == len(texts):
                first = vectors[0]
                if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
                    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
        except Exception:
            pass

        single_embed = embed_func.embed_text if hasattr(embed_func, "embed_text") else embed_func
        output: List[List[float]] = []
        for text in texts:
            vector = single_embed(text)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            output.append(list(vector))
        return output

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
        """Create or update a text memory, replacing old vectors for this path.
        
        Args:
            path: Memory path key within collection
            text: Memory content text
            collection: Collection name
            embed_func: Function/object to embed text into vectors.
                Supports embed_func(text), embed_func.embed_text(text),
                embed_func(texts), or embed_func.embed_texts(texts).
            model: Embedding model name
            importance: Optional importance score (0.0-1.0)
            ttl_seconds: Optional time-to-live in seconds (0 or None = no expiration)
            tags: Optional list of string tags
            _skip_delete: Internal optimization flag for callers that already
                deleted path-scoped vectors in the same namespace.
        
        Returns:
            Content hash of the stored memory
        """
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path is required")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required")

        trace_log("upsert_memory_start", path=normalized_path, collection=collection,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile,
                  importance=importance, ttl_seconds=ttl_seconds, tags=tags, _skip_delete=_skip_delete)


        content_hash = hash_content(text)
        title = extract_title(text, normalized_path)

        if not _skip_delete:
            # Build namespace filter for deletion
            del_filter = f"{_safe_filter('collection', collection)} AND {_safe_filter('file_path', normalized_path)}"
            if user_id is not None:
                del_filter += f" AND {_safe_filter('user_id', user_id)}"
            if session_id is not None:
                del_filter += f" AND {_safe_filter('session_id', session_id)}"
            if project_id is not None:
                del_filter += f" AND {_safe_filter('project_id', project_id)}"
            if profile is not None:
                del_filter += f" AND {_safe_filter('profile', profile)}"

            # Remove prior vectors for this memory path to prevent duplicate chunks.
            try:
                self._embeddings_table.delete(del_filter)
            except Exception as e:
                logger.warning(f"upsert_memory: failed to delete old vectors for {collection}/{normalized_path}: {e}")

        self.insert_content(content_hash, text, "text")
        self.insert_document(
            collection, normalized_path, title, content_hash, "text",
            user_id=user_id, session_id=session_id, project_id=project_id, profile=profile
        )

        chunks = chunk_document(text)
        vectors = self._embed_chunks_batch(chunks, embed_func)
        for i, chunk in enumerate(chunks):
            self.insert_embedding(
                content_hash=content_hash,
                seq=i,
                pos=chunk["pos"],
                vector=vectors[i],
                model=model,
                collection=collection,
                file_path=normalized_path,
                title=title,
                text_body=chunk["text"],
                content_type="text",
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
                importance=importance,
                ttl_seconds=ttl_seconds,
                tags=tags,

            )

        # Schedule debounced FTS rebuild instead of immediate rebuild
        self._schedule_fts_rebuild()

        trace_log("upsert_memory_done", path=normalized_path, hash=content_hash[:8], chunks=len(chunks))
        return content_hash

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
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path is required")

        trace_log("delete_memory_start", path=normalized_path, collection=collection,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        # Build namespace filter
        del_filter = f"{_safe_filter('collection', collection)} AND {_safe_filter('file_path', normalized_path)}"
        if user_id is not None:
            del_filter += f" AND {_safe_filter('user_id', user_id)}"
        if session_id is not None:
            del_filter += f" AND {_safe_filter('session_id', session_id)}"
        if project_id is not None:
            del_filter += f" AND {_safe_filter('project_id', project_id)}"
        if profile is not None:
            del_filter += f" AND {_safe_filter('profile', profile)}"

        removed_vectors = 0
        try:
            removed_vectors = len(
                self._embeddings_table.search()
                .where(del_filter)
                .to_list()
            )
        except Exception as e:
            logger.warning(f"delete_memory: failed to count vectors for {collection}/{normalized_path}: {e}")
            removed_vectors = 0

        try:
            self._embeddings_table.delete(del_filter)
        except Exception as e:
            logger.error(f"delete_memory: failed to delete embeddings for {collection}/{normalized_path}: {e}")

        self.deactivate_document(collection, normalized_path)

        # Schedule debounced FTS rebuild
        self._schedule_fts_rebuild()

        trace_log("delete_memory_done", path=normalized_path, removed_vectors=removed_vectors)
        return {
            "success": True,
            "path": normalized_path,
            "collection": collection,
            "removed_vectors": removed_vectors,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }

    def _video_frames_dir_for_logical_path(self, logical_path: str) -> Path:
        artifact_root = Path(self._store_path or DEFAULT_INDEX_DIR) / "video_frames"
        digest = hashlib.sha1(logical_path.encode("utf-8")).hexdigest()[:16]
        return artifact_root / digest

    def _delete_video_frame_artifacts(self, logical_path: str) -> None:
        output_dir = self._video_frames_dir_for_logical_path(logical_path)
        if not output_dir.exists():
            return

        try:
            trash_bin = shutil.which("trash")
            if trash_bin:
                subprocess.run([trash_bin, str(output_dir)], check=True, capture_output=True, text=True)
            else:
                shutil.rmtree(output_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(
                "delete_path: failed to cleanup video frame artifacts for %s at %s: %s",
                logical_path,
                output_dir,
                e,
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
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path is required")

        removed_vectors = self._delete_path_entries(
            collection=collection,
            logical_path=normalized_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            include_children=include_children,
        )
        if include_children:
            self._delete_video_frame_artifacts(normalized_path)
        self._schedule_fts_rebuild()
        return {
            "success": True,
            "path": normalized_path,
            "collection": collection,
            "removed_vectors": removed_vectors,
            "include_children": include_children,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }

    def _is_text_file(self, file_path: Path) -> bool:
        """Best-effort text file detection."""
        try:
            with file_path.open("rb") as f:
                sample = f.read(8192)
        except Exception:
            return False

        if b"\x00" in sample:
            return False

        return True

    def _read_text_robust(self, file_path: Path) -> Optional[str]:
        """Read text file using common encodings with replacement fallback."""
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception:
                return None

        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    def _is_image_file(self, file_path: Path) -> bool:
        """Best-effort image file detection by extension."""
        return file_path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic"
        }

    def _is_video_file(self, file_path: Path) -> bool:
        """Best-effort video file detection by extension."""
        return is_video_file(file_path)

    def _is_document_file(self, file_path: Path) -> bool:
        """Best-effort office-document detection by extension."""
        return is_document_file(file_path)

    def _iter_folder_files(self, root: Path, recursive: bool):
        """Iterate files while pruning common heavyweight directories."""
        if not recursive:
            for child in sorted(root.iterdir()):
                if child.is_file():
                    yield child
            return

        pruned_dirnames = {".git", "node_modules", "__pycache__", ".venv", "venv"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                dirname for dirname in sorted(dirnames)
                if dirname not in pruned_dirnames and not dirname.startswith(".")
            ]
            for filename in sorted(filenames):
                yield Path(dirpath) / filename

    def _namespace_filters(
        self,
        collection: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> List[str]:
        filters = [_safe_filter("collection", collection)]
        if user_id is not None:
            filters.append(_safe_filter("user_id", user_id))
        if session_id is not None:
            filters.append(_safe_filter("session_id", session_id))
        if project_id is not None:
            filters.append(_safe_filter("project_id", project_id))
        if profile is not None:
            filters.append(_safe_filter("profile", profile))
        return filters

    def _delete_path_entries(
        self,
        collection: str,
        logical_path: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        content_type: Optional[str] = None,
        include_children: bool = False,
    ) -> int:
        filters = self._namespace_filters(
            collection=collection,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )
        # Validate and escape the logical_path for LIKE patterns
        validated_path = _validate_identifier(logical_path, "logical_path")
        escaped_path = validated_path.replace("'", "''")
        if include_children:
            filters.append(f"(file_path = '{escaped_path}' OR file_path LIKE '{escaped_path}::%')")
        else:
            filters.append(f"file_path = '{escaped_path}'")
        if content_type is not None:
            filters.append(_safe_filter("content_type", content_type))

        filter_clause = " AND ".join(filters)
        removed_vectors = 0
        try:
            removed_vectors = len(self._embeddings_table.search().where(filter_clause).to_list())
        except Exception as e:
            logger.debug(f"_delete_path_entries: failed to count rows for {logical_path}: {e}")

        try:
            self._embeddings_table.delete(filter_clause)
        except Exception as e:
            logger.debug(f"_delete_path_entries: failed to delete embeddings for {logical_path}: {e}")

        doc_filters = self._namespace_filters(
            collection=collection,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )
        if include_children:
            doc_filters.append(f"(file_path = '{escaped_path}' OR file_path LIKE '{escaped_path}::%')")
        else:
            doc_filters.append(f"file_path = '{escaped_path}'")
        if content_type is not None:
            doc_filters.append(_safe_filter("content_type", content_type))

        try:
            self._documents_table.update(
                where=" AND ".join(doc_filters),
                values={"active": 0, "updated_at": int(time.time() * 1000)},
            )
        except Exception as e:
            logger.debug(f"_delete_path_entries: failed to deactivate documents for {logical_path}: {e}")

        return removed_vectors

    def _matches_globs(self, rel_path: str, include_globs: Optional[List[str]], exclude_globs: Optional[List[str]]) -> bool:
        include = include_globs or ["**/*"]
        exclude = exclude_globs or []
        if include and not any(fnmatch.fnmatch(rel_path, pattern) for pattern in include):
            return False
        if exclude and any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude):
            return False
        return True

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
        root = Path(folder_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Folder not found: {folder_path}")

        trace_log("index_folder_start", folder=str(root), collection=collection, recursive=recursive,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        include = include_globs or ["**/*"]
        exclude = exclude_globs or []

        indexed = 0
        skipped = 0
        errors = 0
        skipped_details: List[Dict[str, str]] = []

        def mark_skipped(item_path: str, reason: str) -> None:
            nonlocal skipped
            skipped += 1
            skipped_details.append({"path": item_path, "reason": reason})

        # Use bulk mode to defer FTS rebuilds until the end
        with self.bulk_mode():
            for file_path in self._iter_folder_files(root, recursive):
                rel = file_path.relative_to(root).as_posix()
                if include and not any(fnmatch.fnmatch(rel, pattern) for pattern in include):
                    mark_skipped(rel, "glob_mismatch")
                    continue
                if exclude and any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
                    mark_skipped(rel, "excluded")
                    continue

                # Check file size before processing
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > max_file_size_mb * 1024 * 1024:
                        logger.warning("Skipping %s: file size %dMB exceeds limit %dMB",
                                       file_path, file_size // (1024 * 1024), max_file_size_mb)
                        mark_skipped(rel, "file_too_large")
                        continue
                except OSError as e:
                    logger.warning("Could not get size for %s: %s", file_path, e)
                    mark_skipped(rel, "unreadable")
                    continue

                if not self._is_text_file(file_path):
                    mark_skipped(rel, "not_text_file")
                    continue

                text = self._read_text_robust(file_path)
                if text is None or not text.strip():
                    mark_skipped(rel, "empty_content")
                    continue

                try:
                    self.upsert_memory(
                        path=rel,
                        text=text,
                        collection=collection,
                        embed_func=embed_func,
                        model=model,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                    )
                    indexed += 1
                except Exception as e:
                    if "already indexed" in str(e).lower():
                        mark_skipped(rel, "dedup")
                        continue
                    logger.error(f"index_folder: failed to index {rel}: {e}")
                    errors += 1
        # FTS rebuild happens once at context exit

        trace_log("index_folder_done", folder=str(root), indexed=indexed, skipped=skipped, errors=errors)
        return {
            "success": True,
            "folder_path": str(root),
            "collection": collection,
            "indexed": indexed,
            "skipped": skipped,
            "errors": errors,
            "total_seen": indexed + skipped + errors,
            "skipped_details": skipped_details,
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }

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
        """Unified multimodal ingest for text, file, or folder inputs."""
        content_types = content_types or ["text", "image", "video", "document"]
        allowed = set(content_types)
        if not allowed.issubset({"text", "image", "video", "document"}):
            raise ValueError("content_types must be subset of ['text', 'image', 'video', 'document']")

        trace_log("ingest_start", collection=collection, text=text is not None, file_path=file_path, folder_path=folder_path,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        summary = {
            "success": True,
            "collection": collection,
            "indexed_text": 0,
            "indexed_images": 0,
            "indexed_videos": 0,
            "indexed_documents": 0,
            "indexed_document_sections": 0,
            "indexed_video_embeddings": 0,
            "indexed_video_frames": 0,
            "indexed_video_transcripts": 0,
            "skipped": 0,
            "errors": 0,
            "items": [],
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id,
            "profile": profile,
        }

        def mark(
            item_path: str,
            item_type: str,
            status: str,
            error: Optional[str] = None,
            reason: Optional[str] = None,
        ) -> None:
            item = {"path": item_path, "type": item_type, "status": status}
            if error:
                item["error"] = error
            if reason:
                item["reason"] = reason
            summary["items"].append(item)

        def ingest_single(candidate: Path, rel_hint: Optional[str] = None) -> None:
            candidate = candidate.expanduser().resolve()
            if not candidate.exists() or not candidate.is_file():
                summary["errors"] += 1
                mark(str(candidate), "unknown", "error", "file not found")
                return

            item_path = rel_hint or str(candidate)
            is_image = self._is_image_file(candidate)
            is_video = self._is_video_file(candidate)
            is_document = self._is_document_file(candidate)

            try:
                if is_image:
                    if "image" not in allowed:
                        summary["skipped"] += 1
                        mark(item_path, "image", "skipped", reason="not_in_content_types")
                        return
                    self.index_image(
                        path=str(candidate),
                        collection=collection,
                        embed_func=embed_image_func,
                        model=model,
                        stored_path=item_path,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                    )
                    summary["indexed_images"] += 1
                    mark(item_path, "image", "indexed")
                    return

                if is_video:
                    if "video" not in allowed:
                        summary["skipped"] += 1
                        mark(item_path, "video", "skipped", reason="not_in_content_types")
                        return
                    video_summary = self.index_video(
                        path=str(candidate),
                        collection=collection,
                        embed_text_func=embed_text_func,
                        embed_image_func=embed_image_func,
                        embed_video_func=embed_video_func,
                        model=model,
                        stored_path=item_path,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                    )
                    summary["indexed_videos"] += 1
                    summary["indexed_images"] += video_summary["indexed_frames"]
                    summary["indexed_text"] += video_summary["indexed_transcripts"]
                    summary["indexed_video_embeddings"] += video_summary.get("indexed_video_embeddings", 0)
                    summary["indexed_video_frames"] += video_summary["indexed_frames"]
                    summary["indexed_video_transcripts"] += video_summary["indexed_transcripts"]
                    mark(item_path, "video", "indexed")
                    return

                if is_document:
                    if "document" not in allowed:
                        summary["skipped"] += 1
                        mark(item_path, "document", "skipped", reason="not_in_content_types")
                        return
                    document_summary = self.index_document_file(
                        path=str(candidate),
                        collection=collection,
                        embed_func=embed_text_func,
                        model=model,
                        stored_path=item_path,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                    )
                    if document_summary.get("indexed_sections", 0) == 0:
                        summary["skipped"] += 1
                        mark(item_path, "document", "skipped", reason="empty_content")
                    else:
                        summary["indexed_documents"] += 1
                        summary["indexed_document_sections"] += document_summary["indexed_sections"]
                        summary["indexed_text"] += document_summary["indexed_sections"]
                        mark(item_path, "document", "indexed")
                    return

                if "text" not in allowed:
                    summary["skipped"] += 1
                    mark(item_path, "text", "skipped", reason="not_in_content_types")
                    return
                if not self._is_text_file(candidate):
                    summary["skipped"] += 1
                    mark(item_path, "text", "skipped", reason="not_text_file")
                    return
                body = self._read_text_robust(candidate)
                if body is None or not body.strip():
                    summary["skipped"] += 1
                    mark(item_path, "text", "skipped", reason="empty_content")
                    return
                self.upsert_memory(
                    path=item_path,
                    text=body,
                    collection=collection,
                    embed_func=embed_text_func,
                    model=model,
                    user_id=user_id,
                    session_id=session_id,
                    project_id=project_id,
                    profile=profile,
                )
                summary["indexed_text"] += 1
                mark(item_path, "text", "indexed")
            except Exception as e:
                logger.error(f"ingest: failed to index {item_path}: {e}")
                summary["errors"] += 1
                if is_image:
                    item_type = "image"
                elif is_video:
                    item_type = "video"
                elif is_document:
                    item_type = "document"
                else:
                    item_type = "text"
                mark(item_path, item_type, "error", str(e))

        # Use bulk mode to defer FTS rebuilds until the end
        with self.bulk_mode():
            if text is not None:
                if "text" not in allowed:
                    summary["skipped"] += 1
                    mark(path or "inline/skipped", "text", "skipped", reason="not_in_content_types")
                else:
                    if path:
                        text_path = path.strip() or "inline"
                    else:
                        import hashlib
                        text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
                        text_path = f"inline/{text_hash}"
                    self.upsert_memory(
                        path=text_path,
                        text=text,
                        collection=collection,
                        embed_func=embed_text_func,
                        model=model,
                        user_id=user_id,
                        session_id=session_id,
                        project_id=project_id,
                        profile=profile,
                    )
                    summary["indexed_text"] += 1
                    mark(text_path, "text", "indexed")

            if file_path:
                ingest_single(Path(file_path))

            if folder_path:
                root = Path(folder_path).expanduser().resolve()
                if not root.exists() or not root.is_dir():
                    raise ValueError(f"Folder not found: {folder_path}")
                for candidate in self._iter_folder_files(root, recursive):
                    rel = candidate.relative_to(root).as_posix()
                    if include_globs and not any(fnmatch.fnmatch(rel, pattern) for pattern in include_globs):
                        summary["skipped"] += 1
                        mark(rel, "unknown", "skipped", reason="glob_mismatch")
                        continue
                    if exclude_globs and any(fnmatch.fnmatch(rel, pattern) for pattern in exclude_globs):
                        summary["skipped"] += 1
                        mark(rel, "unknown", "skipped", reason="excluded")
                        continue

                    # Check file size before processing
                    try:
                        file_size = os.path.getsize(candidate)
                        if file_size > max_file_size_mb * 1024 * 1024:
                            logger.warning("Skipping %s: file size %dMB exceeds limit %dMB",
                                           candidate, file_size // (1024 * 1024), max_file_size_mb)
                            summary["skipped"] += 1
                            mark(rel, "unknown", "skipped", reason="file_too_large")
                            continue
                    except OSError as e:
                        logger.warning("Could not get size for %s: %s", candidate, e)
                        summary["skipped"] += 1
                        mark(rel, "unknown", "skipped", reason="unreadable")
                        continue

                    ingest_single(candidate, rel)

            if text is None and not file_path and not folder_path:
                raise ValueError("Provide one of: text, file_path, or folder_path")
        # FTS rebuild happens once at context exit

        summary["total_seen"] = len(summary["items"])
        trace_log("ingest_done", collection=collection, indexed_text=summary["indexed_text"], indexed_images=summary["indexed_images"])
        return summary

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
        """
        Index an image file.
        
        Args:
            path: Absolute path to image
            collection: Collection name
            embed_func: Function(path) -> List[float]
            model: Embedding model name
        
        Returns:
            Content hash (hash of file bytes + mtime)
        """
        actual_path = str(Path(path).expanduser().resolve())
        logical_path = stored_path or actual_path

        # Use file bytes + mtime hash for correctness
        # This ensures re-indexing when file content changes
        try:
            content_hash = hash_file_bytes(actual_path)
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"index_image: failed to hash {path}: {e}")
            raise
        
        trace_log("index_image_start", path=path, hash=content_hash[:8])
        
        resolved_title = title or os.path.splitext(os.path.basename(logical_path))[0]
        try:
            modified_at = int(os.path.getmtime(actual_path) * 1000)
            created_at = int(os.path.getctime(actual_path) * 1000)
        except OSError as e:
            logger.warning(f"index_image: failed to get file times for {path}: {e}")
            modified_at = int(time.time() * 1000)
            created_at = modified_at

        # Remove previous image vectors for this logical document path.
        # Deleting only by hash_seq misses changed-content reindex cases.
        self._delete_path_entries(
            collection=collection,
            logical_path=logical_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            content_type="image",
        )

        self.insert_content(content_hash, actual_path, content_type="image")
        self.insert_document(
            collection=collection,
            file_path=logical_path,
            title=resolved_title,
            content_hash=content_hash,
            content_type="image",
            created_at=created_at,
            modified_at=modified_at,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )

        vector = embed_func(actual_path)
        self.insert_embedding(
            content_hash=content_hash,
            seq=0,
            pos=0,
            vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
            model=model,
            collection=collection,
            file_path=logical_path,
            title=resolved_title,
            text_body="",
            content_type="image",
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )

        # Schedule debounced FTS rebuild
        self._schedule_fts_rebuild()

        trace_log("index_image_done", path=logical_path, hash=content_hash[:8])
        return content_hash

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
        actual_path = str(Path(path).expanduser().resolve())
        logical_path = stored_path or actual_path
        resolved_title = os.path.splitext(os.path.basename(logical_path))[0]
        video_embed = embed_video_func or embed_image_func

        artifact_root = Path(self._store_path or DEFAULT_INDEX_DIR) / "video_frames"
        digest = hashlib.sha1(logical_path.encode("utf-8")).hexdigest()[:16]
        output_dir = artifact_root / digest

        self._delete_path_entries(
            collection=collection,
            logical_path=logical_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            include_children=True,
        )

        artifacts = extract_video_artifacts(
            video_path=actual_path,
            output_dir=output_dir,
            logical_path=logical_path,
            frame_interval_seconds=frame_interval_seconds,
            max_frames=max_frames,
        )

        try:
            content_hash = hash_file_bytes(actual_path)
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"index_video: failed to hash {path}: {e}")
            raise

        transcript_summary = "\n".join(
            segment.text.strip()
            for segment in artifacts.transcripts
            if isinstance(segment.text, str) and segment.text.strip()
        ).strip()
        video_body = transcript_summary[:4000]

        try:
            modified_at = int(os.path.getmtime(actual_path) * 1000)
            created_at = int(os.path.getctime(actual_path) * 1000)
        except OSError as e:
            logger.warning(f"index_video: failed to get file times for {path}: {e}")
            modified_at = int(time.time() * 1000)
            created_at = modified_at

        indexed_video_embeddings = 0
        try:
            vector = video_embed(actual_path)
            self.insert_content(content_hash, actual_path, content_type="video")
            self.insert_document(
                collection=collection,
                file_path=logical_path,
                title=resolved_title,
                content_hash=content_hash,
                content_type="video",
                created_at=created_at,
                modified_at=modified_at,
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
            )
            self.insert_embedding(
                content_hash=content_hash,
                seq=0,
                pos=0,
                vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                model=model,
                collection=collection,
                file_path=logical_path,
                title=resolved_title,
                text_body=video_body,
                content_type="video",
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
            )
            indexed_video_embeddings = 1
        except Exception as e:
            logger.warning(
                "index_video: raw video embedding failed for %s; continuing with derived assets: %s",
                actual_path,
                e,
            )

        indexed_frames = 0
        indexed_transcripts = 0

        for frame in artifacts.frames:
            self.index_image(
                path=frame.image_path,
                collection=collection,
                embed_func=embed_image_func,
                model=model,
                stored_path=frame.logical_path,
                title=frame.title,
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
            )
            indexed_frames += 1

        for segment in artifacts.transcripts:
            self.upsert_memory(
                path=segment.logical_path,
                text=segment.text,
                collection=collection,
                embed_func=embed_text_func,
                model=model,
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
            )
            indexed_transcripts += 1

        return {
            "success": True,
            "path": logical_path,
            "collection": collection,
            "hash": content_hash,
            "indexed_video_embeddings": indexed_video_embeddings,
            "indexed_frames": indexed_frames,
            "indexed_transcripts": indexed_transcripts,
            "duration_seconds": artifacts.duration_seconds,
            "transcript_path": artifacts.transcript_path,
            "ffmpeg_available": artifacts.ffmpeg_available,
        }

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
        actual_path = str(Path(path).expanduser().resolve())
        logical_path = stored_path or actual_path

        self._delete_path_entries(
            collection=collection,
            logical_path=logical_path,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            include_children=True,
        )

        artifacts = extract_document_artifacts(actual_path, logical_path)
        indexed_sections = 0

        for section in artifacts.sections:
            self.upsert_memory(
                path=section.logical_path,
                text=section.text,
                collection=collection,
                embed_func=embed_func,
                model=model,
                user_id=user_id,
                session_id=session_id,
                project_id=project_id,
                profile=profile,
                _skip_delete=True,
            )
            indexed_sections += 1

        return {
            "success": True,
            "path": logical_path,
            "collection": collection,
            "document_type": artifacts.document_type,
            "extractor": artifacts.extractor,
            "indexed_sections": indexed_sections,
        }


class _BulkModeContext:
    """Context manager for bulk mode FTS rebuild deferral."""

    def __init__(self, backend: "LanceDBBackend"):
        self._backend = backend
        self._was_in_bulk = False

    def __enter__(self):
        self._was_in_bulk = self._backend._bulk_mode
        self._backend._bulk_mode = True
        trace_log("bulk_mode_enter", was_in_bulk=self._was_in_bulk)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._backend._bulk_mode = self._was_in_bulk
        trace_log("bulk_mode_exit", needs_rebuild=self._backend._fts_needs_rebuild)

        # Trigger a single rebuild at the end of bulk mode if needed
        if not self._was_in_bulk and self._backend._fts_needs_rebuild:
            self._backend._do_fts_rebuild()

        return False  # Don't suppress exceptions
