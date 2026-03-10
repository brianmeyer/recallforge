"""
lancedb_backend.py - LanceDB Storage Backend for RecallForge.

LanceDB + Apache Arrow storage for embeddings, documents, content, and cache.
Provides vector search and full-text search (Tantivy).
"""

import fnmatch
import hashlib
import math
import os
import re
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
# Helper Functions
# =============================================================================

def escape_sql(s: str) -> str:
    """Escape single quotes for SQL filters."""
    return s.replace("'", "''")


def hash_content(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
    
    def close(self) -> None:
        """Close the database connection."""
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
    
    def _ensure_indices(self) -> None:
        """Create scalar indices for faster lookups."""
        try:
            if self._documents_table:
                doc_indices = self._documents_table.list_indices()
                doc_names = {i.name for i in doc_indices}
                if "file_path_scalar" not in doc_names:
                    self._documents_table.create_index("file_path")
                if "content_hash_scalar" not in doc_names:
                    self._documents_table.create_index("content_hash")
        except Exception:
            pass
        
        try:
            if self._content_table:
                content_indices = self._content_table.list_indices()
                if not any(i.name == "hash_scalar" for i in content_indices):
                    self._content_table.create_index("hash")
        except Exception:
            pass
        
        try:
            if self._cache_table:
                cache_indices = self._cache_table.list_indices()
                if not any(i.name == "key_scalar" for i in cache_indices):
                    self._cache_table.create_index("key")
        except Exception:
            pass
    
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
        modified_at: Optional[int] = None
    ) -> str:
        """Insert or update a document."""
        now = int(time.time() * 1000)
        created_ts = created_at or now
        modified_ts = modified_at or now
        
        # Check for existing
        try:
            existing = list(self._documents_table.search()
                .where(f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(file_path)}'")
                .limit(1)
                .to_list())
            
            if len(existing) > 0:
                doc_id = existing[0]["id"]
                self._documents_table.update(
                    where=f"id = '{escape_sql(doc_id)}'",
                    values={
                        "title": title,
                        "content_hash": content_hash,
                        "content_type": content_type,
                        "active": 1,
                        "updated_at": modified_ts,
                    }
                )
                return doc_id
        except Exception:
            pass
        
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
        }])
        
        return doc_id
    
    def find_document(self, collection: str, file_path: str) -> Optional[Document]:
        """Find a document by collection and path."""
        try:
            rows = list(self._documents_table.search()
                .where(f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(file_path)}' AND active = 1")
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
        except Exception:
            return None
    
    def deactivate_document(self, collection: str, file_path: str) -> None:
        """Mark a document as inactive."""
        self._documents_table.update(
            where=f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(file_path)}' AND active = 1",
            values={"active": 0, "updated_at": int(time.time() * 1000)}
        )
    
    # =========================================================================
    # Content Operations
    # =========================================================================
    
    def insert_content(self, hash_str: str, content: str, content_type: str = "text") -> None:
        """Store content by hash."""
        try:
            existing = list(self._content_table.search()
                .where(f"hash = '{escape_sql(hash_str)}'")
                .limit(1)
                .to_list())
            if len(existing) > 0:
                return
        except Exception:
            pass
        
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
                .where(f"hash = '{escape_sql(hash_str)}'")
                .limit(1)
                .to_list())
            if len(rows) == 0:
                return None
            return rows[0]["doc"]
        except Exception:
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
        content_type: str = "text"
    ) -> None:
        """Insert an embedding."""
        hash_seq = f"{content_hash}_{seq}"
        now = int(time.time() * 1000)
        
        # Delete existing
        try:
            self._embeddings_table.delete(f"hash_seq = '{escape_sql(hash_seq)}'")
        except Exception:
            pass
        
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
        }])
    
    def has_vectors(self) -> bool:
        """Check if index has any vectors."""
        try:
            count = self._embeddings_table.count_rows()
            return count > 0
        except Exception:
            return False
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    def ensure_fts_index(self, force_rebuild: bool = False) -> None:
        """Ensure the FTS index exists."""
        if self._embeddings_table is None:
            return
        
        if self._embeddings_table.count_rows() == 0:
            return
        
        if force_rebuild:
            try:
                self._embeddings_table.create_fts_index("text_body", replace=True)
            except Exception:
                pass
            return
        
        indices = self._embeddings_table.list_indices()
        has_fts = any(
            "text_body" in (i.columns or []) and "FTS" in str(i.index_type or i.type or "").upper()
            for i in indices
        )
        
        if not has_fts:
            self._embeddings_table.create_fts_index("text_body", replace=True)
    
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
    ) -> List[SearchResult]:
        """In-memory BM25 fallback when FTS index fails."""
        try:
            rows = self._embeddings_table.to_pandas()
        except Exception:
            return []
        
        if rows.empty:
            return []
        
        if collection:
            rows = rows[rows["collection"] == collection]
        if content_type:
            rows = rows[rows["content_type"] == content_type]
        
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
        content_type: Optional[str] = None
    ) -> List[SearchResult]:
        """Full-text search using LanceDB Tantivy."""
        if self._embeddings_table is None:
            return []
        
        trimmed = query.strip()
        if not trimmed:
            return []
        
        self.ensure_fts_index()
        
        # Build filter
        filter_clause = None
        if collection:
            filter_clause = f"collection = '{escape_sql(collection)}'"
        if content_type:
            if filter_clause:
                filter_clause += f" AND content_type = '{escape_sql(content_type)}'"
            else:
                filter_clause = f"content_type = '{escape_sql(content_type)}'"
        
        # Run FTS search
        try:
            builder = self._embeddings_table.search(trimmed, query_type="fts").limit(limit * 2)
            if filter_clause:
                builder = builder.where(filter_clause)
            results = builder.to_list()
        except Exception as e:
            return self._bm25_fallback(trimmed, limit, collection, content_type)
        
        if not results:
            return self._bm25_fallback(trimmed, limit, collection, content_type)
        
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
        
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:limit]
    
    def search_vec(
        self,
        vector: List[float],
        limit: int = 20,
        collection: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> List[SearchResult]:
        """Vector similarity search."""
        if self._embeddings_table is None:
            return []
        
        if not self.has_vectors():
            return []
        
        # Build filter
        filter_clause = None
        if collection:
            filter_clause = f"collection = '{escape_sql(collection)}'"
        if content_type:
            if filter_clause:
                filter_clause += f" AND content_type = '{escape_sql(content_type)}'"
            else:
                filter_clause = f"content_type = '{escape_sql(content_type)}'"
        
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
        
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:limit]
    
    def _make_search_result(self, row: Dict[str, Any], score: float, source: str) -> SearchResult:
        """Convert LanceDB row to SearchResult."""
        collection = row.get("collection", "")
        file_path = row.get("file_path", "")
        content_hash = row.get("content_hash", "")
        content_type = row.get("content_type", "text")
        
        body = self.get_content(content_hash) or row.get("text_body", "")
        
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
                .where(f"key = '{escape_sql(key)}'")
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
    
    def upsert_memory(
        self,
        path: str,
        text: str,
        collection: str,
        embed_func,
        model: str,
    ) -> str:
        """Create or update a text memory, replacing old vectors for this path."""
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path is required")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required")

        content_hash = hash_content(text)
        title = extract_title(text, normalized_path)

        # Remove prior vectors for this memory path to prevent duplicate chunks.
        self._embeddings_table.delete(
            f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(normalized_path)}'"
        )

        self.insert_content(content_hash, text, "text")
        self.insert_document(collection, normalized_path, title, content_hash, "text")

        chunks = chunk_document(text)
        for i, chunk in enumerate(chunks):
            vector = embed_func(chunk["text"])
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            self.insert_embedding(
                content_hash=content_hash,
                seq=i,
                pos=chunk["pos"],
                vector=vector,
                model=model,
                collection=collection,
                file_path=normalized_path,
                title=title,
                text_body=chunk["text"],
                content_type="text",
            )

        self.ensure_fts_index(force_rebuild=True)
        return content_hash

    def delete_memory(self, path: str, collection: str) -> Dict[str, Any]:
        """Deactivate a memory and remove all associated vectors."""
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError("path is required")

        removed_vectors = 0
        try:
            removed_vectors = len(
                self._embeddings_table.search()
                .where(
                    f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(normalized_path)}'"
                )
                .to_list()
            )
        except Exception:
            removed_vectors = 0

        self._embeddings_table.delete(
            f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(normalized_path)}'"
        )
        self.deactivate_document(collection, normalized_path)
        self.ensure_fts_index(force_rebuild=True)

        return {
            "success": True,
            "path": normalized_path,
            "collection": collection,
            "removed_vectors": removed_vectors,
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
    ) -> Dict[str, Any]:
        """Index text files from a folder and return summary counts."""
        root = Path(folder_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Folder not found: {folder_path}")

        include = include_globs or ["**/*"]
        exclude = exclude_globs or []

        indexed = 0
        skipped = 0
        errors = 0

        iterator = root.rglob("*") if recursive else root.glob("*")
        for file_path in iterator:
            if not file_path.is_file():
                continue

            rel = file_path.relative_to(root).as_posix()
            if include and not any(fnmatch.fnmatch(rel, pattern) for pattern in include):
                skipped += 1
                continue
            if exclude and any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
                skipped += 1
                continue
            if not self._is_text_file(file_path):
                skipped += 1
                continue

            text = self._read_text_robust(file_path)
            if text is None or not text.strip():
                skipped += 1
                continue

            try:
                self.upsert_memory(
                    path=rel,
                    text=text,
                    collection=collection,
                    embed_func=embed_func,
                    model=model,
                )
                indexed += 1
            except Exception:
                errors += 1

        return {
            "success": True,
            "folder_path": str(root),
            "collection": collection,
            "indexed": indexed,
            "skipped": skipped,
            "errors": errors,
            "total_seen": indexed + skipped + errors,
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
        model: str,
    ) -> Dict[str, Any]:
        """Unified multimodal ingest for text, file, or folder inputs."""
        content_types = content_types or ["text", "image"]
        allowed = set(content_types)
        if not allowed.issubset({"text", "image"}):
            raise ValueError("content_types must be subset of ['text', 'image']")

        summary = {
            "success": True,
            "collection": collection,
            "indexed_text": 0,
            "indexed_images": 0,
            "skipped": 0,
            "errors": 0,
            "items": [],
        }

        def mark(item_path: str, item_type: str, status: str, error: Optional[str] = None) -> None:
            item = {"path": item_path, "type": item_type, "status": status}
            if error:
                item["error"] = error
            summary["items"].append(item)

        def ingest_single(candidate: Path, rel_hint: Optional[str] = None) -> None:
            candidate = candidate.expanduser().resolve()
            if not candidate.exists() or not candidate.is_file():
                summary["errors"] += 1
                mark(str(candidate), "unknown", "error", "file not found")
                return

            item_path = rel_hint or str(candidate)
            is_image = self._is_image_file(candidate)

            try:
                if is_image:
                    if "image" not in allowed:
                        summary["skipped"] += 1
                        mark(item_path, "image", "skipped")
                        return
                    self.index_image(path=str(candidate), collection=collection, embed_func=embed_image_func, model=model)
                    summary["indexed_images"] += 1
                    mark(item_path, "image", "indexed")
                    return

                if "text" not in allowed:
                    summary["skipped"] += 1
                    mark(item_path, "text", "skipped")
                    return
                if not self._is_text_file(candidate):
                    summary["skipped"] += 1
                    mark(item_path, "text", "skipped")
                    return
                body = self._read_text_robust(candidate)
                if body is None or not body.strip():
                    summary["skipped"] += 1
                    mark(item_path, "text", "skipped")
                    return
                self.upsert_memory(
                    path=item_path,
                    text=body,
                    collection=collection,
                    embed_func=embed_text_func,
                    model=model,
                )
                summary["indexed_text"] += 1
                mark(item_path, "text", "indexed")
            except Exception as e:
                summary["errors"] += 1
                mark(item_path, "image" if is_image else "text", "error", str(e))

        if text is not None:
            if "text" not in allowed:
                summary["skipped"] += 1
                mark(path or "inline", "text", "skipped")
            else:
                text_path = (path or "inline").strip() or "inline"
                self.upsert_memory(
                    path=text_path,
                    text=text,
                    collection=collection,
                    embed_func=embed_text_func,
                    model=model,
                )
                summary["indexed_text"] += 1
                mark(text_path, "text", "indexed")

        if file_path:
            ingest_single(Path(file_path))

        if folder_path:
            root = Path(folder_path).expanduser().resolve()
            if not root.exists() or not root.is_dir():
                raise ValueError(f"Folder not found: {folder_path}")
            iterator = root.rglob("*") if recursive else root.glob("*")
            for candidate in iterator:
                if not candidate.is_file():
                    continue
                rel = candidate.relative_to(root).as_posix()
                if not self._matches_globs(rel, include_globs, exclude_globs):
                    summary["skipped"] += 1
                    mark(rel, "unknown", "skipped")
                    continue
                ingest_single(candidate, rel)

        if text is None and not file_path and not folder_path:
            raise ValueError("Provide one of: text, file_path, or folder_path")

        self.ensure_fts_index(force_rebuild=True)
        summary["total_seen"] = summary["indexed_text"] + summary["indexed_images"] + summary["skipped"] + summary["errors"]
        return summary

    def index_image(
        self,
        path: str,
        collection: str,
        embed_func,
        model: str = "Qwen3-VL-Embedding-2B"
    ) -> str:
        """
        Index an image file.
        
        Args:
            path: Absolute path to image
            collection: Collection name
            embed_func: Function(path) -> List[float]
            model: Embedding model name
        
        Returns:
            Content hash
        """
        content_hash = hash_content(path)
        title = os.path.splitext(os.path.basename(path))[0]
        modified_at = int(os.path.getmtime(path) * 1000)
        created_at = int(os.path.getctime(path) * 1000)
        
        self.insert_content(content_hash, path, content_type="image")
        self.insert_document(
            collection=collection,
            file_path=path,
            title=title,
            content_hash=content_hash,
            content_type="image",
            created_at=created_at,
            modified_at=modified_at,
        )
        
        vector = embed_func(path)
        
        now = int(time.time() * 1000)
        hash_seq = f"{content_hash}_0"
        
        try:
            self._embeddings_table.delete(f"hash_seq = '{escape_sql(hash_seq)}'")
        except Exception:
            pass
        
        self._embeddings_table.add([{
            "hash_seq": hash_seq,
            "content_hash": content_hash,
            "collection": collection,
            "file_path": path,
            "content_type": "image",
            "title": title,
            "text_body": "",
            "seq": 0,
            "pos": 0,
            "model": model,
            "embedded_at": now,
            "vector": vector,
        }])
        
        self.ensure_fts_index(force_rebuild=True)
        
        return content_hash