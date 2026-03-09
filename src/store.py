"""
store.py - Core data access and retrieval functions for QMD-VL

Document CRUD, chunking, embedding storage, and search (FTS + vector).
"""

import hashlib
import math
import os
import re
import time
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from . import db

# =============================================================================
# Configuration
# =============================================================================

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64  # ~12.5% overlap
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4  # ~2048 chars
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4  # ~256 chars
CHUNK_WINDOW_CHARS = 200  # window for break point detection

# =============================================================================
# Document Types
# =============================================================================

@dataclass
class DocumentResult:
    filepath: str
    display_path: str
    title: str
    context: Optional[str]
    hash: str
    docid: str
    collection: str
    modified_at: str
    body_length: int
    body: Optional[str] = None


@dataclass
class SearchResult:
    filepath: str
    display_path: str
    title: str
    context: Optional[str]
    hash: str
    docid: str
    collection: str
    modified_at: str
    body_length: int
    score: float
    source: str  # 'fts' | 'vec'
    content_type: str = "text"  # 'text' | 'image'
    chunk_pos: int = 0
    body: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def get_docid(hash_str: str) -> str:
    """Generate short docid from hash."""
    return hash_str[:6]


def hash_content(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def escape_sql(s: str) -> str:
    """Escape single quotes for SQL filters."""
    return s.replace("'", "''")


def extract_title(content: str, filename: str) -> str:
    """Extract title from content or filename."""
    # Try markdown heading
    match = re.match(r"^##?\s+(.+)$", content, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        if title not in ("📝 Notes", "Notes"):
            return title
        # Skip "Notes" header, try next heading
        match = re.search(r"\n##\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # Try org-mode
    title_prop = re.search(r"^#\+TITLE:\s*(.+)$", content, re.MULTILINE)
    if title_prop:
        return title_prop.group(1).strip()
    
    # Fallback to filename
    return os.path.splitext(os.path.basename(filename))[0]


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
# Content Storage
# =============================================================================

def insert_content(hash_str: str, content: str, content_type: str = "text") -> None:
    """Insert content into content table (deduped by hash)."""
    if db.content_table is None:
        raise RuntimeError("Database not initialized")
    
    # Check if exists using search
    try:
        existing = list(db.content_table.search().where(f"hash = '{escape_sql(hash_str)}'").limit(1).to_list())
        if len(existing) > 0:
            return  # Content-addressable: skip if already stored
    except Exception:
        # Table may be empty or error on search, just try insert
        pass
    
    db.content_table.add([{
        "hash": hash_str,
        "doc": content,
        "content_type": content_type,
        "created_at": int(time.time() * 1000),
    }])


def get_content(hash_str: str) -> Optional[str]:
    """Retrieve content by hash."""
    if db.content_table is None:
        raise RuntimeError("Database not initialized")
    
    try:
        rows = list(db.content_table.search().where(f"hash = '{escape_sql(hash_str)}'").limit(1).to_list())
        if len(rows) == 0:
            return None
        return rows[0]["doc"]
    except Exception:
        return None


# =============================================================================
# Document Registry
# =============================================================================

def insert_document(
    collection: str,
    file_path: str,
    title: str,
    content_hash: str,
    content_type: str = "text",
    created_at: Optional[int] = None,
    modified_at: Optional[int] = None
) -> str:
    """Insert or update document in registry. Returns document ID."""
    if db.documents_table is None:
        raise RuntimeError("Database not initialized")
    
    now = int(time.time() * 1000)
    created_ts = created_at or now
    modified_ts = modified_at or now
    
    # Check for existing
    try:
        existing = list(db.documents_table.search()
            .where(f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(file_path)}'")
            .limit(1)
            .to_list())
        
        if len(existing) > 0:
            doc_id = existing[0]["id"]
            # Update existing
            db.documents_table.update(
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
        pass  # Table may be empty
    
    # Insert new
    doc_id = str(uuid.uuid4())
    db.documents_table.add([{
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


def find_active_document(collection: str, file_path: str) -> Optional[Dict[str, Any]]:
    """Find active document by collection and path."""
    if db.documents_table is None:
        raise RuntimeError("Database not initialized")
    
    try:
        rows = list(db.documents_table.search()
            .where(f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(file_path)}' AND active = 1")
            .limit(1)
            .to_list())
        
        if len(rows) == 0:
            return None
        
        r = rows[0]
        return {
            "id": r["id"],
            "hash": r["content_hash"],
            "title": r["title"],
        }
    except Exception:
        return None


def get_active_document_paths(collection: str) -> List[str]:
    """Get all active document paths in a collection."""
    if db.documents_table is None:
        raise RuntimeError("Database not initialized")
    
    try:
        rows = list(db.documents_table.search()
            .where(f"collection = '{escape_sql(collection)}' AND active = 1")
            .select(["file_path"])
            .to_list())
        return [r["file_path"] for r in rows]
    except Exception:
        return []


def deactivate_document(collection: str, file_path: str) -> None:
    """Mark document as inactive."""
    if db.documents_table is None:
        raise RuntimeError("Database not initialized")
    
    db.documents_table.update(
        where=f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(file_path)}' AND active = 1",
        values={"active": 0, "updated_at": int(time.time() * 1000)}
    )


# =============================================================================
# Embedding Storage
# =============================================================================

def insert_embedding(
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
    """Insert embedding for a chunk."""
    if db.embeddings_table is None:
        raise RuntimeError("Database not initialized")
    
    hash_seq = f"{content_hash}_{seq}"
    now = int(time.time() * 1000)
    
    # Delete existing
    try:
        db.embeddings_table.delete(f"hash_seq = '{escape_sql(hash_seq)}'")
    except Exception:
        pass
    
    db.embeddings_table.add([{
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


def get_embeddings_for_hash(content_hash: str) -> List[Dict[str, Any]]:
    """Get all embeddings for a content hash."""
    if db.embeddings_table is None:
        raise RuntimeError("Database not initialized")
    
    try:
        rows = list(db.embeddings_table.search()
            .where(f"content_hash = '{escape_sql(content_hash)}'")
            .select(["hash_seq", "seq", "pos", "vector", "model"])
            .to_list())
        
        return [
            {
                "hash_seq": r["hash_seq"],
                "seq": r["seq"],
                "pos": r["pos"],
                "vector": r["vector"],
                "model": r["model"],
            }
            for r in rows
        ]
    except Exception:
        return []


def has_vectors() -> bool:
    """Check if index has any vector embeddings."""
    if db.embeddings_table is None:
        return False
    try:
        count = db.embeddings_table.count_rows()
        return count > 0
    except Exception:
        return False


# =============================================================================
# Cache Operations
# =============================================================================

def get_cache_key(prefix: str, data: Any) -> str:
    """Generate cache key from prefix and data."""
    import json
    hash_obj = hashlib.sha256()
    hash_obj.update(prefix.encode("utf-8"))
    hash_obj.update(json.dumps(data, sort_keys=True).encode("utf-8"))
    return hash_obj.hexdigest()


def get_cached_result(key: str) -> Optional[str]:
    """Get cached result by key."""
    if db.cache_table is None:
        return None
    
    try:
        rows = list(db.cache_table.search()
            .where(f"key = '{escape_sql(key)}'")
            .limit(1)
            .to_list())
        
        if len(rows) == 0:
            return None
        
        return rows[0]["value"]
    except Exception:
        return None


def set_cached_result(key: str, value: str) -> None:
    """Cache a result."""
    if db.cache_table is None:
        return
    
    db.cache_table.merge_insert("key").when_matched_update_all().when_not_matched_insert_all().execute([{
        "key": key,
        "value": value,
        "created_at": int(time.time() * 1000),
    }])


# =============================================================================
# Search - FTS (BM25 via Tantivy)
# =============================================================================

def _bm25_fallback(
    query: str,
    limit: int = 20,
    collection: Optional[str] = None,
    content_type: Optional[str] = None,
) -> List[SearchResult]:
    """In-memory BM25 fallback when LanceDB FTS index fails."""
    if db.embeddings_table is None:
        return []

    try:
        rows = db.embeddings_table.to_pandas()
    except Exception:
        return []

    if rows.empty:
        return []

    if collection:
        rows = rows[rows["collection"] == collection]
        if rows.empty:
            return []

    if content_type:
        rows = rows[rows["content_type"] == content_type]
        if rows.empty:
            return []

    query_terms = re.findall(r'\w+', query.lower())
    if not query_terms:
        return []

    from collections import defaultdict
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
            results.append(_make_search_result(dict(row), score, "fts"))

    results.sort(key=lambda x: x.score, reverse=True)
    if results:
        max_s = results[0].score
        for r in results:
            r.score = r.score / max_s if max_s > 0 else 0
    return results[:limit]


def search_fts(
    query: str,
    limit: int = 20,
    collection: Optional[str] = None,
    content_type: Optional[str] = None
) -> List[SearchResult]:
    """Full-text search using LanceDB Tantivy on embeddings.text_body.
    Falls back to in-memory BM25 if Tantivy fails."""
    if db.embeddings_table is None:
        raise RuntimeError("Database not initialized")
    
    trimmed = query.strip()
    if not trimmed:
        return []
    
    db.ensure_fts_index()
    
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
        builder = db.embeddings_table.search(trimmed, query_type="fts").limit(limit * 2)
        if filter_clause:
            builder = builder.where(filter_clause)
        
        results = builder.to_list()
    except Exception as e:
        print(f"FTS search error, falling back to in-memory BM25: {e}")
        return _bm25_fallback(trimmed, limit, collection, content_type)
    
    if not results:
        return _bm25_fallback(trimmed, limit, collection, content_type)
    
    # Normalize scores to [0, 1]
    max_score = max(r.get("_score", 0) for r in results) or 1
    
    # Dedupe by filepath, keep best score
    seen: Dict[str, SearchResult] = {}
    for r in results:
        filepath = f"qmd://{r['collection']}/{r['file_path']}"
        score = r.get("_score", 0) / max_score
        
        if filepath in seen:
            if score > seen[filepath].score:
                seen[filepath] = _make_search_result(r, score, "fts")
        else:
            seen[filepath] = _make_search_result(r, score, "fts")
    
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:limit]


# =============================================================================
# Search - Vector (ANN via LanceDB)
# =============================================================================

def search_vec(
    vector: List[float],
    limit: int = 20,
    collection: Optional[str] = None,
    content_type: Optional[str] = None
) -> List[SearchResult]:
    """Vector nearest-neighbor search."""
    if db.embeddings_table is None:
        raise RuntimeError("Database not initialized")
    
    if not has_vectors():
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
    
    # Run vector search with cosine metric (distance in [0, 2])
    builder = db.embeddings_table.search(vector).metric("cosine").limit(limit * 2)
    if filter_clause:
        builder = builder.where(filter_clause)
    
    results = builder.to_list()
    
    if not results:
        return []
    
    # Dedupe by filepath, keep best (smallest) distance
    seen: Dict[str, SearchResult] = {}
    for r in results:
        filepath = f"qmd://{r['collection']}/{r['file_path']}"
        distance = r.get("_distance", 1.0)
        # Convert distance to similarity score (1 - dist/2 for cosine-like)
        score = 1.0 - distance / 2.0
        
        if filepath in seen:
            if score > seen[filepath].score:
                seen[filepath] = _make_search_result(r, score, "vec")
        else:
            seen[filepath] = _make_search_result(r, score, "vec")
    
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)[:limit]


# =============================================================================
# Full Document Indexing Pipeline
# =============================================================================

def index_document(
    path: str,
    text: str,
    collection: str,
    model: str,
    embed_func,
    content_type: str = "text"
) -> str:
    """
    Full document indexing pipeline.
    
    Args:
        path: File path within collection
        text: Full document text
        collection: Collection name
        model: Embedding model name
        embed_func: Function(text) -> List[float]
        content_type: 'text' or 'image'
    
    Returns:
        Content hash
    """
    # 1. Hash content
    content_hash = hash_content(text)
    
    # 2. Extract title
    title = extract_title(text, path)
    
    # 3. Store content
    insert_content(content_hash, text, content_type)
    
    # 4. Insert document registry
    insert_document(collection, path, title, content_hash, content_type)
    
    # 5. Chunk and embed
    chunks = chunk_document(text)
    
    for i, chunk in enumerate(chunks):
        # Embed chunk
        vector = embed_func(chunk["text"])
        
        # Store embedding
        insert_embedding(
            content_hash=content_hash,
            seq=i,
            pos=chunk["pos"],
            vector=vector,
            model=model,
            collection=collection,
            file_path=path,
            title=title,
            text_body=chunk["text"],
            content_type=content_type,
        )
    
    # 6. Rebuild FTS index to include new data
    db.ensure_fts_index(force_rebuild=True)
    
    return content_hash


def delete_document(collection: str, path: str) -> None:
    """Delete document and all its embeddings."""
    # Find document
    doc = find_active_document(collection, path)
    if not doc:
        return
    
    content_hash = doc["hash"]
    
    # Deactivate document
    deactivate_document(collection, path)
    
    # Delete embeddings
    if db.embeddings_table:
        try:
            db.embeddings_table.delete(f"content_hash = '{escape_sql(content_hash)}'")
        except Exception:
            pass


# =============================================================================
# Helper
# =============================================================================

def _make_search_result(row: Dict[str, Any], score: float, source: str) -> SearchResult:
    """Convert LanceDB row to SearchResult."""
    collection = row.get("collection", "")
    file_path = row.get("file_path", "")
    content_hash = row.get("content_hash", "")
    content_type = row.get("content_type", "text")
    
    # Get body from content table
    body = get_content(content_hash) or row.get("text_body", "")
    
    return SearchResult(
        filepath=f"qmd://{collection}/{file_path}",
        display_path=f"{collection}/{file_path}",
        title=row.get("title", file_path) or "",
        context=None,  # Would need collections config
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


# =============================================================================
# Image Indexing
# =============================================================================

def insert_image(
    path: str,
    collection: str,
    embed_func,
    model: str = "Qwen3-VL-Embedding-2B"
) -> str:
    """
    Index an image file.
    
    Args:
        path: Absolute path to image file
        collection: Collection name
        embed_func: Function(path) -> List[float]
        model: Embedding model name
    
    Returns:
        Content hash of the indexed image
    """
    # Generate content hash from file path for dedup
    content_hash = hash_content(path)
    
    # Extract metadata
    title = os.path.splitext(os.path.basename(path))[0]
    modified_at = int(os.path.getmtime(path) * 1000)
    created_at = int(os.path.getctime(path) * 1000)
    
    # Store content (as path reference for images)
    insert_content(content_hash, path, content_type="image")
    
    # Insert document registry
    insert_document(
        collection=collection,
        file_path=path,
        title=title,
        content_hash=content_hash,
        content_type="image",
        created_at=created_at,
        modified_at=modified_at,
    )
    
    # Embed the image
    vector = embed_func(path)
    
    # Insert embedding with content_type='image'
    now = int(time.time() * 1000)
    hash_seq = f"{content_hash}_0"
    
    # Clean up existing
    try:
        if db.embeddings_table:
            db.embeddings_table.delete(f"hash_seq = '{escape_sql(hash_seq)}'")
    except Exception:
        pass
    
    if db.embeddings_table:
        db.embeddings_table.add([{
            "hash_seq": hash_seq,
            "content_hash": content_hash,
            "collection": collection,
            "file_path": path,
            "content_type": "image",
            "title": title,
            "text_body": "",  # Images don't have text body
            "seq": 0,
            "pos": 0,
            "model": model,
            "embedded_at": now,
            "vector": vector,
        }])
    
    # Rebuild FTS index
    db.ensure_fts_index(force_rebuild=True)
    
    return content_hash


# =============================================================================
# Cross-Modal Search
# =============================================================================

def search_cross_modal(
    query: str,
    vector: List[float],
    limit: int = 20,
    collection: Optional[str] = None,
    content_type: Optional[str] = None
) -> List[SearchResult]:
    """
    Cross-modal search combining FTS and vector search.
    
    Text queries find relevant images via vector similarity.
    Image queries find relevant text via vector similarity.
    
    Args:
        query: Text query (for FTS)
        vector: Embedding vector (for semantic search)
        limit: Max results
        collection: Optional collection filter
        content_type: Optional content type filter ('text' or 'image')
    
    Returns:
        Merged and ranked results
    """
    from typing import Dict
    
    # Run both searches
    fts_results = search_fts(query, limit=limit, collection=collection, content_type=content_type)
    vec_results = search_vec(vector, limit=limit, collection=collection, content_type=content_type)
    
    # Merge with RRF-style fusion
    seen: Dict[str, SearchResult] = {}
    k = 60  # RRF constant
    
    # Process FTS results
    for rank, result in enumerate(fts_results):
        key = result.filepath
        rrf_score = 1.0 / (k + rank + 1)
        if key in seen:
            seen[key].score += rrf_score
        else:
            result.score = rrf_score
            result.source = "fts+vec" if any(r.filepath == key for r in vec_results) else "fts"
            seen[key] = result
    
    # Process vector results
    for rank, result in enumerate(vec_results):
        key = result.filepath
        rrf_score = 1.0 / (k + rank + 1)
        if key in seen:
            seen[key].score += rrf_score
            seen[key].source = "fts+vec"
        else:
            result.score = rrf_score
            result.source = "vec"
            seen[key] = result
    
    # Sort by combined score
    results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    
    return results[:limit]