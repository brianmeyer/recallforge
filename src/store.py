"""
store.py - Core data access and retrieval functions for QMD-VL

LanceDB-based storage with BM25 full-text search and vector similarity search.
"""

import hashlib
import time
import re
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import uuid

import numpy as np

import db
from db import (
    initialize_database,
    ensure_indices,
    has_vector_index,
    escape_sql,
    EMBED_DIM,
    get_lance_store_path,
)


# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4  # ~2048 chars
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4  # ~256 chars
CHUNK_WINDOW_CHARS = 200

BREAK_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r'\n#{1}(?!#)'), 100, 'h1'),
    (re.compile(r'\n#{2}(?!#)'), 90, 'h2'),
    (re.compile(r'\n#{3}(?!#)'), 80, 'h3'),
    (re.compile(r'\n#{4}(?!#)'), 70, 'h4'),
    (re.compile(r'\n#{5}(?!#)'), 60, 'h5'),
    (re.compile(r'\n#{6}(?!#)'), 50, 'h6'),
    (re.compile(r'\n```'), 80, 'codeblock'),
    (re.compile(r'\n(?:---|\*\*\*|___)\s*\n'), 60, 'hr'),
    (re.compile(r'\n\n+'), 20, 'blank'),
    (re.compile(r'\n[-*]\s'), 5, 'list'),
    (re.compile(r'\n\d+\.\s'), 5, 'numlist'),
    (re.compile(r'\n'), 1, 'newline'),
]


def scan_break_points(text: str) -> List[Dict[str, Any]]:
    """Find natural break points in text."""
    seen = {}
    for pattern, score, btype in BREAK_PATTERNS:
        for match in pattern.finditer(text):
            pos = match.start()
            if pos not in seen or score > seen[pos]['score']:
                seen[pos] = {'pos': pos, 'score': score, 'type': btype}
    return sorted(seen.values(), key=lambda x: x['pos'])


def find_code_fences(text: str) -> List[Dict[str, int]]:
    """Find code fence regions in text."""
    regions = []
    in_fence = False
    fence_start = 0
    for match in re.finditer(r'\n```', text):
        if not in_fence:
            fence_start = match.start()
            in_fence = True
        else:
            regions.append({'start': fence_start, 'end': match.end()})
            in_fence = False
    if in_fence:
        regions.append({'start': fence_start, 'end': len(text)})
    return regions


def is_inside_code_fence(pos: int, fences: List[Dict[str, int]]) -> bool:
    """Check if position is inside a code fence."""
    for f in fences:
        if f['start'] < pos < f['end']:
            return True
    return False


def find_best_cutoff(
    break_points: List[Dict[str, Any]],
    target_pos: int,
    window_chars: int = CHUNK_WINDOW_CHARS,
    decay_factor: float = 0.7,
    code_fences: Optional[List[Dict[str, int]]] = None
) -> int:
    """Find the best break point within a window."""
    if code_fences is None:
        code_fences = []
    
    window_start = target_pos - window_chars
    best_score = -1
    best_pos = target_pos
    
    for bp in break_points:
        pos = bp['pos']
        if pos < window_start:
            continue
        if pos > target_pos:
            break
        if is_inside_code_fence(pos, code_fences):
            continue
        
        distance = target_pos - pos
        normalized_dist = distance / window_chars
        multiplier = 1.0 - (normalized_dist * normalized_dist) * decay_factor
        final_score = bp['score'] * multiplier
        
        if final_score > best_score:
            best_score = final_score
            best_pos = pos
    
    return best_pos


def chunk_document(
    content: str,
    max_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    window_chars: int = CHUNK_WINDOW_CHARS
) -> List[Dict[str, Any]]:
    """Split document into overlapping chunks at natural break points."""
    if len(content) <= max_chars:
        return [{'text': content, 'pos': 0}]
    
    break_points = scan_break_points(content)
    code_fences = find_code_fences(content)
    chunks = []
    char_pos = 0
    
    while char_pos < len(content):
        target_end = min(char_pos + max_chars, len(content))
        end_pos = target_end
        
        if end_pos < len(content):
            best_cutoff = find_best_cutoff(break_points, target_end, window_chars, 0.7, code_fences)
            if best_cutoff > char_pos and best_cutoff <= target_end:
                end_pos = best_cutoff
        
        if end_pos <= char_pos:
            end_pos = min(char_pos + max_chars, len(content))
        
        chunks.append({'text': content[char_pos:end_pos], 'pos': char_pos})
        
        if end_pos >= len(content):
            break
        
        char_pos = end_pos - overlap_chars
        last_chunk_pos = chunks[-1]['pos']
        if char_pos <= last_chunk_pos:
            char_pos = end_pos
    
    return chunks


# -----------------------------------------------------------------------------
# Document operations
# -----------------------------------------------------------------------------

def hash_content(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def extract_title(content: str, filename: str) -> str:
    """Extract title from document content or filename."""
    # Try markdown heading
    md_match = re.match(r'^##?\s+(.+)$', content, re.MULTILINE)
    if md_match:
        title = md_match.group(1).strip()
        if title and title != "📝 Notes" and title != "Notes":
            return title
    
    # Try org-mode title
    org_match = re.match(r'^#\+TITLE:\s*(.+)$', content, re.MULTILINE | re.IGNORECASE)
    if org_match:
        return org_match.group(1).strip()
    
    # Fallback to filename
    return Path(filename).stem


async def insert_content(hash_val: str, content: str) -> None:
    """Insert content into the content table (deduplicated)."""
    if db.content_table is None:
        raise RuntimeError("Database not initialized")
    
    # Check if already exists
    existing = db.content_table.search().where(f"hash = '{escape_sql(hash_val)}'").limit(1).to_list()
    if existing:
        return
    
    db.content_table.add([{
        'hash': hash_val,
        'doc': content,
        'content_type': 'text',
        'created_at': int(time.time() * 1000),
    }])


async def insert_document(
    collection: str,
    file_path: str,
    title: str,
    content_hash: str,
    content_type: str = 'text',
    created_at: Optional[int] = None,
    updated_at: Optional[int] = None,
) -> str:
    """Insert or update a document record. Returns document ID."""
    if db.documents_table is None:
        raise RuntimeError("Database not initialized")
    
    now = int(time.time() * 1000)
    created = created_at or now
    updated = updated_at or now
    
    # Check for existing
    existing = db.documents_table.search().where(
        f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(file_path)}'"
    ).limit(1).to_list()
    
    if existing:
        doc_id = existing[0]['id']
        # Update existing
        db.documents_table.merge_insert('id').when_matched_update_all().when_not_matched_insert_all().execute([{
            'id': doc_id,
            'collection': collection,
            'file_path': file_path,
            'title': title,
            'content_hash': content_hash,
            'content_type': content_type,
            'active': 1,
            'created_at': existing[0]['created_at'],
            'updated_at': updated,
        }])
        return doc_id
    
    # Create new
    doc_id = str(uuid.uuid4())
    db.documents_table.add([{
        'id': doc_id,
        'collection': collection,
        'file_path': file_path,
        'title': title,
        'content_hash': content_hash,
        'content_type': content_type,
        'active': 1,
        'created_at': created,
        'updated_at': updated,
    }])
    return doc_id


async def get_document(collection: str, file_path: str) -> Optional[Dict[str, Any]]:
    """Get active document by collection and path."""
    if db.documents_table is None:
        raise RuntimeError("Database not initialized")
    
    rows = db.documents_table.search().where(
        f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(file_path)}' AND active = 1"
    ).limit(1).to_list()
    
    if not rows:
        return None
    
    return rows[0]


async def get_content(hash_val: str) -> Optional[str]:
    """Get document content by hash."""
    if db.content_table is None:
        raise RuntimeError("Database not initialized")
    
    rows = db.content_table.search().where(f"hash = '{escape_sql(hash_val)}'").limit(1).to_list()
    if not rows:
        return None
    return rows[0]['doc']


async def delete_document(collection: str, file_path: str) -> bool:
    """Deactivate a document (soft delete)."""
    if db.documents_table is None:
        return False
    
    rows = db.documents_table.search().where(
        f"collection = '{escape_sql(collection)}' AND file_path = '{escape_sql(file_path)}' AND active = 1"
    ).limit(1).to_list()
    
    if not rows:
        return False
    
    doc = rows[0]
    db.documents_table.merge_insert('id').when_matched_update_all().execute([{
        **doc,
        'active': 0,
        'updated_at': int(time.time() * 1000),
    }])
    return True


# -----------------------------------------------------------------------------
# Embedding operations
# -----------------------------------------------------------------------------

async def insert_embedding(
    content_hash: str,
    seq: int,
    pos: int,
    vector: List[float],
    model: str,
    text_body: str,
    collection: str = '',
    file_path: str = '',
    content_type: str = 'text',
    title: str = '',
) -> None:
    """Insert an embedding for a document chunk."""
    if db.embeddings_table is None:
        raise RuntimeError("Database not initialized")
    
    hash_seq = f"{content_hash}_{seq}"
    now = int(time.time() * 1000)
    
    # Delete existing if any
    try:
        db.embeddings_table.delete(f"hash_seq = '{escape_sql(hash_seq)}'")
    except Exception:
        pass
    
    db.embeddings_table.add([{
        'hash_seq': hash_seq,
        'content_hash': content_hash,
        'collection': collection,
        'file_path': file_path,
        'content_type': content_type,
        'title': title,
        'text_body': text_body,
        'seq': seq,
        'pos': pos,
        'model': model,
        'embedded_at': now,
        'vector': vector,
    }])


# -----------------------------------------------------------------------------
# Search operations
# -----------------------------------------------------------------------------

def normalize_scores(results: List[Dict[str, Any]], score_key: str = '_score') -> List[Dict[str, Any]]:
    """Normalize scores to [0, 1] range based on max score."""
    if not results:
        return results
    
    max_score = max(r.get(score_key, 0) for r in results)
    if max_score <= 0:
        return results
    
    for r in results:
        r['score'] = r.get(score_key, 0) / max_score
    
    return results


async def search_fts(query: str, limit: int = 20, collection: Optional[str] = None) -> List[Dict[str, Any]]:
    """Full-text search using Tantivy BM25 on embeddings.text_body."""
    if db.embeddings_table is None:
        raise RuntimeError("Database not initialized")
    
    if not query.strip():
        return []
    
    # Build search
    search = db.embeddings_table.search(query, query_type="fts")
    
    if collection:
        search = search.where(f"collection = '{escape_sql(collection)}'")
    
    # Add fts column
    search = search.select(["hash_seq", "content_hash", "collection", "file_path", "title", "text_body", "seq", "pos", "model"])
    
    rows = search.limit(limit).to_list()
    
    # Normalize scores
    rows = normalize_scores(rows, '_score')
    
    # Dedupe by file_path
    seen = {}
    for r in rows:
        fp = r.get('file_path', '')
        if fp and fp not in seen:
            seen[fp] = r
    
    return list(seen.values())[:limit]


async def search_vec(
    query_embedding: List[float],
    limit: int = 20,
    collection: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Vector similarity search using LanceDB ANN."""
    if db.embeddings_table is None:
        raise RuntimeError("Database not initialized")
    
    # Build search - specify vector column explicitly
    search = db.embeddings_table.search(query_embedding, query_type="vector", vector_column_name="vector")
    
    if collection:
        search = search.where(f"collection = '{escape_sql(collection)}'")
    
    rows = search.limit(limit).to_list()
    
    # Convert distance to score (1 - distance/2 for cosine-like)
    for r in rows:
        dist = r.get('_distance', 1.0)
        r['score'] = max(0, 1 - dist / 2)
    
    # Dedupe by file_path
    seen = {}
    for r in rows:
        fp = r.get('file_path', '')
        if fp and fp not in seen:
            seen[fp] = r
    
    return list(seen.values())[:limit]


async def has_vectors() -> bool:
    """Check if there are any vectors in the embeddings table."""
    if db.embeddings_table is None:
        return False
    try:
        return db.embeddings_table.count_rows() > 0
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Cache operations
# -----------------------------------------------------------------------------

def get_cache_key(prefix: str, data: Any) -> str:
    """Generate a cache key from prefix and data."""
    h = hashlib.sha256()
    h.update(prefix.encode())
    h.update(str(data).encode())
    return h.hexdigest()


async def get_cached(key: str) -> Optional[str]:
    """Get cached value."""
    if db.cache_table is None:
        return None
    
    rows = db.cache_table.search().where(f"key = '{escape_sql(key)}'").limit(1).to_list()
    if not rows:
        return None
    return rows[0].get('value')


async def set_cached(key: str, value: str) -> None:
    """Set cached value."""
    if db.cache_table is None:
        return
    
    db.cache_table.merge_insert('key').when_matched_update_all().when_not_matched_insert_all().execute([{
        'key': key,
        'value': value,
        'created_at': int(time.time() * 1000),
    }])


async def clear_cache() -> None:
    """Clear all cached values."""
    if db.cache_table is None:
        return
    try:
        db.cache_table.delete("1 = 1")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Convenience: Full indexing pipeline
# -----------------------------------------------------------------------------

async def insert_document_with_embedding(
    path: str,
    text: str,
    collection: str,
    embed_func,  # Callable: text -> List[float]
    model: str = 'qwen3-vl-embedding-2b',
    max_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> str:
    """
    Full pipeline: hash content, insert document, chunk, embed, and store.
    
    Args:
        path: File path or identifier
        text: Full document text
        collection: Collection name
        embed_func: Async function(text) -> List[float]
        model: Embedding model name
        max_chars: Max chunk size in characters
        overlap_chars: Overlap between chunks
    
    Returns:
        Content hash
    """
    # Hash and store content
    content_hash = hash_content(text)
    await insert_content(content_hash, text)
    
    # Extract title
    title = extract_title(text, Path(path).name)
    
    # Insert document record
    await insert_document(
        collection=collection,
        file_path=path,
        title=title,
        content_hash=content_hash,
    )
    
    # Chunk document
    chunks = chunk_document(text, max_chars, overlap_chars)
    
    # Embed each chunk
    for seq, chunk in enumerate(chunks):
        chunk_text = chunk['text']
        pos = chunk['pos']
        
        # Get embedding
        embedding = await embed_func(chunk_text)
        
        # Store embedding
        await insert_embedding(
            content_hash=content_hash,
            seq=seq,
            pos=pos,
            vector=embedding,
            model=model,
            text_body=chunk_text,
            collection=collection,
            file_path=path,
            title=title,
        )
    
    return content_hash