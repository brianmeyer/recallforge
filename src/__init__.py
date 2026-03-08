"""
QMD-VL: Vision-Language Memory Search System

A LanceDB-based memory search system with Qwen3-VL embeddings.
"""

from .db import (
    initialize_database,
    close_database,
    get_lance_store_path,
    EMBED_DIM,
)
from .store import (
    chunk_document,
    hash_content,
    extract_title,
    insert_content,
    insert_document,
    insert_document_with_embedding,
    insert_embedding,
    get_document,
    get_content,
    delete_document,
    search_fts,
    search_vec,
    has_vectors,
    get_cache_key,
    get_cached,
    set_cached,
    clear_cache,
)
from .embed import (
    Qwen3VLEmbedderWrapper,
    get_embedder,
    embed_text_async,
    embed_texts_async,
)

__all__ = [
    # db
    'initialize_database',
    'close_database',
    'get_lance_store_path',
    'EMBED_DIM',
    # store
    'chunk_document',
    'hash_content',
    'extract_title',
    'insert_content',
    'insert_document',
    'insert_document_with_embedding',
    'insert_embedding',
    'get_document',
    'get_content',
    'delete_document',
    'search_fts',
    'search_vec',
    'has_vectors',
    'get_cache_key',
    'get_cached',
    'set_cached',
    'clear_cache',
    # embed
    'Qwen3VLEmbedderWrapper',
    'get_embedder',
    'embed_text_async',
    'embed_texts_async',
]