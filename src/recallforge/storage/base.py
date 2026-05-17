"""
base.py - Abstract Base Class for Storage Backends.

Defines the interface for storage backends that handle:
- Document storage and retrieval
- Vector indexing and search
- Full-text search
- Caching
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import numpy as np


@dataclass
class SearchResult:
    """Result from search operations."""
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
    source: str  # 'fts', 'vec', or 'fts+vec'
    content_type: str = "text"
    chunk_pos: int = 0
    body: Optional[str] = None
    # Namespace fields for multi-tenant isolation
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    profile: Optional[str] = None
    memory_id: Optional[str] = None
    memory_role: str = "root"
    memory_root_path: Optional[str] = None
    tags: Optional[List[str]] = None
    importance: Optional[float] = None
    expires_at: Optional[int] = None


@dataclass
class Document:
    """Document in the index."""
    id: str
    collection: str
    file_path: str
    title: str
    content_hash: str
    content_type: str
    active: bool
    created_at: int
    updated_at: int
    # Namespace fields for multi-tenant isolation
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    profile: Optional[str] = None
    memory_id: Optional[str] = None
    memory_role: str = "root"
    memory_root_path: Optional[str] = None


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    
    Storage backends handle:
    - Document CRUD operations
    - Vector storage and ANN search
    - Full-text search (BM25/TF-IDF)
    - Result caching
    
    All backends must implement these methods for a unified interface.
    """
    
    @abstractmethod
    def initialize(self, store_path: str) -> None:
        """
        Initialize the storage backend.
        
        Args:
            store_path: Path to storage directory
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close connections and release resources."""
        pass
    
    # =========================================================================
    # Document Operations
    # =========================================================================
    
    @abstractmethod
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
        profile: Optional[str] = None,
        memory_id: Optional[str] = None,
        memory_role: str = "root",
        memory_root_path: Optional[str] = None,
    ) -> str:
        """
        Insert or update a document.
        
        Returns:
            Document ID
        """
        pass
    
    @abstractmethod
    def find_document(self, collection: str, file_path: str) -> Optional[Document]:
        """Find a document by collection and path."""
        pass
    
    @abstractmethod
    def deactivate_document(self, collection: str, file_path: str) -> None:
        """Mark a document as inactive (soft delete)."""
        pass
    
    # =========================================================================
    # Content Operations
    # =========================================================================
    
    @abstractmethod
    def insert_content(self, hash_str: str, content: str, content_type: str = "text") -> None:
        """Store content by hash (deduplication)."""
        pass
    
    @abstractmethod
    def get_content(self, hash_str: str) -> Optional[str]:
        """Retrieve content by hash."""
        pass
    
    # =========================================================================
    # Embedding Operations
    # =========================================================================
    
    @abstractmethod
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
        memory_id: Optional[str] = None,
        memory_role: str = "root",
        memory_root_path: Optional[str] = None,
        importance: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Insert an embedding for a document chunk."""
        pass
    
    @abstractmethod
    def has_vectors(self) -> bool:
        """Check if the index has any vectors."""
        pass
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    @abstractmethod
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
        """Full-text search (BM25)."""
        pass
    
    @abstractmethod
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
        """Vector similarity search (ANN)."""
        pass
    
    @abstractmethod
    def rebuild_fts_index(self) -> None:
        """Rebuild the full-text search index."""
        pass
    
    # =========================================================================
    # Cache Operations
    # =========================================================================
    
    @abstractmethod
    def get_cached(self, key: str) -> Optional[str]:
        """Get a cached value."""
        pass
    
    @abstractmethod
    def set_cached(self, key: str, value: str) -> None:
        """Set a cached value."""
        pass
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    @abstractmethod
    def count_embeddings(self) -> int:
        """Count total embeddings in the index."""
        pass
    
    @abstractmethod
    def count_documents(self) -> int:
        """Count total documents in the index."""
        pass

    # =========================================================================
    # Inline Memory Operations
    # =========================================================================

    @abstractmethod
    def upsert_memory(
        self,
        path: str,
        text: str,
        collection: str,
        embed_func: Callable[[str], List[float]],
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        importance: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        _skip_delete: bool = False,
        memory_role: str = "root",
        memory_root_path: Optional[str] = None,
    ) -> str:
        """Create or update a memory entry and its embeddings."""
        pass

    @abstractmethod
    def index_conversation(
        self,
        path: str,
        turns: List[Dict[str, Any]],
        collection: str,
        embed_func: Callable[[str], List[float]],
        model: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        importance: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Index a conversation root plus turn-level child memories."""
        pass

    @abstractmethod
    def list_memory_entities(
        self,
        *,
        memory_id: Optional[str] = None,
        path: Optional[str] = None,
        entity: Optional[str] = None,
        collection: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List extracted entity mentions with source evidence."""
        pass

    @abstractmethod
    def find_related_memories(
        self,
        *,
        memory_id: Optional[str] = None,
        path: Optional[str] = None,
        entity: Optional[str] = None,
        collection: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find memories related by shared extracted entities."""
        pass

    @abstractmethod
    def delete_memory(
        self,
        path: str,
        collection: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deactivate a memory entry and remove associated embeddings."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def index_folder(
        self,
        folder_path: str,
        collection: str,
        recursive: bool,
        include_globs: Optional[List[str]],
        exclude_globs: Optional[List[str]],
        embed_func: Callable[[str], List[float]],
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index text files from a folder and return summary counts."""
        pass

    @abstractmethod
    def index_document_file(
        self,
        path: str,
        collection: str,
        embed_func: Callable[[str], List[float]],
        model: str,
        stored_path: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract and index a document file into structured text assets."""
        pass

    @abstractmethod
    def index_audio(
        self,
        path: str,
        collection: str,
        embed_text_func: Callable[[str], List[float]],
        model: str = "Qwen3-VL-Embedding-2B",
        stored_path: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index an audio file through transcript sidecars."""
        pass

    @abstractmethod
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
        embed_text_func: Callable[[str], List[float]],
        embed_image_func: Callable[[str], List[float]],
        embed_video_func: Optional[Callable[[str], List[float]]],
        model: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        caption_media: bool = True,
    ) -> Dict[str, Any]:
        """Unified multimodal ingest for text/image/file/folder inputs."""
        pass

    def rename_collection(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """Rename a collection. Override in subclass."""
        raise NotImplementedError("rename_collection not supported by this backend")

    def delete_collection(self, name: str) -> Dict[str, Any]:
        """Delete all data for a collection. Override in subclass."""
        raise NotImplementedError("delete_collection not supported by this backend")
