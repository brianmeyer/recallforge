"""
base.py - Abstract Base Class for Model Backends.

Defines the interface that all backends must implement.
Backends handle model loading, device selection, and inference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class BackendInfo:
    """Information about a backend's capabilities and state."""
    name: str
    device: str
    dtype: str
    embedder_loaded: bool = False
    reranker_loaded: bool = False
    memory_allocated_gb: float = 0.0
    supports_images: bool = True
    quantization: Optional[str] = None  # "4bit", "8bit", or None


class ModelBackend(ABC):
    """
    Abstract base class for model backends.
    
    All backends must implement:
    - embed_text: Embed text strings
    - embed_image: Embed images
    - rerank: Rerank documents
    - warm_up: Preload all models
    - get_info: Return backend status
    
    Model IDs are defined per backend:
    - Torch: Qwen/Qwen3-VL-Embedding-2B, Qwen/Qwen3-VL-Reranker-2B
    - MLX BF16: arthurcollet/Qwen3-VL-Embedding-2B-mlx, arthurcollet/Qwen3-VL-Reranker-2B-mlx
    - MLX 4-bit: arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit, arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
        
        Returns:
            2048-dimensional numpy array (float32)
        """
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text strings in a batch.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            N x 2048 numpy array (float32)
        """
        pass
    
    @abstractmethod
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Embed a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            2048-dimensional numpy array (float32)
        """
        pass
    
    @abstractmethod
    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Embed multiple images in a batch.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            N x 2048 numpy array (float32)
        """
        pass

    def embed_video(self, video_path: str) -> np.ndarray:
        """
        Embed a single video.

        Backends that support native raw-video queries should override this.
        """
        return self.embed_videos([video_path])[0]

    def embed_videos(self, video_paths: List[str]) -> np.ndarray:
        """
        Embed multiple videos in a batch.

        Default implementation uses per-item fallback so older backends/mocks can
        opt into `embed_video()` only.
        """
        if video_paths is None:
            raise ValueError("Video batch is None; expected a list of video paths.")
        if not isinstance(video_paths, list):
            raise TypeError(
                f"Video batch must be a list[str], got {type(video_paths).__name__}."
            )
        vectors = [self.embed_video(path) for path in video_paths]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(vectors).astype(np.float32)
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """
        Rerank documents for a query.
        
        Args:
            query: Search query
            documents: List of document dicts with 'text' or 'text_body' field
        
        Returns:
            List of relevance scores (0.0 to 1.0) in same order as documents
        """
        pass
    
    @abstractmethod
    def warm_up(self) -> None:
        """
        Preload all models.
        
        Call this at server startup to avoid slow first queries.
        """
        pass
    
    @abstractmethod
    def get_info(self) -> BackendInfo:
        """
        Return information about the backend.
        """
        pass
    
    # Mode support: which models are active
    # Modes: embed (embedder only), hybrid (embedder + reranker)
    
    _mode: str = "hybrid"
    
    def set_mode(self, mode: str) -> None:
        """Set the search mode. Modes: embed (vector+BM25), hybrid (+reranker)."""
        if mode not in ("embed", "hybrid"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'embed' or 'hybrid'")
        self._mode = mode
    
    def get_mode(self) -> str:
        """Get current search mode."""
        return self._mode
    
    def needs_reranker(self) -> bool:
        """Check if current mode needs reranker."""
        return self._mode == "hybrid"
