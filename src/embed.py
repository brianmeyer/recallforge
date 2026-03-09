"""
embed.py - Qwen3-VL-Embedding wrapper for text and image embeddings.

DEPRECATED: Use models.py registry instead for new code.
This module is kept for backward compatibility and re-exports from models.
"""

import numpy as np
from typing import List, Optional

# Re-export from models.py for centralized model management
from src.models import (
    get_registry,
    embed_text as _embed_text,
    embed_texts as _embed_texts,
    embed_image as _embed_image,
    embed_images as _embed_images,
)


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string."""
    return _embed_text(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed multiple text strings.
    
    Uses single MPS call for efficiency. All texts embedded in one batch.
    """
    return _embed_texts(texts)


def embed_image(image_path: str) -> np.ndarray:
    """Embed a single image."""
    return _embed_image(image_path)


def embed_images(image_paths: List[str]) -> np.ndarray:
    """Embed multiple images."""
    return _embed_images(image_paths)


class Embedder:
    """
    Backward-compatible Embedder wrapper.
    
    DEPRECATED: Use models.get_registry() directly.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Embedding-2B", device: str = "auto", dtype: str = "float16"):
        """Initialize embedder (just gets the registry)."""
        self._registry = get_registry()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text."""
        return self._registry.embed_text(text)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return self._registry.embed_texts(texts)
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed single image."""
        return self._registry.embed_image(image_path)
    
    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """Embed multiple images."""
        return self._registry.embed_images(image_paths)


# Backward-compatible singleton getter
_embedder: Optional[Embedder] = None


def get_embedder(model_name: str = "Qwen/Qwen3-VL-Embedding-2B") -> Embedder:
    """Get or create singleton embedder (backward compatible)."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(model_name=model_name)
    return _embedder
