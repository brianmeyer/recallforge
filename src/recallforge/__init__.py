"""
RecallForge - Cross-Modal Vision-Language Search Engine.

A powerful semantic search system combining:
- BM25 full-text search
- Vector similarity search
- Query expansion
- Cross-encoder reranking
- Image-text cross-modal search

Model Backends:
- torch: PyTorch (CUDA > MPS > CPU)
- mlx: Apple Silicon MLX-VLM with optional 4-bit quantization

Storage Backends:
- lancedb: LanceDB with vector + FTS (default)

Tiered Search Modes:
- embed: Embedder only (1 model, ~4GB)
- hybrid: Embedder + Reranker (2 models, ~8GB)
- full: All three models (3 models, ~12GB)
"""

import os
from typing import Optional

__version__ = "1.0.0"

# Backend selection via environment
RECALLFORGE_BACKEND = os.environ.get("RECALLFORGE_BACKEND", "auto")
RECALLFORGE_MODE = os.environ.get("RECALLFORGE_MODE", "full")
RECALLFORGE_MLX_QUANTIZE = os.environ.get("RECALLFORGE_MLX_QUANTIZE", "bf16")
RECALLFORGE_STORAGE = os.environ.get("RECALLFORGE_STORAGE", "lancedb")


def get_backend():
    """
    Get the appropriate model backend based on configuration.
    
    Selection order:
    1. If RECALLFORGE_BACKEND=mlx and MLX available, use MLX
    2. If RECALLFORGE_BACKEND=torch, use Torch
    3. If RECALLFORGE_BACKEND=auto, detect best option
    
    Returns:
        ModelBackend instance
    """
    from .backends import TorchBackend, MLXBackend, MLX_AVAILABLE
    
    backend_type = RECALLFORGE_BACKEND.lower()
    mode = RECALLFORGE_MODE.lower()
    
    if backend_type == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX backend requested but not available. "
                "Install with: pip install mlx mlx-vlm"
            )
        return MLXBackend(
            mode=mode,
            quantization=RECALLFORGE_MLX_QUANTIZE,
        )
    
    elif backend_type == "torch":
        return TorchBackend(mode=mode)
    
    elif backend_type == "auto":
        # Auto-detect: prefer MLX on Apple Silicon, else Torch
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if MLX_AVAILABLE:
                return MLXBackend(
                    mode=mode,
                    quantization=RECALLFORGE_MLX_QUANTIZE,
                )
        return TorchBackend(mode=mode)
    
    else:
        raise ValueError(f"Unknown backend: {backend_type}. Use 'torch', 'mlx', or 'auto'")


def get_storage(store_path: Optional[str] = None):
    """
    Get the storage backend.
    
    Currently only LanceDB is supported.
    
    Args:
        store_path: Optional path to storage directory
    
    Returns:
        StorageBackend instance
    """
    from .storage import LanceDBBackend
    
    storage_type = RECALLFORGE_STORAGE.lower()
    
    if storage_type == "lancedb":
        backend = LanceDBBackend(store_path)
        backend.initialize(store_path)
        return backend
    
    else:
        raise ValueError(f"Unknown storage backend: {storage_type}. Use 'lancedb'")


# Convenience imports
from .backends import ModelBackend, TorchBackend, MLXBackend, MLX_AVAILABLE
from .storage import StorageBackend, LanceDBBackend
from .search import HybridSearcher, hybrid_query