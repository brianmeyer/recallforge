"""
RecallForge - Cross-Modal Vision-Language Search Engine.

A powerful semantic search system combining:
- BM25 full-text search
- Vector similarity search
- Cross-encoder reranking
- Image-text cross-modal search

Model Backends:
- torch: PyTorch (CUDA > MPS > CPU)
- mlx: Apple Silicon MLX-VLM with optional 4-bit quantization

Storage Backends:
- lancedb: LanceDB with vector + FTS (default)

Tiered Search Modes (MLX 4-bit):
- embed: Embedder only (1 model, ~1.7GB)
- hybrid: Embedder + Reranker (2 models, ~3.4GB)
"""

import importlib.util
import os
import warnings
from typing import Optional

__version__ = "0.2.0"


def _has_torch() -> bool:
    """Check if torch is importable without actually importing it."""
    return importlib.util.find_spec("torch") is not None


# Backend selection via environment
RECALLFORGE_BACKEND = os.environ.get("RECALLFORGE_BACKEND", "auto")
RECALLFORGE_MODE = os.environ.get("RECALLFORGE_MODE", "hybrid")
RECALLFORGE_MLX_QUANTIZE = os.environ.get("RECALLFORGE_MLX_QUANTIZE", "4bit")
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
    from .backends import (
        TorchBackend,
        MLX_AVAILABLE,
        get_mlx_backend_class,
        get_mlx_probe_reason,
    )

    backend_type = os.environ.get("RECALLFORGE_BACKEND", RECALLFORGE_BACKEND).lower()
    mode = os.environ.get("RECALLFORGE_MODE", RECALLFORGE_MODE).lower()
    quantization = os.environ.get("RECALLFORGE_MLX_QUANTIZE", RECALLFORGE_MLX_QUANTIZE)
    
    # Handle deprecated "full" mode with backward compatibility
    if mode == "full":
        warnings.warn(
            "[RecallForge] Mode 'full' is deprecated (query expander removed). "
            "Falling back to 'hybrid'. See REC-108 for details.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "hybrid"
    
    if backend_type == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX backend requested but not available. "
                f"Reason: {get_mlx_probe_reason()}. "
                "Use RECALLFORGE_BACKEND=torch to force torch fallback."
            )
        MLXBackend = get_mlx_backend_class()
        return MLXBackend(
            mode=mode,
            quantization=quantization,
        )
    
    elif backend_type == "torch":
        if not _has_torch():
            raise ImportError(
                "PyTorch backend requested but torch is not installed. "
                "Install with: pip install recallforge[torch]"
            )
        return TorchBackend(mode=mode)
    
    elif backend_type == "auto":
        # Auto-detect: prefer MLX on Apple Silicon, else Torch
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if MLX_AVAILABLE:
                try:
                    MLXBackend = get_mlx_backend_class()
                    return MLXBackend(
                        mode=mode,
                        quantization=quantization,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"MLX auto-selection failed ({exc}); falling back to torch.",
                        RuntimeWarning,
                    )
        if not _has_torch():
            raise ImportError(
                "No inference backend available. RecallForge requires either MLX or PyTorch.\n\n"
                "Install a backend for your platform:\n"
                "  Apple Silicon:  pip install recallforge[mlx]\n"
                "  NVIDIA GPU:     pip install recallforge[cuda]\n"
                "  CPU/other:      pip install recallforge[torch]\n"
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
    
    storage_type = os.environ.get("RECALLFORGE_STORAGE", RECALLFORGE_STORAGE).lower()
    
    if storage_type == "lancedb":
        backend = LanceDBBackend(store_path)
        backend.initialize(store_path)
        return backend
    
    else:
        raise ValueError(f"Unknown storage backend: {storage_type}. Use 'lancedb'")


def __getattr__(name: str):
    """Lazy-exports MLXBackend so importing recallforge stays crash-safe."""
    if name == "MLXBackend":
        from .backends import get_mlx_backend_class
        return get_mlx_backend_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Convenience imports
from .backends import ModelBackend, TorchBackend, MLX_AVAILABLE
from .storage import StorageBackend, LanceDBBackend
from .search import HybridSearcher, hybrid_query, hybrid_query_image, hybrid_query_video
