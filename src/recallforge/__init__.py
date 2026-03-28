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
from typing import Optional, Tuple

__version__ = "0.2.0"


def _has_torch() -> bool:
    """Check if torch is importable without actually importing it."""
    return importlib.util.find_spec("torch") is not None


# Backend selection via environment
RECALLFORGE_BACKEND = os.environ.get("RECALLFORGE_BACKEND", "auto")
RECALLFORGE_MODE = os.environ.get("RECALLFORGE_MODE", "hybrid")
RECALLFORGE_MLX_QUANTIZE = os.environ.get("RECALLFORGE_MLX_QUANTIZE", "4bit")
RECALLFORGE_STORAGE = os.environ.get("RECALLFORGE_STORAGE", "lancedb")

# Central env-var reference used across CLI, server, search, and storage paths.
# Keep this list in sync with all RECALLFORGE_* lookups in the codebase.
RECALLFORGE_ENV_VARS = {
    "RECALLFORGE_BACKEND": "Backend selector: auto | torch | mlx.",
    "RECALLFORGE_MODE": "Search mode: embed | hybrid.",
    "RECALLFORGE_MLX_QUANTIZE": "MLX quantization mode: bf16 | 4bit.",
    "RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY": "Concurrency ceiling for heavy MLX multimodal ops (default 1 for local safety).",
    "RECALLFORGE_MLX_VIDEO_SAMPLE_FPS": "Sampling rate for MLX raw-video processing (lower is safer).",
    "RECALLFORGE_MLX_VIDEO_MAX_FRAMES": "Frame cap for MLX raw-video processing (default tuned for local safety).",
    "RECALLFORGE_MLX_VIDEO_FALLBACK_MAX_FRAMES": "Frame cap for ffmpeg frame-averaging fallback when native video embedding is unavailable.",
    "RECALLFORGE_STORAGE": "Storage backend selector (currently lancedb).",
    "RECALLFORGE_STORE_PATH": "Path to RecallForge data store.",
    "RECALLFORGE_TRACE": "Enable verbose MCP server trace logging (1=true).",
    "RECALLFORGE_MCP_MAX_CONCURRENCY": "Maximum concurrent MCP tool executions.",
    "RECALLFORGE_OVERFETCH_FACTOR": "RRF candidate overfetch multiplier.",
    "RECALLFORGE_MAX_CANDIDATES": "Hard cap for candidate pool before reranking.",
    "RECALLFORGE_RERANK_TOP_K": "Number of top RRF candidates sent to reranker.",
    "RECALLFORGE_ENABLE_MEDIA_RERANKING": "Enable multimodal reranking for image/video-involved searches (disabled by default).",
    "RECALLFORGE_MEDIA_QUERY_RERANK_TOP_K": "Rerank cap for query-side image/video searches.",
    "RECALLFORGE_MEDIA_RESULT_RERANK_TOP_K": "Rerank cap when text queries retrieve image/video candidates.",
    "RECALLFORGE_DISABLE_MLX": "Force-disable MLX backend detection (1=true).",
    "RECALLFORGE_BM25_FALLBACK_MAX_ROWS": "Row limit for BM25 fallback recovery path.",
    "RECALLFORGE_BULK_FLUSH_DOCS": "Batch flush threshold for document table writes.",
    "RECALLFORGE_BULK_FLUSH_EMBEDDINGS": "Batch flush threshold for embedding table writes.",
}

import threading

_BACKEND_SINGLETON = None
_BACKEND_SINGLETON_CONFIG: Optional[Tuple[str, str, str]] = None
_BACKEND_LOCK = threading.Lock()


def _resolve_backend_config() -> tuple[str, str, str]:
    """Resolve normalized backend selection config from environment."""
    backend_type = os.environ.get("RECALLFORGE_BACKEND", RECALLFORGE_BACKEND).lower()
    mode = os.environ.get("RECALLFORGE_MODE", RECALLFORGE_MODE).lower()
    quantization = os.environ.get("RECALLFORGE_MLX_QUANTIZE", RECALLFORGE_MLX_QUANTIZE)
    return backend_type, mode, quantization


from .backends import (
    TorchBackend,
    MLX_AVAILABLE,
    get_mlx_backend_class,
    get_mlx_probe_reason,
)


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
    global _BACKEND_SINGLETON, _BACKEND_SINGLETON_CONFIG

    backend_type, mode, quantization = _resolve_backend_config()

    with _BACKEND_LOCK:
        if _BACKEND_SINGLETON is not None and _BACKEND_SINGLETON_CONFIG == (backend_type, mode, quantization):
            return _BACKEND_SINGLETON
        return _create_backend_locked(backend_type, mode, quantization)


def _create_backend_locked(backend_type: str, mode: str, quantization: str):
    """Create backend while holding _BACKEND_LOCK. Called by get_backend()."""
    global _BACKEND_SINGLETON, _BACKEND_SINGLETON_CONFIG

    if mode not in ("embed", "hybrid"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'embed' or 'hybrid'")
    
    if backend_type == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX backend requested but not available. "
                f"Reason: {get_mlx_probe_reason()}. "
                "Use RECALLFORGE_BACKEND=torch to force torch fallback."
            )
        MLXBackend = get_mlx_backend_class()
        backend = MLXBackend(
            mode=mode,
            quantization=quantization,
        )

    elif backend_type == "torch":
        if not _has_torch():
            raise ImportError(
                "PyTorch backend requested but torch is not installed. "
                "Install with: pip install recallforge[torch]"
            )
        backend = TorchBackend(mode=mode)

    elif backend_type == "auto":
        # Auto-detect: prefer MLX on Apple Silicon, else Torch
        import platform
        backend = None
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if MLX_AVAILABLE:
                try:
                    MLXBackend = get_mlx_backend_class()
                    backend = MLXBackend(
                        mode=mode,
                        quantization=quantization,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"MLX auto-selection failed ({exc}); falling back to torch.",
                        RuntimeWarning,
                    )
        if backend is None:
            if not _has_torch():
                raise ImportError(
                    "No inference backend available. RecallForge requires either MLX or PyTorch.\n\n"
                    "Install a backend for your platform:\n"
                    "  Apple Silicon:  pip install recallforge[mlx]\n"
                    "  NVIDIA GPU:     pip install recallforge[cuda]\n"
                    "  CPU/other:      pip install recallforge[torch]\n"
                )
            backend = TorchBackend(mode=mode)

    else:
        raise ValueError(f"Unknown backend: {backend_type}. Use 'torch', 'mlx', or 'auto'")

    _BACKEND_SINGLETON = backend
    _BACKEND_SINGLETON_CONFIG = (backend_type, mode, quantization)
    return backend


def warmup_backend():
    """Load backend singleton and warm all configured models."""
    backend = get_backend()
    backend.warm_up()
    return backend


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
