"""
RecallForge Backend System.

Multi-backend model loading supporting:
- torch: PyTorch (CUDA > MPS > CPU)
- mlx: Apple Silicon MLX-VLM (optional, 4-bit quantization)
"""

from .base import ModelBackend
from .torch_backend import TorchBackend
from .mlx_backend import MLXBackend, MLX_AVAILABLE

__all__ = [
    "ModelBackend",
    "TorchBackend",
    "MLXBackend",
    "MLX_AVAILABLE",
]