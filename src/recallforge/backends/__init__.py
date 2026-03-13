"""
RecallForge Backend System.

Multi-backend model loading supporting:
- torch: PyTorch (CUDA > MPS > CPU)
- mlx: Apple Silicon MLX-VLM (optional, 4-bit quantization)
"""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from functools import lru_cache
from typing import Tuple

from .base import ModelBackend
from .torch_backend import TorchBackend


def _truthy(value: str | None) -> bool:
    """Interpret common true-ish environment variable values."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _probe_mlx_runtime() -> Tuple[bool, str]:
    """
    Probe MLX in a child process so host Python never crashes on bad MLX installs.

    Some MLX/Metal runtime failures abort the interpreter at import time.
    Running the check in a subprocess allows safe fallback to torch.
    """
    if _truthy(os.environ.get("RECALLFORGE_DISABLE_MLX")):
        return False, "disabled by RECALLFORGE_DISABLE_MLX"

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, "not running on macOS arm64"

    if importlib.util.find_spec("mlx") is None:
        return False, "mlx package not installed"

    if importlib.util.find_spec("mlx_vlm") is None:
        return False, "mlx_vlm package not installed"

    probe_code = (
        "import mlx.core as mx; "
        "arr = mx.array([0], dtype=mx.int32); "
        "mx.eval(arr); "
        "print('ok')"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return False, f"mlx probe failed: {exc}"

    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout or "").strip().splitlines()[:1]
        summary = reason[0] if reason else f"probe exited {proc.returncode}"
        return False, f"mlx probe failed: {summary}"

    return True, "ok"


def get_mlx_probe_reason() -> str:
    """Return MLX runtime probe status message."""
    return _probe_mlx_runtime()[1]


MLX_AVAILABLE = _probe_mlx_runtime()[0]


def get_mlx_backend_class():
    """Lazily import MLXBackend only after a successful runtime probe."""
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX backend is unavailable: "
            f"{get_mlx_probe_reason()}. "
            "Use RECALLFORGE_BACKEND=torch to force torch fallback."
        )
    from .mlx_backend import MLXBackend
    return MLXBackend


def __getattr__(name: str):
    """Lazy export for MLXBackend to avoid import-time MLX crashes."""
    if name == "MLXBackend":
        return get_mlx_backend_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelBackend",
    "TorchBackend",
    "MLXBackend",
    "MLX_AVAILABLE",
    "get_mlx_backend_class",
    "get_mlx_probe_reason",
]
