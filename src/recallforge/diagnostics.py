"""
diagnostics.py - Local-only diagnostic report helpers for RecallForge.

No network transport lives here. Users must explicitly generate and share a
report, which keeps crash reporting opt-in and inspectable.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from . import __version__
from .feature_flags import list_feature_flags


_ALLOWLIST_ENV_KEYS = {
    "RECALLFORGE_BACKEND",
    "RECALLFORGE_MODE",
    "RECALLFORGE_MLX_QUANTIZE",
    "RECALLFORGE_DISABLE_MLX",
    "RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY",
    "RECALLFORGE_MLX_VIDEO_SAMPLE_FPS",
    "RECALLFORGE_MLX_VIDEO_MAX_FRAMES",
    "RECALLFORGE_MLX_VIDEO_FALLBACK_MAX_FRAMES",
    "RECALLFORGE_MLX_MIN_PIXELS",
    "RECALLFORGE_MLX_MAX_PIXELS",
    "RECALLFORGE_ENABLE_MLX_NATIVE_VIDEO_PROCESSING",
    "RECALLFORGE_CAPTIONER_IDLE_SECONDS",
    "RECALLFORGE_STORAGE",
    "RECALLFORGE_STORE_PATH",
    "RECALLFORGE_OVERFETCH_FACTOR",
    "RECALLFORGE_MAX_CANDIDATES",
    "RECALLFORGE_RERANK_TOP_K",
    "RECALLFORGE_ENABLE_MEDIA_RERANKING",
    "RECALLFORGE_MEDIA_QUERY_RERANK_TOP_K",
    "RECALLFORGE_MEDIA_RESULT_RERANK_TOP_K",
    "RECALLFORGE_MEDIA_RERANK_REQUIRE_AMBIGUITY",
    "RECALLFORGE_MEDIA_RERANK_MIN_RRF_MARGIN",
    "RECALLFORGE_ENABLE_RAW_VIDEO_QUERY_EMBEDDING",
    "RECALLFORGE_TRACE",
    "RECALLFORGE_MCP_MAX_CONCURRENCY",
    "RECALLFORGE_BM25_FALLBACK_MAX_ROWS",
    "RECALLFORGE_BULK_FLUSH_DOCS",
    "RECALLFORGE_BULK_FLUSH_EMBEDDINGS",
}


def _redact_path(value: str) -> str:
    """Redact the user's home directory from path-like environment values."""
    home = str(Path.home())
    if home and value.startswith(home):
        return "~" + value[len(home):]
    return value


def sanitized_recallforge_env(environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return allowlisted RecallForge env vars with home paths redacted."""
    env = os.environ if environ is None else environ
    sanitized: dict[str, str] = {}
    for key in sorted(_ALLOWLIST_ENV_KEYS):
        if key not in env:
            continue
        value = str(env[key])
        if key.endswith("_PATH") or key == "RECALLFORGE_STORE_PATH":
            value = _redact_path(value)
        sanitized[key] = value
    return sanitized


def collect_crash_report(
    *,
    message: str = "",
    include_env: bool = False,
    environ: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build a local-only crash report payload for manual sharing."""
    env = os.environ if environ is None else environ
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recallforge_version": __version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "feature_flags": list_feature_flags(env),
        "user_message": message.strip(),
        "privacy": {
            "network_sent": False,
            "sharing": "manual",
            "notes": "Generated locally. Review before attaching to GitHub Discussions or Issues.",
        },
    }
    if include_env:
        report["environment"] = sanitized_recallforge_env(env)
    return report


def write_crash_report(
    output_path: str | os.PathLike[str],
    *,
    message: str = "",
    include_env: bool = False,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """Write a crash report JSON file and return its resolved path."""
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = collect_crash_report(message=message, include_env=include_env, environ=environ)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output.resolve()
