"""
feature_flags.py - Central registry for RecallForge beta/experimental flags.

Feature flags are environment-variable backed so CLI, MCP, tests, and local
agent hosts all see the same behavior without a separate config service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class FeatureFlag:
    """Documented environment-backed feature flag."""

    name: str
    default: str
    description: str
    stage: str = "beta"
    choices: tuple[str, ...] = ()
    restart_required: bool = True

    def value_from(self, environ: Mapping[str, str]) -> str:
        return str(environ.get(self.name, self.default))

    def enabled_from(self, environ: Mapping[str, str]) -> Optional[bool]:
        if not self.choices:
            return None
        value = self.value_from(environ).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return None

    def as_dict(self, environ: Mapping[str, str]) -> dict:
        value = self.value_from(environ)
        return {
            "name": self.name,
            "value": value,
            "default": self.default,
            "enabled": self.enabled_from(environ),
            "stage": self.stage,
            "choices": list(self.choices),
            "restart_required": self.restart_required,
            "description": self.description,
        }


FEATURE_FLAGS: tuple[FeatureFlag, ...] = (
    FeatureFlag(
        name="RECALLFORGE_ENABLE_MEDIA_RERANKING",
        default="0",
        description="Enable capped multimodal reranking for image/video-involved searches.",
        stage="beta",
        choices=("0", "1"),
        restart_required=False,
    ),
    FeatureFlag(
        name="RECALLFORGE_ENABLE_RAW_VIDEO_QUERY_EMBEDDING",
        default="0",
        description="Enable raw video query embedding instead of safer caption/transcript-first retrieval.",
        stage="experimental",
        choices=("0", "1"),
    ),
    FeatureFlag(
        name="RECALLFORGE_ENABLE_MLX_NATIVE_VIDEO_PROCESSING",
        default="0",
        description="Enable qwen-vl-utils native video decoding on MLX.",
        stage="experimental",
        choices=("0", "1"),
    ),
    FeatureFlag(
        name="RECALLFORGE_MEDIA_RERANK_REQUIRE_AMBIGUITY",
        default="1",
        description="Only run media reranking when cheap RRF results are close enough to need it.",
        stage="beta",
        choices=("0", "1"),
        restart_required=False,
    ),
    FeatureFlag(
        name="RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY",
        default="1",
        description="Concurrency ceiling for heavy MLX multimodal operations.",
        stage="safety",
    ),
    FeatureFlag(
        name="RECALLFORGE_MCP_MAX_CONCURRENCY",
        default="2",
        description="Maximum number of blocking MCP tool operations run concurrently.",
        stage="safety",
    ),
    FeatureFlag(
        name="RECALLFORGE_TRACE",
        default="0",
        description="Enable structured trace logging for MCP tool handlers.",
        stage="diagnostics",
        choices=("0", "1"),
        restart_required=False,
    ),
)


def list_feature_flags(environ: Optional[Mapping[str, str]] = None) -> list[dict]:
    """Return the effective feature flag registry for display or diagnostics."""
    env = os.environ if environ is None else environ
    return [flag.as_dict(env) for flag in FEATURE_FLAGS]
