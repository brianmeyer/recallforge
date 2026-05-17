#!/usr/bin/env python3
"""Diagnose weak cross-modal benchmark categories.

This offline tool reads a saved ``cross_modal_ablation.py`` JSON payload and
turns the raw per-stage metrics into a ranked diagnosis. It is intentionally
stdlib-only so it can run after a long benchmark session without loading model
stacks or touching the local RecallForge index.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_INPUT = Path("benchmarks/results/cross_modal_ablation_results.json")
DEFAULT_JSON_OUTPUT = Path("benchmarks/results/cross_modal_diagnostics.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/research/cross-modal-diagnostics.md")

WEAK_RECALL_AT_5 = 0.60
MIN_CATEGORY_QUERIES = 20
MEANINGFUL_DELTA = 0.10

DOCUMENT_FILE_TYPES = {"pdf", "docx", "pptx"}
DOCUMENT_CATEGORIES = {
    "text_to_document",
    "image_to_document",
    "video_to_document",
}
GENERIC_MEDIA_QUERIES = {
    "",
    "related text",
    "related image",
    "related video",
    "related document",
}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stage_role(stage_name: str) -> str:
    normalized = stage_name.lower()
    if "vector-only" in normalized or normalized == "vector":
        return "vector"
    if "bm25-only" in normalized or normalized == "bm25":
        return "bm25"
    if "reranker" in normalized or "hybrid" in normalized:
        return "hybrid"
    if "rrf" in normalized:
        return "rrf"
    return "other"


def _stage_lookup(stages: Mapping[str, Any]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for stage_name in stages:
        role = _stage_role(stage_name)
        if role not in roles:
            roles[role] = stage_name
    return roles


def _category_names(payload: Mapping[str, Any]) -> List[str]:
    categories = set(payload.get("categories", {}) or {})
    for stage_data in (payload.get("stages", {}) or {}).values():
        if isinstance(stage_data, Mapping):
            categories.update(stage_data)
    return sorted(categories)


def _category_query_count(payload: Mapping[str, Any], category: str) -> int:
    metadata = (payload.get("categories", {}) or {}).get(category)
    if isinstance(metadata, Mapping):
        queries = metadata.get("queries")
        if queries is not None:
            return int(queries)
    for stage_data in (payload.get("stages", {}) or {}).values():
        metrics = stage_data.get(category) if isinstance(stage_data, Mapping) else None
        if isinstance(metrics, Mapping) and metrics.get("total_queries") is not None:
            return int(metrics["total_queries"])
    return 0


def _metric(
    stages: Mapping[str, Mapping[str, Any]],
    stage_name: Optional[str],
    category: str,
    metric_name: str,
) -> Optional[float]:
    if not stage_name:
        return None
    metrics = stages.get(stage_name, {}).get(category)
    if not isinstance(metrics, Mapping) or metrics.get("skipped"):
        return None
    return _safe_float(metrics.get(metric_name))


def _asset_metric(
    stages: Mapping[str, Mapping[str, Any]],
    stage_name: Optional[str],
    category: str,
    metric_name: str,
) -> Optional[float]:
    if not stage_name:
        return None
    metrics = stages.get(stage_name, {}).get(category)
    if not isinstance(metrics, Mapping) or metrics.get("skipped"):
        return None
    asset_level = metrics.get("asset_level")
    if not isinstance(asset_level, Mapping):
        return None
    return _safe_float(asset_level.get(metric_name))


def _best_stage(
    stages: Mapping[str, Mapping[str, Any]],
    category: str,
    metric_name: str = "recall_at_5",
) -> Tuple[Optional[str], Optional[float]]:
    best_name: Optional[str] = None
    best_value: Optional[float] = None
    for stage_name, stage_data in stages.items():
        metrics = stage_data.get(category)
        if not isinstance(metrics, Mapping) or metrics.get("skipped"):
            continue
        value = _safe_float(metrics.get(metric_name))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_name = stage_name
            best_value = value
    return best_name, best_value


def _is_media_query_category(category: str, per_query_results: Sequence[Mapping[str, Any]]) -> bool:
    if category.startswith(("image_to_", "video_to_")):
        return True
    return any(q.get("query_type") in {"image", "video"} for q in per_query_results)


def _first_stage_queries(
    stages: Mapping[str, Mapping[str, Any]],
    category: str,
) -> List[Mapping[str, Any]]:
    for stage_data in stages.values():
        metrics = stage_data.get(category)
        if isinstance(metrics, Mapping):
            rows = metrics.get("per_query_results")
            if isinstance(rows, list) and rows:
                return [row for row in rows if isinstance(row, Mapping)]
    return []


def _stage_queries(
    stages: Mapping[str, Mapping[str, Any]],
    stage_name: Optional[str],
    category: str,
) -> List[Mapping[str, Any]]:
    if not stage_name:
        return []
    metrics = stages.get(stage_name, {}).get(category)
    if not isinstance(metrics, Mapping):
        return []
    rows = metrics.get("per_query_results")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _query_values(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    return [str(row.get("query") or "").strip().lower() for row in rows]


def _has_generic_media_queries(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    query_values = _query_values(rows)
    generic_count = sum(1 for query in query_values if query in GENERIC_MEDIA_QUERIES)
    return generic_count > 0 and generic_count >= max(1, len(query_values) // 2)


def _missing_media_query_paths(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    missing: List[str] = []
    for row in rows:
        query_type = row.get("query_type")
        if query_type == "image" and not row.get("image_query_path"):
            missing.append(str(row.get("query") or "image query"))
        if query_type == "video" and not row.get("video_query_path"):
            missing.append(str(row.get("query") or "video query"))
    return missing


def _has_asset_level_metrics(stages: Mapping[str, Mapping[str, Any]], category: str) -> bool:
    for stage_data in stages.values():
        metrics = stage_data.get(category)
        if isinstance(metrics, Mapping) and isinstance(metrics.get("asset_level"), Mapping):
            return True
    return False


def _audit_source_counts(rows: Sequence[Mapping[str, Any]], top_k: int = 5) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for result in (row.get("results") or [])[:top_k]:
            if not isinstance(result, Mapping):
                continue
            audit = result.get("audit")
            if not isinstance(audit, Mapping):
                continue
            sources = audit.get("rrf_sources")
            if not isinstance(sources, Mapping):
                continue
            for source, rank in sources.items():
                try:
                    if int(rank) > 0:
                        counts[str(source)] += 1
                except (TypeError, ValueError):
                    continue
    return dict(sorted(counts.items()))


def _normalize_result_path(filepath: str) -> str:
    raw = str(filepath or "")
    if not raw:
        return ""
    if raw.startswith("recallforge://"):
        raw = raw.split("/", 3)[-1]
    marker = "/tests/uat/corpus/"
    if marker in raw:
        raw = raw.split(marker, 1)[1]
    return raw


def _failure_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("hit_at_5") is True:
            continue
        top_results = []
        for result in (row.get("results") or [])[:3]:
            if isinstance(result, Mapping):
                top_results.append(_normalize_result_path(str(result.get("filepath") or "")))
        examples.append(
            {
                "query": row.get("query") or "",
                "query_type": row.get("query_type") or "text",
                "image_query_path": row.get("image_query_path"),
                "video_query_path": row.get("video_query_path"),
                "relevant_paths": row.get("relevant_paths") or [],
                "top_results": top_results,
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _issue(code: str, severity: str, evidence: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "evidence": evidence}


def _action(owner: str, priority: int, title: str, rationale: str) -> Dict[str, Any]:
    return {
        "owner": owner,
        "priority": priority,
        "title": title,
        "rationale": rationale,
    }


def _target_content_type(category: str) -> Optional[str]:
    if category.endswith("_text"):
        return "text"
    if category.endswith("_image"):
        return "image"
    if category.endswith("_video"):
        return "video"
    if category.endswith("_document"):
        return "document_family"
    return None


def _document_filter_gap(
    category: str,
    content_type_filters: Mapping[str, Optional[str]],
) -> bool:
    if category not in DOCUMENT_CATEGORIES:
        return False
    configured = content_type_filters.get(category)
    return configured is None or configured == "document"


def _build_category_diagnosis(
    payload: Mapping[str, Any],
    category: str,
    *,
    content_type_filters: Mapping[str, Optional[str]],
    min_queries: int,
    weak_recall_at_5: float,
    meaningful_delta: float,
) -> Dict[str, Any]:
    stages = payload.get("stages", {}) or {}
    role_to_stage = _stage_lookup(stages)
    first_rows = _first_stage_queries(stages, category)
    best_name, best_r5 = _best_stage(stages, category, "recall_at_5")

    vector_stage = role_to_stage.get("vector")
    bm25_stage = role_to_stage.get("bm25")
    rrf_stage = role_to_stage.get("rrf")
    hybrid_stage = role_to_stage.get("hybrid")

    vector_r5 = _metric(stages, vector_stage, category, "recall_at_5")
    vector_r10 = _metric(stages, vector_stage, category, "recall_at_10")
    bm25_r5 = _metric(stages, bm25_stage, category, "recall_at_5")
    rrf_r5 = _metric(stages, rrf_stage, category, "recall_at_5")
    rrf_r10 = _metric(stages, rrf_stage, category, "recall_at_10")
    hybrid_r5 = _metric(stages, hybrid_stage, category, "recall_at_5")
    hybrid_r10 = _metric(stages, hybrid_stage, category, "recall_at_10")
    best_rows = _stage_queries(stages, best_name, category)

    total_queries = _category_query_count(payload, category)
    issues: List[Dict[str, str]] = []
    media_query = _is_media_query_category(category, first_rows)

    if total_queries < min_queries:
        issues.append(
            _issue(
                "under_sampled_category",
                "high" if total_queries < 5 else "medium",
                f"{total_queries} queries is below the {min_queries}-query diagnostic floor.",
            )
        )

    bm25_metrics = stages.get(bm25_stage or "", {}).get(category)
    if isinstance(bm25_metrics, Mapping) and bm25_metrics.get("skipped"):
        issues.append(
            _issue(
                "bm25_modality_blind",
                "medium",
                str(bm25_metrics.get("skip_reason") or "BM25 stage skipped."),
            )
        )

    if vector_r5 is not None and vector_r5 < weak_recall_at_5 and media_query:
        issues.append(
            _issue(
                "embedding_alignment_gap",
                "high",
                f"Vector-only R@5={vector_r5:.3f}; media-query categories need raw embedding alignment above {weak_recall_at_5:.2f}.",
            )
        )

    if vector_r5 is not None and rrf_r5 is not None:
        lift = rrf_r5 - vector_r5
        r10_lift = (rrf_r10 or 0.0) - (vector_r10 or 0.0)
        if lift >= meaningful_delta or r10_lift >= meaningful_delta:
            issues.append(
                _issue(
                    "derived_text_probe_lift",
                    "positive",
                    f"RRF improves R@5 by {lift:+.3f} and R@10 by {r10_lift:+.3f} over raw vector search.",
                )
            )
        elif media_query and best_r5 is not None and best_r5 < weak_recall_at_5:
            issues.append(
                _issue(
                    "derived_text_probe_insufficient",
                    "high",
                    f"RRF R@5={rrf_r5:.3f} does not materially lift vector R@5={vector_r5:.3f}.",
                )
            )

    if rrf_r5 is not None and hybrid_r5 is not None:
        rerank_delta = hybrid_r5 - rrf_r5
        if rerank_delta <= -meaningful_delta:
            issues.append(
                _issue(
                    "reranker_regression",
                    "high",
                    f"Hybrid reranker R@5={hybrid_r5:.3f} trails RRF R@5={rrf_r5:.3f}.",
                )
            )
        elif best_r5 is not None and best_r5 < weak_recall_at_5 and abs(rerank_delta) < meaningful_delta:
            issues.append(
                _issue(
                    "reranker_no_lift",
                    "medium",
                    f"Hybrid reranker changes R@5 by only {rerank_delta:+.3f} versus RRF.",
                )
            )

    if not _has_asset_level_metrics(stages, category):
        issues.append(
            _issue(
                "parent_asset_metrics_missing",
                "medium",
                "Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.",
            )
        )
    elif best_name:
        memory_r5 = _metric(stages, best_name, category, "recall_at_5")
        asset_r5 = _asset_metric(stages, best_name, category, "recall_at_5")
        if memory_r5 is not None and asset_r5 is not None and memory_r5 - asset_r5 >= meaningful_delta:
            issues.append(
                _issue(
                    "parent_rollup_matters",
                    "positive",
                    f"Parent-memory R@5={memory_r5:.3f} is {memory_r5 - asset_r5:+.3f} above raw asset R@5.",
                )
            )

    if _document_filter_gap(category, content_type_filters):
        issues.append(
            _issue(
                "document_family_filter_gap",
                "medium",
                "Benchmark cannot currently constrain results to the pdf/docx/pptx document family with a single content_type filter.",
            )
        )

    if _has_generic_media_queries(first_rows):
        issues.append(
            _issue(
                "generic_query_artifact",
                "medium",
                "Most media-query prompts are generic placeholders such as 'related document', so scores mix retrieval quality with query-definition ambiguity.",
            )
        )

    missing_paths = _missing_media_query_paths(first_rows)
    if missing_paths:
        issues.append(
            _issue(
                "media_query_path_missing",
                "low",
                f"{len(missing_paths)} per-query rows omit image_query_path or video_query_path instrumentation.",
            )
        )

    weakness = max(0.0, weak_recall_at_5 - (best_r5 or 0.0))
    sample_penalty = 0.15 if total_queries < min_queries else 0.0
    media_penalty = 0.05 if media_query else 0.0
    document_penalty = 0.05 if category in DOCUMENT_CATEGORIES else 0.0
    priority_score = round(weakness + sample_penalty + media_penalty + document_penalty, 4)

    rrf_rows = _stage_queries(stages, rrf_stage, category)
    hybrid_rows = _stage_queries(stages, hybrid_stage, category)
    audit_sources = _audit_source_counts(hybrid_rows or rrf_rows)

    return {
        "category": category,
        "target_content_type": _target_content_type(category),
        "configured_content_type_filter": content_type_filters.get(category),
        "total_queries": total_queries,
        "best_stage": best_name,
        "best_recall_at_5": best_r5,
        "priority_score": priority_score,
        "metrics": {
            "vector_recall_at_5": vector_r5,
            "vector_recall_at_10": vector_r10,
            "bm25_recall_at_5": bm25_r5,
            "rrf_recall_at_5": rrf_r5,
            "rrf_recall_at_10": rrf_r10,
            "hybrid_recall_at_5": hybrid_r5,
            "hybrid_recall_at_10": hybrid_r10,
        },
        "issues": issues,
        "audit_source_counts_top5": audit_sources,
        "failure_examples": _failure_examples(best_rows or first_rows),
    }


def _aggregate_actions(diagnoses: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    issue_codes_by_category: Dict[str, List[str]] = {}
    for diagnosis in diagnoses:
        issue_codes_by_category[str(diagnosis["category"])] = [
            str(issue["code"]) for issue in diagnosis.get("issues", [])
        ]

    actions: List[Dict[str, Any]] = []

    weak_media = [
        category
        for category, codes in issue_codes_by_category.items()
        if "embedding_alignment_gap" in codes or "derived_text_probe_insufficient" in codes
    ]
    if weak_media:
        actions.append(
            _action(
                "search",
                1,
                "Add bounded cascaded media reranking only after cheap top-K retrieval.",
                "The weakest media-query categories are not rescued by current RRF/reranker stages, so REC-130 should target a strict top-K cascade instead of broad expensive scoring.",
            )
        )

    if any("under_sampled_category" in codes for codes in issue_codes_by_category.values()):
        actions.append(
            _action(
                "evals",
                2,
                "Expand weak categories to at least 20 queries and keep parent-memory scoring.",
                "Several weak categories have 1-3 examples, which is too small to distinguish model weakness from benchmark noise; this maps directly to REC-160.",
            )
        )

    if any("document_family_filter_gap" in codes for codes in issue_codes_by_category.values()):
        actions.append(
            _action(
                "indexing",
                3,
                "Represent document-family filters explicitly across pdf/docx/pptx roots and children.",
                "Document retrieval categories are evaluated without a proper document-family content filter, so unrelated images/videos can dominate media-query results.",
            )
        )

    if any("generic_query_artifact" in codes for codes in issue_codes_by_category.values()):
        actions.append(
            _action(
                "evals",
                4,
                "Replace placeholder media prompts with grounded intent labels and provenance.",
                "Queries such as 'related document' are useful smoke probes but too ambiguous for release-quality diagnostics.",
            )
        )

    if any("parent_asset_metrics_missing" in codes for codes in issue_codes_by_category.values()):
        actions.append(
            _action(
                "evals",
                5,
                "Rerun cross-modal ablation with the current harness to populate asset_level metrics.",
                "The checked-in result is from v0.2.0 and predates serialized asset-level rollups, so it cannot fully separate child-asset hits from parent-memory hits.",
            )
        )

    if any("derived_text_probe_lift" in codes for codes in issue_codes_by_category.values()):
        actions.append(
            _action(
                "ingest",
                6,
                "Keep strengthening captions, transcripts, and OCR as first-class retrieval text.",
                "Where RRF improves over vector-only, the improvement is evidence that derived text is helping and should be cached/versioned rather than recomputed ad hoc.",
            )
        )

    actions.append(
        _action(
            "model_research",
            7,
            "Benchmark visual/document-specialized retrievers against the weak categories.",
            "ViDoRe-style visual document retrieval and MTEB/BEIR-style qrels offer better external baselines for document-heavy failures than anecdotes from one synthetic corpus.",
        )
    )

    return actions


def _issue_summary(diagnoses: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for diagnosis in diagnoses:
        for issue in diagnosis.get("issues", []):
            counts[str(issue["code"])] += 1
    return dict(sorted(counts.items()))


def _load_content_type_filters(categories: Sequence[str]) -> Dict[str, Optional[str]]:
    """Inspect the current benchmark helper without making it a hard dependency."""
    try:
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from benchmarks import cross_modal_ablation  # type: ignore
    except Exception:
        return {}
    return {
        category: cross_modal_ablation._result_content_type_for_category(category)
        for category in categories
    }


def build_diagnostics(
    payload: Mapping[str, Any],
    *,
    content_type_filters: Optional[Mapping[str, Optional[str]]] = None,
    min_queries: int = MIN_CATEGORY_QUERIES,
    weak_recall_at_5: float = WEAK_RECALL_AT_5,
    meaningful_delta: float = MEANINGFUL_DELTA,
) -> Dict[str, Any]:
    categories = _category_names(payload)
    filters = dict(content_type_filters or _load_content_type_filters(categories))
    diagnoses = [
        _build_category_diagnosis(
            payload,
            category,
            content_type_filters=filters,
            min_queries=min_queries,
            weak_recall_at_5=weak_recall_at_5,
            meaningful_delta=meaningful_delta,
        )
        for category in categories
    ]
    diagnoses.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            str(row["category"]),
        )
    )

    weak_categories = [
        diagnosis
        for diagnosis in diagnoses
        if diagnosis["best_recall_at_5"] is None
        or float(diagnosis["best_recall_at_5"]) < weak_recall_at_5
        or float(diagnosis["priority_score"]) >= 0.20
    ]

    return {
        "diagnostic": "cross_modal_retrieval",
        "diagnosed_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "benchmark": payload.get("benchmark"),
            "version": payload.get("version"),
            "generated_at": payload.get("generated_at"),
            "run_status": payload.get("run_status"),
        },
        "thresholds": {
            "weak_recall_at_5": weak_recall_at_5,
            "min_category_queries": min_queries,
            "meaningful_delta": meaningful_delta,
        },
        "summary": {
            "categories": len(diagnoses),
            "weak_categories": len(weak_categories),
            "issue_counts": _issue_summary(diagnoses),
        },
        "weak_categories": weak_categories,
        "all_categories": diagnoses,
        "prioritized_actions": _aggregate_actions(diagnoses),
        "method_notes": [
            "Vector-only is treated as the raw embedding baseline.",
            "RRF lift over vector-only is treated as evidence from derived text probes such as captions, transcripts, OCR, or BM25 text.",
            "Hybrid-minus-RRF isolates the current reranker contribution.",
            "Parent-memory versus asset-level scoring is only available when the source payload includes asset_level metrics.",
        ],
    }


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _fmt_filter(value: Optional[str]) -> str:
    return "`None`" if value is None else f"`{value}`"


def render_markdown(diagnostics: Mapping[str, Any]) -> str:
    source = diagnostics.get("source", {}) or {}
    thresholds = diagnostics.get("thresholds", {}) or {}
    weak_categories = diagnostics.get("weak_categories", []) or []
    all_categories = diagnostics.get("all_categories", []) or []
    actions = diagnostics.get("prioritized_actions", []) or []
    issue_counts = (diagnostics.get("summary", {}) or {}).get("issue_counts", {}) or {}

    lines: List[str] = [
        "# Cross-Modal Retrieval Diagnostics",
        "",
        "This report is generated from the saved cross-modal ablation JSON. It separates raw embedding alignment, derived-text contribution, reranker contribution, benchmark artifacts, and parent-memory scoring coverage.",
        "",
        "## Source",
        "",
        f"- Benchmark: `{source.get('benchmark') or 'unknown'}`",
        f"- Source version: `{source.get('version') or 'unknown'}`",
        f"- Source generated at: `{source.get('generated_at') or 'unknown'}`",
        f"- Run status: `{source.get('run_status') or 'unknown'}`",
        f"- Weak threshold: R@5 < {_fmt_pct(_safe_float(thresholds.get('weak_recall_at_5')))}",
        f"- Query floor: {thresholds.get('min_category_queries')} queries per category",
        "",
        "## Weak And At-Risk Category Ranking",
        "",
        "| Priority | Category | Queries | Best stage | Best R@5 | Vector R@5 | RRF R@5 | Hybrid R@5 | Key issues |",
        "|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]

    for diagnosis in weak_categories:
        metrics = diagnosis.get("metrics", {}) or {}
        issues = ", ".join(issue["code"] for issue in diagnosis.get("issues", [])[:4])
        lines.append(
            "| "
            f"{diagnosis.get('priority_score'):.2f} | "
            f"`{diagnosis.get('category')}` | "
            f"{diagnosis.get('total_queries')} | "
            f"{diagnosis.get('best_stage') or 'n/a'} | "
            f"{_fmt_pct(_safe_float(diagnosis.get('best_recall_at_5')))} | "
            f"{_fmt_pct(_safe_float(metrics.get('vector_recall_at_5')))} | "
            f"{_fmt_pct(_safe_float(metrics.get('rrf_recall_at_5')))} | "
            f"{_fmt_pct(_safe_float(metrics.get('hybrid_recall_at_5')))} | "
            f"{issues or 'none'} |"
        )

    lines.extend(
        [
            "",
            "## Diagnosis Summary",
            "",
        ]
    )
    for code, count in issue_counts.items():
        lines.append(f"- `{code}`: {count}")

    lines.extend(
        [
            "",
            "## Prioritized Fix List",
            "",
        ]
    )
    for action in actions:
        lines.append(
            f"{action['priority']}. **{action['owner']}** - {action['title']} {action['rationale']}"
        )

    lines.extend(
        [
            "",
            "## Category Evidence",
            "",
        ]
    )
    for diagnosis in all_categories:
        metrics = diagnosis.get("metrics", {}) or {}
        lines.extend(
            [
                f"### `{diagnosis.get('category')}`",
                "",
                f"- Queries: {diagnosis.get('total_queries')}",
                f"- Target result family: `{diagnosis.get('target_content_type') or 'mixed'}`",
                f"- Configured benchmark content filter: {_fmt_filter(diagnosis.get('configured_content_type_filter'))}",
                f"- Best stage/R@5: {diagnosis.get('best_stage') or 'n/a'} / {_fmt_pct(_safe_float(diagnosis.get('best_recall_at_5')))}",
                f"- Raw vector R@5/R@10: {_fmt_pct(_safe_float(metrics.get('vector_recall_at_5')))} / {_fmt_pct(_safe_float(metrics.get('vector_recall_at_10')))}",
                f"- RRF R@5/R@10: {_fmt_pct(_safe_float(metrics.get('rrf_recall_at_5')))} / {_fmt_pct(_safe_float(metrics.get('rrf_recall_at_10')))}",
                f"- Hybrid R@5/R@10: {_fmt_pct(_safe_float(metrics.get('hybrid_recall_at_5')))} / {_fmt_pct(_safe_float(metrics.get('hybrid_recall_at_10')))}",
            ]
        )
        audit_counts = diagnosis.get("audit_source_counts_top5") or {}
        if audit_counts:
            source_text = ", ".join(f"`{key}`={value}" for key, value in audit_counts.items())
            lines.append(f"- Top-5 audit source counts: {source_text}")
        for issue in diagnosis.get("issues", []):
            lines.append(f"- `{issue['code']}` ({issue['severity']}): {issue['evidence']}")
        examples = diagnosis.get("failure_examples") or []
        if examples:
            lines.append("- Example misses:")
            for example in examples:
                top_results = ", ".join(
                    f"`{path}`" for path in (example.get("top_results") or []) if path
                )
                query = example.get("query") or "<empty media query>"
                lines.append(
                    f"  - `{query}` ({example.get('query_type')}): expected {example.get('relevant_paths')}; top results {top_results or 'n/a'}"
                )
        lines.append("")

    lines.extend(
        [
            "## Method Notes",
            "",
        ]
    )
    for note in diagnostics.get("method_notes", []) or []:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## External Evaluation References",
            "",
            "- [BEIR](https://github.com/beir-cellar/beir) structures retrieval evaluation around corpus, queries, qrels, run results, and metrics such as NDCG, MAP, Recall, Precision, and MRR.",
            "- [MTEB](https://github.com/embeddings-benchmark/mteb) is the broader embedding and retrieval evaluation framework now used by ViDoRe for single-model retriever submissions.",
            "- [ViDoRe pipeline evaluation](https://github.com/illuin-tech/vidore-benchmark) explicitly covers multi-stage, hybrid, reranking, OCR, and custom preprocessing pipelines for visual document retrieval.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="cross_modal_ablation JSON to diagnose")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT, help="diagnostic JSON output path")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MARKDOWN_OUTPUT, help="diagnostic Markdown output path")
    parser.add_argument("--min-queries", type=int, default=MIN_CATEGORY_QUERIES, help="minimum desired queries per category")
    parser.add_argument("--weak-recall-at-5", type=float, default=WEAK_RECALL_AT_5, help="R@5 threshold for weak categories")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    diagnostics = build_diagnostics(
        payload,
        min_queries=args.min_queries,
        weak_recall_at_5=args.weak_recall_at_5,
    )
    _write_json(args.output_json, diagnostics)
    _write_text(args.output_md, render_markdown(diagnostics))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
