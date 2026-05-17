"""
search.py - Hybrid Search Pipeline for RecallForge.

Combines BM25, vector search, and reranking with tiered modes:
- embed: Embedder only (fastest, lowest memory)
- hybrid: Embedder + Reranker

Intent-aware query steering:
- exact_lookup: Boost BM25 weight in RRF fusion, lower vector weight
- semantic: Boost vector weight, lower BM25
- broad: Equal weights for all sources
- None: Default behavior (unchanged)

Uses true concurrency with ThreadPoolExecutor for parallel searches.
"""

import concurrent.futures
import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import List, Dict, Any, Optional, Union

from .backends.base import ModelBackend
from .cache import EmbeddingCache
from .storage.base import StorageBackend, SearchResult

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    """Parse boolean environment flags with a safe default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_mlx_backend(backend: ModelBackend) -> bool:
    """Best-effort detection for the MLX backend without importing it here."""
    return type(backend).__name__ == "MLXBackend"


def _log_stage_metrics(
    stage: str,
    results: List[Any],
    start_time: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit structured per-stage metrics for observability.

    Args:
        stage: Name of the pipeline stage (bm25, vector, rrf, reranker, blend)
        results: List of results (SearchResult or HybridResult)
        start_time: Optional start time for latency calculation
        extra: Optional extra fields to include in log
    """
    # Calculate latency if start_time provided
    latency_ms = 0.0
    if start_time is not None:
        latency_ms = (time.perf_counter() - start_time) * 1000

    # Count candidates by content_type
    total_count = len(results)
    counts_by_type: Dict[str, int] = {}
    scores_by_type: Dict[str, List[float]] = {}

    for r in results:
        # Handle both SearchResult and HybridResult
        content_type = getattr(r, 'content_type', 'unknown')
        score = getattr(r, 'score', 0.0)

        counts_by_type[content_type] = counts_by_type.get(content_type, 0) + 1
        if content_type not in scores_by_type:
            scores_by_type[content_type] = []
        scores_by_type[content_type].append(score)

    # Build log message with key=value format
    log_parts = [
        f"stage={stage}",
        f"candidate_count={total_count}",
        f"latency_ms={latency_ms:.2f}",
    ]

    # Add per-content-type counts
    for ct, count in sorted(counts_by_type.items()):
        log_parts.append(f"count_{ct}={count}")

    # Add per-content-type score stats
    for ct, scores in sorted(scores_by_type.items()):
        if scores:
            log_parts.append(f"score_min_{ct}={min(scores):.4f}")
            log_parts.append(f"score_max_{ct}={max(scores):.4f}")
            log_parts.append(f"score_mean_{ct}={sum(scores)/len(scores):.4f}")

    # Add extra fields
    if extra:
        for key, value in extra.items():
            log_parts.append(f"{key}={value}")

    logger.debug("stage_metrics " + " ".join(log_parts))


def _canonicalize_memory_result_paths(result: "HybridResult", canonical_root_path: Optional[str]) -> tuple[str, str]:
    """Return a stable result filepath plus display path for rolled memory hits."""
    if not canonical_root_path:
        return result.filepath, result.display_path

    if str(result.filepath or "").startswith("recallforge://"):
        return (
            f"recallforge://{result.collection}/{canonical_root_path}",
            f"{result.collection}/{canonical_root_path}",
        )

    return canonical_root_path, canonical_root_path


# Intent-to-weight mappings for RRF fusion
# Each intent maps source names to weight multipliers
INTENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "exact_lookup": {"original_fts": 2.5, "original_vec": 0.8},
    "semantic": {"original_fts": 0.8, "original_vec": 2.5},
    "broad": {"original_fts": 1.0, "original_vec": 1.0},
}


@dataclass
class SearchAudit:
    """Per-result audit trail capturing all scoring provenance.

    This dataclass stores detailed information about how a result was scored
    through the pipeline, enabling post-hoc analysis and debugging.
    """
    # Result identification
    filepath: str
    content_type: str

    # RRF provenance
    rrf_sources: Dict[str, int] = field(default_factory=dict)  # source_name -> rank in that source list
    rrf_score: float = 0.0  # Final RRF score after fusion

    # Reranker provenance
    reranker_raw_score: float = 0.5
    reranker_normalized_score: float = 0.5
    reranker_scoring_path: str = "unknown"  # text, vl_image, vl_video, or fallback

    # Blend provenance
    blend_weights: Dict[str, float] = field(default_factory=dict)  # rrf_weight, rerank_weight
    media_compensation_applied: bool = False  # Whether media boost was applied in RRF
    memory_rollup_boost: float = 1.0  # Multiplier applied when sibling assets are rolled up
    memory_primary_evidence_path: Optional[str] = None
    memory_supporting_paths: List[str] = field(default_factory=list)
    final_blended_score: float = 0.0


@dataclass
class HybridResult:
    """Final hybrid search result with blended score."""
    filepath: str
    display_path: str
    title: str
    context: Optional[str]
    hash: str
    docid: str
    collection: str
    modified_at: str
    body_length: int
    body: Optional[str]
    score: float  # Blended score (RRF + reranker)
    rrf_rank: int  # Position in RRF output
    rerank_score: float  # Reranker score
    source: str  # Sources that contributed to this result
    content_type: str = "text"  # Content modality (text, image, video)
    memory_id: Optional[str] = None
    memory_role: str = "root"
    memory_root_path: Optional[str] = None
    memory_hit_count: int = 1
    memory_primary_evidence_path: Optional[str] = None
    memory_supporting_paths: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    audit: Optional[SearchAudit] = None  # Per-result audit trail


# Visual query indicators - multi-word phrases checked first (high precision),
# then single-word tokens checked with word-boundary matching to avoid false
# positives like "mapreduce" matching "map" or "tableau" matching "table".
_VISUAL_PHRASE_INDICATORS = [
    "show me", "image of", "photo of", "picture of", "architecture diagram",
    "flow chart", "mind map", "that photo", "that image", "the diagram",
    "the chart", "the picture", "the screenshot", "visual representation",
]
_VISUAL_WORD_INDICATORS = [
    "show", "diagram", "chart", "screenshot", "illustration", "drawing", "infographic",
    "wireframe", "mockup", "sketch", "whiteboard", "portrait",
]
# Deliberately excluded: "graph", "table", "map", "figure", "landscape",
# "scene", "visual" — too many false positives as substrings or common non-visual usage.

_VISUAL_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _VISUAL_WORD_INDICATORS) + r")\b",
    re.IGNORECASE,
)


def _is_visual_query(query: str) -> bool:
    """Check if query implies visual content.

    Uses phrase matching first (high precision), then word-boundary matching
    for single tokens to avoid false positives like 'mapreduce' or 'tableau'.
    """
    query_lower = query.lower().strip()
    if any(phrase in query_lower for phrase in _VISUAL_PHRASE_INDICATORS):
        return True
    if query_lower == "show":
        return True
    if "show" in query_lower.split():
        # Bare 'show' inside a normal sentence is too ambiguous.
        safe_query = re.sub(r"\bshow\b", "", query_lower).strip()
        return bool(safe_query) and bool(_VISUAL_WORD_PATTERN.search(safe_query))
    return bool(_VISUAL_WORD_PATTERN.search(query_lower))


def _generate_text_variants(query: str, _backend: ModelBackend) -> List[str]:
    """Generate 1-2 semantic variants of a text query.

    Prefer the backend's lightweight text generator when available, and fall
    back to the legacy heuristic expansion rules when generation is missing or
    returns unusable output.
    """
    def _normalize_variant(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        cleaned = cleaned.strip("\"'` ")
        cleaned = re.sub(r"^\s*(?:[-*•]\s*|\d+[\.\)]\s*)", "", cleaned)
        return cleaned.strip()

    def _dedupe_variants(items: List[str]) -> List[str]:
        seen: set[str] = set()
        deduped: List[str] = []
        original = query.strip().lower()
        for item in items:
            cleaned = _normalize_variant(item)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered == original or lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(cleaned)
            if len(deduped) >= 2:
                break
        return deduped

    def _heuristic_variants() -> List[str]:
        variants = []
        query_lower = query.lower().strip()

        expansions = {
            "how to": ["guide for", "steps to", "tutorial on"],
            "what is": ["definition of", "explaining", "introduction to"],
            "best way to": ["optimal method for", "recommended approach to"],
            "difference between": ["comparison of", "vs", "versus"],
            "example of": ["sample", "instance of", "demonstration of"],
            "how do": ["how does", "how can", "ways to"],
        }

        for pattern, alternatives in expansions.items():
            if pattern in query_lower:
                for alt in alternatives[:1]:
                    variant = query_lower.replace(pattern, alt, 1)
                    if variant != query_lower:
                        variants.append(variant)
                        break
                if variants:
                    break

        if not variants and (query_lower.startswith("what ") or query_lower.startswith("how ")):
            words = query_lower.split()
            if len(words) > 2:
                variants.append(" ".join(words[2:]))
                variants.append(f"information about {' '.join(words[2:])}")

        return _dedupe_variants(variants)

    generate_text = getattr(_backend, "generate_text", None)
    if callable(generate_text):
        prompt = (
            "Rewrite the following search query into up to 2 short alternative "
            "search queries for retrieval.\n"
            "Rules:\n"
            "- Preserve the original intent\n"
            "- Use different wording that may match other documents\n"
            "- Keep each variant concise\n"
            "- Return only the alternative queries, one per line\n\n"
            f"Query: {query.strip()}"
        )
        try:
            raw = generate_text(prompt, max_tokens=80) or ""
        except Exception as exc:
            logger.debug("query_expansion_generate_text failed: %s", exc)
            raw = ""

        generated: List[str] = []
        if raw.strip():
            stripped = raw.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    payload = json.loads(stripped)
                    if isinstance(payload, list):
                        generated.extend(str(item) for item in payload)
                except json.JSONDecodeError:
                    pass

            if not generated:
                for line in (line.strip() for line in raw.splitlines() if line.strip()):
                    lowered = line.lower()
                    if lowered.startswith("here are"):
                        continue
                    if lowered.startswith("query:") or lowered.startswith("queries:"):
                        line = line.split(":", 1)[1]
                    generated.append(line)

            parsed = _dedupe_variants(generated)
            if parsed:
                return parsed

    return _heuristic_variants()


def _generate_visual_description(query: str) -> str:
    """Generate descriptive text for what the image likely contains.

    For queries that imply visual content, generate a description that
    describes what the image likely shows. This helps match against
    image embeddings which encode visual content.
    """
    query_lower = query.lower().strip()

    # Remove visual indicators to get the core subject
    core_query = query_lower
    for phrase in _VISUAL_PHRASE_INDICATORS:
        core_query = core_query.replace(phrase, "")
    for word in _VISUAL_WORD_INDICATORS:
        core_query = re.sub(r"\b" + re.escape(word) + r"\b", "", core_query)
    # Strip leading prepositions as whole words (not char-strip which corrupts terms)
    core_query = re.sub(r"^\s*(of|a|an|the|with|for|in)\b\s*", "", core_query.strip()).strip()

    if not core_query:
        return query_lower

    # Generate a visual description based on the core subject
    # This describes what the image likely contains
    descriptions = [
        f"A photograph or image showing {core_query}",
        f"Visual representation of {core_query} with details and context",
        f"Image depicting {core_query} in a clear view",
    ]

    return descriptions[0]


def expand_query(query: str, backend: ModelBackend, expand: bool = False) -> List[str]:
    """Expand a query into multiple variants for improved retrieval.

    Args:
        query: Original search query
        backend: Model backend for generating expansions
        expand: Whether to enable query expansion (default: False, opt-in)

    Returns:
        List of query variants. Always includes the original query as first element.
        Additional variants are generated based on query type:
        - Text queries: 1-2 semantic variants
        - Visual queries: descriptive text of what the image likely contains
    """
    if not expand or not query or not query.strip():
        return [query] if query else []

    variants = [query]  # Always keep original

    if _is_visual_query(query):
        # For visual queries, generate a description of what the image contains
        visual_desc = _generate_visual_description(query)
        if visual_desc and visual_desc != query:
            variants.append(visual_desc)
    else:
        # For text queries, generate semantic variants
        text_variants = _generate_text_variants(query, backend)
        variants.extend(text_variants)

    return variants


class HybridSearcher:
    """
    Full hybrid search pipeline with tiered modes.

    Modes:
    - embed: Embedder only. Vector + FTS, no reranking.
    - hybrid: + Reranker. Cross-encoder refinement.

    Uses ThreadPoolExecutor for concurrent searches.
    """

    def __init__(
        self,
        backend: ModelBackend,
        storage: StorageBackend,
        limit: int = 10,
        fts_probe_limit: int = 20,
        rrf_k: int = 60,
        collection: Optional[str] = None,
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
        max_workers: int = 8,
        overfetch_factor: int = 10,
        max_candidates: int = 200,
        rerank_top_k: int = 20,
        cache: Optional[EmbeddingCache] = None,
        intent: Optional[str] = None,
        expand: bool = False,
        enable_media_query_probe: bool = True,
    ):
        """
        Initialize hybrid searcher.

        Args:
            backend: Model backend (TorchBackend or MLXBackend)
            storage: Storage backend (LanceDBBackend)
            limit: Final number of results to return
            fts_probe_limit: How many BM25 results to retrieve
            rrf_k: RRF fusion constant
            collection: Optional collection filter
            content_type: Optional content type filter
            user_id: Optional user namespace filter
            session_id: Optional session namespace filter
            project_id: Optional project namespace filter
            profile: Optional profile namespace filter
            max_workers: ThreadPoolExecutor workers for parallel searches
            overfetch_factor: Candidate overfetch multiplier before final trim
            max_candidates: Hard cap on candidate pool size
            rerank_top_k: Maximum number of top RRF candidates to rerank
            cache: Optional EmbeddingCache; created with default maxsize if None
            intent: Optional intent for query steering ("exact_lookup", "semantic", "broad")
            expand: Whether to enable VL-aware query expansion (default: False, opt-in)
            enable_media_query_probe: Whether image/video queries should generate
                caption/transcript BM25 probe text before fusion. Default True.
        """
        self.backend = backend
        self.storage = storage
        self.limit = limit
        self.fts_probe_limit = fts_probe_limit
        self.rrf_k = rrf_k
        self.collection = collection
        self.content_type = content_type
        self.user_id = user_id
        self.session_id = session_id
        self.project_id = project_id
        self.profile = profile
        self.max_workers = max_workers
        env_overfetch = int(os.environ.get("RECALLFORGE_OVERFETCH_FACTOR", overfetch_factor))
        env_max_candidates = int(os.environ.get("RECALLFORGE_MAX_CANDIDATES", max_candidates))
        env_rerank_top_k = int(os.environ.get("RECALLFORGE_RERANK_TOP_K", rerank_top_k))
        env_media_query_rerank_top_k = int(
            os.environ.get(
                "RECALLFORGE_MEDIA_QUERY_RERANK_TOP_K",
                min(env_rerank_top_k, 5),
            )
        )
        env_media_result_rerank_top_k = int(
            os.environ.get(
                "RECALLFORGE_MEDIA_RESULT_RERANK_TOP_K",
                min(env_rerank_top_k, 10),
            )
        )
        env_media_rerank_min_rrf_margin = float(
            os.environ.get("RECALLFORGE_MEDIA_RERANK_MIN_RRF_MARGIN", "0.25")
        )
        self.enable_media_reranking = _env_flag(
            "RECALLFORGE_ENABLE_MEDIA_RERANKING",
            False,
        )
        self.media_rerank_require_ambiguity = _env_flag(
            "RECALLFORGE_MEDIA_RERANK_REQUIRE_AMBIGUITY",
            True,
        )
        self.enable_raw_video_query_embedding = _env_flag(
            "RECALLFORGE_ENABLE_RAW_VIDEO_QUERY_EMBEDDING",
            not _is_mlx_backend(self.backend),
        )
        self.overfetch_factor = max(2, env_overfetch)
        self.max_candidates = max(self.limit, env_max_candidates)
        self.candidate_limit = min(self.max_candidates, self.limit * self.overfetch_factor)
        self.rerank_top_k = max(0, env_rerank_top_k)
        self.media_query_rerank_top_k = max(
            0,
            min(self.rerank_top_k, env_media_query_rerank_top_k),
        )
        self.media_result_rerank_top_k = max(
            0,
            min(self.rerank_top_k, env_media_result_rerank_top_k),
        )
        self.media_rerank_min_rrf_margin = max(0.0, env_media_rerank_min_rrf_margin)
        self.cache: EmbeddingCache = cache if cache is not None else EmbeddingCache()
        self.intent = intent
        self.expand = expand
        self.enable_media_query_probe = enable_media_query_probe

    def _vector_results_to_hybrid(self, results: List[SearchResult]) -> List[HybridResult]:
        """Convert raw vector results into HybridResult objects."""
        hybrid_results: List[HybridResult] = []
        for rank, result in enumerate(results, start=1):
            hybrid_results.append(HybridResult(
                filepath=result.filepath,
                display_path=result.display_path,
                title=result.title,
                context=result.context,
                hash=result.hash,
                docid=result.docid,
                collection=result.collection,
                modified_at=result.modified_at,
                body_length=result.body_length,
                body=result.body,
                score=result.score,
                rrf_rank=rank,
                rerank_score=0.5,
                source=result.source,
                content_type=getattr(result, 'content_type', 'text'),
                memory_id=getattr(result, "memory_id", None),
                memory_role=getattr(result, "memory_role", "root"),
                memory_root_path=getattr(result, "memory_root_path", None),
                tags=getattr(result, "tags", None),
            ))
        return hybrid_results

    def _bm25_probe(self, query: str) -> List[SearchResult]:
        """Run initial BM25 probe."""
        t0 = time.perf_counter()
        results = self.storage.search_fts(
            query,
            limit=self.fts_probe_limit,
            collection=self.collection,
            content_type=self.content_type,
            user_id=self.user_id,
            session_id=self.session_id,
            project_id=self.project_id,
            profile=self.profile,
        )
        _log_stage_metrics("bm25", results, start_time=t0)
        return results

    def _vector_search(self, query: str) -> List[SearchResult]:
        """Run vector search."""
        t0 = time.perf_counter()
        cache_key = self.cache.make_key("text", query)
        vector = self.cache.get(cache_key)
        if vector is not None:
            logger.debug("Embedding cache hit for text query (key=%s…)", cache_key[:8])
        else:
            vector = self.backend.embed_text(query)
            self.cache.put(cache_key, vector)
        results = self.storage.search_vec(
            vector.tolist() if hasattr(vector, 'tolist') else list(vector),
            limit=self.fts_probe_limit,
            collection=self.collection,
            content_type=self.content_type,
            user_id=self.user_id,
            session_id=self.session_id,
            project_id=self.project_id,
            profile=self.profile,
        )
        _log_stage_metrics("vector", results, start_time=t0)
        return results

    def _embed_image_cached(self, image_path: str):
        """Embed an image with cache lookup keyed by content hash."""
        # Key by file content hash so cache survives path renames but busts on edits.
        try:
            h = sha256()
            with open(image_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            file_hash = h.hexdigest()
        except OSError:
            file_hash = image_path  # fall back to path string if unreadable

        cache_key = self.cache.make_key("image", file_hash)
        vector = self.cache.get(cache_key)
        if vector is not None:
            logger.debug("Embedding cache hit for image (key=%s…)", cache_key[:8])
            return vector

        vector = self.backend.embed_image(image_path)
        self.cache.put(cache_key, vector)
        return vector

    def _caption_image_query(self, image_path: str) -> str:
        """Caption an image query for BM25 probing when the backend supports it."""
        try:
            caption = self.backend.caption_image(image_path)
        except (NotImplementedError, Exception) as exc:
            logger.debug("image_query_caption failed: %s", exc)
            return ""
        return (caption or "").strip()

    def _caption_video_query(self, video_path: str) -> str:
        """Build safe BM25 probe text from a video transcript and first-frame caption."""
        query_parts: List[str] = []
        try:
            from .video import load_transcript_segments

            segments, _ = load_transcript_segments(video_path, video_path)
            transcript_text = " ".join(
                segment.text.strip()
                for segment in segments
                if isinstance(segment.text, str) and segment.text.strip()
            ).strip()
            if transcript_text:
                query_parts.append(transcript_text)
        except Exception as exc:
            logger.debug("video query transcript probe failed: %s", exc)

        try:
            import subprocess
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                frame_path = os.path.join(tmpdir, "frame.jpg")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-vf",
                        "select=eq(n\\,0)",
                        "-vframes",
                        "1",
                        frame_path,
                    ],
                    capture_output=True,
                    timeout=10,
                )
                if os.path.exists(frame_path):
                    frame_caption = self._caption_image_query(frame_path)
                    if frame_caption:
                        query_parts.append(frame_caption)
        except Exception as exc:
            logger.warning("video query caption failed (is ffmpeg installed?): %s", exc)

        return " ".join(dict.fromkeys(query_parts))[:2000].strip()

    def _query_media_probe(
        self,
        *,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> tuple[str, List[SearchResult]]:
        """Generate a text probe from query media and run BM25 when possible."""
        if not self.enable_media_query_probe:
            return "", []

        query_text = ""
        if image_path:
            query_text = self._caption_image_query(image_path)
        elif video_path:
            query_text = self._caption_video_query(video_path)

        if not query_text:
            return "", []

        logger.debug(
            "media_bm25_probe type=%s caption=%s",
            "image" if image_path else "video",
            query_text[:100],
        )
        return query_text, self._bm25_probe(query_text)

    def _add_text_expansion_branches(
        self,
        all_results: Dict[str, List[SearchResult]],
        query_text: str,
    ) -> None:
        """Add BM25/vector branches for generated text expansions."""
        if not self.expand or not query_text.strip():
            return

        query_variants = expand_query(query_text, self.backend, expand=True)
        if len(query_variants) <= 1:
            return

        for i, variant in enumerate(query_variants[1:], start=1):
            try:
                variant_results = self._run_parallel_searches(variant)
                for key, results in variant_results.items():
                    all_results[f"{key}_exp{i}"] = results
                all_results[f"original_fts_exp{i}"] = self._bm25_probe(variant)
            except Exception as exc:
                logger.warning(
                    "Skipping failed query expansion branch variant_index=%d variant=%r error=%s",
                    i,
                    variant[:120],
                    exc,
                )

    def search_image(self, image_path: str) -> List[HybridResult]:
        """Run image-query search through hybrid pipeline (RRF + optional rerank)."""
        # Image query always contributes vector candidates.
        all_results: Dict[str, List[SearchResult]] = {}
        query_image_path_for_rerank: Optional[str] = image_path
        try:
            vector = self._embed_image_cached(image_path)
            all_results["original_vec"] = self.storage.search_vec(
                vector.tolist() if hasattr(vector, 'tolist') else list(vector),
                limit=self.fts_probe_limit,
                collection=self.collection,
                content_type=self.content_type,
                user_id=self.user_id,
                session_id=self.session_id,
                project_id=self.project_id,
                profile=self.profile,
            )
        except Exception as exc:
            logger.warning("image query embedding failed for %s: %s", image_path, exc)
            query_image_path_for_rerank = None

        query_text, bm25_results = self._query_media_probe(image_path=image_path)
        if bm25_results:
            all_results["original_fts"] = bm25_results
        self._add_text_expansion_branches(all_results, query_text)

        if not all_results:
            return []

        candidates, rrf_audit_info = self._reciprocal_rank_fusion(all_results)
        rerank_scores, reranker_path = self._rerank_candidates(
            candidates, query=query_text, query_image_path=query_image_path_for_rerank
        )
        return self._blend_scores(candidates, rerank_scores, rrf_audit_info, reranker_path)

    def search_video(self, video_path: str) -> List[HybridResult]:
        """Run video-query search through hybrid pipeline (RRF + optional rerank).

        Raises:
            NotImplementedError: If the backend does not support native video embedding.
        """
        all_results: Dict[str, List[SearchResult]] = {}
        query_video_path_for_rerank: Optional[str] = None
        embed_video = getattr(self.backend, "embed_video", None)
        if self.enable_raw_video_query_embedding:
            if not callable(embed_video):
                raise NotImplementedError(
                    f"Backend {type(self.backend).__name__} does not support raw video queries. "
                    "Install a backend with video support (e.g. recallforge[mlx] or recallforge[torch])."
                )
            query_video_path_for_rerank = video_path
            try:
                vector = embed_video(video_path)
                all_results["original_vec"] = self.storage.search_vec(
                    vector.tolist() if hasattr(vector, 'tolist') else list(vector),
                    limit=self.fts_probe_limit,
                    collection=self.collection,
                    content_type=self.content_type,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    project_id=self.project_id,
                    profile=self.profile,
                )
            except Exception as exc:
                logger.warning("video query embedding failed for %s: %s", video_path, exc)
                query_video_path_for_rerank = None
        else:
            logger.info(
                "raw video query embedding disabled for backend=%s; using caption/transcript-first retrieval",
                type(self.backend).__name__,
            )

        query_text, bm25_results = self._query_media_probe(video_path=video_path)
        if bm25_results:
            all_results["original_fts"] = bm25_results
        self._add_text_expansion_branches(all_results, query_text)

        if not all_results:
            return []

        candidates, rrf_audit_info = self._reciprocal_rank_fusion(all_results)
        rerank_scores, reranker_path = self._rerank_candidates(
            candidates, query=query_text, query_video_path=query_video_path_for_rerank
        )
        return self._blend_scores(candidates, rerank_scores, rrf_audit_info, reranker_path)

    def _search_vector(self, vector) -> List[HybridResult]:
        """Run a direct vector search and convert to hybrid-style results."""
        results = self.storage.search_vec(
            vector.tolist() if hasattr(vector, 'tolist') else list(vector),
            limit=self.limit,
            collection=self.collection,
            content_type=self.content_type,
            user_id=self.user_id,
            session_id=self.session_id,
            project_id=self.project_id,
            profile=self.profile,
        )
        return self._vector_results_to_hybrid(results)

    def _run_parallel_searches(
        self,
        query: str,
    ) -> Dict[str, List[SearchResult]]:
        """Run all searches in parallel."""
        all_results: Dict[str, List[SearchResult]] = {}

        # Build search tasks
        search_tasks: List[tuple] = []

        # NOTE: original BM25 is run separately via _bm25_probe() in search()
        # and injected into all_results after this method returns.
        # Do NOT duplicate it here.

        # Original vector - use cache like _vector_search() does
        try:
            cache_key = self.cache.make_key("text", query)
            vector = self.cache.get(cache_key)
            if vector is not None:
                logger.debug("Embedding cache hit for text query in parallel search (key=%s…)", cache_key[:8])
            else:
                vector = self.backend.embed_text(query)
                self.cache.put(cache_key, vector)
            search_tasks.append((
                'original_vec',
                lambda v=vector: self.storage.search_vec(
                    v.tolist() if hasattr(v, 'tolist') else list(v),
                    limit=self.fts_probe_limit,
                    collection=self.collection,
                    content_type=self.content_type,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    project_id=self.project_id,
                    profile=self.profile,
                ),
                ()
            ))
        except Exception as e:
            logger.error("Embedding failed: %s", e)

        # Execute all searches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(func, *args): key
                for key, func, args in search_tasks
            }

            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result(timeout=30)
                    all_results[key] = result
                except Exception as e:
                    logger.error("Search task %s failed: %s", key, e)
                    all_results[key] = []

        return all_results

    def _default_rrf_parent_weight(self, source_name: str) -> float:
        """Return the default RRF weight for a primary source."""
        if self.content_type == "text":
            return {"original_vec": 2.5, "original_fts": 1.5}.get(source_name, 2.0)
        if self.content_type in ("image", "video"):
            return {"original_vec": 1.5, "original_fts": 2.5}.get(source_name, 2.0)
        return 2.0
    
    def _reciprocal_rank_fusion(
        self,
        all_results: Dict[str, List[SearchResult]],
    ) -> tuple[List[SearchResult], Dict[str, Dict[str, Any]]]:
        """Apply RRF (Reciprocal Rank Fusion) to combine results.
        
        Returns:
            Tuple of (fused_results, rrf_audit_info) where rrf_audit_info maps
            filepath to {sources: {name: rank}, rrf_score: float, media_compensation: bool}.
        """
        combined: Dict[str, Dict[str, Any]] = {}
        k = self.rrf_k

        # Determine weights based on intent
        weights: Dict[str, float] = {}
        list_names = list(all_results.keys())

        if self.intent and self.intent in INTENT_WEIGHTS:
            # Apply intent-specific weights. Expansion sources (e.g.
            # original_vec_exp1) inherit the weight of their parent source
            # so that intent steering remains stable regardless of expansion.
            intent_weights = INTENT_WEIGHTS[self.intent]
            for name in list_names:
                if name in intent_weights:
                    weights[name] = intent_weights[name]
                else:
                    # Derive parent name by stripping _expN suffix
                    parent = re.sub(r"_exp\d+$", "", name)
                    parent_weight = intent_weights.get(parent, 1.0)
                    # Expansion variants get reduced weight (0.5x parent)
                    # to avoid diluting the primary signal
                    weights[name] = parent_weight * 0.5
        else:
            # Default weights are result-modality aware for category-specific runs:
            # text results lean vector, image/video results lean BM25 captions.
            for name in list_names:
                if re.search(r"_exp\d+$", name):
                    parent = re.sub(r"_exp\d+$", "", name)
                    parent_weight = weights.get(parent, self._default_rrf_parent_weight(parent))
                    weights[name] = parent_weight * 0.5
                else:
                    weights[name] = self._default_rrf_parent_weight(name)

        # Log blend weights applied
        for name, weight in weights.items():
            logger.debug("blend_weight source=%s weight=%.2f intent=%s", name, weight, self.intent or "default")
        
        # Track which lists exist so we can compensate media candidates
        # that structurally cannot appear in text-based lists (BM25).
        has_bm25 = any("bm25" in name.lower() or "fts" in name.lower() for name in list_names)

        for list_name, results in all_results.items():
            weight = weights.get(list_name, 1.0)
            for rank, result in enumerate(results):
                filepath = result.filepath
                if filepath not in combined:
                    combined[filepath] = {
                        'result': result,
                        'rrf_score': 0.0,
                        'sources': {},  # source_name -> rank
                        'best_rank': float('inf'),
                        'content_type': result.content_type,
                        'media_compensation': False,
                    }
                
                combined[filepath]['rrf_score'] += weight / (k + rank + 1)
                combined[filepath]['sources'][list_name] = rank  # Track rank per source
                combined[filepath]['best_rank'] = min(
                    combined[filepath]['best_rank'],
                    rank
                )
        
        # Compensate media candidates for structural BM25 absence.
        # Only apply boost to media that did NOT appear in any BM25/FTS list
        # (i.e., has no text_body/caption). Captioned media CAN appear in BM25
        # and should not get an unconditional modality boost.
        should_apply_media_compensation = self.content_type in ("image", "video")
        if has_bm25 and should_apply_media_compensation:
            total_weight = sum(weights.values())
            bm25_weight = sum(
                w for name, w in weights.items()
                if "bm25" in name.lower() or "fts" in name.lower()
            )
            non_bm25_weight = total_weight - bm25_weight
            bm25_list_names = {
                name for name in list_names
                if "bm25" in name.lower() or "fts" in name.lower()
            }
            if non_bm25_weight > 0 and bm25_weight > 0:
                media_boost = total_weight / non_bm25_weight
                for filepath, data in combined.items():
                    if data['content_type'] in ("image", "video"):
                        # Only boost if this candidate was NOT found by BM25
                        in_bm25 = bool(data['sources'].keys() & bm25_list_names)
                        if not in_bm25:
                            data['rrf_score'] *= media_boost
                            data['media_compensation'] = True

        # Convert to list and sort
        final_results = []
        for filepath, data in combined.items():
            result = data['result']
            result.score = data['rrf_score']
            result.source = '+'.join(sorted(data['sources'].keys()))
            final_results.append(result)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Build audit info for each result
        rrf_audit_info: Dict[str, Dict[str, Any]] = {}
        for filepath, data in combined.items():
            rrf_audit_info[filepath] = {
                'sources': data['sources'],  # {source_name: rank}
                'rrf_score': data['rrf_score'],
                'media_compensation': data['media_compensation'],
                'weights': dict(weights),  # Capture the weights used
            }
        
        return final_results[:self.candidate_limit], rrf_audit_info
    
    def _select_best_chunk(self, result: SearchResult) -> Dict[str, Any]:
        """Select the best chunk from a document for reranking.
        
        For image results, includes the image_path so the reranker
        can use its vision-language capabilities instead of scoring
        an empty text string.
        """
        chunk: Dict[str, Any] = {
            'text': result.body or result.context or "",
            'filepath': result.filepath,
            'content_type': result.content_type,
            'hash': result.hash,
        }
        # For image/video content, resolve the actual file path for VL reranking
        if result.content_type in ("image", "video") and result.filepath:
            # filepath may be recallforge://collection/path or absolute
            raw = result.filepath
            if raw.startswith("recallforge://"):
                # Strip scheme + collection prefix
                parts = raw.split("/", 3)
                raw = "/" + parts[-1] if len(parts) > 3 else raw
            from pathlib import Path
            p = Path(raw)
            if p.is_file():
                if result.content_type == "image":
                    chunk['image_path'] = str(p)
                elif result.content_type == "video":
                    chunk['video_path'] = str(p)
        return chunk

    def _chunk_has_rerank_content(self, chunk: Dict[str, Any]) -> bool:
        """Return whether a candidate has enough content for expensive reranking."""
        text = str(chunk.get("text") or chunk.get("text_body") or "").strip()
        return bool(text or chunk.get("image_path") or chunk.get("video_path"))

    def _media_rerank_skip_reason(
        self,
        candidates_by_rrf: List[SearchResult],
        *,
        has_query_media: bool,
        has_media_candidates: bool,
    ) -> Optional[str]:
        """Return a media rerank skip reason when the cheap stage is decisive."""
        if not (has_query_media or has_media_candidates):
            return None
        if not self.media_rerank_require_ambiguity:
            return None
        if len(candidates_by_rrf) < 2:
            return None

        top_score = float(getattr(candidates_by_rrf[0], "score", 0.0) or 0.0)
        second_score = float(getattr(candidates_by_rrf[1], "score", 0.0) or 0.0)
        if top_score <= 0:
            return None

        relative_margin = (top_score - second_score) / max(abs(top_score), 1e-9)
        if relative_margin >= self.media_rerank_min_rrf_margin:
            logger.debug(
                "reranker_path path=media_confident_skip reason=rrf_margin "
                "relative_margin=%.4f threshold=%.4f top_score=%.6f second_score=%.6f",
                relative_margin,
                self.media_rerank_min_rrf_margin,
                top_score,
                second_score,
            )
            return "media_confident_skip"
        return None
    
    def _rerank_candidates(
        self,
        candidates: List[SearchResult],
        query: str,
        query_image_path: Optional[str] = None,
        query_video_path: Optional[str] = None,
    ) -> tuple[Dict[str, float], str]:
        """Rerank candidates with cross-encoder.

        Args:
            candidates: Candidate results from RRF fusion.
            query: Text query string. When query_image_path or query_video_path is
                provided this can be empty ("") — the media IS the query.
            query_image_path: Path to query image (for image-query searches).
            query_video_path: Path to query video (for video-query searches).

        Returns:
            Tuple of (scores_dict, scoring_path) where scoring_path is one of:
            'text', 'vl_image', 'vl_video', 'fallback', or 'skipped'.
        """
        t0 = time.perf_counter()
        if not candidates:
            return {}, "skipped"

        if not self.backend.needs_reranker():
            logger.debug("reranker_path path=skipped reason=no_reranker_needed")
            return {c.filepath: 0.5 for c in candidates}, "text"

        candidates_by_rrf = sorted(candidates, key=lambda c: c.score, reverse=True)
        rerank_limit = min(len(candidates_by_rrf), self.rerank_top_k)
        if rerank_limit <= 0:
            return {c.filepath: 0.5 for c in candidates}, "skipped"

        preview_candidates = candidates_by_rrf[:rerank_limit]
        has_query_media = bool(query_image_path or query_video_path)
        has_media_candidates = any(
            getattr(candidate, "content_type", "text") in ("image", "video")
            for candidate in preview_candidates
        )
        if (has_query_media or has_media_candidates) and not self.enable_media_reranking:
            logger.debug(
                "reranker_path path=media_disabled reason=media_reranker_disabled "
                "query_media=%s media_candidates=%s",
                has_query_media,
                has_media_candidates,
            )
            _log_stage_metrics(
                "reranker",
                candidates,
                start_time=t0,
                extra={"path": "media_disabled"},
            )
            return {c.filepath: 0.5 for c in candidates}, "media_disabled"

        if has_query_media:
            rerank_limit = min(rerank_limit, self.media_query_rerank_top_k)
        elif has_media_candidates:
            rerank_limit = min(rerank_limit, self.media_result_rerank_top_k)
        if rerank_limit <= 0:
            return {c.filepath: 0.5 for c in candidates}, "skipped"

        media_skip_reason = self._media_rerank_skip_reason(
            candidates_by_rrf[:rerank_limit],
            has_query_media=has_query_media,
            has_media_candidates=has_media_candidates,
        )
        if media_skip_reason:
            _log_stage_metrics(
                "reranker",
                candidates,
                start_time=t0,
                extra={"path": media_skip_reason, "rerank_top_k": rerank_limit},
            )
            return {c.filepath: 0.5 for c in candidates}, media_skip_reason

        rerank_candidates: List[SearchResult] = []
        chunks: List[Dict[str, Any]] = []
        for candidate in candidates_by_rrf[:rerank_limit]:
            chunk = self._select_best_chunk(candidate)
            if self._chunk_has_rerank_content(chunk):
                rerank_candidates.append(candidate)
                chunks.append(chunk)
            else:
                logger.debug(
                    "reranker_prefilter skip=empty_candidate filepath=%s content_type=%s",
                    getattr(candidate, "filepath", ""),
                    getattr(candidate, "content_type", "unknown"),
                )

        if not rerank_candidates:
            path = "media_no_rerankable_candidates" if (has_query_media or has_media_candidates) else "no_rerankable_candidates"
            logger.debug("reranker_path path=%s reason=empty_prefilter", path)
            _log_stage_metrics(
                "reranker",
                candidates,
                start_time=t0,
                extra={"path": path, "rerank_top_k": rerank_limit},
            )
            return {c.filepath: 0.5 for c in candidates}, path

        effective_query = query or ""

        # Determine expected reranker scoring path for telemetry
        has_doc_image = any(c.get('image_path') for c in chunks)
        has_doc_video = any(c.get('video_path') for c in chunks)
        if query_image_path:
            path = "vl_image"
        elif query_video_path:
            path = "vl_video"
        elif has_doc_image:
            path = "vl_image"
        elif has_doc_video:
            path = "vl_video"
        else:
            path = "text"

        try:
            scores = self.backend.rerank(
                effective_query,
                chunks,
                query_image_path=query_image_path,
                query_video_path=query_video_path,
            )
            logger.debug(
                "reranker_path path=%s candidate_count=%d base_candidate_count=%d rerank_top_k=%d",
                path,
                len(rerank_candidates),
                len(candidates_by_rrf),
                rerank_limit,
            )
            rerank_scores = {c.filepath: 0.5 for c in candidates}
            rerank_scores.update({c.filepath: s for c, s in zip(rerank_candidates, scores)})
            _log_stage_metrics(
                "reranker",
                candidates,
                start_time=t0,
                extra={
                    "path": path,
                    "rerank_top_k": rerank_limit,
                    "reranked_count": len(rerank_candidates),
                },
            )
            return rerank_scores, path
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            logger.debug("reranker_path path=fallback reason=error")
            _log_stage_metrics("reranker", candidates, start_time=t0, extra={"path": "error_fallback"})
            return {c.filepath: 0.5 for c in candidates}, "fallback"
    
    def _blend_scores(
        self,
        rrf_results: List[SearchResult],
        rerank_scores: Dict[str, float],
        rrf_audit_info: Optional[Dict[str, Dict[str, Any]]] = None,
        reranker_path: str = "unknown",
    ) -> List[HybridResult]:
        """Blend RRF scores with reranker scores.
        
        Args:
            rrf_results: RRF-fused results from _reciprocal_rank_fusion
            rerank_scores: Raw reranker scores per filepath
            rrf_audit_info: Audit info from _reciprocal_rank_fusion with sources, weights, etc.
            reranker_path: The scoring path used (text, vl_image, vl_video, etc.)
        """
        t0 = time.perf_counter()
        hybrid_results = []

        if not rrf_results:
            return hybrid_results

        def _normalize(values: Dict[str, float], neutral: float = 0.5) -> Dict[str, float]:
            if not values:
                return {}
            min_v = min(values.values())
            max_v = max(values.values())
            if abs(max_v - min_v) < 1e-9:
                return {k: neutral for k in values}
            return {k: (v - min_v) / (max_v - min_v) for k, v in values.items()}

        rrf_raw = {r.filepath: r.score for r in rrf_results}
        rrf_norm = _normalize(rrf_raw, neutral=0.0)

        # Check if reranking was actually performed (embed mode returns all 0.5)
        # In embed mode, rerank_scores are uniform - use RRF scores directly
        unique_rerank_scores = set(rerank_scores.values())
        has_meaningful_rerank = len(unique_rerank_scores) > 1 or (
            len(unique_rerank_scores) == 1 and 0.5 not in unique_rerank_scores
        )

        # Normalize reranker scores per-modality so text and media scores
        # are on the same 0-1 scale independently.  The VL reranker produces
        # valid discrimination within each modality but the raw score ranges
        # differ (text: 0.03-0.18, VL: 0.07-0.12).  Cross-modality min-max
        # would let text dominate.
        result_types = {r.filepath: r.content_type for r in rrf_results}
        text_rerank = {
            fp: s for fp, s in rerank_scores.items()
            if result_types.get(fp, "text") not in ("image", "video")
        }
        media_rerank = {
            fp: s for fp, s in rerank_scores.items()
            if result_types.get(fp, "text") in ("image", "video")
        }
        text_rerank_norm = _normalize(text_rerank, neutral=0.5)
        media_rerank_norm = _normalize(media_rerank, neutral=0.5)
        # Merge into single dict
        rerank_norm = {}
        rerank_norm.update(text_rerank_norm)
        rerank_norm.update(media_rerank_norm)
        all_text_results = all(
            result_types.get(fp, "text") not in ("image", "video")
            for fp in rrf_raw
        )

        for rank, result in enumerate(rrf_results):
            rrf_rank = rank + 1
            rrf_score = rrf_norm.get(result.filepath, 0.0)
            rerank_score_raw = rerank_scores.get(result.filepath, 0.5)
            rerank_score = rerank_norm.get(result.filepath, 0.5)

            if has_meaningful_rerank:
                # Blend RRF with reranker scores
                # Text-only reranking should be conservative; the VL reranker is
                # stronger as a refinement layer for cross-modal candidates.
                if all_text_results and reranker_path == "text":
                    if rrf_rank <= 3:
                        rrf_weight = 0.90
                    elif rrf_rank <= 10:
                        rrf_weight = 0.85
                    else:
                        rrf_weight = 0.75
                else:
                    if rrf_rank <= 3:
                        rrf_weight = 0.75
                    elif rrf_rank <= 10:
                        rrf_weight = 0.60
                    else:
                        rrf_weight = 0.40

                blended = rrf_weight * rrf_score + (1 - rrf_weight) * rerank_score
            else:
                # Embed mode: use RRF scores directly (rerank_scores are uniform)
                blended = rrf_score
                rrf_weight = 1.0

            # Get audit info from RRF step if available
            audit_info = rrf_audit_info.get(result.filepath, {}) if rrf_audit_info else {}
            rrf_sources = audit_info.get('sources', {})
            media_compensation = audit_info.get('media_compensation', False)
            rrf_raw_score = audit_info.get('rrf_score', result.score)

            # Build audit trail for this result
            audit = SearchAudit(
                filepath=result.filepath,
                content_type=result.content_type or "unknown",
                rrf_sources=rrf_sources,
                rrf_score=rrf_raw_score,
                reranker_raw_score=rerank_score_raw,
                reranker_normalized_score=rerank_score,
                reranker_scoring_path=reranker_path,
                blend_weights={"rrf": rrf_weight, "rerank": 1 - rrf_weight} if has_meaningful_rerank else {"rrf": 1.0},
                media_compensation_applied=media_compensation,
                final_blended_score=blended,
            )

            hybrid_results.append(HybridResult(
                filepath=result.filepath,
                display_path=result.display_path,
                title=result.title,
                context=result.context,
                hash=result.hash,
                docid=result.docid,
                collection=result.collection,
                modified_at=result.modified_at,
                body_length=result.body_length,
                body=result.body,
                score=blended,
                rrf_rank=rrf_rank,
                rerank_score=rerank_score_raw,
                source=result.source,
                content_type=getattr(result, 'content_type', 'text'),
                memory_id=getattr(result, "memory_id", None),
                memory_role=getattr(result, "memory_role", "root"),
                memory_root_path=getattr(result, "memory_root_path", None),
                tags=getattr(result, "tags", None),
                audit=audit,
            ))

        hybrid_results.sort(key=lambda x: x.score, reverse=True)

        # Log final ranking with scores and audit trail
        for rank, hr in enumerate(hybrid_results[:self.limit], start=1):
            audit_info = ""
            if hr.audit:
                audit_info = f" rrf_weight={hr.audit.blend_weights.get('rrf', 0):.2f} rerank_weight={hr.audit.blend_weights.get('rerank', 0):.2f}"
            logger.debug("final_ranking rank=%d score=%.4f rrf_rank=%d rerank_score=%.4f%s filepath=%s",
                         rank, hr.score, hr.rrf_rank, hr.rerank_score, audit_info, hr.filepath)

        rolled_results = self._roll_up_memory_hits(hybrid_results)
        # Log score audit trail for the post-rollup results so the audit
        # payload matches the score the caller receives.
        for hr in rolled_results[:self.limit]:
            if hr.audit:
                logger.debug("score_audit filepath=%s content_type=%s rrf_sources=%s reranker_raw=%.4f reranker_norm=%.4f scoring_path=%s blend_weights=%s memory_rollup_boost=%.4f final_score=%.4f",
                             hr.audit.filepath, hr.audit.content_type, json.dumps(hr.audit.rrf_sources),
                             hr.audit.reranker_raw_score, hr.audit.reranker_normalized_score,
                             hr.audit.reranker_scoring_path, json.dumps(hr.audit.blend_weights),
                             hr.audit.memory_rollup_boost, hr.audit.final_blended_score)

        _log_stage_metrics("blend", rolled_results, start_time=t0)
        return rolled_results[:self.limit]

    def _roll_up_memory_hits(self, results: List[HybridResult]) -> List[HybridResult]:
        """Collapse sibling assets into one representative memory result."""
        if not results:
            return []

        def _merge_tags(items: List[HybridResult]) -> Optional[List[str]]:
            merged: List[str] = []
            seen: set[str] = set()
            for item in items:
                for tag in getattr(item, "tags", None) or []:
                    cleaned = str(tag or "").strip().lower()
                    if not cleaned or cleaned in seen:
                        continue
                    seen.add(cleaned)
                    merged.append(cleaned)
                    if len(merged) >= 8:
                        return merged
            return merged or None

        grouped: Dict[str, List[HybridResult]] = {}
        order: List[str] = []
        for result in results:
            key = result.memory_id or result.filepath
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(result)

        rolled: List[HybridResult] = []
        for key in order:
            group = sorted(grouped[key], key=lambda item: item.score, reverse=True)
            top_hit = group[0]
            root_candidate = next(
                (item for item in group if item.memory_role == "root"),
                None,
            )
            representative = replace(root_candidate or top_hit)
            representative.score = top_hit.score
            representative.rrf_rank = top_hit.rrf_rank
            representative.rerank_score = top_hit.rerank_score
            representative.source = top_hit.source
            representative.audit = copy.deepcopy(top_hit.audit) if top_hit.audit else None
            representative.context = representative.context or top_hit.context
            representative.body = representative.body or top_hit.body
            representative.hash = representative.hash or top_hit.hash
            representative.docid = representative.docid or top_hit.docid
            representative.modified_at = representative.modified_at or top_hit.modified_at
            representative.body_length = representative.body_length or top_hit.body_length

            canonical_path = representative.memory_root_path or top_hit.memory_root_path
            if canonical_path:
                representative.filepath, representative.display_path = _canonicalize_memory_result_paths(
                    representative,
                    canonical_path,
                )
                if not root_candidate:
                    representative.title = os.path.basename(canonical_path)
                representative.memory_root_path = canonical_path
            else:
                representative.memory_root_path = representative.filepath

            representative.memory_role = "root"
            representative.memory_hit_count = len(group)
            representative.tags = _merge_tags(group)
            representative.memory_primary_evidence_path = top_hit.filepath
            representative.memory_supporting_paths = [
                item.filepath
                for item in group
                if item.filepath not in {representative.filepath, top_hit.filepath}
            ][:5]
            memory_rollup_boost = 1.0
            if len(group) > 1:
                memory_rollup_boost += min(0.15, 0.03 * (len(group) - 1))
                representative.score *= memory_rollup_boost
            if representative.audit:
                representative.audit.filepath = representative.filepath
                representative.audit.content_type = representative.content_type
                representative.audit.memory_rollup_boost = memory_rollup_boost
                representative.audit.memory_primary_evidence_path = top_hit.filepath
                representative.audit.memory_supporting_paths = list(
                    representative.memory_supporting_paths or []
                )
                representative.audit.final_blended_score = representative.score
            rolled.append(representative)

        rolled.sort(key=lambda item: item.score, reverse=True)
        return rolled
    
    def search(self, query: str) -> List[HybridResult]:
        """
        Run full hybrid search pipeline.

        Steps:
        1. Optional: Expand query into variants (VL-aware)
        2. BM25 probe (on original query)
        3. Vector search (on all query variants)
        4. RRF fusion across all branches
        5. Cross-encoder reranking (if hybrid mode)
        6. Score blending

        Args:
            query: User's search query

        Returns:
            List of HybridResult objects (top K)
        """
        # Log query routing decision
        logger.debug("query_routing intent=%s collection=%s content_type=%s limit=%d candidate_limit=%d expand=%s",
                     self.intent or "default", self.collection or "none", self.content_type or "none",
                     self.limit, self.candidate_limit, self.expand)

        # Step 1: Expand query if enabled
        query_variants = expand_query(query, self.backend, expand=self.expand)
        logger.debug("query_expansion enabled=%s variants=%d", self.expand, len(query_variants))

        # Step 2: BM25 probe (only on original query to avoid noise)
        fts_results = self._bm25_probe(query)
        logger.debug("candidate_count stage=bm25 count=%d", len(fts_results))

        # Step 3: Parallel searches across all query variants
        all_results: Dict[str, List[SearchResult]] = {}

        # Run vector search for each query variant
        for i, variant in enumerate(query_variants):
            variant_results = self._run_parallel_searches(variant)
            # Prefix with variant index to keep them distinct in RRF
            for key, results in variant_results.items():
                if i == 0:
                    # Original query keeps original naming
                    all_results[key] = results
                else:
                    # Expanded variants get prefixed names
                    all_results[f"{key}_exp{i}"] = results

        # Add original FTS results
        all_results['original_fts'] = fts_results

        # Log candidate counts per modality
        for source, results in all_results.items():
            logger.debug("candidate_count stage=parallel source=%s count=%d", source, len(results))

        # Step 4: RRF fusion across all branches (original + expansions)
        candidates, rrf_audit_info = self._reciprocal_rank_fusion(all_results)
        logger.debug("candidate_count stage=rrf count=%d", len(candidates))

        # Step 5: Reranking (hybrid mode) - use original query for reranking
        rerank_scores, reranker_path = self._rerank_candidates(candidates, query)
        logger.debug("candidate_count stage=reranker count=%d", len(rerank_scores))

        # Step 6: Blend scores
        final_results = self._blend_scores(candidates, rerank_scores, rrf_audit_info, reranker_path)
        logger.debug("final_ranking count=%d top_score=%.4f", len(final_results),
                     final_results[0].score if final_results else 0.0)

        return final_results


def hybrid_query(
    query: str,
    backend: Optional[ModelBackend] = None,
    storage: Optional[StorageBackend] = None,
    limit: int = 10,
    collection: Optional[str] = None,
    content_type: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    profile: Optional[str] = None,
    intent: Optional[str] = None,
    expand: bool = False,
) -> List[HybridResult]:
    """
    Convenience function for hybrid search.

    Args:
        query: Search query
        backend: Model backend (default: get_backend())
        storage: Storage backend (default: get_storage())
        limit: Max results
        collection: Optional collection filter
        content_type: Optional content type filter
        user_id: Optional user namespace filter
        session_id: Optional session namespace filter
        project_id: Optional project namespace filter
        profile: Optional profile namespace filter
        intent: Optional intent for query steering ("exact_lookup", "semantic", "broad")
        expand: Whether to enable VL-aware query expansion (default: False, opt-in)

    Returns:
        List of HybridResult objects
    """
    if backend is None:
        from . import get_backend
        backend = get_backend()

    if storage is None:
        from . import get_storage
        storage = get_storage()

    searcher = HybridSearcher(
        backend=backend,
        storage=storage,
        limit=limit,
        collection=collection,
        content_type=content_type,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        intent=intent,
        expand=expand,
    )

    return searcher.search(query)


def hybrid_query_image(
    image_path: str,
    backend: Optional[ModelBackend] = None,
    storage: Optional[StorageBackend] = None,
    limit: int = 10,
    collection: Optional[str] = None,
    content_type: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    profile: Optional[str] = None,
    intent: Optional[str] = None,
) -> List[HybridResult]:
    """Convenience function for image-query vector search."""
    if backend is None:
        from . import get_backend
        backend = get_backend()
    if storage is None:
        from . import get_storage
        storage = get_storage()

    searcher = HybridSearcher(
        backend=backend,
        storage=storage,
        limit=limit,
        collection=collection,
        content_type=content_type,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        intent=intent,
    )
    return searcher.search_image(image_path)


def hybrid_query_video(
    video_path: str,
    backend: Optional[ModelBackend] = None,
    storage: Optional[StorageBackend] = None,
    limit: int = 10,
    collection: Optional[str] = None,
    content_type: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    profile: Optional[str] = None,
    intent: Optional[str] = None,
) -> List[HybridResult]:
    """Convenience function for raw-video vector search."""
    if backend is None:
        from . import get_backend
        backend = get_backend()
    if storage is None:
        from . import get_storage
        storage = get_storage()

    searcher = HybridSearcher(
        backend=backend,
        storage=storage,
        limit=limit,
        collection=collection,
        content_type=content_type,
        user_id=user_id,
        session_id=session_id,
        project_id=project_id,
        profile=profile,
        intent=intent,
    )
    return searcher.search_video(video_path)


@dataclass
class BatchQuery:
    """A single query in a batch search request."""
    query: str
    mode: Optional[str] = None  # "hybrid" (default), "fts", "vec"
    intent: Optional[str] = None
    weight: float = 1.0  # Optional weight for result merging


@dataclass
class BatchSearchResult:
    """Result from a batch search with per-query provenance."""
    filepath: str
    display_path: str
    title: str
    context: Optional[str]
    hash: str
    docid: str
    collection: str
    modified_at: str
    body_length: int
    body: Optional[str]
    score: float  # Best score across queries
    source: str  # Comma-separated list of query indices that found this result
    query_scores: Dict[int, float]  # Map of query_index -> score
    tags: Optional[List[str]] = None


def search_batch(
    queries: List[Union[str, BatchQuery]],
    backend: ModelBackend,
    storage: StorageBackend,
    limit: int = 10,
    collection: Optional[str] = None,
    content_type: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    profile: Optional[str] = None,
    max_workers: int = 4,
    rrf_k: int = 60,
) -> List[BatchSearchResult]:
    """
    Run multiple search queries in parallel and merge results using RRF.

    Each query runs independently in a thread pool, then results are merged
    using Reciprocal Rank Fusion with best-score-wins deduplication.

    Args:
        queries: List of query strings or BatchQuery objects
        backend: Model backend
        storage: Storage backend
        limit: Maximum final results to return
        collection: Optional collection filter
        content_type: Optional content type filter
        user_id: Optional user namespace filter
        session_id: Optional session namespace filter
        project_id: Optional project namespace filter
        profile: Optional profile namespace filter
        max_workers: Maximum parallel threads
        rrf_k: RRF fusion constant

    Returns:
        List of BatchSearchResult objects, sorted by best merged score
    """
    # Normalize queries to BatchQuery objects
    batch_queries: List[BatchQuery] = []
    for q in queries:
        if isinstance(q, str):
            batch_queries.append(BatchQuery(query=q))
        else:
            batch_queries.append(q)

    if not batch_queries:
        return []

    def _merge_tags(items: List[Any]) -> Optional[List[str]]:
        merged: List[str] = []
        seen: set[str] = set()
        for item in items:
            for tag in getattr(item, "tags", None) or []:
                cleaned = str(tag or "").strip().lower()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                merged.append(cleaned)
        return merged or None

    def run_single_query(q: BatchQuery) -> List[tuple]:
        """Run a single query and return (result, score) tuples."""
        mode = q.mode or "hybrid"
        searcher = HybridSearcher(
            backend=backend,
            storage=storage,
            limit=limit * 3,  # Overfetch for better merging
            collection=collection,
            content_type=content_type,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            intent=q.intent,
        )

        if mode == "fts":
            results = searcher._bm25_probe(q.query)
            # Convert SearchResult to HybridResult-like for consistency
            return [(r, r.score) for r in results]
        elif mode == "vec":
            results = searcher._vector_search(q.query)
            return [(r, r.score) for r in results]
        else:  # hybrid
            results = searcher.search(q.query)
            return [(r, r.score) for r in results]

    # Run all queries in parallel
    all_results: List[List[tuple]] = [[] for _ in batch_queries]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(run_single_query, q): i
            for i, q in enumerate(batch_queries)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_results[idx] = future.result(timeout=30)
            except Exception as e:
                logger.error("Batch query %d failed: %s", idx, e)
                all_results[idx] = []

    # Merge results using RRF with best-score-wins
    merged: Dict[str, Dict[str, Any]] = {}

    for idx, results in enumerate(all_results):
        weight = batch_queries[idx].weight
        for rank, (result, score) in enumerate(results):
            filepath = result.filepath
            if filepath not in merged:
                merged[filepath] = {
                    'result': result,
                    'results': [result],
                    'rrf_score': 0.0,
                    'query_indices': set(),
                    'query_scores': {},
                    'best_score': 0.0,
                }
            else:
                merged[filepath]['results'].append(result)

            # RRF contribution: rank-based, not insertion-order-based
            merged[filepath]['rrf_score'] += weight / (rrf_k + rank + 1)
            merged[filepath]['query_indices'].add(idx)
            merged[filepath]['query_scores'][idx] = score
            merged[filepath]['best_score'] = max(merged[filepath]['best_score'], score)

    # Build final results sorted by RRF score
    final_results: List[BatchSearchResult] = []
    for filepath, data in merged.items():
        result = data['result']
        final_results.append(BatchSearchResult(
            filepath=result.filepath,
            display_path=result.display_path,
            title=result.title,
            context=result.context,
            hash=result.hash,
            docid=result.docid,
            collection=result.collection,
            modified_at=result.modified_at,
            body_length=result.body_length,
            body=result.body,
            score=data['rrf_score'],
            source=','.join(str(i) for i in sorted(data['query_indices'])),
            query_scores=data['query_scores'],
            tags=_merge_tags(data['results']),
        ))

    final_results.sort(key=lambda x: x.score, reverse=True)
    return final_results[:limit]
