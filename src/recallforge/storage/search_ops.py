"""Search operations service for LanceDB storage backend."""

import json
import math
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import SearchResult
from .lancedb_shared import _safe_filter, get_docid, logger, resolve_memory_identity, trace_log

if TYPE_CHECKING:
    from .lancedb_backend import LanceDBBackend


class SearchOps:
    """Service class for search operations."""

    def __init__(self, backend: "LanceDBBackend"):
        self._backend = backend

    def _bm25_fallback(
        self,
        query: str,
        limit: int = 20,
        collection: Optional[str] = None,
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> List[SearchResult]:
        """In-memory BM25 fallback when FTS index fails."""
        try:
            filter_parts = [self._get_ttl_filter()]
            if collection:
                filter_parts.append(_safe_filter("collection", collection))
            if content_type:
                filter_parts.append(_safe_filter("content_type", content_type))
            if user_id is not None:
                filter_parts.append(_safe_filter("user_id", user_id))
            if session_id is not None:
                filter_parts.append(_safe_filter("session_id", session_id))
            if project_id is not None:
                filter_parts.append(_safe_filter("project_id", project_id))
            if profile is not None:
                filter_parts.append(_safe_filter("profile", profile))
            filter_clause = " AND ".join(filter_parts)

            # Keep fallback bounded to avoid OOM on large corpora.
            row_limit = min(self._backend._bm25_fallback_max_rows, max(limit * 50, 200))
            builder = (
                self._backend._embeddings_table.search()
                .where(filter_clause)
                .select(["collection", "file_path", "content_hash", "content_type", "title",
                         "text_body", "embedded_at", "modified_at", "user_id", "session_id",
                         "project_id", "profile", "memory_id", "memory_role",
                         "memory_root_path", "importance", "tags", "expires_at"])
                .limit(row_limit)
            )
            rows = builder.to_pandas()
        except Exception:
            return []

        if rows.empty:
            return []

        query_terms = re.findall(r'\w+', query.lower())
        if not query_terms:
            return []

        N = len(rows)
        avgdl = rows["text_body"].str.len().mean() or 1
        k1, b = 1.5, 0.75

        doc_freqs: Dict[str, int] = defaultdict(int)
        for text in rows["text_body"]:
            seen_terms = set(re.findall(r'\w+', (text or "").lower()))
            for t in seen_terms:
                doc_freqs[t] += 1

        results: List[SearchResult] = []
        for _, row in rows.iterrows():
            text = row.get("text_body") or ""
            text_lower = text.lower()
            doc_len = len(text)
            score = 0.0
            for term in query_terms:
                df_t = doc_freqs.get(term, 0)
                if df_t == 0:
                    continue
                idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
                tf = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                tf_comp = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
                score += idf * tf_comp
            if score > 0:
                row_dict = dict(row)
                results.append(
                    self._make_search_result(
                        row_dict,
                        score * self._memory_policy_multiplier(row_dict),
                        "fts",
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        if results:
            max_s = results[0].score
            for r in results:
                r.score = r.score / max_s if max_s > 0 else 0
        return results[:limit]

    def search_fts(
        self,
        query: str,
        limit: int = 20,
        collection: Optional[str] = None,
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None
    ) -> List[SearchResult]:
        """Full-text search using LanceDB Tantivy."""
        if self._backend._embeddings_table is None:
            return []

        trimmed = query.strip()
        if not trimmed:
            return []

        trace_log("search_fts_start", query=trimmed[:50], limit=limit, collection=collection, content_type=content_type,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        self._backend._fts.ensure_fts_index()

        # Build filter including TTL and namespace fields
        filter_parts = [self._get_ttl_filter()]

        if collection:
            filter_parts.append(_safe_filter("collection", collection))
        if content_type:
            filter_parts.append(_safe_filter("content_type", content_type))
        if user_id is not None:
            filter_parts.append(_safe_filter("user_id", user_id))
        if session_id is not None:
            filter_parts.append(_safe_filter("session_id", session_id))
        if project_id is not None:
            filter_parts.append(_safe_filter("project_id", project_id))
        if profile is not None:
            filter_parts.append(_safe_filter("profile", profile))

        filter_clause = " AND ".join(filter_parts) if filter_parts else None

        # Run FTS search
        try:
            builder = self._backend._embeddings_table.search(trimmed, query_type="fts").limit(limit * 2)
            if filter_clause:
                builder = builder.where(filter_clause)
            results = builder.to_list()
        except Exception as e:
            logger.warning(f"search_fts: FTS index failed, using BM25 fallback: {e}")
            return self._bm25_fallback(trimmed, limit, collection, content_type, user_id, session_id, project_id, profile)

        # Empty FTS results are normal (no matches), not an error.
        # Do NOT run full-table BM25 fallback - only use fallback on true FTS errors.
        if not results:
            trace_log("search_fts_empty", query=trimmed[:50])
            return []

        # Normalize scores
        max_score = max(r.get("_score", 0) for r in results) or 1

        # Dedupe by filepath
        seen: Dict[str, SearchResult] = {}
        for r in results:
            filepath = f"recallforge://{r['collection']}/{r['file_path']}"
            score = (r.get("_score", 0) / max_score) * self._memory_policy_multiplier(r)

            if filepath in seen:
                if score > seen[filepath].score:
                    seen[filepath] = self._make_search_result(r, score, "fts")
            else:
                seen[filepath] = self._make_search_result(r, score, "fts")

        final_results = self._normalize_ranked_scores(
            sorted(seen.values(), key=lambda x: x.score, reverse=True)
        )[:limit]
        trace_log("search_fts_done", count=len(final_results), query=trimmed[:50])
        return final_results

    def search_vec(
        self,
        vector: List[float],
        limit: int = 20,
        collection: Optional[str] = None,
        content_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None
    ) -> List[SearchResult]:
        """Vector similarity search."""
        if self._backend._embeddings_table is None:
            return []

        if not self._backend.has_vectors():
            return []

        trace_log("search_vec_start", limit=limit, collection=collection, content_type=content_type,
                  user_id=user_id, session_id=session_id, project_id=project_id, profile=profile)

        # Build filter including TTL and namespace fields
        filter_parts = [self._get_ttl_filter()]

        if collection:
            filter_parts.append(_safe_filter("collection", collection))
        if content_type:
            filter_parts.append(_safe_filter("content_type", content_type))
        if user_id is not None:
            filter_parts.append(_safe_filter("user_id", user_id))
        if session_id is not None:
            filter_parts.append(_safe_filter("session_id", session_id))
        if project_id is not None:
            filter_parts.append(_safe_filter("project_id", project_id))
        if profile is not None:
            filter_parts.append(_safe_filter("profile", profile))

        filter_clause = " AND ".join(filter_parts) if filter_parts else None

        # Run vector search
        builder = self._backend._embeddings_table.search(vector, query_type="vector").metric("cosine").limit(limit * 2)
        if filter_clause:
            builder = builder.where(filter_clause)

        results = builder.to_list()

        if not results:
            return []

        # Dedupe by filepath
        seen: Dict[str, SearchResult] = {}
        for r in results:
            filepath = f"recallforge://{r['collection']}/{r['file_path']}"
            distance = r.get("_distance", 1.0)
            score = (1.0 - distance / 2.0) * self._memory_policy_multiplier(r)

            if filepath in seen:
                if score > seen[filepath].score:
                    seen[filepath] = self._make_search_result(r, score, "vec")
            else:
                seen[filepath] = self._make_search_result(r, score, "vec")

        final_results = self._normalize_ranked_scores(
            sorted(seen.values(), key=lambda x: x.score, reverse=True)
        )[:limit]
        trace_log("search_vec_done", count=len(final_results))
        return final_results

    def _memory_policy_multiplier(self, row: Dict[str, Any]) -> float:
        """Return a modest ranking multiplier from memory metadata."""
        boost = 1.0

        raw_importance = row.get("importance")
        if raw_importance is not None:
            try:
                importance = max(0.0, min(1.0, float(raw_importance)))
                boost += 0.15 * importance
            except (TypeError, ValueError):
                pass

        embedded_at = row.get("embedded_at")
        try:
            age_ms = int(time.time() * 1000) - int(embedded_at)
        except (TypeError, ValueError):
            age_ms = None
        if age_ms is not None and age_ms >= 0:
            age_days = age_ms / 86_400_000
            if age_days <= 7:
                boost += 0.05
            elif age_days <= 30:
                boost += 0.025

        return boost

    def _normalize_ranked_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Keep policy-boosted scores on the familiar 0..1 scale."""
        if not results:
            return results
        max_score = max((result.score for result in results), default=0.0)
        if max_score <= 0:
            return results
        for result in results:
            result.score = result.score / max_score
        return results

    def _make_search_result(self, row: Dict[str, Any], score: float, source: str) -> SearchResult:
        """Convert LanceDB row to SearchResult.

        PERFORMANCE OPTIMIZATION: Prefer text_body from embeddings row over get_content() lookup.
        - text_body is already available in the row (from embeddings table query)
        - Only call get_content() as fallback when text_body is empty/None
        This avoids N+1 lookups to content table for every search result.
        """
        collection = row.get("collection", "")
        file_path = row.get("file_path", "")
        content_hash = row.get("content_hash", "")
        content_type = row.get("content_type", "text")
        user_id = row.get("user_id")
        session_id = row.get("session_id")
        project_id = row.get("project_id")
        profile = row.get("profile")
        memory_id, memory_role, memory_root_path = resolve_memory_identity(
            collection=collection,
            file_path=file_path,
            memory_id=row.get("memory_id"),
            memory_role=row.get("memory_role"),
            memory_root_path=row.get("memory_root_path"),
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
        )

        # P0 OPTIMIZATION: Prefer text_body (already in row) over get_content() lookup
        body = row.get("text_body") or ""
        if not body:
            # Fallback only when text_body is empty - lazy load for final output
            body = self._backend.get_content(content_hash) or ""

        raw_tags = row.get("tags")
        decoded_tags: Optional[List[str]] = None
        if isinstance(raw_tags, list):
            decoded_tags = [str(tag).strip().lower() for tag in raw_tags if str(tag).strip()]
        elif isinstance(raw_tags, str) and raw_tags.strip():
            try:
                payload = json.loads(raw_tags)
                if isinstance(payload, list):
                    decoded_tags = [str(tag).strip().lower() for tag in payload if str(tag).strip()]
            except json.JSONDecodeError:
                decoded_tags = [
                    part.strip().lower()
                    for part in raw_tags.split(",")
                    if part.strip()
                ]
        if decoded_tags == []:
            decoded_tags = None

        return SearchResult(
            filepath=f"recallforge://{collection}/{file_path}",
            display_path=f"{collection}/{file_path}",
            title=row.get("title", file_path) or "",
            context=None,
            hash=content_hash,
            docid=get_docid(content_hash),
            collection=collection,
            modified_at="",
            body_length=len(body),
            score=score,
            source=source,
            content_type=content_type,
            chunk_pos=row.get("pos", 0) or 0,
            body=body,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            profile=profile,
            memory_id=memory_id,
            memory_role=memory_role,
            memory_root_path=memory_root_path,
            tags=decoded_tags,
            importance=row.get("importance"),
            expires_at=row.get("expires_at"),
        )

    def _get_ttl_filter(self) -> str:
        """Generate filter clause to exclude expired entries.

        Returns SQL WHERE clause fragment that filters out expired entries:
        - expires_at IS NULL (no TTL set)
        - expires_at > current_time (not yet expired)
        """
        now_ms = int(time.time() * 1000)
        # Exclude entries where expires_at is set and less than now
        return f"(expires_at IS NULL OR expires_at > {now_ms})"

    def list_collections(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> List[str]:
        """Return sorted list of unique collection names, with optional namespace filters."""
        if self._backend._embeddings_table is None:
            return []

        try:
            filter_parts: List[str] = []
            if user_id is not None:
                filter_parts.append(_safe_filter("user_id", user_id))
            if session_id is not None:
                filter_parts.append(_safe_filter("session_id", session_id))
            if project_id is not None:
                filter_parts.append(_safe_filter("project_id", project_id))
            if profile is not None:
                filter_parts.append(_safe_filter("profile", profile))

            builder = self._backend._embeddings_table.search().select(["collection"])
            if filter_parts:
                builder = builder.where(" AND ".join(filter_parts))

            rows = builder.limit(100_000).to_list()
            seen: set = set()
            for row in rows:
                val = row.get("collection")
                if val:
                    seen.add(val)
            return sorted(seen)
        except Exception as e:
            logger.warning(f"list_collections: failed: {e}")
            return []

    def list_namespaces(
        self,
        collection: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Return unique namespace combinations (user_id, session_id, project_id, profile)."""
        if self._backend._embeddings_table is None:
            return []

        try:
            filter_parts: List[str] = []
            if collection is not None:
                filter_parts.append(_safe_filter("collection", collection))

            builder = self._backend._embeddings_table.search().select(
                ["user_id", "session_id", "project_id", "profile"]
            )
            if filter_parts:
                builder = builder.where(" AND ".join(filter_parts))

            rows = builder.limit(100_000).to_list()
            seen: set = set()
            for row in rows:
                key = (
                    row.get("user_id") or "",
                    row.get("session_id") or "",
                    row.get("project_id") or "",
                    row.get("profile") or "",
                )
                seen.add(key)

            result = []
            for user_id, session_id, project_id, profile in sorted(seen):
                ns: Dict[str, str] = {}
                if user_id:
                    ns["user_id"] = user_id
                if session_id:
                    ns["session_id"] = session_id
                if project_id:
                    ns["project_id"] = project_id
                if profile:
                    ns["profile"] = profile
                result.append(ns)
            return result
        except Exception as e:
            logger.warning(f"list_namespaces: failed: {e}")
            return []
