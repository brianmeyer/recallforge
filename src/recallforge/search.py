"""
search.py - Hybrid Search Pipeline for RecallForge.

Combines BM25, vector search, query expansion, and reranking
with tiered modes:
- embed: Embedder only (fastest, lowest memory)
- hybrid: Embedder + Reranker
- full: Embedder + Reranker + Query Expander (best quality)

Uses true concurrency with ThreadPoolExecutor for parallel searches.
"""

import concurrent.futures
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .backends.base import ModelBackend
from .storage.base import StorageBackend, SearchResult


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


class HybridSearcher:
    """
    Full hybrid search pipeline with tiered modes.

    Modes:
    - embed: Embedder only. Vector + FTS, no reranking or expansion.
    - hybrid: + Reranker. Cross-encoder refinement.
    - full: + Query Expander. Lex/vec/hyde expansions.

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
        self.overfetch_factor = max(2, env_overfetch)
        self.max_candidates = max(self.limit, env_max_candidates)
        self.candidate_limit = min(self.max_candidates, self.limit * self.overfetch_factor)

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
            ))
        return hybrid_results

    def _bm25_probe(self, query: str) -> List[SearchResult]:
        """Run initial BM25 probe."""
        return self.storage.search_fts(
            query,
            limit=self.fts_probe_limit,
            collection=self.collection,
            content_type=self.content_type,
            user_id=self.user_id,
            session_id=self.session_id,
            project_id=self.project_id,
            profile=self.profile,
        )

    def _bm25_search(self, query: str) -> List[SearchResult]:
        """Run BM25 search."""
        return self.storage.search_fts(
            query,
            limit=self.fts_probe_limit,
            collection=self.collection,
            content_type=self.content_type,
            user_id=self.user_id,
            session_id=self.session_id,
            project_id=self.project_id,
            profile=self.profile,
        )

    def _vector_search(self, query: str) -> List[SearchResult]:
        """Run vector search."""
        vector = self.backend.embed_text(query)
        return self.storage.search_vec(
            vector.tolist() if hasattr(vector, 'tolist') else list(vector),
            limit=self.fts_probe_limit,
            collection=self.collection,
            content_type=self.content_type,
            user_id=self.user_id,
            session_id=self.session_id,
            project_id=self.project_id,
            profile=self.profile,
        )

    def search_image(self, image_path: str) -> List[HybridResult]:
        """Run image-query search through the vector path."""
        vector = self.backend.embed_image(image_path)
        return self._search_vector(vector)

    def search_video(self, video_path: str) -> List[HybridResult]:
        """Run raw-video query search through the vector path.

        Raises:
            NotImplementedError: If the backend does not support native video embedding.
        """
        embed_video = getattr(self.backend, "embed_video", None)
        if not callable(embed_video):
            raise NotImplementedError(
                f"Backend {type(self.backend).__name__} does not support raw video queries. "
                "Install a backend with video support (e.g. recallforge[mlx] or recallforge[torch])."
            )
        vector = embed_video(video_path)
        return self._search_vector(vector)

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
    
    def _expand_query(self, query: str) -> List[Dict[str, str]]:
        """Expand query into variants (full mode only)."""
        if not self.backend.needs_expander():
            return []
        
        expansions = self.backend.expand_query(query)
        return [
            {"type": "lex", "text": expansions.get("lex", query)},
            {"type": "vec", "text": expansions.get("vec", query)},
            {"type": "hyde", "text": expansions.get("hyde", query)},
        ]
    
    def _strong_signal_detected(self, fts_results: List[SearchResult]) -> bool:
        """Detect if BM25 has strong signal (skip expansion)."""
        if len(fts_results) < 2:
            return False
        top_score = fts_results[0].score
        second_score = fts_results[1].score
        return (top_score - second_score) > 0.3
    
    def _run_parallel_searches(
        self,
        query: str,
        expansions: List[Dict[str, str]],
    ) -> Dict[str, List[SearchResult]]:
        """Run all searches in parallel with batch embedding optimization."""
        all_results: Dict[str, List[SearchResult]] = {}
        
        # Collect all queries for vector search
        all_vec_queries = [query]
        if expansions:
            vec_expansions = [e["text"] for e in expansions if e["type"] == "vec"][:3]
            hyde_expansions = [e["text"] for e in expansions if e["type"] == "hyde"][:3]
            all_vec_queries.extend(vec_expansions)
            all_vec_queries.extend(hyde_expansions)
        
        # Batch embed all vector queries
        vectors_by_query = {}
        if all_vec_queries:
            try:
                all_vectors = self.backend.embed_texts(all_vec_queries)
                vectors_by_query = {q: v for q, v in zip(all_vec_queries, all_vectors)}
            except Exception as e:
                print(f"Batch embedding failed: {e}")
        
        # Build search tasks
        search_tasks: List[tuple] = []
        
        # NOTE: original BM25 is run separately via _bm25_probe() in search()
        # and injected into all_results after this method returns.
        # Do NOT duplicate it here.
        
        # Original vector
        if query in vectors_by_query:
            vec = vectors_by_query[query]
            search_tasks.append((
                'original_vec',
                lambda v=vec: self.storage.search_vec(
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

        # Lexical expansions - BM25 only
        lex_queries = [e["text"] for e in expansions if e["type"] == "lex"] if expansions else []
        for i, lex_q in enumerate(lex_queries[:3]):
            search_tasks.append((f'lex_{i}', self._bm25_search, (lex_q,)))

        # Vector expansions
        for i, vec_q in enumerate([e["text"] for e in expansions if e["type"] == "vec"][:3] if expansions else []):
            if vec_q in vectors_by_query:
                vec = vectors_by_query[vec_q]
                search_tasks.append((
                    f'vec_{i}',
                    lambda v=vec: self.storage.search_vec(
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

        # Hyde expansions
        for i, hyde_q in enumerate([e["text"] for e in expansions if e["type"] == "hyde"][:3] if expansions else []):
            if hyde_q in vectors_by_query:
                vec = vectors_by_query[hyde_q]
                search_tasks.append((
                    f'hyde_{i}',
                    lambda v=vec: self.storage.search_vec(
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
                    print(f"Search task {key} failed: {e}")
                    all_results[key] = []
        
        return all_results
    
    def _reciprocal_rank_fusion(
        self,
        all_results: Dict[str, List[SearchResult]],
    ) -> List[SearchResult]:
        """Apply RRF (Reciprocal Rank Fusion) to combine results."""
        combined: Dict[str, Dict[str, Any]] = {}
        k = self.rrf_k
        
        # Weights: first 2 lists = 2.0, rest = 1.0
        weights = {}
        list_names = list(all_results.keys())
        for i, name in enumerate(list_names):
            weights[name] = 2.0 if i < 2 else 1.0
        
        for list_name, results in all_results.items():
            weight = weights.get(list_name, 1.0)
            for rank, result in enumerate(results):
                filepath = result.filepath
                if filepath not in combined:
                    combined[filepath] = {
                        'result': result,
                        'rrf_score': 0.0,
                        'sources': set(),
                        'best_rank': float('inf'),
                    }
                
                combined[filepath]['rrf_score'] += weight / (k + rank + 1)
                combined[filepath]['sources'].add(list_name)
                combined[filepath]['best_rank'] = min(
                    combined[filepath]['best_rank'],
                    rank
                )
        
        # Convert to list and sort
        final_results = []
        for filepath, data in combined.items():
            result = data['result']
            result.score = data['rrf_score']
            result.source = '+'.join(sorted(data['sources']))
            final_results.append(result)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:self.candidate_limit]
    
    def _select_best_chunk(self, result: SearchResult) -> Dict[str, Any]:
        """Select the best chunk from a document for reranking."""
        return {
            'text': result.body or result.context or "",
            'filepath': result.filepath,
            'content_type': result.content_type,
            'hash': result.hash,
        }
    
    def _rerank_candidates(
        self,
        candidates: List[SearchResult],
        query: str,
    ) -> Dict[str, float]:
        """Rerank candidates with cross-encoder."""
        if not candidates:
            return {}
        
        if not self.backend.needs_reranker():
            return {c.filepath: 0.5 for c in candidates}
        
        chunks = [self._select_best_chunk(c) for c in candidates]
        
        try:
            scores = self.backend.rerank(query, chunks)
            return {c.filepath: s for c, s in zip(candidates, scores)}
        except Exception as e:
            print(f"Reranking failed: {e}")
            return {c.filepath: 0.5 for c in candidates}
    
    def _blend_scores(
        self,
        rrf_results: List[SearchResult],
        rerank_scores: Dict[str, float],
    ) -> List[HybridResult]:
        """Blend RRF scores with reranker scores."""
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
        
        rerank_norm = _normalize(rerank_scores, neutral=0.5)

        for rank, result in enumerate(rrf_results):
            rrf_rank = rank + 1
            rrf_score = rrf_norm.get(result.filepath, 0.0)
            rerank_score_raw = rerank_scores.get(result.filepath, 0.5)
            rerank_score = rerank_norm.get(result.filepath, 0.5)
            
            if has_meaningful_rerank:
                # Blend RRF with reranker scores
                # RRF weight: higher for top results
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
            ))
        
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:self.limit]
    
    def search(self, query: str) -> List[HybridResult]:
        """
        Run full hybrid search pipeline.
        
        Steps:
        1. BM25 probe + query expansion (parallel if full mode)
        2. All searches in parallel
        3. RRF fusion
        4. Cross-encoder reranking (if hybrid/full)
        5. Score blending
        
        Args:
            query: User's search query
        
        Returns:
            List of HybridResult objects (top K)
        """
        # Step 1: BM25 probe
        fts_results = self._bm25_probe(query)
        
        # Step 2: Query expansion (full mode only)
        expansions = []
        if self.backend.needs_expander() and not self._strong_signal_detected(fts_results):
            expansions = self._expand_query(query)
        
        # Step 3: Parallel searches
        all_results = self._run_parallel_searches(query, expansions)
        
        # Add original FTS results
        all_results['original_fts'] = fts_results
        
        # Step 4: RRF fusion
        candidates = self._reciprocal_rank_fusion(all_results)
        
        # Step 5: Reranking (hybrid/full mode)
        rerank_scores = self._rerank_candidates(candidates, query)
        
        # Step 6: Blend scores
        return self._blend_scores(candidates, rerank_scores)


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
    )
    return searcher.search_video(video_path)
