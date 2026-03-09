"""
search.py - Full hybrid search pipeline for QMD-VL.

Combines BM25, vector search, query expansion, reranking, and RRF fusion.
Implements the complete multi-stage search pipeline:

1. BM25 probe (20 results) - detect strong signal
2. Query expansion (if no strong signal) - generate lex/vec/hyde variants
3. Parallel searches - run all variants simultaneously
4. RRF fusion - combine result lists
5. Chunk selection - pick best chunk per document for reranking
6. Cross-encoder reranking - refine relevance scores
7. Score blending - combine RRF and reranker scores
8. Return top K results
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math

from src import db
from src.store import SearchResult, search_fts, search_vec
from src.embed import embed_text
from src.expand import expand_query, _get_expand_cache_key, clear_expand_cache
from src.rerank import rerank, RerankResult, get_reranker


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
    Full hybrid search pipeline.
    
    Implements multi-stage retrieval with:
    - BM25 (FTS) for keyword search
    - Vector search (ANN) for semantic search
    - Query expansion for robustness
    - RRF fusion for result combination
    - Cross-encoder reranking for precision
    """
    
    def __init__(
        self,
        limit: int = 10,
        fts_probe_limit: int = 20,
        rrf_k: int = 60,
        collection: Optional[str] = None,
        content_type: Optional[str] = None,
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            limit: Final number of results to return
            fts_probe_limit: How many BM25 results to get for probing
            rrf_k: RRF fusion constant (higher = more weight to lower ranks)
            collection: Optional collection filter
            content_type: Optional content type filter ('text' or 'image')
        """
        self.limit = limit
        self.fts_probe_limit = fts_probe_limit
        self.rrf_k = rrf_k
        self.collection = collection
        self.content_type = content_type
    
    def _bm25_probe(self, query: str) -> List[SearchResult]:
        """Run initial BM25 probe to detect strong signal."""
        return search_fts(
            query,
            limit=self.fts_probe_limit,
            collection=self.collection,
            content_type=self.content_type,
        )
    
    def _vector_search(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Run vector search for a query."""
        vector = embed_text(query)
        return search_vec(
            vector.tolist() if hasattr(vector, 'tolist') else list(vector),
            limit=limit,
            collection=self.collection,
            content_type=self.content_type,
        )
    
    def _vector_search_batch(
        self,
        queries: List[str],
        limit_per_query: int = 20
    ) -> List[Tuple[str, List[SearchResult]]]:
        """Run vector search for multiple queries (e.g., expanded queries)."""
        results = []
        for query in queries:
            vector = embed_text(query)
            vec_results = search_vec(
                vector.tolist() if hasattr(vector, 'tolist') else list(vector),
                limit=limit_per_query,
                collection=self.collection,
                content_type=self.content_type,
            )
            results.append((query, vec_results))
        return results
    
    def _run_parallel_searches(
        self,
        query: str,
        expansions: List[Dict[str, str]],
    ) -> Dict[str, List[SearchResult]]:
        """
        Run all search variants in parallel.
        
        Returns dict mapping search type to results:
        - 'original': Original query (BM25 + Vector)
        - 'lex': Lexical variants (BM25 only)
        - 'vec': Vector variants (Vector only)
        - 'hyde': Hyde variants (Vector only)
        """
        all_results = {}
        
        # Original query searches
        all_results['original_fts'] = search_fts(
            query,
            limit=self.fts_probe_limit,
            collection=self.collection,
            content_type=self.content_type,
        )
        all_results['original_vec'] = self._vector_search(query, limit=self.fts_probe_limit)
        
        # Lexical expansions - BM25 only
        lex_queries = [exp['text'] for exp in expansions if exp['type'] == 'lex']
        for i, lex_q in enumerate(lex_queries[:3]):  # Limit to 3 lex variants
            key = f'lex_{i}'
            all_results[key] = search_fts(
                lex_q,
                limit=self.fts_probe_limit,
                collection=self.collection,
                content_type=self.content_type,
            )
        
        # Vector expansions - vector search
        vec_queries = [exp['text'] for exp in expansions if exp['type'] == 'vec']
        for i, vec_q in enumerate(vec_queries[:3]):
            key = f'vec_{i}'
            all_results[key] = self._vector_search(vec_q, limit=self.fts_probe_limit)
        
        # Hyde expansions - vector search
        hyde_queries = [exp['text'] for exp in expansions if exp['type'] == 'hyde']
        for i, hyde_q in enumerate(hyde_queries[:3]):
            key = f'hyde_{i}'
            all_results[key] = self._vector_search(hyde_q, limit=self.fts_probe_limit)
        
        return all_results
    
    def _reciprocal_rank_fusion(
        self,
        all_results: Dict[str, List[SearchResult]],
    ) -> List[SearchResult]:
        """
        Apply RRF (Reciprocal Rank Fusion) to combine result lists.
        
        Each doc gets score = Σ weight_i / (k + rank_i + 1) from each list it appears in.
        """
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
        return final_results[:self.limit * 2]  # Over-select for reranking
    
    def _select_best_chunk(self, result: SearchResult) -> Dict[str, Any]:
        """
        Select the best chunk from a document for reranking.
        
        For now, return the full result as-is. In a more sophisticated
        implementation, we'd split the document into chunks and pick
        the one with the best query term overlap.
        """
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
        """
        Rerank candidate documents with cross-encoder.
        
        Returns mapping of filepath -> rerank score.
        """
        if not candidates:
            return {}
        
        # Select best chunks for reranking
        chunks = [self._select_best_chunk(r) for r in candidates]
        
        # Rerank
        rerank_results = rerank(query, chunks)
        
        # Build score mapping
        scores = {}
        for rr in rerank_results:
            filepath = rr.document.get('filepath', '')
            if filepath:
                scores[filepath] = rr.score
        
        return scores
    
    def _blend_scores(
        self,
        rrf_results: List[SearchResult],
        rerank_scores: Dict[str, float],
    ) -> List[HybridResult]:
        """
        Blend RRF scores with reranker scores.
        
        Top-ranked RRF results keep more of their RRF signal.
        """
        hybrid_results = []
        
        for rank, result in enumerate(rrf_results):
            rrf_rank = rank + 1
            
            # RRF weight: high for top results, decaying lower
            if rrf_rank <= 3:
                rrf_weight = 0.75
            elif rrf_rank <= 10:
                rrf_weight = 0.60
            else:
                rrf_weight = 0.40
            
            # RRF score (already normalized by RRF)
            rrf_score = result.score
            
            # Reranker score (0-1)
            rerank_score = rerank_scores.get(result.filepath, 0.0)
            
            # Blended score
            blended = rrf_weight * rrf_score + (1 - rrf_weight) * rerank_score
            
            hybrid_results.append(HybridResult(
                filepath=result.filepath,
                display_path=result.display_path,
                title=result.title,
                context=result.context,
                hash=result.hash,
                docid=result.docid,
                collection=result.collection,
                modified_at="",
                body_length=result.body_length,
                body=result.body,
                score=blended,
                rrf_rank=rrf_rank,
                rerank_score=rerank_score,
                source=result.source,
            ))
        
        # Sort by blended score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:self.limit]
    
    def search(self, query: str) -> List[HybridResult]:
        """
        Run full hybrid search pipeline.
        
        1. BM25 probe for strong signal detection
        2. Query expansion (if needed)
        3. Parallel searches (BM25 + Vector for all variants)
        4. RRF fusion
        5. Cross-encoder reranking
        6. Score blending
        
        Args:
            query: User's search query
        
        Returns:
            List of HybridResult objects (top K)
        """
        # Step 1: BM25 probe
        fts_results = self._bm25_probe(query)
        
        # Check for strong signal (skip expansion if clear winner)
        if len(fts_results) >= 2:
            top_score = fts_results[0].score
            second_score = fts_results[1].score
            if (top_score - second_score) > 0.3:
                # Strong signal - use original query only
                fts_results = self._run_parallel_searches(query, [])
                candidates = self._reciprocal_rank_fusion(fts_results)
                
                # Rerank
                rerank_scores = self._rerank_candidates(candidates, query)
                
                # Blend
                return self._blend_scores(candidates, rerank_scores)
        
        # Step 2: Query expansion (if no strong signal)
        expansions = expand_query(query, fts_results)
        
        # Convert to dict format for search
        expansion_dicts = [
            {'type': e.type, 'text': e.text} for e in expansions
        ]
        
        # Step 3: Parallel searches
        all_results = self._run_parallel_searches(query, expansion_dicts)
        
        # Step 4: RRF fusion
        candidates = self._reciprocal_rank_fusion(all_results)
        
        # Step 5: Rerank candidates
        rerank_scores = self._rerank_candidates(candidates, query)
        
        # Step 6: Blend scores
        return self._blend_scores(candidates, rerank_scores)


# Convenience function for single call
def hybrid_query(
    query: str,
    limit: int = 10,
    collection: Optional[str] = None,
    content_type: Optional[str] = None,
) -> List[HybridResult]:
    """Convenience: run full hybrid search."""
    searcher = HybridSearcher(
        limit=limit,
        collection=collection,
        content_type=content_type,
    )
    return searcher.search(query)


def clear_expand_and_rerank_caches() -> None:
    """Clear both expand and rerank caches."""
    clear_expand_cache()
    get_reranker()  # Initialize to ensure we have the wrapper
    # Clear rerank cache via the wrapper's method (need to access internals)
    if db.cache_table is None:
        return
    try:
        rows = list(db.cache_table.search().limit(1000).to_list())
        to_delete = [r["key"] for r in rows if r.get("key", "").startswith("expand:") or r.get("key", "").startswith("rerank:")]
        for key in to_delete:
            db.cache_table.delete(f"key = '{db.escape_sql(key)}'")
    except Exception as e:
        print(f"Error clearing caches: {e}")
