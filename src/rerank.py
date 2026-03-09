"""
rerank.py - Qwen3VLReranker wrapper for relevance scoring.

Uses the official Qwen3VLReranker from the Qwen3-VL-Embedding repo.
Returns float scores in [0.0, 1.0] for (query, document) pairs.

Caches results per (query, doc_hash) to avoid redundant reranking.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np

from src import db, store

# Add Qwen3-VL-Embedding repo to path for imports
QWEN_REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Qwen3-VL-Embedding")
QWEN_SRC_PATH = os.path.join(QWEN_REPO_PATH, "src") if QWEN_REPO_PATH else None

if QWEN_SRC_PATH and os.path.exists(QWEN_SRC_PATH):
    import sys
    if QWEN_SRC_PATH not in sys.path:
        sys.path.insert(0, QWEN_SRC_PATH)

# Import Qwen3VLReranker
Qwen3VLReranker = None
try:
    from models.qwen3_vl_reranker import Qwen3VLReranker as Qwen3VLRerankerClass
    Qwen3VLReranker = Qwen3VLRerankerClass
except ImportError as e:
    print(f"Warning: Could not import Qwen3VLReranker: {e}")


@dataclass
class RerankResult:
    """Result from reranker with relevance score."""
    document: Dict[str, Any]
    score: float  # 0.0 to 1.0
    rank: int = 0


class Qwen3VLRerankerWrapper:
    """
    Wrapper around Qwen3VLReranker with caching.
    
    The reranker takes a query and list of documents, returning relevance scores.
    Caching is done per (query, doc_hash) pair.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Reranker-2B",
        device: str = "auto"
    ):
        """
        Initialize reranker wrapper.
        
        Args:
            model_name: HuggingFace model name for the reranker
            device: 'auto', 'mps', 'cuda', or 'cpu'
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._initialized = False
        
        # Determine device
        import torch
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
    
    def _ensure_model(self):
        """Lazy-load model on first use."""
        if self._model is not None:
            return
        
        if Qwen3VLReranker is None:
            raise ImportError(
                "Qwen3VLReranker not found. "
                "Clone the repo: git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git "
                "and ensure models/qwen3_vl_reranker.py is in the path."
            )
        
        import torch
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = torch.float16  # Reranker works well with float16
        
        # MPS doesn't support bfloat16, force float16
        if self.device == "mps" and torch_dtype == torch.bfloat16:
            torch_dtype = torch.float16
        
        # Initialize model with eager attention on MPS
        attn_implementation = "eager" if self.device == "mps" else "flash_attention_2"
        
        self._model = Qwen3VLReranker(
            model_name_or_path=self.model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        
        # Move to device
        self._model.device = torch.device(self.device)
        self._model.model.to(self._model.device)
        self._model.score_linear.to(self._model.device)
        
        self._initialized = True
    
    def _get_doc_hash(self, document: Dict[str, Any]) -> str:
        """Compute hash for a document."""
        text = document.get("text", "") or document.get("text_body", "") or ""
        content = document.get("content", "") or ""
        
        combined = text + content + str(document.get("filepath", ""))
        hash_obj = hashlib.sha256()
        hash_obj.update(combined.encode("utf-8"))
        return hash_obj.hexdigest()[:16]
    
    def _get_rerank_cache_key(self, query: str, doc_hash: str) -> str:
        """Generate cache key for rerank result."""
        hash_obj = hashlib.sha256()
        hash_obj.update(f"rerank:{query}:{doc_hash}".encode("utf-8"))
        return hash_obj.hexdigest()
    
    def _rerank_single_pair(self, query: str, document: Dict[str, Any]) -> float:
        """
        Rerank a single query-document pair.
        
        Args:
            query: Search query
            document: Document dict with 'text' or 'text_body' field
        
        Returns:
            Relevance score in [0.0, 1.0]
        """
        self._ensure_model()
        
        text = document.get("text", "") or document.get("text_body", "") or ""
        content_type = document.get("content_type", "text")
        
        # Build inputs for Qwen3VLReranker.process()
        inputs = {
            "query": {"text": query},
            "documents": [{"text": text}],
            "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
        }
        
        if content_type == "image":
            image_path = document.get("filepath", "")
            if image_path:
                inputs["query"]["image"] = image_path
                inputs["documents"][0]["image"] = image_path
        
        # Get scores
        scores = self._model.process(inputs)
        
        if scores:
            return scores[0]
        return 0.0
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[RerankResult]:
        """
        Rerank a list of documents for a query.
        
        Args:
            query: Search query
            documents: List of document dicts with 'text', 'filepath', etc.
        
        Returns:
            List of RerankResult objects sorted by score (descending)
        """
        if not documents:
            return []
        
        self._ensure_model()
        
        # Build input for batch processing
        doc_texts = []
        doc_metadatas = []
        
        for doc in documents:
            text = doc.get("text", "") or doc.get("text_body", "") or ""
            doc_texts.append(text)
            doc_metadatas.append(doc)
        
        # Check cache for all documents
        uncached_docs = []
        cached_scores = []
        
        for i, doc in enumerate(documents):
            doc_hash = self._get_doc_hash(doc)
            cache_key = self._get_rerank_cache_key(query, doc_hash)
            cached = store.get_cached_result(cache_key)
            
            if cached:
                try:
                    cached_scores.append((i, float(cached)))
                except (ValueError, TypeError):
                    uncached_docs.append((i, doc))
            else:
                uncached_docs.append((i, doc))
        
        # Rerank uncached documents
        if uncached_docs:
            indices, docs = zip(*uncached_docs)
            doc_texts_to_score = [
                d.get("text", "") or d.get("text_body", "") or "" for d in docs
            ]
            
            # Build inputs
            inputs = {
                "query": {"text": query},
                "documents": [{"text": t} for t in doc_texts_to_score],
                "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
            }
            
            try:
                scores = self._model.process(inputs)
                
                # Cache and collect scores
                for i, idx in enumerate(indices):
                    score = scores[i] if i < len(scores) else 0.0
                    doc = documents[idx]
                    doc_hash = self._get_doc_hash(doc)
                    cache_key = self._get_rerank_cache_key(query, doc_hash)
                    store.set_cached_result(cache_key, str(score))
                    cached_scores.append((idx, score))
            except Exception as e:
                print(f"Rerank error: {e}")
                # Fall back to zero scores
                for idx, doc in uncached_docs:
                    cached_scores.append((idx, 0.0))
        
        # Sort by score descending
        cached_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for rank, (orig_idx, score) in enumerate(cached_scores):
            results.append(RerankResult(
                document=documents[orig_idx],
                score=score,
                rank=rank + 1
            ))
        
        return results


# Singleton wrapper
_reranker: Optional[Qwen3VLRerankerWrapper] = None


def get_reranker() -> Qwen3VLRerankerWrapper:
    """Get or create singleton reranker."""
    global _reranker
    if _reranker is None:
        _reranker = Qwen3VLRerankerWrapper()
    return _reranker


def rerank(query: str, documents: List[Dict[str, Any]]) -> List[RerankResult]:
    """Convenience: rerank documents for a query."""
    return get_reranker().rerank(query, documents)


def clear_rerank_cache() -> None:
    """Clear all rerank cache entries."""
    if db.cache_table is None:
        return
    
    try:
        rows = list(db.cache_table.search().limit(1000).to_list())
        to_delete = [r["key"] for r in rows if r.get("key", "").startswith("rerank:")]
        
        for key in to_delete:
            db.cache_table.delete(f"key = '{db.escape_sql(key)}'")
    except Exception as e:
        print(f"Error clearing rerank cache: {e}")
