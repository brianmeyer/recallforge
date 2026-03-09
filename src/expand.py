"""
expand.py - Query expansion using HuggingFace transformers.

Loads a small instruct model (Qwen/Qwen2.5-0.5B-Instruct) lazily on first call.
Runs on MPS (Apple Silicon) via transformers AutoModelForCausalLM.

Generates 3 types of expansions:
- lex: keywords/synonyms for BM25 search
- vec: semantic reformulations for vector search
- hyde: hypothetical document for embedding-based search

Caches results and detects strong signal (skip expansion if BM25 already found perfect match).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src import db, store


@dataclass
class ExpandedQuery:
    """A single expanded query."""
    original: str
    type: str  # 'lex', 'vec', or 'hyde'
    text: str


# Model globals - lazy loaded on first use
_expand_model = None
_expand_tokenizer = None


def _load_expand_model():
    """Lazy-load the Qwen2.5-0.5B-Instruct model for query expansion."""
    global _expand_model, _expand_tokenizer
    
    if _expand_model is not None:
        return
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Use Qwen2.5-0.5B-Instruct - small enough to coexist with 2B embedder
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Determine device (MPS for Apple Silicon)
    device = "auto"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    
    # Load tokenizer and model
    _expand_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _expand_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager" if device == "mps" else "flash_attention_2",
    ).to(device)
    
    _expand_model.eval()


def _generate_expansions(query: str) -> List[ExpandedQuery]:
    """Generate query expansions using the small instruct model."""
    _load_expand_model()
    
    import torch
    
    # Prompt template for query expansion
    prompt = f"""<|im_start|>system
You are a query expansion assistant. Given a search query, generate 3 variations:
1. A lexical variant (lex) - keywords and synonyms for BM25/Fuzzy matching
2. A vector variant (vec) - semantic rephrasing for vector/ANN search
3. A hypothetical document (hyde) - what a perfect answer would look like

Format as JSON with keys: lex, vec, hyde
Each value is a string. Keep variations concise (under 20 words each).<|im_end|>
<|im_start|>user
Query: {query}<|im_end|>
<|im_start|>assistant
{{"""
    
    inputs = _expand_tokenizer(prompt, return_tensors="pt").to(_expand_model.device)
    
    with torch.no_grad():
        outputs = _expand_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=_expand_tokenizer.eos_token_id,
        )
    
    response = _expand_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response (might be after the assistant prompt)
    json_start = response.find("{")
    if json_start == -1:
        # Try to parse as-is
        json_str = response
    else:
        # Find the closing brace, handling nested structures
        json_str = response[json_start:]
        brace_count = 0
        end_pos = -1
        for i, c in enumerate(json_str):
            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        if end_pos != -1:
            json_str = json_str[:end_pos]
    
    try:
        expansions = json.loads(json_str)
        results = []
        
        for exp_type in ["lex", "vec", "hyde"]:
            text = expansions.get(exp_type, query)
            if text and text != query:
                results.append(ExpandedQuery(
                    original=query,
                    type=exp_type,
                    text=text
                ))
        
        return results
    except (json.JSONDecodeError, TypeError):
        # Fallback: return original query as vector type
        return [ExpandedQuery(original=query, type="vec", text=query)]


def _generate_expansions_fast(query: str) -> List[ExpandedQuery]:
    """Fast fallback expansion without model (rules-based)."""
    results = []
    
    # Lex: Extract keywords, add common variants
    import re
    words = re.findall(r'\w+', query.lower())
    keywords = [w for w in words if len(w) > 2]  # Filter short words
    
    if keywords:
        # Add common technical variants
        lex_variants = [
            " ".join(keywords),
            " ".join(keywords) + " implementation",
            " ".join(keywords) + " example",
            " ".join(keywords) + " tutorial",
        ]
        results.append(ExpandedQuery(
            original=query,
            type="lex",
            text=lex_variants[0]
        ))
    
    # Vec: Semantic rephrasing patterns
    vec_patterns = [
        f"documents about {query}",
        f"information regarding {query}",
        f"{query} guide",
        f"explained: {query}",
    ]
    results.append(ExpandedQuery(
        original=query,
        type="vec",
        text=vec_patterns[0]
    ))
    
    # Hyde: Hypothetical document snippet
    hyde_template = f"""An ideal document about {query} would explain:
- Core concepts and definitions
- Practical applications and examples
- Common pitfalls and best practices
- Related techniques and approaches"""
    results.append(ExpandedQuery(
        original=query,
        type="hyde",
        text=hyde_template
    ))
    
    return results


def strong_signal_detected(fts_results: List) -> bool:
    """
    Detect if BM25 search returned a strong signal (skip expansion).
    
    Strong signal: top score >> second score (score gap > 0.3)
    """
    if len(fts_results) < 2:
        return True  # Single result = strong signal
    
    top_score = fts_results[0].score
    second_score = fts_results[1].score
    
    # Large gap = clear winner, skip expansion
    return (top_score - second_score) > 0.3


def expand_query(query: str, fts_results: Optional[List] = None) -> List[ExpandedQuery]:
    """
    Expand a query into multiple variants for hybrid search.
    
    Args:
        query: Original search query
        fts_results: Optional BM25 results for strong signal detection
    
    Returns:
        List of ExpandedQuery objects (lex, vec, hyde)
    
    Caches results per query to avoid redundant model calls.
    """
    # Check cache first
    cache_key = _get_expand_cache_key(query)
    cached = store.get_cached_result(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            return [ExpandedQuery(**item) for item in data]
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Detect strong signal - skip expansion if BM25 already has clear winner
    if fts_results is not None and strong_signal_detected(fts_results):
        # Still return vector variant for completeness
        results = [ExpandedQuery(original=query, type="vec", text=query)]
    else:
        # Try model-based expansion first
        try:
            results = _generate_expansions(query)
        except Exception as e:
            # Fallback to rule-based
            print(f"Query expansion model failed, using fallback: {e}")
            results = _generate_expansions_fast(query)
    
    # Cache results
    cache_data = [{"original": r.original, "type": r.type, "text": r.text} for r in results]
    store.set_cached_result(cache_key, json.dumps(cache_data))
    
    return results


def _get_expand_cache_key(query: str) -> str:
    """Generate cache key for query expansion."""
    hash_obj = hashlib.sha256()
    hash_obj.update(f"expand:{query}".encode("utf-8"))
    return hash_obj.hexdigest()


def clear_expand_cache() -> None:
    """Clear all query expansion cache entries."""
    if db.cache_table is None:
        return
    
    try:
        # Delete all cache entries starting with 'expand:'
        rows = list(db.cache_table.search().limit(1000).to_list())
        to_delete = [r["key"] for r in rows if r.get("key", "").startswith("expand:")]
        
        for key in to_delete:
            db.cache_table.delete(f"key = '{db.escape_sql(key)}'")
    except Exception as e:
        print(f"Error clearing expand cache: {e}")
