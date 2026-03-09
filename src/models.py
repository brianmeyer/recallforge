"""
models.py - Model Registry for QMD-VL.

Singleton holding ALL THREE models resident in memory simultaneously:
- Qwen3-VL-Embedding-2B (~4GB fp16) — embedder
- Qwen3-VL-Reranker-2B (~4GB fp16) — reranker
- Qwen2.5-0.5B-Instruct (~1GB fp16) — query expander

Total ~9GB. Mac mini handles this with headroom. NO swapping/unloading.

All modules (embed.py, rerank.py, expand.py) should import from models.py.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

# =============================================================================
# Model Registry
# =============================================================================

@dataclass
class ModelStatus:
    """Status of a loaded model."""
    name: str
    loaded: bool
    memory_mb: float = 0.0


class ModelRegistry:
    """
    Singleton registry holding all models resident in memory.
    
    Models load lazily on first use, then stay resident forever.
    warm_up() loads all three at once for predictable latency.
    
    Total memory: ~9GB for all three models (4GB + 4GB + 1GB).
    """
    
    _instance: Optional['ModelRegistry'] = None
    
    def __new__(cls) -> 'ModelRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Lazy-loaded model instances
        self._embedder = None
        self._reranker = None
        self._expander = None
        
        # Device and dtype
        self._device = None
        self._dtype = None
        
        # Apply transformers compatibility patch
        self._apply_transformers_patch()
        
        # Add Qwen3-VL-Embedding repo to path
        self._add_qwen_repo_to_path()
    
    def _apply_transformers_patch(self):
        """Apply compatibility patch for transformers 5.x."""
        try:
            import transformers.utils.generic as _generic
            if not hasattr(_generic, 'check_model_inputs'):
                _generic.check_model_inputs = lambda *args, **kwargs: None
        except Exception:
            pass
        
        try:
            import transformers.utils
            if not hasattr(transformers.utils, 'check_model_inputs'):
                transformers.utils.check_model_inputs = lambda *args, **kwargs: None
        except Exception:
            pass
    
    def _add_qwen_repo_to_path(self):
        """Add Qwen3-VL-Embedding repo to sys.path."""
        qwen_repo = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Qwen3-VL-Embedding")
        qwen_src = os.path.join(qwen_repo, "src")
        
        if os.path.exists(qwen_src) and qwen_src not in sys.path:
            sys.path.insert(0, qwen_src)
    
    def _get_device(self) -> str:
        """Determine best device for inference."""
        if self._device is not None:
            return self._device
        
        import torch
        
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        
        return self._device
    
    def _get_dtype(self):
        """Get torch dtype for model loading."""
        import torch
        
        if self._dtype is not None:
            return self._dtype
        
        device = self._get_device()
        
        # MPS doesn't support bfloat16 well
        if device == "mps":
            self._dtype = torch.float16
        else:
            self._dtype = torch.float16  # Consistent across devices
        
        return self._dtype
    
    # =========================================================================
    # Embedder (Qwen3-VL-Embedding-2B, ~4GB)
    # =========================================================================
    
    def _load_embedder(self):
        """Lazy-load the embedder model."""
        if self._embedder is not None:
            return
        
        import torch
        from models.qwen3_vl_embedding import Qwen3VLEmbedder
        
        device = self._get_device()
        dtype = self._get_dtype()
        
        # MPS doesn't support flash attention
        attn = "eager" if device == "mps" else "flash_attention_2"
        
        self._embedder = Qwen3VLEmbedder(
            model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
            torch_dtype=dtype,
            attn_implementation=attn,
        )
        
        print(f"[ModelRegistry] Loaded embedder on {device} with {dtype}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text strings in a SINGLE MPS call.
        
        This is the key optimization: batch all text embeddings together
        for much better throughput than sequential calls.
        """
        self._load_embedder()
        
        # Format inputs for Qwen3VLEmbedder
        inputs = [
            {
                "text": text,
                "instruction": "Retrieve documents relevant to the user's query.",
            }
            for text in texts
        ]
        
        # Get embeddings
        embeddings = self._embedder.process(inputs)
        
        # Convert to numpy float32
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
        else:
            import torch
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy().astype(np.float32)
            return np.array(embeddings, dtype=np.float32)
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed a single image."""
        return self.embed_images([image_path])[0]
    
    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """Embed multiple images in a single call."""
        self._load_embedder()
        
        inputs = [
            {
                "image": path,
                "instruction": "Retrieve content relevant to the image.",
            }
            for path in image_paths
        ]
        
        embeddings = self._embedder.process(inputs)
        
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
        else:
            import torch
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy().astype(np.float32)
            return np.array(embeddings, dtype=np.float32)
    
    def embed_mixed(self, items: List[Dict[str, Any]]) -> np.ndarray:
        """Embed mixed text/image content."""
        self._load_embedder()
        
        inputs = []
        for item in items:
            inp = {"instruction": "Retrieve content relevant to the query."}
            if "text" in item:
                inp["text"] = item["text"]
            if "image" in item:
                inp["image"] = item["image"]
            inputs.append(inp)
        
        embeddings = self._embedder.process(inputs)
        
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
        else:
            import torch
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy().astype(np.float32)
            return np.array(embeddings, dtype=np.float32)
    
    # =========================================================================
    # Reranker (Qwen3-VL-Reranker-2B, ~4GB)
    # =========================================================================
    
    def _load_reranker(self):
        """Lazy-load the reranker model."""
        if self._reranker is not None:
            return
        
        import torch
        from models.qwen3_vl_reranker import Qwen3VLReranker
        
        device = self._get_device()
        dtype = self._get_dtype()
        
        attn = "eager" if device == "mps" else "flash_attention_2"
        
        self._reranker = Qwen3VLReranker(
            model_name_or_path="Qwen/Qwen3-VL-Reranker-2B",
            torch_dtype=dtype,
            attn_implementation=attn,
        )
        
        # Move to device
        self._reranker.device = torch.device(device)
        self._reranker.model.to(self._reranker.device)
        self._reranker.score_linear.to(self._reranker.device)
        
        print(f"[ModelRegistry] Loaded reranker on {device} with {dtype}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """
        Rerank documents for a query.
        
        Returns list of relevance scores (0.0 to 1.0) in same order as documents.
        """
        if not documents:
            return []
        
        self._load_reranker()
        
        # Build inputs
        doc_texts = [
            d.get("text", "") or d.get("text_body", "") or ""
            for d in documents
        ]
        
        inputs = {
            "query": {"text": query},
            "documents": [{"text": t} for t in doc_texts],
            "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
        }
        
        try:
            scores = self._reranker.process(inputs)
            return list(scores) if scores else [0.0] * len(documents)
        except Exception as e:
            print(f"[ModelRegistry] Rerank error: {e}")
            return [0.5] * len(documents)  # Neutral fallback
    
    # =========================================================================
    # Query Expander (Qwen2.5-0.5B-Instruct, ~1GB)
    # =========================================================================
    
    def _load_expander(self):
        """Lazy-load the query expansion model."""
        if self._expander is not None:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        device = self._get_device()
        dtype = self._get_dtype()
        
        attn = "eager" if device == "mps" else "flash_attention_2"
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        self._expander = {
            "tokenizer": AutoTokenizer.from_pretrained(model_name),
            "model": AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                attn_implementation=attn,
            ).to(device),
        }
        
        self._expander["model"].eval()
        
        print(f"[ModelRegistry] Loaded expander ({model_name}) on {device} with {dtype}")
    
    def expand_query(self, query: str) -> Dict[str, str]:
        """
        Generate query expansions.
        
        Returns dict with keys: 'lex', 'vec', 'hyde'
        Each value is a string expansion.
        """
        self._load_expander()
        
        import torch
        import json
        
        tokenizer = self._expander["tokenizer"]
        model = self._expander["model"]
        
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
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON
        json_start = response.find("{")
        if json_start == -1:
            return {"lex": query, "vec": query, "hyde": query}
        
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
            return {
                "lex": expansions.get("lex", query),
                "vec": expansions.get("vec", query),
                "hyde": expansions.get("hyde", query),
            }
        except (json.JSONDecodeError, TypeError):
            return {"lex": query, "vec": query, "hyde": query}
    
    # =========================================================================
    # Warm-up and Status
    # =========================================================================
    
    def warm_up(self):
        """
        Load all three models at once.
        
        Call this on server start so first query isn't slow.
        """
        import time
        
        print("[ModelRegistry] Warming up all models...")
        start = time.time()
        
        # Load embedder first (largest, most commonly used)
        self._load_embedder()
        t1 = time.time()
        print(f"[ModelRegistry]   Embedder loaded in {t1 - start:.1f}s")
        
        # Load reranker
        self._load_reranker()
        t2 = time.time()
        print(f"[ModelRegistry]   Reranker loaded in {t2 - t1:.1f}s")
        
        # Load expander
        self._load_expander()
        t3 = time.time()
        print(f"[ModelRegistry]   Expander loaded in {t3 - t2:.1f}s")
        
        print(f"[ModelRegistry] All models ready in {t3 - start:.1f}s total")
    
    def status(self) -> Dict[str, Any]:
        """Get status of all models."""
        import torch
        
        result = {
            "embedder_loaded": self._embedder is not None,
            "reranker_loaded": self._reranker is not None,
            "expander_loaded": self._expander is not None,
            "device": self._get_device(),
            "dtype": str(self._get_dtype()),
        }
        
        # Memory usage
        if torch.backends.mps.is_available():
            try:
                # MPS doesn't have a direct memory query, estimate from loaded models
                mem = 0
                if self._embedder:
                    mem += 4000  # ~4GB
                if self._reranker:
                    mem += 4000  # ~4GB
                if self._expander:
                    mem += 1000  # ~1GB
                result["memory_allocated_gb"] = mem / 1000
            except:
                pass
        
        return result


# =============================================================================
# Module-level convenience functions
# =============================================================================

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the singleton ModelRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def warm_up():
    """Load all models at once."""
    get_registry().warm_up()


def status() -> Dict[str, Any]:
    """Get model status."""
    return get_registry().status()


def embed_text(text: str) -> np.ndarray:
    """Convenience: embed single text."""
    return get_registry().embed_text(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Convenience: embed multiple texts (batch)."""
    return get_registry().embed_texts(texts)


def embed_image(image_path: str) -> np.ndarray:
    """Convenience: embed single image."""
    return get_registry().embed_image(image_path)


def embed_images(image_paths: List[str]) -> np.ndarray:
    """Convenience: embed multiple images (batch)."""
    return get_registry().embed_images(image_paths)