"""
mlx_backend.py - MLX Backend for RecallForge (Apple Silicon).

Uses mlx-vlm for Apple Silicon MLX inference.
Optional 4-bit quantization for memory efficiency.

Model IDs:
- MLX BF16: arthurcollet/Qwen3-VL-Embedding-2B-mlx, arthurcollet/Qwen3-VL-Reranker-2B-mlx
- MLX 4-bit: arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit, arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit
- Expander: Uses torch fallback (MLX doesn't support the expander model well)
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

from .base import ModelBackend, BackendInfo

# Check if MLX is available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Optional torch import for expander fallback
_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    pass


class MLXBackend(ModelBackend):
    """
    MLX-based model backend for Apple Silicon.
    
    Supports bf16 and 4-bit quantization.
    Uses torch fallback for query expansion (MLX doesn't support the expander well).
    """
    
    def __init__(
        self,
        mode: str = "full",
        quantization: str = "bf16",  # "bf16" or "4bit"
    ):
        """
        Initialize MLX backend.
        
        Args:
            mode: Search mode - 'embed', 'hybrid', or 'full'
            quantization: 'bf16' or '4bit'
        """
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX is not available. Install with: pip install mlx mlx-vlm"
            )
        
        self._mode = mode
        self._quantization = quantization
        
        # Lazy-loaded models
        self._embedder = None
        self._reranker = None
        self._expander = None  # Torch fallback
        self._expander_tokenizer = None
        
        # Model IDs based on quantization
        if quantization == "4bit":
            self.EMBEDDER_MODEL = "arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit"
            self.RERANKER_MODEL = "arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit"
        else:  # bf16
            self.EMBEDDER_MODEL = "arthurcollet/Qwen3-VL-Embedding-2B-mlx"
            self.RERANKER_MODEL = "arthurcollet/Qwen3-VL-Reranker-2B-mlx"
        
        # Expander always uses torch (MLX doesn't support this model well)
        self.EXPANDER_MODEL = "tobil/qmd-query-expansion-qwen3.5-2B"
        
        # Add Qwen3-VL-Embedding repo to path for reranker logic
        self._add_qwen_repo_to_path()
    
    def _add_qwen_repo_to_path(self):
        """Add Qwen3-VL-Embedding repo to sys.path."""
        qwen_repo = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Qwen3-VL-Embedding")
        qwen_src = os.path.join(qwen_repo, "src")
        
        if os.path.exists(qwen_src) and qwen_src not in sys.path:
            sys.path.insert(0, qwen_src)
    
    # =========================================================================
    # Embedder
    # =========================================================================
    
    def _load_embedder(self):
        """Lazy-load the embedder model."""
        if self._embedder is not None:
            return
        
        # MLX-VLM loading
        try:
            from mlx_vlm import load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.tokenizer_utils import TokenizerWrapper
        except ImportError:
            raise ImportError(
                "mlx-vlm is required. Install with: pip install mlx-vlm"
            )
        
        print(f"[MLXBackend] Loading embedder: {self.EMBEDDER_MODEL} ({self._quantization})")
        
        # Load model and processor
        self._embedder = {}
        self._embedder["model"], self._embedder["processor"] = load(
            self.EMBEDDER_MODEL,
            trust_remote_code=True,
        )
        self._embedder["tokenizer"] = TokenizerWrapper(self._embedder["processor"])
        
        print(f"[MLXBackend] Loaded embedder")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple text strings in a batch."""
        self._load_embedder()
        
        # MLX-VLM embedding for text
        embeddings = []
        
        for text in texts:
            # Format input for Qwen3-VL embedding
            messages = [
                {"role": "system", "content": "Retrieve documents relevant to the user's query."},
                {"role": "user", "content": text}
            ]
            
            # Get embedding
            # Note: This is a simplified version - actual implementation
            # depends on the specific MLX model API
            emb = self._get_embedding_for_text(text)
            embeddings.append(emb)
        
        return np.array(embeddings, dtype=np.float32)
    
    def _get_embedding_for_text(self, text: str) -> np.ndarray:
        """Get embedding for a single text using MLX model."""
        # Use the model's embedding method if available
        # This is model-specific and may need adjustment
        import mlx.core as mx
        
        model = self._embedder["model"]
        processor = self._embedder["processor"]
        
        # Tokenize
        inputs = processor(text, return_tensors="mlx")
        
        # Get hidden states
        output = model(**inputs, output_hidden_states=True)
        
        # Use last hidden state, mean pool
        hidden_states = output.hidden_states[-1]  # Last layer
        embedding = mx.mean(hidden_states, axis=1)  # Mean pool
        
        # Convert to numpy
        return np.array(embedding, dtype=np.float32).flatten()
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed a single image."""
        return self.embed_images([image_path])[0]
    
    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """Embed multiple images in a batch."""
        self._load_embedder()
        
        from PIL import Image
        import mlx.core as mx
        
        embeddings = []
        
        for path in image_paths:
            # Load image
            image = Image.open(path).convert("RGB")
            
            # Get embedding
            emb = self._get_embedding_for_image(image)
            embeddings.append(emb)
        
        return np.array(embeddings, dtype=np.float32)
    
    def _get_embedding_for_image(self, image) -> np.ndarray:
        """Get embedding for a single image using MLX model."""
        import mlx.core as mx
        
        model = self._embedder["model"]
        processor = self._embedder["processor"]
        
        # Process image
        inputs = processor(images=image, return_tensors="mlx")
        
        # Get hidden states
        output = model(**inputs, output_hidden_states=True)
        
        # Use last hidden state, mean pool
        hidden_states = output.hidden_states[-1]
        embedding = mx.mean(hidden_states, axis=1)
        
        return np.array(embedding, dtype=np.float32).flatten()
    
    # =========================================================================
    # Reranker
    # =========================================================================
    
    def _load_reranker(self):
        """Lazy-load the reranker model."""
        if self._reranker is not None:
            return
        
        # MLX-VLM loading for reranker
        try:
            from mlx_vlm import load
            from mlx_vlm.tokenizer_utils import TokenizerWrapper
        except ImportError:
            raise ImportError("mlx-vlm is required for reranker")
        
        print(f"[MLXBackend] Loading reranker: {self.RERANKER_MODEL} ({self._quantization})")
        
        self._reranker = {}
        self._reranker["model"], self._reranker["processor"] = load(
            self.RERANKER_MODEL,
            trust_remote_code=True,
        )
        self._reranker["tokenizer"] = TokenizerWrapper(self._reranker["processor"])
        
        print(f"[MLXBackend] Loaded reranker")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Rerank documents for a query."""
        if not documents:
            return []
        
        if not self.needs_reranker():
            return [0.5] * len(documents)
        
        self._load_reranker()
        
        # Get reranking scores
        # This is model-specific and depends on the Qwen3-VL reranker MLX implementation
        try:
            scores = self._compute_rerank_scores(query, documents)
            return scores
        except Exception as e:
            print(f"[MLXBackend] Rerank error: {e}")
            return [0.5] * len(documents)
    
    def _compute_rerank_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Compute reranking scores for query-document pairs."""
        import mlx.core as mx
        
        model = self._reranker["model"]
        processor = self._reranker["processor"]
        
        scores = []
        for doc in documents:
            text = doc.get("text", "") or doc.get("text_body", "") or ""
            
            # Format for reranking
            # This depends on the specific model's expected input format
            inputs = processor(
                text=query,
                text_pair=text,
                return_tensors="mlx"
            )
            
            # Get score from model
            output = model(**inputs)
            
            # Extract relevance score (model-specific)
            # For Qwen reranker, typically a sigmoid of a linear output
            score = float(mx.sigmoid(output.logits[0, 0]))
            scores.append(score)
        
        return scores
    
    # =========================================================================
    # Query Expander (Torch Fallback)
    # =========================================================================
    
    def _load_expander(self):
        """Load expander using torch backend (MLX fallback)."""
        if self._expander is not None:
            return
        
        if not _torch_available:
            raise ImportError(
                "PyTorch is required for query expansion. "
                "Install with: pip install torch"
            )
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"[MLXBackend] Loading expander via torch fallback on {device}")
        
        self._expander_tokenizer = AutoTokenizer.from_pretrained(self.EXPANDER_MODEL)
        self._expander = AutoModelForCausalLM.from_pretrained(
            self.EXPANDER_MODEL,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        ).to(device)
        
        self._expander.eval()
        
        print(f"[MLXBackend] Loaded expander (torch fallback)")
    
    def expand_query(self, query: str) -> Dict[str, str]:
        """Generate query expansions using torch fallback."""
        if not self.needs_expander():
            return {"lex": query, "vec": query, "hyde": query}
        
        self._load_expander()
        
        import json
        
        tokenizer = self._expander_tokenizer
        model = self._expander
        
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
    
    def warm_up(self) -> None:
        """Preload all models for current mode."""
        import time
        
        print(f"[MLXBackend] Warming up models (mode={self._mode}, quant={self._quantization})...")
        start = time.time()
        
        # Always load embedder
        self._load_embedder()
        t1 = time.time()
        print(f"[MLXBackend]   Embedder loaded in {t1 - start:.1f}s")
        
        # Load reranker for hybrid/full
        if self.needs_reranker():
            self._load_reranker()
            t2 = time.time()
            print(f"[MLXBackend]   Reranker loaded in {t2 - t1:.1f}s")
        
        # Load expander for full (torch fallback)
        if self.needs_expander():
            self._load_expander()
            t3 = time.time()
            print(f"[MLXBackend]   Expander loaded in {t3 - t2:.1f}s")
        
        print(f"[MLXBackend] All models ready in {time.time() - start:.1f}s total")
    
    def get_info(self) -> BackendInfo:
        """Return backend information."""
        device = "mps"  # MLX always uses MPS
        
        # Estimate memory
        mem = 0
        if self._embedder:
            mem += 2000 if self._quantization == "4bit" else 4000
        if self._reranker:
            mem += 2000 if self._quantization == "4bit" else 4000
        if self._expander:
            mem += 4000  # Torch fallback
        
        return BackendInfo(
            name="mlx",
            device=device,
            dtype=self._quantization,
            embedder_loaded=self._embedder is not None,
            reranker_loaded=self._reranker is not None,
            expander_loaded=self._expander is not None,
            memory_allocated_gb=mem / 1000,
            supports_images=True,
            quantization=self._quantization,
        )
    
    def set_mode(self, mode: str) -> None:
        """Set the search mode."""
        if mode not in ("embed", "hybrid", "full"):
            raise ValueError(f"Invalid mode: {mode}")
        self._mode = mode
    
    def get_mode(self) -> str:
        """Get current search mode."""
        return self._mode