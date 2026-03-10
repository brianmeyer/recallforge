"""
torch_backend.py - PyTorch Backend for RecallForge.

Uses transformers + torch for model inference.
Device selection: CUDA > MPS > CPU.

Model IDs:
- Embedder: Qwen/Qwen3-VL-Embedding-2B
- Reranker: Qwen/Qwen3-VL-Reranker-2B
- Expander: tobil/qmd-query-expansion-qwen3.5-2B (MULTIMODAL Image-Text-to-Text)
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

from .base import ModelBackend, BackendInfo


class TorchBackend(ModelBackend):
    """
    PyTorch-based model backend.
    
    Supports CUDA, MPS (Apple Silicon), and CPU.
    Uses float16 for efficiency on GPU/MPS.
    """
    
    def __init__(
        self,
        mode: str = "full",
        device: str = "auto",
        dtype: str = "float16",
    ):
        """
        Initialize PyTorch backend.
        
        Args:
            mode: Search mode - 'embed', 'hybrid', or 'full'
            device: 'auto', 'cuda', 'mps', or 'cpu'
            dtype: 'float16', 'bfloat16', or 'float32'
        """
        self._mode = mode
        self._device_requested = device
        self._dtype_requested = dtype
        
        # Lazy-loaded models
        self._embedder = None
        self._reranker = None
        self._expander = None
        self._expander_tokenizer = None
        
        # Resolved device/dtype
        self._device = None
        self._dtype = None
        
        # Model IDs
        self.EMBEDDER_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
        self.RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
        self.EXPANDER_MODEL = "tobil/qmd-query-expansion-qwen3.5-2B"
        
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
        # Find the repo relative to this file
        qwen_repo = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Qwen3-VL-Embedding")
        qwen_src = os.path.join(qwen_repo, "src")
        
        if os.path.exists(qwen_src) and qwen_src not in sys.path:
            sys.path.insert(0, qwen_src)
    
    def _get_device(self) -> str:
        """Determine best device for inference."""
        if self._device is not None:
            return self._device
        
        import torch
        
        if self._device_requested != "auto":
            self._device = self._device_requested
            return self._device
        
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        
        return self._device
    
    def _get_dtype(self):
        """Get torch dtype for model loading."""
        import torch
        
        if self._dtype is not None:
            return self._dtype
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        requested = dtype_map.get(self._dtype_requested, torch.float16)
        
        # MPS doesn't support bfloat16
        device = self._get_device()
        if device == "mps" and requested == torch.bfloat16:
            requested = torch.float16
        
        self._dtype = requested
        return self._dtype
    
    def _get_attention_implementation(self) -> str:
        """Get attention implementation for device."""
        device = self._get_device()
        if device == "mps":
            return "eager"
        elif device == "cuda":
            return "flash_attention_2"
        else:
            return "eager"
    
    # =========================================================================
    # Embedder (Qwen3-VL-Embedding-2B, ~4GB)
    # =========================================================================
    
    def _load_embedder(self):
        """Lazy-load the embedder model."""
        if self._embedder is not None:
            return
        
        import torch
        
        # Import from Qwen3-VL-Embedding repo
        # Try multiple import paths: installed package, repo src/models, direct
        try:
            from models.qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError:
            try:
                from qwen3_vl_embedding import Qwen3VLEmbedder
            except ImportError:
                # Find the Qwen3-VL-Embedding repo relative to this package
                import importlib.util
                _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _repo_root = os.path.dirname(os.path.dirname(_pkg_root))
                _qwen_src = os.path.join(_repo_root, "Qwen3-VL-Embedding", "src")
                if os.path.isdir(_qwen_src) and _qwen_src not in sys.path:
                    sys.path.insert(0, _qwen_src)
                from models.qwen3_vl_embedding import Qwen3VLEmbedder
        
        device = self._get_device()
        dtype = self._get_dtype()
        attn = self._get_attention_implementation()
        
        self._embedder = Qwen3VLEmbedder(
            model_name_or_path=self.EMBEDDER_MODEL,
            torch_dtype=dtype,
            attn_implementation=attn,
        )
        
        print(f"[TorchBackend] Loaded embedder on {device} with {dtype}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple text strings in a batch."""
        self._load_embedder()
        
        inputs = [
            {
                "text": text,
                "instruction": "Retrieve documents relevant to the user's query.",
            }
            for text in texts
        ]
        
        embeddings = self._embedder.process(inputs)
        
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
        """Embed multiple images in a batch."""
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
    
    # =========================================================================
    # Reranker (Qwen3-VL-Reranker-2B, ~4GB)
    # =========================================================================
    
    def _load_reranker(self):
        """Lazy-load the reranker model.
        
        On MPS, loads with float32 to avoid Apple Metal GEMV kernel crash
        (LORADOWN matrixRowPadElements overflow in float16).
        Embedder is fine with float16 — only the reranker triggers this bug.
        """
        if self._reranker is not None:
            return
        
        import torch
        
        try:
            from models.qwen3_vl_reranker import Qwen3VLReranker
        except ImportError:
            try:
                from qwen3_vl_reranker import Qwen3VLReranker
            except ImportError:
                # Fallback: find via Qwen3-VL-Embedding repo path
                _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _repo_root = os.path.dirname(os.path.dirname(_pkg_root))
                _qwen_src = os.path.join(_repo_root, "Qwen3-VL-Embedding", "src")
                if os.path.isdir(_qwen_src) and _qwen_src not in sys.path:
                    sys.path.insert(0, _qwen_src)
                from models.qwen3_vl_reranker import Qwen3VLReranker
        
        device = self._get_device()
        attn = self._get_attention_implementation()
        
        # MPS float16 triggers Apple Metal GEMV kernel crash on reranker.
        # Use float32 on MPS — still GPU-accelerated, ~50ms penalty on 10 docs.
        if device == "mps":
            dtype = torch.float32
        else:
            dtype = self._get_dtype()
        
        self._reranker = Qwen3VLReranker(
            model_name_or_path=self.RERANKER_MODEL,
            torch_dtype=dtype,
            attn_implementation=attn,
        )
        
        # Move to device
        self._reranker.device = torch.device(device)
        self._reranker.model.to(self._reranker.device)
        self._reranker.score_linear.to(self._reranker.device)
        
        print(f"[TorchBackend] Loaded reranker on {device} with {dtype}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Rerank documents for a query."""
        if not documents:
            return []
        
        if not self.needs_reranker():
            # Hybrid/full mode not active, return neutral scores
            return [0.5] * len(documents)
        
        self._load_reranker()
        
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
            print(f"[TorchBackend] Rerank error: {e}")
            return [0.5] * len(documents)
    
    # =========================================================================
    # Query Expander (tobil/qmd-query-expansion-qwen3.5-2B, ~4GB)
    # =========================================================================
    
    def _load_expander(self):
        """Lazy-load the query expansion model."""
        if self._expander is not None:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        device = self._get_device()
        dtype = self._get_dtype()
        attn = self._get_attention_implementation()
        
        model_name = self.EXPANDER_MODEL
        
        self._expander_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._expander = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn,
        ).to(device)
        
        self._expander.eval()
        
        print(f"[TorchBackend] Loaded expander ({model_name}) on {device} with {dtype}")
    
    def expand_query(self, query: str) -> Dict[str, str]:
        """Generate query expansions."""
        if not self.needs_expander():
            # Full mode not active, return original query
            return {"lex": query, "vec": query, "hyde": query}
        
        self._load_expander()
        
        import torch
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
        
        print(f"[TorchBackend] Warming up models (mode={self._mode})...")
        start = time.time()
        
        # Always load embedder
        self._load_embedder()
        t1 = time.time()
        print(f"[TorchBackend]   Embedder loaded in {t1 - start:.1f}s")
        
        # Load reranker for hybrid/full
        if self.needs_reranker():
            self._load_reranker()
            t2 = time.time()
            print(f"[TorchBackend]   Reranker loaded in {t2 - t1:.1f}s")
        
        # Load expander for full
        if self.needs_expander():
            self._load_expander()
            t3 = time.time()
            print(f"[TorchBackend]   Expander loaded in {t3 - t2:.1f}s")
        
        print(f"[TorchBackend] All models ready in {time.time() - start:.1f}s total")
    
    def get_info(self) -> BackendInfo:
        """Return backend information."""
        import torch
        
        device = self._get_device()
        dtype = str(self._get_dtype())
        
        # Estimate memory
        mem = 0
        if self._embedder:
            mem += 4000  # ~4GB
        if self._reranker:
            mem += 4000  # ~4GB
        if self._expander:
            mem += 4000  # ~4GB (larger model)
        
        return BackendInfo(
            name="torch",
            device=device,
            dtype=dtype,
            embedder_loaded=self._embedder is not None,
            reranker_loaded=self._reranker is not None,
            expander_loaded=self._expander is not None,
            memory_allocated_gb=mem / 1000,
            supports_images=True,
            quantization=None,
        )
    
    def set_mode(self, mode: str) -> None:
        """Set the search mode."""
        if mode not in ("embed", "hybrid", "full"):
            raise ValueError(f"Invalid mode: {mode}")
        self._mode = mode
    
    def get_mode(self) -> str:
        """Get current search mode."""
        return self._mode