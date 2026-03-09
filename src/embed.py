"""
embed.py - Qwen3-VL-Embedding wrapper for text and image embeddings.

Uses the official Qwen3VLEmbedder from the Qwen3-VL-Embedding repo.
MPS device with float16 for Apple Silicon compatibility.
"""

import os
import sys
from typing import List, Union, Optional
import numpy as np

# Add Qwen3-VL-Embedding repo to path
QWEN_REPO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Qwen3-VL-Embedding")

# Monkey-patch for transformers 5.x compatibility BEFORE any imports
# The qwen3_vl_embedding module imports check_model_inputs directly
def _apply_transformers_patch():
    """Apply compatibility patch for transformers 5.x."""
    try:
        import transformers.utils.generic as _generic
        if not hasattr(_generic, 'check_model_inputs'):
            _generic.check_model_inputs = lambda *args, **kwargs: None
    except Exception:
        pass
    
    # Also add to module-level if it doesn't exist
    try:
        import transformers.utils
        if not hasattr(transformers.utils, 'check_model_inputs'):
            transformers.utils.check_model_inputs = lambda *args, **kwargs: None
    except Exception:
        pass

_apply_transformers_patch()

# Now add paths and import
Qwen3VLEmbedder = None
if os.path.exists(QWEN_REPO_PATH):
    _src_path = os.path.join(QWEN_REPO_PATH, "src")
    if os.path.exists(_src_path):
        sys.path.insert(0, _src_path)
        try:
            from models.qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError as e:
            print(f"Warning: Could not import Qwen3VLEmbedder: {e}")
            Qwen3VLEmbedder = None


class Embedder:
    """Qwen3-VL-Embedding wrapper for text embeddings."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = "auto",
        dtype: str = "float16"
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model name
            device: 'auto', 'mps', 'cuda', or 'cpu'
            dtype: 'float16' or 'bfloat16'
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model = None
        self._initialized = False
        
        # Determine device
        if device == "auto":
            import torch
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
        
        import torch
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)
        
        # MPS doesn't support bfloat16 well, force float16
        if self.device == "mps" and torch_dtype == torch.bfloat16:
            torch_dtype = torch.float16
        
        # Import embedder
        if Qwen3VLEmbedder is None:
            raise ImportError(
                "Qwen3VLEmbedder not found. "
                "Clone the repo: git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git "
                "and ensure qwen3_vl_embedding.py is in the path."
            )
        
        # Initialize model
        # Note: flash_attention_2 doesn't work on MPS, use eager
        attn_implementation = "eager" if self.device == "mps" else "flash_attention_2"
        
        self._model = Qwen3VLEmbedder(
            model_name_or_path=self.model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        
        self._initialized = True
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
        
        Returns:
            2048-dim numpy array (float32)
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            N x 2048 numpy array (float32)
        """
        self._ensure_model()
        
        # Format inputs for Qwen3VLEmbedder
        inputs = []
        for text in texts:
            inputs.append({
                "text": text,
                "instruction": "Retrieve documents relevant to the user's query.",
            })
        
        # Get embeddings
        embeddings = self._model.process(inputs)
        
        # Convert to numpy
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
        else:
            # Torch tensor
            import torch
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy().astype(np.float32)
            # List
            return np.array(embeddings, dtype=np.float32)
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Embed a single image.
        
        Args:
            image_path: Path or URL to image
        
        Returns:
            2048-dim numpy array (float32)
        """
        return self.embed_images([image_path])[0]
    
    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Embed multiple images.
        
        Args:
            image_paths: List of image paths or URLs
        
        Returns:
            N x 2048 numpy array (float32)
        """
        self._ensure_model()
        
        # Format inputs
        inputs = []
        for path in image_paths:
            inputs.append({
                "image": path,
                "instruction": "Retrieve content relevant to the image.",
            })
        
        # Get embeddings
        embeddings = self._model.process(inputs)
        
        # Convert to numpy
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
        else:
            import torch
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy().astype(np.float32)
            return np.array(embeddings, dtype=np.float32)
    
    def embed_mixed(
        self,
        items: List[dict]
    ) -> np.ndarray:
        """
        Embed mixed text/image content.
        
        Args:
            items: List of {"text": ..., "image": ...} dicts
        
        Returns:
            N x 2048 numpy array (float32)
        """
        self._ensure_model()
        
        # Format inputs
        inputs = []
        for item in items:
            inp = {"instruction": "Retrieve content relevant to the query."}
            if "text" in item:
                inp["text"] = item["text"]
            if "image" in item:
                inp["image"] = item["image"]
            inputs.append(inp)
        
        embeddings = self._model.process(inputs)
        
        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(np.float32)
        else:
            import torch
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy().astype(np.float32)
            return np.array(embeddings, dtype=np.float32)


# Singleton for reuse
_embedder: Optional[Embedder] = None


def get_embedder(model_name: str = "Qwen/Qwen3-VL-Embedding-2B") -> Embedder:
    """Get or create singleton embedder."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(model_name=model_name)
    return _embedder


def embed_text(text: str) -> np.ndarray:
    """Convenience: embed single text."""
    return get_embedder().embed_text(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Convenience: embed multiple texts."""
    return get_embedder().embed_texts(texts)


def embed_image(image_path: str) -> np.ndarray:
    """Convenience: embed single image."""
    return get_embedder().embed_image(image_path)


def embed_images(image_paths: List[str]) -> np.ndarray:
    """Convenience: embed multiple images."""
    return get_embedder().embed_images(image_paths)