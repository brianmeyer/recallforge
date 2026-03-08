"""
embed.py - Qwen3-VL-Embedding wrapper for text and image embeddings

Uses Qwen3-VL-Embedding-2B for 2048-dim embeddings via MPS/CUDA/CPU.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

# Add Qwen3-VL-Embedding to path
QWEN_REPO = Path(__file__).parent.parent / "Qwen3-VL-Embedding"
if QWEN_REPO.exists():
    sys.path.insert(0, str(QWEN_REPO / "src"))

import torch
from PIL import Image

# Lazy imports to avoid loading model until needed
_model = None
_processor = None
_device = None


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def _load_model(model_name: str = "Qwen/Qwen3-VL-Embedding-2B"):
    """Load model and processor lazily."""
    global _model, _processor, _device
    
    if _model is not None:
        return _model, _processor, _device
    
    _device = get_device()
    
    # Import from Qwen3-VL-Embedding repo
    from models.qwen3_vl_embedding import Qwen3VLEmbedder
    
    # Load with MPS support, float16, eager attention
    _model = Qwen3VLEmbedder(
        model_name_or_path=model_name,
        torch_dtype=torch.float16,
        device_map=str(_device),
        # Disable flash_attention_2 for MPS compatibility
        attn_implementation="eager",
    )
    
    _processor = _model.processor
    _model.model.eval()
    
    return _model, _processor, _device


class Qwen3VLEmbedderWrapper:
    """Wrapper for Qwen3-VL-Embedding with convenient API."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._model = None
        self._processor = None
    
    def _ensure_loaded(self):
        """Lazily load model on first use."""
        if self._model is None:
            from models.qwen3_vl_embedding import Qwen3VLEmbedder
            
            device_obj = torch.device(self.device)
            
            self._model = Qwen3VLEmbedder(
                model_name_or_path=self.model_name,
                torch_dtype=self.dtype,
                device_map=str(device_obj),
                attn_implementation="eager",
            )
            self._processor = self._model.processor
            self._model.model.eval()
    
    @property
    def model(self):
        self._ensure_loaded()
        return self._model
    
    @property
    def processor(self):
        self._ensure_loaded()
        return self._processor
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension (2048 for 2B model)."""
        return 2048
    
    def embed_text(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            instruction: Optional instruction for the embedding
        
        Returns:
            np.ndarray of shape (2048,) with float32 values
        """
        embeddings = self.embed_texts([text], instruction=instruction)
        return embeddings[0]
    
    def embed_texts(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed
            instruction: Optional instruction for all texts
            batch_size: Batch size for processing
        
        Returns:
            np.ndarray of shape (N, 2048) with float32 values
        """
        self._ensure_loaded()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Format inputs
            inputs = [
                self._model.format_model_input(text=text, instruction=instruction)
                for text in batch_texts
            ]
            
            # Process
            with torch.no_grad():
                embeddings = self._model.process(inputs, normalize=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed_image(
        self,
        image: Union[str, Image.Image, Path],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Embed an image (path, URL, or PIL Image).
        
        Args:
            image: Image path, URL, or PIL Image object
            instruction: Optional instruction for the embedding
        
        Returns:
            np.ndarray of shape (2048,) with float32 values
        """
        self._ensure_loaded()
        
        # Format input
        input_dict = {'image': image}
        if instruction:
            input_dict['instruction'] = instruction
        
        inputs = [self._model.format_model_input(**input_dict)]
        
        with torch.no_grad():
            embeddings = self._model.process(inputs, normalize=True)
            return embeddings[0].cpu().numpy()
    
    def embed_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        video: Optional[Union[str, List]] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Embed multimodal content (text + image/video).
        
        Args:
            text: Optional text
            image: Optional image (path, URL, or PIL Image)
            video: Optional video (path, URL, or frame list)
            instruction: Optional instruction
        
        Returns:
            np.ndarray of shape (2048,)
        """
        self._ensure_loaded()
        
        inputs = [self._model.format_model_input(
            text=text,
            image=image,
            video=video,
            instruction=instruction,
        )]
        
        with torch.no_grad():
            embeddings = self._model.process(inputs, normalize=True)
            return embeddings[0].cpu().numpy()


# Singleton for convenience
_embedder: Optional[Qwen3VLEmbedderWrapper] = None


def get_embedder(model_name: str = "Qwen/Qwen3-VL-Embedding-2B") -> Qwen3VLEmbedderWrapper:
    """Get the singleton embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Qwen3VLEmbedderWrapper(model_name=model_name)
    return _embedder


async def embed_text_async(text: str, instruction: Optional[str] = None) -> List[float]:
    """Async wrapper for embed_text."""
    embedder = get_embedder()
    embedding = embedder.embed_text(text, instruction)
    return embedding.tolist()


async def embed_texts_async(texts: List[str], instruction: Optional[str] = None) -> List[List[float]]:
    """Async wrapper for embed_texts."""
    embedder = get_embedder()
    embeddings = embedder.embed_texts(texts, instruction)
    return embeddings.tolist()