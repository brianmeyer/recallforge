"""
mlx_backend.py - MLX Backend for RecallForge (Apple Silicon).

Uses mlx-vlm for native Apple Silicon inference with Qwen3-VL models.
Supports bf16 and 4-bit quantization (4-bit default, ~2GB memory).

Model IDs:
- MLX BF16: arthurcollet/Qwen3-VL-Embedding-2B-mlx, arthurcollet/Qwen3-VL-Reranker-2B-mlx
- MLX 4-bit: arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit, arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit
"""

import os
import logging
import importlib.util
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from .base import ModelBackend, BackendInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace cache helpers
# ---------------------------------------------------------------------------

def _check_model_cached(repo_id: str) -> bool:
    """Return True if *repo_id* already exists in the local HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(repo_id, "config.json")
        return result is not None
    except Exception:
        return False

mx = None

# Check if MLX is installed without importing runtime at module import time.
# Importing mlx.core can abort Python on some broken Metal/MLX setups.
MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None


def _load_mlx_core():
    """Import mlx.core lazily and cache the module reference."""
    global mx
    if mx is None:
        import mlx.core as _mx
        mx = _mx
    return mx

def _make_cache(num_layers):
    """Create KVCache objects for each transformer layer."""
    try:
        from mlx_vlm.models.qwen3_vl.language import KVCache
    except ImportError as exc:
        raise ImportError(
            "Failed to import KVCache from mlx_vlm. "
            "Ensure mlx-vlm is installed and up to date."
        ) from exc

    try:
        return [KVCache() for _ in range(num_layers)]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create KVCache objects for {num_layers} layers."
        ) from exc


class MLXEmbeddingError(RuntimeError):
    """Raised when the MLX embedding pipeline fails."""


class MLXBackend(ModelBackend):
    """
    MLX-based model backend for Apple Silicon.

    Default path for macOS arm64. Uses 4-bit quantization (~2GB)
    for embedder and reranker.
    """

    # Chat template for embedding queries
    _EMBED_SYSTEM = "You are a helpful assistant."
    _EMBED_TEMPLATE = (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    _EMBED_DIM = 2048
    _MAX_TEXT_TOKENS_FALLBACK = 32768
    _TOKENIZER_MAX_SENTINEL = 1_000_000
    _RERANK_SYSTEM = (
        "Judge whether the Document meets the requirements based on the Query and "
        "the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    )
    _RERANK_DEFAULT_INSTRUCTION = (
        "Given a search query, retrieve relevant candidates that answer the query."
    )

    def __init__(
        self,
        mode: str = "hybrid",
        quantization: str = "4bit",
    ):
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX is not available. Install with: pip install recallforge[mlx]"
            )
        try:
            _load_mlx_core()
        except Exception as exc:
            raise ImportError(
                "MLX runtime failed to initialize in this environment."
            ) from exc

        self._mode = mode
        self._quantization = quantization

        # Lazy-loaded models
        self._embedder_model = None
        self._embedder_processor = None
        self._reranker_model = None
        self._reranker_processor = None
        self._reranker_score_linear = None
        self._reranker_score_weight = None
        self._reranker_score_bias = None
        self._reranker_yes_token_id = None
        self._reranker_no_token_id = None
        self._reranker_use_direct_logits = False
        self._embedder_num_layers = None
        self._embed_text_max_tokens = None
        self._embed_warmed = False

        # Model IDs based on quantization
        if quantization == "4bit":
            self.EMBEDDER_MODEL = "arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit"
            self.RERANKER_MODEL = "arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit"
        else:
            self.EMBEDDER_MODEL = "arthurcollet/Qwen3-VL-Embedding-2B-mlx"
            self.RERANKER_MODEL = "arthurcollet/Qwen3-VL-Reranker-2B-mlx"

    # =========================================================================
    # Embedder
    # =========================================================================

    def _load_embedder(self):
        """Lazy-load the MLX embedding model with explicit failure context."""
        if self._embedder_model is not None:
            return

        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise ImportError(
                "mlx-vlm is required for MLX embeddings. "
                "Install with: pip install recallforge[mlx]"
            ) from exc

        if not _check_model_cached(self.EMBEDDER_MODEL):
            size = "~800MB" if self._quantization == "4bit" else "~4GB"
            logger.info(
                f"[RecallForge] Downloading embedder model "
                f"({self.EMBEDDER_MODEL.split('/')[-1]}, {size})... first run only."
            )
        logger.info(f"[MLXBackend] Loading embedder: {self.EMBEDDER_MODEL}")

        try:
            self._embedder_model, self._embedder_processor = load(
                self.EMBEDDER_MODEL,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise MLXEmbeddingError(
                f"Failed to load MLX embedder '{self.EMBEDDER_MODEL}'."
            ) from exc

        try:
            self._embedder_num_layers = int(
                self._embedder_model.language_model.model.num_hidden_layers
            )
        except Exception as exc:
            raise MLXEmbeddingError(
                "Loaded MLX embedder does not expose num_hidden_layers."
            ) from exc

        self._embed_text_max_tokens = self._resolve_max_text_tokens()
        logger.info(f"[MLXBackend] Loaded embedder ({self._quantization})")

    def _embed_hidden(self, input_ids: "mx.array", cache) -> "mx.array":
        """
        Text pipeline (MLX): token IDs -> Qwen transformer -> hidden states.

        This is the proven path:
        processor(text=...) -> mx.array(input_ids) -> qwen_model(input_ids, cache=...).
        """
        qwen_model = self._embedder_model.language_model.model
        h = qwen_model(input_ids, cache=cache)
        return h

    def _embed_hidden_with_media(
        self,
        input_ids: "mx.array",
        pixel_values: "mx.array",
        cache,
        image_grid_thw: Optional["mx.array"] = None,
        video_grid_thw: Optional["mx.array"] = None,
    ) -> "mx.array":
        """
        Multimodal vision pipeline (MLX): processor output -> input embeddings -> transformer.

        This is the proven path:
        get_input_embeddings(...) -> qwen_model(None, inputs_embeds=..., cache=...).
        """
        # Merge vision features with text embeddings
        emb_features = self._embedder_model.get_input_embeddings(
            input_ids,
            pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        try:
            inputs_embeds = emb_features.to_dict()["inputs_embeds"]
        except Exception as exc:
            raise MLXEmbeddingError(
                "Vision embedding features did not contain 'inputs_embeds'."
            ) from exc

        # Forward through transformer with pre-computed embeddings
        qwen_model = self._embedder_model.language_model.model
        h = qwen_model(None, inputs_embeds=inputs_embeds, cache=cache)
        return h

    def _pool_and_normalize(self, hidden_states: "mx.array") -> np.ndarray:
        """
        Last-token pooling + L2 normalization in MLX, then cast to float32 numpy.

        MLX tensors are cast to float32 before numpy conversion for stable downstream math.
        """
        emb = hidden_states[:, -1, :]
        emb = emb.astype(mx.float32)
        norm = mx.sqrt(mx.sum(emb * emb, axis=-1, keepdims=True))
        emb = emb / mx.maximum(norm, mx.array(1e-12))
        mx.eval(emb)
        return np.array(emb, dtype=np.float32)

    def _format_text_prompt(self, text: str, system: str = None) -> str:
        """Format text for embedding using chat template."""
        return self._EMBED_TEMPLATE.format(
            system=system or self._EMBED_SYSTEM,
            text=text,
        )

    def _resolve_max_text_tokens(self) -> int:
        """Resolve a safe tokenizer max length for truncation."""
        tokenizer = getattr(self._embedder_processor, "tokenizer", None)
        max_len = getattr(tokenizer, "model_max_length", None)

        if (
            not isinstance(max_len, int)
            or max_len <= 0
            or max_len > self._TOKENIZER_MAX_SENTINEL
        ):
            max_len = self._MAX_TEXT_TOKENS_FALLBACK

        return max_len

    def _get_embedder_num_layers(self) -> int:
        """Return embedder layer count after model load."""
        if self._embedder_num_layers is None:
            raise MLXEmbeddingError(
                "Embedder layer count is unavailable. Reload the embedder model."
            )
        return self._embedder_num_layers

    def _validate_texts(self, texts: List[str]) -> List[str]:
        """Validate text batch input and reject empty/invalid values."""
        if texts is None:
            raise ValueError("Text batch is None; expected a list of strings.")

        if not isinstance(texts, list):
            raise TypeError(
                f"Text batch must be a list[str], got {type(texts).__name__}."
            )

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Text at index {idx} must be a string, got {type(text).__name__}."
                )
            if not text.strip():
                raise ValueError(
                    f"Text at index {idx} is empty. Provide non-empty input text."
                )

        return texts

    def _validate_image_paths(self, image_paths: List[str]) -> List[str]:
        """Validate image paths and surface clear missing/corrupt image errors."""
        if image_paths is None:
            raise ValueError("Image batch is None; expected a list of image paths.")

        if not isinstance(image_paths, list):
            raise TypeError(
                f"Image batch must be a list[str], got {type(image_paths).__name__}."
            )

        normalized_paths = []
        for idx, raw_path in enumerate(image_paths):
            if not isinstance(raw_path, str):
                raise TypeError(
                    f"Image path at index {idx} must be a string, got {type(raw_path).__name__}."
                )
            if not raw_path.strip():
                raise ValueError(
                    f"Image path at index {idx} is empty. Provide a valid file path."
                )

            path = os.path.expanduser(raw_path)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Image file not found at index {idx}: '{raw_path}'."
                )
            if not os.path.isfile(path):
                raise ValueError(
                    f"Image path at index {idx} is not a file: '{raw_path}'."
                )

            try:
                with Image.open(path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ValueError(
                    f"Image file at index {idx} is unreadable or corrupt: '{raw_path}'."
                ) from exc

            normalized_paths.append(path)

        return normalized_paths

    def _validate_video_paths(self, video_paths: List[str]) -> List[str]:
        """Validate video paths and surface clear missing-file errors."""
        if video_paths is None:
            raise ValueError("Video batch is None; expected a list of video paths.")

        if not isinstance(video_paths, list):
            raise TypeError(
                f"Video batch must be a list[str], got {type(video_paths).__name__}."
            )

        normalized_paths = []
        for idx, raw_path in enumerate(video_paths):
            if not isinstance(raw_path, str):
                raise TypeError(
                    f"Video path at index {idx} must be a string, got {type(raw_path).__name__}."
                )
            if not raw_path.strip():
                raise ValueError(
                    f"Video path at index {idx} is empty. Provide a valid file path."
                )

            path = os.path.expanduser(raw_path)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Video file not found at index {idx}: '{raw_path}'."
                )
            if not os.path.isfile(path):
                raise ValueError(
                    f"Video path at index {idx} is not a file: '{raw_path}'."
                )

            normalized_paths.append(path)

        return normalized_paths

    def _to_mx_array(self, value: Any, name: str = None) -> Optional["mx.array"]:
        """Convert torch/np/python arrays to MLX arrays with optional error context."""
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            elif hasattr(value, "numpy") and callable(value.numpy):
                value = value.numpy()
            return mx.array(value)
        except Exception as exc:
            label = f"'{name}'" if name else "value"
            raise MLXEmbeddingError(
                f"Failed to convert {label} to MLX array."
            ) from exc

    def _warm_embed(self) -> None:
        """
        Prime MLX compilation with a lightweight dummy embedding pass.

        MLX compiles kernels on first execution; this removes the first-query spike.
        """
        if self._embed_warmed:
            return

        try:
            self.embed_texts(["warmup"])
        except Exception as exc:
            raise MLXEmbeddingError(
                "Failed to warm MLX embedder with a dummy forward pass."
            ) from exc

        self._embed_warmed = True

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts using the batched MLX text pipeline.

        Pipeline:
        1) Build chat-formatted prompts
        2) Tokenize with truncation (`return_tensors="np"`)
        3) Convert `input_ids` to MLX
        4) Single batched transformer forward pass with KVCache
        5) Last-token pool + L2 normalize + float32 numpy
        """
        self._validate_texts(texts)
        if not texts:
            return np.empty((0, self._EMBED_DIM), dtype=np.float32)

        self._load_embedder()
        num_layers = self._get_embedder_num_layers()
        max_text_tokens = self._embed_text_max_tokens or self._MAX_TEXT_TOKENS_FALLBACK

        prompts = [self._format_text_prompt(text) for text in texts]

        try:
            inputs = self._embedder_processor(
                text=prompts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=max_text_tokens,
            )
        except Exception as exc:
            raise MLXEmbeddingError(
                "Failed to tokenize text batch for MLX embedding."
            ) from exc

        if "input_ids" not in inputs:
            raise MLXEmbeddingError(
                "Text processor output is missing 'input_ids'."
            )

        input_ids = self._to_mx_array(inputs["input_ids"], "input_ids")

        try:
            cache = _make_cache(num_layers)
        except Exception as exc:
            raise MLXEmbeddingError(
                "Failed to initialize KV cache for text embedding batch."
            ) from exc

        try:
            h = self._embed_hidden(input_ids, cache=cache)
        except Exception as exc:
            raise MLXEmbeddingError(
                "MLX text forward pass failed."
            ) from exc

        try:
            embeddings = self._pool_and_normalize(h)
        except Exception as exc:
            raise MLXEmbeddingError(
                "Failed to pool and normalize text embeddings."
            ) from exc

        if embeddings.shape[0] != len(texts):
            raise MLXEmbeddingError(
                f"Text embedding batch size mismatch: expected {len(texts)}, got {embeddings.shape[0]}."
            )

        return embeddings

    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed a single image."""
        return self.embed_images([image_path])[0]

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Embed multiple images using the batched MLX vision pipeline.

        Pipeline:
        1) Validate image files and integrity
        2) Build multimodal chat messages
        3) `process_vision_info` + processor (`return_tensors="pt"`)
        4) Convert tensors to MLX arrays
        5) `get_input_embeddings(...)` -> transformer forward with `inputs_embeds`
        6) Last-token pool + L2 normalize + float32 numpy
        """
        image_paths = self._validate_image_paths(image_paths)
        if not image_paths:
            return np.empty((0, self._EMBED_DIM), dtype=np.float32)

        self._load_embedder()
        num_layers = self._get_embedder_num_layers()

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise MLXEmbeddingError(
                "qwen-vl-utils vision dependencies are missing. "
                "Install qwen-vl-utils and torchvision for image embeddings."
            ) from exc

        messages_batch = []
        for path in image_paths:
            messages_batch.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": "Describe this image."},
                ],
            }])

        try:
            chat_texts = [
                self._embedder_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                for messages in messages_batch
            ]
        except Exception as exc:
            raise MLXEmbeddingError(
                "Failed to build chat templates for image embedding batch."
            ) from exc

        try:
            image_inputs, _ = process_vision_info(messages_batch)
        except Exception as exc:
            raise MLXEmbeddingError(
                "Failed to process vision inputs for image embedding batch."
            ) from exc

        if not image_inputs:
            raise MLXEmbeddingError(
                "Vision pre-processing produced no image inputs."
            )
        if len(image_inputs) != len(image_paths):
            raise MLXEmbeddingError(
                f"Vision pre-processing count mismatch: expected {len(image_paths)}, got {len(image_inputs)}."
            )

        try:
            # Image processor requires PyTorch tensors, convert to MLX after
            inputs = self._embedder_processor(
                text=chat_texts, images=image_inputs,
                return_tensors="pt", padding=True,
            )
        except Exception as exc:
            raise MLXEmbeddingError(
                "Failed to tokenize image batch with processor."
            ) from exc

        for required_key in ("input_ids", "pixel_values", "image_grid_thw"):
            if required_key not in inputs:
                raise MLXEmbeddingError(
                    f"Image processor output is missing '{required_key}'."
                )

        input_ids = self._to_mx_array(inputs["input_ids"], "input_ids")
        pixel_values = self._to_mx_array(inputs["pixel_values"], "pixel_values")
        image_grid_thw = self._to_mx_array(inputs["image_grid_thw"], "image_grid_thw")

        try:
            cache = _make_cache(num_layers)
        except Exception as exc:
            raise MLXEmbeddingError(
                "Failed to initialize KV cache for image embedding batch."
            ) from exc

        try:
            h = self._embed_hidden_with_media(
                input_ids,
                pixel_values,
                cache,
                image_grid_thw=image_grid_thw,
            )
        except Exception as exc:
            raise MLXEmbeddingError(
                "MLX vision forward pass failed."
            ) from exc

        try:
            embeddings = self._pool_and_normalize(h)
        except Exception as exc:
            raise MLXEmbeddingError(
                "Failed to pool and normalize image embeddings."
            ) from exc

        if embeddings.shape[0] != len(image_paths):
            raise MLXEmbeddingError(
                f"Image embedding batch size mismatch: expected {len(image_paths)}, got {embeddings.shape[0]}."
            )

        return embeddings

    def embed_video(self, video_path: str) -> np.ndarray:
        """Embed a single video."""
        return self.embed_videos([video_path])[0]

    def embed_videos(self, video_paths: List[str]) -> np.ndarray:
        """
        Embed multiple videos using the MLX video pipeline.

        Qwen3-VL's video processor currently expects per-video sampling kwargs,
        so we process each video independently and stack the resulting vectors.
        """
        video_paths = self._validate_video_paths(video_paths)
        if not video_paths:
            return np.empty((0, self._EMBED_DIM), dtype=np.float32)

        self._load_embedder()
        num_layers = self._get_embedder_num_layers()

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise MLXEmbeddingError(
                "qwen-vl-utils vision dependencies are missing. "
                "Install qwen-vl-utils and torchvision for video embeddings."
            ) from exc

        embeddings: List[np.ndarray] = []
        for path in video_paths:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": path},
                    {"type": "text", "text": "Describe this video."},
                ],
            }]

            try:
                chat_text = self._embedder_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception as exc:
                raise MLXEmbeddingError(
                    f"Failed to build chat template for video '{path}'."
                ) from exc

            try:
                _, video_inputs, video_kwargs = process_vision_info(
                    [messages],
                    return_video_kwargs=True,
                )
            except Exception as exc:
                raise MLXEmbeddingError(
                    f"Failed to process video inputs for '{path}'."
                ) from exc

            if not video_inputs:
                raise MLXEmbeddingError(
                    f"Video pre-processing produced no video inputs for '{path}'."
                )

            normalized_video_kwargs = dict(video_kwargs or {})
            fps_value = normalized_video_kwargs.get("fps")
            if isinstance(fps_value, list):
                normalized_video_kwargs["fps"] = fps_value[0] if fps_value else None

            try:
                inputs = self._embedder_processor(
                    text=[chat_text],
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                    **normalized_video_kwargs,
                )
            except Exception as exc:
                raise MLXEmbeddingError(
                    f"Failed to tokenize video '{path}' with processor."
                ) from exc

            for required_key in ("input_ids", "pixel_values_videos", "video_grid_thw"):
                if required_key not in inputs:
                    raise MLXEmbeddingError(
                        f"Video processor output is missing '{required_key}' for '{path}'."
                    )

            input_ids = self._to_mx_array(inputs["input_ids"], "input_ids")
            pixel_values = self._to_mx_array(inputs["pixel_values_videos"], "pixel_values_videos")
            video_grid_thw = self._to_mx_array(inputs["video_grid_thw"], "video_grid_thw")

            try:
                cache = _make_cache(num_layers)
            except Exception as exc:
                raise MLXEmbeddingError(
                    f"Failed to initialize KV cache for video '{path}'."
                ) from exc

            try:
                h = self._embed_hidden_with_media(
                    input_ids,
                    pixel_values,
                    cache,
                    video_grid_thw=video_grid_thw,
                )
            except Exception as exc:
                raise MLXEmbeddingError(
                    f"MLX video forward pass failed for '{path}'."
                ) from exc

            try:
                embedding = self._pool_and_normalize(h)
            except Exception as exc:
                raise MLXEmbeddingError(
                    f"Failed to pool and normalize video embedding for '{path}'."
                ) from exc

            embeddings.append(embedding[0])

        return np.stack(embeddings).astype(np.float32)

    # =========================================================================
    # Reranker
    # =========================================================================

    def _resolve_attr(self, root: Any, candidates: List[str]) -> Optional[Any]:
        """Resolve the first matching dotted attribute path from candidates."""
        for path in candidates:
            current = root
            found = True
            for part in path.split("."):
                if not hasattr(current, part):
                    found = False
                    break
                current = getattr(current, part)
            if found:
                return current
        return None

    def _dequantize_weight(self, module: Any) -> Optional["mx.array"]:
        """Return a dense weight matrix, dequantizing quantized layers when needed."""
        weight = getattr(module, "weight", module)
        weight_mx = self._to_mx_array(weight)
        if weight_mx is None:
            return None

        scales = getattr(module, "scales", None)
        group_size = getattr(module, "group_size", None)
        bits = getattr(module, "bits", None)
        if scales is None or group_size is None or bits is None:
            return weight_mx.astype(mx.float32)

        try:
            dequant_kwargs = {
                "scales": self._to_mx_array(scales),
                "group_size": int(group_size),
                "bits": int(bits),
            }
            biases = getattr(module, "biases", None)
            if biases is not None:
                dequant_kwargs["biases"] = self._to_mx_array(biases)
            mode = getattr(module, "mode", None)
            if mode is not None:
                dequant_kwargs["mode"] = mode
            return mx.dequantize(weight_mx, **dequant_kwargs).astype(mx.float32)
        except Exception as e:
            logger.debug(f"[MLXBackend] Failed to dequantize weight: {e}")
            return None

    def _derive_binary_head_from_lm(self) -> Optional["mx.array"]:
        """Derive yes-no projection from lm_head when score_linear is unavailable."""
        tokenizer = getattr(self._reranker_processor, "tokenizer", self._reranker_processor)
        if not hasattr(tokenizer, "get_vocab"):
            return None

        vocab = tokenizer.get_vocab()
        yes_id = vocab.get("yes")
        no_id = vocab.get("no")
        if yes_id is None or no_id is None:
            return None

        lm_head = self._resolve_attr(
            self._reranker_model,
            ["lm_head", "language_model.lm_head", "language_model.model.lm_head"],
        )
        # Qwen3-VL ties lm_head to embed_tokens (tie_word_embeddings=True).
        # When lm_head is absent (e.g. mlx_vlm conversion), use embed_tokens.
        if lm_head is None:
            lm_head = self._resolve_attr(
                self._reranker_model,
                ["language_model.model.embed_tokens", "language_model.embed_tokens",
                 "model.embed_tokens", "embed_tokens"],
            )
        if lm_head is None:
            return None

        # Use dequantization helper to handle 4-bit quantized weights
        lm_head_weight = self._dequantize_weight(lm_head)
        hidden_size = self._resolve_attr(
            self._reranker_model,
            [
                "language_model.model.args.hidden_size",
                "language_model.args.hidden_size",
                "config.text_config.hidden_size",
                "language_model.model.embed_tokens.dims",
            ],
        )
        if hidden_size is not None:
            hidden_size = int(hidden_size)

        if (
            lm_head_weight is None
            or len(lm_head_weight.shape) != 2
            or (hidden_size is not None and lm_head_weight.shape[1] != hidden_size)
        ):
            # Fallback: calling embedding modules with token IDs returns dequantized rows.
            if callable(lm_head):
                try:
                    yes_emb = self._to_mx_array(lm_head(mx.array([yes_id])))
                    no_emb = self._to_mx_array(lm_head(mx.array([no_id])))
                    if yes_emb is not None and no_emb is not None:
                        return (yes_emb[0] - no_emb[0]).astype(mx.float32)
                except Exception as e:
                    logger.debug(f"[MLXBackend] Failed to get embeddings via module call: {e}")
            return None

        return (lm_head_weight[yes_id] - lm_head_weight[no_id]).astype(mx.float32)

    def _init_reranker_scoring(self) -> None:
        """Resolve score_linear or derive an equivalent projection from lm_head."""
        tokenizer = getattr(self._reranker_processor, "tokenizer", self._reranker_processor)
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
            self._reranker_yes_token_id = vocab.get("yes")
            self._reranker_no_token_id = vocab.get("no")
        else:
            self._reranker_yes_token_id = None
            self._reranker_no_token_id = None
        self._reranker_use_direct_logits = False

        self._reranker_score_linear = self._resolve_attr(
            self._reranker_model,
            ["score_linear", "language_model.score_linear", "language_model.model.score_linear"],
        )
        self._reranker_score_weight = None
        self._reranker_score_bias = None

        if self._reranker_score_linear is not None:
            weight = getattr(self._reranker_score_linear, "weight", None)
            bias = getattr(self._reranker_score_linear, "bias", None)
            if weight is not None:
                self._reranker_score_weight = self._to_mx_array(weight).astype(mx.float32)
            if bias is not None:
                self._reranker_score_bias = self._to_mx_array(bias).astype(mx.float32)

            if self._reranker_score_weight is not None:
                logger.debug("[MLXBackend] Using native score_linear reranker head")
                return
            if callable(self._reranker_score_linear):
                logger.debug("[MLXBackend] Using callable score_linear reranker head")
                return

        derived_weight = self._derive_binary_head_from_lm()
        if derived_weight is not None:
            self._reranker_score_weight = derived_weight
            self._reranker_score_bias = None
            self._reranker_score_linear = None
            logger.debug("[MLXBackend] score_linear missing; using lm_head yes-no projection fallback")
            return

        # Last-resort fallback for quantized tied embeddings: use direct language-model
        # logits and compare yes/no token logits.
        if (
            self._reranker_yes_token_id is not None
            and self._reranker_no_token_id is not None
        ):
            self._reranker_use_direct_logits = True
            self._reranker_score_linear = None
            self._reranker_score_weight = None
            self._reranker_score_bias = None
            logger.debug("[MLXBackend] score head missing; using direct yes/no logits fallback")
            return

        raise RuntimeError("Unable to locate reranker score head (score_linear/lm_head)")

    def _format_reranker_prompt(self, query: str, document: str, instruction: str) -> str:
        """Format a query-document pair as chat input for reranking."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._RERANK_SYSTEM}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<Instruct>: {instruction}"},
                    {"type": "text", "text": "<Query>:"},
                    {"type": "text", "text": query or "NULL"},
                    {"type": "text", "text": "\n<Document>:"},
                    {"type": "text", "text": document or "NULL"},
                ],
            },
        ]

        try:
            return self._reranker_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Keep a manual fallback so reranking still works even if template API differs.
            return (
                f"<|im_start|>system\n{self._RERANK_SYSTEM}<|im_end|>\n"
                "<|im_start|>user\n"
                f"<Instruct>: {instruction}\n<Query>:\n{query or 'NULL'}\n"
                f"<Document>:\n{document or 'NULL'}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

    def _apply_reranker_linear(self, last_hidden: "mx.array") -> "mx.array":
        """Apply resolved reranker linear head to final hidden states."""
        hidden = last_hidden.astype(mx.float32)

        # Preferred path: call native score_linear directly if available.
        if callable(self._reranker_score_linear):
            try:
                logits = self._reranker_score_linear(hidden)
                return self._to_mx_array(logits).astype(mx.float32)
            except Exception:
                # Fall through to manual matmul with extracted/derived weights.
                pass

        if self._reranker_score_weight is None:
            raise RuntimeError("Reranker score weights are not initialized")

        weight = self._reranker_score_weight.astype(mx.float32)
        if len(weight.shape) == 1:
            logits = mx.sum(hidden * weight, axis=-1, keepdims=True)
        elif len(weight.shape) == 2:
            if weight.shape[0] == 1:
                logits = mx.matmul(hidden, mx.transpose(weight))
            elif weight.shape[1] == 1:
                logits = mx.matmul(hidden, weight)
            elif weight.shape[0] == hidden.shape[-1]:
                logits = mx.matmul(hidden, weight)
            elif weight.shape[1] == hidden.shape[-1]:
                logits = mx.matmul(hidden, mx.transpose(weight))
            else:
                raise RuntimeError(
                    f"Unsupported score_linear weight shape: {tuple(weight.shape)}"
                )
        else:
            raise RuntimeError(f"Unsupported score_linear rank: {len(weight.shape)}")

        if self._reranker_score_bias is not None:
            logits = logits + self._reranker_score_bias.astype(mx.float32)

        return logits

    def _score_reranker_prompt(self, prompt: str, num_layers: int) -> float:
        """Run one query-document pair and return sigmoid(score_linear(last_hidden))."""
        inputs = self._reranker_processor(text=prompt, return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])

        cache = _make_cache(num_layers)
        if self._reranker_use_direct_logits:
            if self._reranker_yes_token_id is None or self._reranker_no_token_id is None:
                raise RuntimeError("yes/no token ids are unavailable for reranker fallback")
            lm_out = self._reranker_model.language_model(input_ids, cache=cache)
            full_logits = self._to_mx_array(getattr(lm_out, "logits", lm_out)).astype(mx.float32)
            last_logits = full_logits[:, -1, :]
            logits = (
                last_logits[:, self._reranker_yes_token_id]
                - last_logits[:, self._reranker_no_token_id]
            ).reshape(-1, 1)
        else:
            qwen_model = self._reranker_model.language_model.model
            hidden = qwen_model(input_ids, cache=cache).astype(mx.float32)
            last_hidden = hidden[:, -1, :]
            logits = self._apply_reranker_linear(last_hidden)

        probs = 1.0 / (1.0 + mx.exp(-logits))
        mx.eval(probs)
        score = float(np.array(probs).reshape(-1)[0])
        if not np.isfinite(score):
            raise RuntimeError("Reranker produced non-finite score")
        return score

    def _load_reranker(self):
        """Lazy-load the MLX reranker model."""
        if self._reranker_model is not None:
            return

        from mlx_vlm import load

        if not _check_model_cached(self.RERANKER_MODEL):
            size = "~800MB" if self._quantization == "4bit" else "~4GB"
            logger.info(
                f"[RecallForge] Downloading reranker model "
                f"({self.RERANKER_MODEL.split('/')[-1]}, {size})... first run only."
            )
        logger.info(f"[MLXBackend] Loading reranker: {self.RERANKER_MODEL}")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*The fast path is not available.*")
                warnings.filterwarnings("ignore", message=".*causal_conv1d.*")
                warnings.filterwarnings(
                    "ignore",
                    message=".*`torch_dtype` is deprecated! Use `dtype` instead!.*",
                )

                hf_logging = None
                prev_hf_verbosity = None
                try:
                    from transformers.utils import logging as hf_logging  # type: ignore

                    prev_hf_verbosity = hf_logging.get_verbosity()
                    hf_logging.set_verbosity_error()
                except Exception:
                    hf_logging = None
                    prev_hf_verbosity = None

                try:
                    self._reranker_model, self._reranker_processor = load(
                        self.RERANKER_MODEL,
                        trust_remote_code=True,
                    )
                finally:
                    if hf_logging is not None and prev_hf_verbosity is not None:
                        hf_logging.set_verbosity(prev_hf_verbosity)
            self._init_reranker_scoring()
        except Exception:
            self._reranker_model = None
            self._reranker_processor = None
            self._reranker_score_linear = None
            self._reranker_score_weight = None
            self._reranker_score_bias = None
            self._reranker_yes_token_id = None
            self._reranker_no_token_id = None
            self._reranker_use_direct_logits = False
            raise
        logger.info(f"[MLXBackend] Loaded reranker ({self._quantization})")

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Rerank documents for a query."""
        if not documents:
            return []

        if not self.needs_reranker():
            return [0.5] * len(documents)

        try:
            self._load_reranker()
            num_layers = self._reranker_model.language_model.model.num_hidden_layers
        except Exception as e:
            logger.error(f"[MLXBackend] Failed to initialize reranker: {e}")
            return [0.5] * len(documents)

        instruction = self._RERANK_DEFAULT_INSTRUCTION
        scores: List[float] = []
        for idx, doc in enumerate(documents):
            try:
                text = doc.get("text", "") or doc.get("text_body", "") or ""
                prompt = self._format_reranker_prompt(query, text, instruction)
                score = self._score_reranker_prompt(prompt, num_layers)
                scores.append(score)
            except Exception as e:
                logger.error(f"[MLXBackend] Rerank error at doc {idx}: {e}")
                scores.append(0.5)

        return scores

    # =========================================================================
    # Warm-up and Status
    # =========================================================================

    def warm_up(self) -> None:
        """Preload models and run a dummy embed pass to prime MLX compilation."""
        import time

        logger.info(f"[MLXBackend] Warming up (mode={self._mode}, quant={self._quantization})...")
        start = time.time()

        self._load_embedder()
        self._warm_embed()
        t1 = time.time()
        logger.info(f"[MLXBackend]   Embedder+compile: {t1 - start:.1f}s")

        if self.needs_reranker():
            self._load_reranker()
            t2 = time.time()
            logger.info(f"[MLXBackend]   Reranker: {t2 - t1:.1f}s")

        logger.info(f"[MLXBackend] Ready in {time.time() - start:.1f}s")

    def get_info(self) -> BackendInfo:
        """Return backend information."""
        mem = 0
        if self._embedder_model:
            mem += 2000 if self._quantization == "4bit" else 4000
        if self._reranker_model:
            mem += 2000 if self._quantization == "4bit" else 4000

        return BackendInfo(
            name="mlx",
            device="mps",
            dtype=self._quantization,
            embedder_loaded=self._embedder_model is not None,
            reranker_loaded=self._reranker_model is not None,
            memory_allocated_gb=mem / 1000,
            supports_images=True,
            quantization=self._quantization,
        )

    def set_mode(self, mode: str) -> None:
        """Set the search mode."""
        import logging
        logger = logging.getLogger(__name__)
        
        if mode == "full":
            logger.warning(
                "[MLXBackend] Mode 'full' is deprecated (query expander removed). "
                "Falling back to 'hybrid'. See REC-108 for details."
            )
            mode = "hybrid"
        
        if mode not in ("embed", "hybrid"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'embed' or 'hybrid'")
        self._mode = mode

    def get_mode(self) -> str:
        """Get current search mode."""
        return self._mode
