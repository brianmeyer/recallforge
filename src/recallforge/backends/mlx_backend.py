"""
mlx_backend.py - MLX Backend for RecallForge (Apple Silicon).

Uses mlx-vlm for native Apple Silicon inference with Qwen3-VL models.
Supports bf16 and 4-bit quantization (4-bit default, ~2GB memory).

Model IDs:
- MLX BF16: arthurcollet/Qwen3-VL-Embedding-2B-mlx, arthurcollet/Qwen3-VL-Reranker-2B-mlx
- MLX 4-bit: arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit, arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit
"""

import gc
import os
import logging
import importlib.util
import threading
import tempfile
import warnings
import re
from contextlib import contextmanager
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


_HEAVY_OP_GATE_INIT_LOCK = threading.Lock()
_HEAVY_OP_GATE = None
_HEAVY_OP_GATE_LIMIT = None


class _HeavyOpGate:
    """Serialize the heaviest MLX operations to avoid local memory pileups."""

    def __init__(self, limit: int, lock_path: Optional[str] = None):
        self.limit = max(1, int(limit))
        self.lock_path = lock_path or os.path.join(
            tempfile.gettempdir(), "recallforge-mlx-heavy-op.lock"
        )
        self._semaphore = threading.BoundedSemaphore(self.limit)
        self._thread_state = threading.local()

    def _acquire_file_lock(self, op_name: str):
        if self.limit != 1:
            return None
        try:
            import fcntl
        except ImportError:
            return None

        handle = open(self.lock_path, "a+", encoding="utf-8")
        logger.debug(
            "mlx_heavy_op_wait_host_lock op=%s lock_path=%s",
            op_name,
            self.lock_path,
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _release_file_lock(self, handle) -> None:
        if handle is None:
            return
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        finally:
            handle.close()

    @contextmanager
    def hold(self, op_name: str):
        depth = getattr(self._thread_state, "depth", 0)
        acquired = False
        file_handle = None
        if depth == 0:
            logger.debug("mlx_heavy_op_wait op=%s limit=%d", op_name, self.limit)
            self._semaphore.acquire()
            acquired = True
            file_handle = self._acquire_file_lock(op_name)
            logger.debug("mlx_heavy_op_acquired op=%s limit=%d", op_name, self.limit)

        self._thread_state.depth = depth + 1
        try:
            yield
        finally:
            new_depth = getattr(self._thread_state, "depth", 1) - 1
            if new_depth <= 0:
                if hasattr(self._thread_state, "depth"):
                    delattr(self._thread_state, "depth")
                self._release_file_lock(file_handle)
                if acquired:
                    self._semaphore.release()
                logger.debug("mlx_heavy_op_release op=%s", op_name)
            else:
                self._thread_state.depth = new_depth


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

    # Video sampling: 1 fps adapts to video length (30s video = 30 frames).
    # Keep the default cap conservative for local-agent safety; callers can
    # raise it explicitly via env vars when they want heavier runs.
    _VIDEO_SAMPLE_FPS = 1.0
    _VIDEO_MAX_FRAMES = 32
    _VIDEO_FALLBACK_MAX_FRAMES = 8
    _DEFAULT_HEAVY_OP_CONCURRENCY = 1
    # Captioning descriptors removed — they produced captions too generic for BM25.
    # See REC-129 for dedicated captioning model support.

    # Default model IDs (can be overridden via env vars or set_config)
    _DEFAULT_EMBEDDER_MODEL_4BIT = "arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit"
    _DEFAULT_EMBEDDER_MODEL_BF16 = "arthurcollet/Qwen3-VL-Embedding-2B-mlx"
    _DEFAULT_RERANKER_MODEL_4BIT = "arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit"
    _DEFAULT_RERANKER_MODEL_BF16 = "arthurcollet/Qwen3-VL-Reranker-2B-mlx"
    _DEFAULT_CAPTION_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"

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
        self._model_lock = threading.Lock()

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
        self._captioner_model = None
        self._captioner_processor = None

        # Model IDs - configurable via env vars (REC-116)
        # Priority: env var > default
        if quantization == "4bit":
            default_embedder = self._DEFAULT_EMBEDDER_MODEL_4BIT
            default_reranker = self._DEFAULT_RERANKER_MODEL_4BIT
        else:
            default_embedder = self._DEFAULT_EMBEDDER_MODEL_BF16
            default_reranker = self._DEFAULT_RERANKER_MODEL_BF16

        self.EMBEDDER_MODEL = os.environ.get(
            "RECALLFORGE_EMBEDDER_MODEL", default_embedder
        )
        self.RERANKER_MODEL = os.environ.get(
            "RECALLFORGE_RERANKER_MODEL", default_reranker
        )
        self.CAPTION_MODEL = os.environ.get(
            "RECALLFORGE_CAPTIONER_MODEL", self._DEFAULT_CAPTION_MODEL
        )
        self._VIDEO_SAMPLE_FPS = self._resolve_positive_float_env(
            "RECALLFORGE_MLX_VIDEO_SAMPLE_FPS",
            self._VIDEO_SAMPLE_FPS,
        )
        self._VIDEO_MAX_FRAMES = self._resolve_positive_int_env(
            "RECALLFORGE_MLX_VIDEO_MAX_FRAMES",
            self._VIDEO_MAX_FRAMES,
        )
        self._VIDEO_FALLBACK_MAX_FRAMES = min(
            self._VIDEO_MAX_FRAMES,
            self._resolve_positive_int_env(
                "RECALLFORGE_MLX_VIDEO_FALLBACK_MAX_FRAMES",
                self._VIDEO_FALLBACK_MAX_FRAMES,
            ),
        )

    def _resolve_heavy_op_concurrency(self) -> int:
        """Return the configured MLX heavy-op concurrency ceiling."""
        raw = os.environ.get(
            "RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY",
            str(self._DEFAULT_HEAVY_OP_CONCURRENCY),
        ).strip()
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "Invalid RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY=%r; using %d",
                raw,
                self._DEFAULT_HEAVY_OP_CONCURRENCY,
            )
            return self._DEFAULT_HEAVY_OP_CONCURRENCY
        return max(1, value)

    def _resolve_positive_int_env(self, name: str, default: int) -> int:
        """Read a positive integer env var with graceful fallback."""
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw.strip())
        except ValueError:
            logger.warning("Invalid %s=%r; using %d", name, raw, default)
            return default
        return value if value > 0 else default

    def _resolve_positive_float_env(self, name: str, default: float) -> float:
        """Read a positive float env var with graceful fallback."""
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = float(raw.strip())
        except ValueError:
            logger.warning("Invalid %s=%r; using %.3f", name, raw, default)
            return default
        return value if value > 0 else default

    def _get_heavy_op_gate(self) -> _HeavyOpGate:
        """Return the shared gate used to limit overlapping MLX heavy ops."""
        global _HEAVY_OP_GATE, _HEAVY_OP_GATE_LIMIT

        limit = self._resolve_heavy_op_concurrency()
        with _HEAVY_OP_GATE_INIT_LOCK:
            if _HEAVY_OP_GATE is None or _HEAVY_OP_GATE_LIMIT != limit:
                _HEAVY_OP_GATE = _HeavyOpGate(limit)
                _HEAVY_OP_GATE_LIMIT = limit
        return _HEAVY_OP_GATE

    @contextmanager
    def _hold_heavy_op(self, op_name: str):
        """Serialize the heaviest MLX operations for local safety."""
        with self._get_heavy_op_gate().hold(op_name):
            yield

    # =========================================================================
    # Embedder
    # =========================================================================

    def _load_embedder(self):
        """Lazy-load the MLX embedding model with explicit failure context."""
        if self._embedder_model is not None:
            return
        with self._model_lock:
            # Double-check after acquiring lock
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

    def _apply_chat_template(
        self,
        processor: Any,
        messages: List[Dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> Any:
        """Apply a chat template with fallback to tokenizer templates.

        Some current MLX/Qwen processor objects expose a tokenizer chat template
        but raise when `processor.apply_chat_template(...)` is called directly.
        Fall back to the tokenizer so the live MLX path remains compatible.
        """
        apply_template = getattr(processor, "apply_chat_template", None)
        if callable(apply_template):
            try:
                return apply_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                )
            except ValueError as exc:
                if "does not have a chat template" not in str(exc):
                    raise
            except AttributeError:
                pass

        tokenizer = getattr(processor, "tokenizer", None)
        tokenizer_apply = getattr(tokenizer, "apply_chat_template", None)
        if callable(tokenizer_apply):
            return tokenizer_apply(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )

        raise ValueError("Processor does not expose a usable chat template.")

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
        with self._hold_heavy_op("embed_images"):
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
                    self._apply_chat_template(
                        self._embedder_processor,
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

            # Free PyTorch tensors and vision intermediates now that we have MLX arrays
            del inputs, image_inputs, chat_texts, messages_batch
            gc.collect()

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

    def _video_content(self, path: str) -> dict:
        """Build a video content dict with adaptive frame sampling.

        Uses fps-based sampling (1 frame/sec) so longer videos get more frames.
        Caps at _VIDEO_MAX_FRAMES to bound memory on very long videos.
        A 30s video → 30 frames. A 10min video → 128 frames (capped).
        """
        return {"type": "video", "video": path, "fps": self._VIDEO_SAMPLE_FPS,
                "max_frames": self._VIDEO_MAX_FRAMES}

    def embed_video(self, video_path: str) -> np.ndarray:
        """Embed a single video."""
        return self.embed_videos([video_path])[0]

    # Captioning & Generation configuration
    _CAPTION_MAX_TOKENS = 60
    _CAPTION_PROMPT = "Describe this image in one concise sentence for search indexing. No thinking."

    def _load_captioner(self) -> None:
        """Lazily load the captioning model (Qwen3.5-0.8B)."""
        if getattr(self, "_captioner_model", None) is not None:
            return
        with self._model_lock:
            # Double-check after acquiring lock
            if getattr(self, "_captioner_model", None) is not None:
                return
            from mlx_vlm import load as vlm_load

            logger.debug("captioner_load model=%s", self.CAPTION_MODEL)
            self._captioner_model, self._captioner_processor = vlm_load(
                self.CAPTION_MODEL
            )

    def _unload_captioner(self) -> None:
        """Free captioner memory when no longer needed."""
        if getattr(self, "_captioner_model", None) is not None:
            del self._captioner_model
            del self._captioner_processor
            self._captioner_model = None
            self._captioner_processor = None
            import gc
            gc.collect()

    def _format_caption_prompt(self, image_path: str) -> str:
        """Build a chat-template prompt with vision tokens for captioning."""
        import os
        abs_path = os.path.abspath(image_path)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{abs_path}"},
                {"type": "text", "text": self._CAPTION_PROMPT},
            ]}
        ]
        try:
            return self._apply_chat_template(
                self._captioner_processor,
                messages, tokenize=False, add_generation_prompt=True
            )
        finally:
            del messages
            gc.collect()

    def caption_image(self, image_path: str) -> str:
        """Generate a one-sentence image caption using Qwen3.5-0.8B.

        The captioning model is loaded lazily on first use and kept in memory
        for the duration of the ingest batch.  Call _unload_captioner() after
        batch completion to reclaim ~0.9 GB.
        """
        with self._hold_heavy_op("caption_image"):
            try:
                self._load_captioner()
                from mlx_vlm import generate as vlm_generate

                prompt = self._format_caption_prompt(image_path)
                output = vlm_generate(
                    self._captioner_model,
                    self._captioner_processor,
                    prompt=prompt,
                    image=[image_path],
                    max_tokens=self._CAPTION_MAX_TOKENS,
                )
                text = output.text if hasattr(output, "text") else str(output)
                caption = re.sub(r"\s+", " ", text).strip()
                logger.debug(
                    "caption_image path=%s caption_len=%d caption=%s",
                    image_path, len(caption), caption[:100],
                )
                return caption[:512]  # Safety cap for BM25 field length
            except Exception as exc:
                logger.warning("caption_image failed for %s: %s", image_path, exc)
                return ""
            finally:
                try:
                    del prompt, output, text, caption
                except Exception:
                    pass
                gc.collect()

    def describe_image(self, image_path: str) -> str:
        """Backward-compatible alias for caption_image."""
        return self.caption_image(image_path)

    def describe_video(self, video_path: str, frame_paths: Optional[List[str]] = None) -> str:
        """Describe a video by captioning its keyframes.

        Generates captions for up to 3 keyframes and merges them.
        Falls back to empty string if no frames are available.
        """
        if not frame_paths:
            return ""

        parts: List[str] = []
        for frame_path in frame_paths[:3]:
            try:
                caption = self.caption_image(frame_path)
                if caption:
                    parts.append(caption)
            except Exception:
                continue

        return " ".join(parts).strip()[:420] if parts else ""

    def generate_text(self, prompt: str, max_tokens: int = 60) -> str:
        """Generate text using the captioner model (Qwen3.5-0.8B).

        Used for query expansion and other lightweight generation tasks.
        Text-only — no image input.
        """
        with self._hold_heavy_op("generate_text"):
            try:
                self._load_captioner()
                from mlx_vlm import generate as vlm_generate

                messages = [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                ]}]
                formatted = self._apply_chat_template(
                    self._captioner_processor,
                    messages, tokenize=False, add_generation_prompt=True
                )
                output = vlm_generate(
                    self._captioner_model,
                    self._captioner_processor,
                    prompt=formatted,
                    max_tokens=max_tokens,
                )
                text = output.text if hasattr(output, "text") else str(output)
                return text.strip()
            except Exception as exc:
                logger.warning("generate_text failed: %s", exc)
                return ""
            finally:
                try:
                    del messages, formatted, output, text
                except Exception:
                    pass
                gc.collect()

    def embed_videos(self, video_paths: List[str]) -> np.ndarray:
        """
        Embed multiple videos using the MLX video pipeline.

        Qwen3-VL's video processor currently expects per-video sampling kwargs,
        so we process each video independently and stack the resulting vectors.
        """
        with self._hold_heavy_op("embed_videos"):
            video_paths = self._validate_video_paths(video_paths)
            if not video_paths:
                return np.empty((0, self._EMBED_DIM), dtype=np.float32)

            self._load_embedder()
            num_layers = self._get_embedder_num_layers()

            embeddings: List[np.ndarray] = []
            for path in video_paths:
                try:
                    embedding = self._embed_video_native(path, num_layers)
                except Exception as exc:
                    logger.warning(
                        "native_video_embedding failed for %s: %s; falling back to frame embeddings",
                        path,
                        exc,
                    )
                    embedding = self._embed_video_via_frames(path)

                embeddings.append(embedding)

            return np.stack(embeddings).astype(np.float32)

    def _embed_video_native(self, path: str, num_layers: int) -> np.ndarray:
        """Embed a video via qwen-vl-utils native video preprocessing."""
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise MLXEmbeddingError(
                "qwen-vl-utils vision dependencies are missing. "
                "Install qwen-vl-utils and torchvision for video embeddings."
            ) from exc

        messages = [{
            "role": "user",
            "content": [
                self._video_content(path),
                {"type": "text", "text": "Describe this video."},
            ],
        }]

        try:
            chat_text = self._apply_chat_template(
                self._embedder_processor,
                messages,
                tokenize=False,
                add_generation_prompt=True,
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

        del inputs, video_inputs, messages, chat_text, normalized_video_kwargs, video_kwargs
        gc.collect()

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
        finally:
            del input_ids, pixel_values, video_grid_thw, cache, h

        return embedding[0]

    def _embed_video_via_frames(self, path: str) -> np.ndarray:
        """Fallback raw-video embedding by averaging ffmpeg-extracted frame vectors."""
        from ..video import extract_video_frames

        with tempfile.TemporaryDirectory(prefix="recallforge_video_query_") as temp_dir:
            frames, _ = extract_video_frames(
                path,
                temp_dir,
                logical_path=os.path.basename(path),
                frame_interval_seconds=5.0,
                max_frames=self._VIDEO_FALLBACK_MAX_FRAMES,
            )
            frame_paths = [frame.image_path for frame in frames]
            if not frame_paths:
                raise MLXEmbeddingError(
                    f"Video fallback produced no frames for '{path}'. Ensure ffmpeg/ffprobe are installed."
                )

            frame_embeddings = self.embed_images(frame_paths)
            if frame_embeddings.size == 0:
                raise MLXEmbeddingError(
                    f"Video fallback frame embeddings were empty for '{path}'."
                )

            pooled = frame_embeddings.mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            if norm > 0:
                pooled = pooled / norm
            return pooled.astype(np.float32)

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

    def _render_reranker_prompt(
        self,
        messages: List[Dict[str, Any]],
        query: str,
        document: str,
        instruction: str,
    ) -> tuple[str, bool]:
        """Render reranker chat messages to text and report whether vision tokens survived."""
        try:
            return (
                self._apply_chat_template(
                    self._reranker_processor,
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                True,
            )
        except Exception:
            # Keep a manual fallback so reranking still works even if template API differs.
            return (
                (
                    f"<|im_start|>system\n{self._RERANK_SYSTEM}<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"<Instruct>: {instruction}\n<Query>:\n{query or 'NULL'}\n"
                    f"<Document>:\n{document or 'NULL'}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
                False,
            )

    def _format_reranker_prompt(
        self, query: str, document: str, instruction: str,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        query_image_path: Optional[str] = None,
        query_video_path: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format a query-document pair as chat input for reranking.

        Document-side media (image_path / video_path):
            The document being scored contains visual content.  It is embedded in
            the <Document> section so the reranker can cross-encode text against it.

        Query-side media (query_image_path / query_video_path):
            The *search query itself* is an image or video.  It is embedded in the
            <Query> section.  The reranker cross-encodes the visual query against
            each document's content.  This is the fix for REC-138/REC-139/REC-150:
            previously the literal string "image_query:/path/to/file.png" was used,
            which made scores meaningless.
        """
        messages = messages or self._build_reranker_messages(
            query,
            document,
            instruction,
            image_path=image_path,
            video_path=video_path,
            query_image_path=query_image_path,
            query_video_path=query_video_path,
        )

        prompt, _ = self._render_reranker_prompt(messages, query, document, instruction)
        return prompt

    def _as_file_uri(self, path: str) -> str:
        """Normalize local media paths for Qwen chat-template vision blocks."""
        if path.startswith("file://"):
            return path
        return f"file://{os.path.abspath(path)}"

    def _build_reranker_messages(
        self,
        query: str,
        document: str,
        instruction: str,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        query_image_path: Optional[str] = None,
        query_video_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the multimodal chat messages used for reranker prompting."""
        query_content: list = [{"type": "text", "text": "<Query>:"}]
        if query_image_path:
            query_content.append({"type": "image", "image": self._as_file_uri(query_image_path)})
            if query:
                query_content.append({"type": "text", "text": query})
        elif query_video_path:
            query_content.append(self._video_content(self._as_file_uri(query_video_path)))
            if query:
                query_content.append({"type": "text", "text": query})
        else:
            query_content.append({"type": "text", "text": query or "NULL"})

        doc_content: list = [{"type": "text", "text": "\n<Document>:"}]
        if image_path:
            doc_content.append({"type": "image", "image": self._as_file_uri(image_path)})
            if document:
                doc_content.append({"type": "text", "text": document})
        elif video_path:
            doc_content.append(self._video_content(self._as_file_uri(video_path)))
            if document:
                doc_content.append({"type": "text", "text": document})
        else:
            doc_content.append({"type": "text", "text": document or "NULL"})

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._RERANK_SYSTEM}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<Instruct>: {instruction}"},
                    *query_content,
                    *doc_content,
                ],
            },
        ]

    def _build_reranker_processor_inputs(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build reranker processor inputs from the same message structure as the prompt."""
        if not messages:
            return self._reranker_processor(text=prompt, return_tensors="np")

        user_messages = [m for m in messages if m.get("role") == "user"]
        content_blocks = []
        for message in user_messages:
            content_blocks.extend(message.get("content", []))

        has_vision = any(
            block.get("type") in {"image", "video"}
            for block in content_blocks
            if isinstance(block, dict)
        )
        if not has_vision:
            return self._reranker_processor(text=prompt, return_tensors="np")

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [user_messages],
            return_video_kwargs=True,
        )
        normalized_video_kwargs = dict(video_kwargs or {})
        fps = normalized_video_kwargs.get("fps")
        if isinstance(fps, list):
            normalized_video_kwargs["fps"] = fps[0] if fps else None

        proc_kwargs = {"text": [prompt], "return_tensors": "pt", "padding": True}
        if image_inputs:
            proc_kwargs["images"] = image_inputs
        if video_inputs:
            proc_kwargs["videos"] = video_inputs
            proc_kwargs.update(normalized_video_kwargs)

        pt_inputs = self._reranker_processor(**proc_kwargs)
        return {
            key: value.numpy() if hasattr(value, "numpy") else value
            for key, value in pt_inputs.items()
        }

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

    def _score_reranker_prompt(
        self, prompt: str, num_layers: int,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[float, str, float]:
        """Run one query-document pair and return (score, scoring_path, raw_score_before_sigmoid).

        When multimodal messages are supplied, the same message structure is used to
        build both the chat template and the processor vision inputs. This keeps the
        number and order of vision tokens aligned with the extracted pixel features.

        Returns:
            Tuple of (sigmoid_score, scoring_path, raw_score_before_sigmoid)
        """
        scoring_path = "unknown"
        vl_fallback_triggered = False
        raw_score_before_sigmoid = 0.0

        try:
            try:
                inputs = self._build_reranker_processor_inputs(prompt, messages)
            except Exception as e:
                logger.warning(
                    f"[MLXBackend] Failed to build multimodal reranker inputs: {e}; "
                    "falling back to text-only"
                )
                vl_fallback_triggered = True
                inputs = self._reranker_processor(text=prompt, return_tensors="np")

            input_ids = mx.array(inputs["input_ids"])

            # Check for vision inputs (image or video reranking)
            has_image = "pixel_values" in inputs and "image_grid_thw" in inputs
            has_video = "pixel_values_videos" in inputs and "video_grid_thw" in inputs
            has_vision = has_image or has_video
            pixel_values = None
            image_grid_thw = None
            video_grid_thw = None
            if has_image:
                pixel_values = self._to_mx_array(inputs["pixel_values"], "pixel_values")
                image_grid_thw = self._to_mx_array(inputs["image_grid_thw"], "image_grid_thw")
            if has_video:
                # When both image and video are present (e.g., query_image + doc_video),
                # video pixels go into a separate tensor. get_input_embeddings handles both.
                video_pixel_values = self._to_mx_array(inputs["pixel_values_videos"], "pixel_values_videos")
                video_grid_thw = self._to_mx_array(inputs["video_grid_thw"], "video_grid_thw")
                if pixel_values is None:
                    # Video-only: use video pixels as the primary pixel_values
                    pixel_values = video_pixel_values
                # else: both are set — get_input_embeddings receives pixel_values for
                # images and video_grid_thw for videos separately

            # Free processor outputs after conversion to MLX arrays
            del inputs
            gc.collect()

            cache = _make_cache(num_layers)

            if has_vision:
                # VL path: merge vision features with text embeddings.
                # Must use direct yes/no logit comparison instead of the derived
                # weight projection.  The derived projection (lm_head[yes] - lm_head[no])
                # is a linear shortcut that assumes text-only hidden-state distributions.
                # Vision features shift the distribution, making the shortcut unreliable.
                if self._reranker_yes_token_id is None or self._reranker_no_token_id is None:
                    raise RuntimeError(
                        "yes/no token ids are unavailable; cannot score VL reranker inputs"
                    )
                try:
                    emb_features = self._reranker_model.get_input_embeddings(
                        input_ids, pixel_values,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                    )
                    inputs_embeds = emb_features.to_dict()["inputs_embeds"]
                    # language_model.__call__ accesses inputs.shape for mask validation,
                    # so pass a dummy input_ids matching the sequence length to avoid
                    # NoneType.shape crash when inputs_embeds is the real input.
                    dummy_ids = mx.zeros((1, inputs_embeds.shape[1]), dtype=mx.int32)
                    lm_out = self._reranker_model.language_model(
                        dummy_ids, inputs_embeds=inputs_embeds, cache=cache,
                    )
                    full_logits = self._to_mx_array(
                        getattr(lm_out, "logits", lm_out)
                    ).astype(mx.float32)
                    last_logits = full_logits[:, -1, :]
                    logits = (
                        last_logits[:, self._reranker_yes_token_id]
                        - last_logits[:, self._reranker_no_token_id]
                    ).reshape(-1, 1)
                    raw_score_before_sigmoid = float(np.array(logits).reshape(-1)[0])
                    scoring_path = "vl_image" if has_image else "vl_video"
                except Exception as e:
                    logger.warning(f"[MLXBackend] VL reranking failed, falling back to text-only: {e}")
                    vl_fallback_triggered = True
                    # Fall back to text-only path
                    qwen_model = self._reranker_model.language_model.model
                    hidden = qwen_model(input_ids, cache=cache).astype(mx.float32)
                    last_hidden = hidden[:, -1, :]
                    logits = self._apply_reranker_linear(last_hidden)
                    raw_score_before_sigmoid = float(np.array(logits).reshape(-1)[0])
                    scoring_path = "fallback_text"
            elif self._reranker_use_direct_logits:
                if self._reranker_yes_token_id is None or self._reranker_no_token_id is None:
                    raise RuntimeError("yes/no token ids are unavailable for reranker fallback")
                lm_out = self._reranker_model.language_model(input_ids, cache=cache)
                full_logits = self._to_mx_array(getattr(lm_out, "logits", lm_out)).astype(mx.float32)
                last_logits = full_logits[:, -1, :]
                logits = (
                    last_logits[:, self._reranker_yes_token_id]
                    - last_logits[:, self._reranker_no_token_id]
                ).reshape(-1, 1)
                raw_score_before_sigmoid = float(np.array(logits).reshape(-1)[0])
                scoring_path = "text_direct_logits"
            else:
                qwen_model = self._reranker_model.language_model.model
                hidden = qwen_model(input_ids, cache=cache).astype(mx.float32)
                last_hidden = hidden[:, -1, :]
                logits = self._apply_reranker_linear(last_hidden)
                raw_score_before_sigmoid = float(np.array(logits).reshape(-1)[0])
                scoring_path = "text_derived"

            probs = 1.0 / (1.0 + mx.exp(-logits))
            mx.eval(probs)
            score = float(np.array(probs).reshape(-1)[0])
            if not np.isfinite(score):
                raise RuntimeError("Reranker produced non-finite score")

            # Log reranker path tracing
            logger.debug("reranker_score path=%s raw_score=%.4f final_score=%.4f vl_fallback=%s",
                         scoring_path, raw_score_before_sigmoid, score, vl_fallback_triggered)

            return score, scoring_path, raw_score_before_sigmoid
        finally:
            try:
                del input_ids, pixel_values, image_grid_thw, video_grid_thw
            except Exception:
                pass
            try:
                del cache, logits, probs
            except Exception:
                pass
            gc.collect()

    def _load_reranker(self):
        """Lazy-load the MLX reranker model."""
        if self._reranker_model is not None:
            return
        with self._model_lock:
            # Double-check after acquiring lock
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

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_image_path: Optional[str] = None,
        query_video_path: Optional[str] = None,
    ) -> List[float]:
        """Rerank documents for a query.

        Args:
            query: Text query.  May be empty when query_image_path/query_video_path
                is set; the media is then used directly as the query side of the
                cross-encoder prompt.
            documents: Document dicts (with 'text', optional 'image_path'/'video_path').
            query_image_path: Path to the query image for image-query searches.
            query_video_path: Path to the query video for video-query searches.

        Returns list of scores. Scoring path information is logged via logger.debug.
        """
        with self._hold_heavy_op("rerank"):
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
                    doc_image_path = doc.get("image_path")
                    doc_video_path = doc.get("video_path")
                    messages = self._build_reranker_messages(
                        query,
                        text,
                        instruction,
                        image_path=doc_image_path, video_path=doc_video_path,
                        query_image_path=query_image_path,
                        query_video_path=query_video_path,
                    )
                    prompt, template_ok = self._render_reranker_prompt(
                        messages,
                        query,
                        text,
                        instruction,
                    )
                    score, scoring_path, raw_score = self._score_reranker_prompt(
                        prompt, num_layers,
                        messages=messages if template_ok else None,
                    )
                    scores.append(score)
                    logger.debug(
                        "reranker_doc idx=%d path=%s raw_score=%.4f final_score=%.4f content_type=%s",
                        idx, scoring_path, raw_score, score, doc.get("content_type", "unknown")
                    )
                except Exception as e:
                    logger.error(f"[MLXBackend] Rerank error at doc {idx}: {e}")
                    scores.append(0.5)
                    logger.debug(
                        "reranker_doc idx=%d path=error_fallback raw_score=0.0 final_score=0.5 content_type=%s",
                        idx, doc.get("content_type", "unknown")
                    )

            return scores

    # =========================================================================
    # Warm-up and Status
    # =========================================================================

    def warm_up(self) -> None:
        """Preload models and run a dummy embed pass to prime MLX compilation."""
        import time

        with self._hold_heavy_op("warm_up"):
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
        if mode not in ("embed", "hybrid"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'embed' or 'hybrid'")
        self._mode = mode

    def get_mode(self) -> str:
        """Get current search mode."""
        return self._mode

    def get_model_ids(self) -> Dict[str, str]:
        """Return current model IDs for embedder, reranker, and captioner."""
        return {
            "embedder_model": self.EMBEDDER_MODEL,
            "reranker_model": self.RERANKER_MODEL,
            "captioner_model": self.CAPTION_MODEL,
        }

    def set_model_ids(
        self,
        embedder_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        captioner_model: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Update model IDs and unload cached models so they reload on next use.

        Args:
            embedder_model: New embedder model ID (optional)
            reranker_model: New reranker model ID (optional)
            captioner_model: New captioner model ID (optional)

        Returns:
            Dict with the updated model IDs
        """
        with self._model_lock:
            changed = []
            if embedder_model is not None and embedder_model != self.EMBEDDER_MODEL:
                self.EMBEDDER_MODEL = embedder_model
                old = self._embedder_model
                self._embedder_model = None
                self._embedder_processor = None
                self._embedder_num_layers = None
                self._embed_text_max_tokens = None
                self._embed_warmed = False
                del old
                changed.append("embedder")

            if reranker_model is not None and reranker_model != self.RERANKER_MODEL:
                self.RERANKER_MODEL = reranker_model
                old = self._reranker_model
                self._reranker_model = None
                self._reranker_processor = None
                self._reranker_score_linear = None
                self._reranker_score_weight = None
                self._reranker_score_bias = None
                self._reranker_yes_token_id = None
                self._reranker_no_token_id = None
                self._reranker_use_direct_logits = False
                del old
                changed.append("reranker")

            if captioner_model is not None and captioner_model != self.CAPTION_MODEL:
                self.CAPTION_MODEL = captioner_model
                self._unload_captioner()
                changed.append("captioner")

            if changed:
                gc.collect()
                logger.info(
                    "model_swap changed=%s", ", ".join(changed)
                )

        return self.get_model_ids()
