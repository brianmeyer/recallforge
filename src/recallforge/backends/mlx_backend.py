"""
mlx_backend.py - MLX Backend for RecallForge (Apple Silicon).

Uses mlx-vlm for native Apple Silicon inference with Qwen3-VL models.
Supports bf16 and 4-bit quantization (4-bit default, ~2GB memory).

Model IDs:
- MLX BF16: arthurcollet/Qwen3-VL-Embedding-2B-mlx, arthurcollet/Qwen3-VL-Reranker-2B-mlx
- MLX 4-bit: arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit, arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit
- Expander: Uses torch fallback (MLX doesn't support the expander model well)
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

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
    for embedder and reranker. Query expander falls back to torch.
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
        mode: str = "full",
        quantization: str = "4bit",
    ):
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX is not available. Install with: pip install recallforge[mlx]"
            )

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
        self._expander = None
        self._expander_tokenizer = None
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

        self.EXPANDER_MODEL = "tobil/qmd-query-expansion-qwen3.5-2B"

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

        print(f"[MLXBackend] Loading embedder: {self.EMBEDDER_MODEL}")

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
        print(f"[MLXBackend] Loaded embedder ({self._quantization})")

    def _embed_hidden(self, input_ids: "mx.array", cache) -> "mx.array":
        """
        Text pipeline (MLX): token IDs -> Qwen transformer -> hidden states.

        This is the proven path:
        processor(text=...) -> mx.array(input_ids) -> qwen_model(input_ids, cache=...).
        """
        qwen_model = self._embedder_model.language_model.model
        h = qwen_model(input_ids, cache=cache)
        return h

    def _embed_hidden_with_vision(
        self, input_ids: "mx.array", pixel_values: "mx.array",
        image_grid_thw: "mx.array", cache,
    ) -> "mx.array":
        """
        Vision pipeline (MLX): processor + process_vision_info -> input embeddings -> transformer.

        This is the proven path:
        get_input_embeddings(...) -> qwen_model(None, inputs_embeds=..., cache=...).
        """
        # Merge vision features with text embeddings
        emb_features = self._embedder_model.get_input_embeddings(
            input_ids, pixel_values, image_grid_thw=image_grid_thw,
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
            h = self._embed_hidden_with_vision(
                input_ids, pixel_values, image_grid_thw, cache,
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

        lm_head_weight = getattr(lm_head, "weight", lm_head)
        lm_head_weight = self._to_mx_array(lm_head_weight).astype(mx.float32)
        return (lm_head_weight[yes_id] - lm_head_weight[no_id]).astype(mx.float32)

    def _init_reranker_scoring(self) -> None:
        """Resolve score_linear or derive an equivalent projection from lm_head."""
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
                print("[MLXBackend] Using native score_linear reranker head")
                return
            if callable(self._reranker_score_linear):
                print("[MLXBackend] Using callable score_linear reranker head")
                return

        derived_weight = self._derive_binary_head_from_lm()
        if derived_weight is None:
            raise RuntimeError("Unable to locate reranker score head (score_linear/lm_head)")

        self._reranker_score_weight = derived_weight
        self._reranker_score_bias = None
        self._reranker_score_linear = None
        print("[MLXBackend] score_linear missing; using lm_head yes-no projection fallback")

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

        print(f"[MLXBackend] Loading reranker: {self.RERANKER_MODEL}")
        try:
            self._reranker_model, self._reranker_processor = load(
                self.RERANKER_MODEL,
                trust_remote_code=True,
            )
            self._init_reranker_scoring()
        except Exception:
            self._reranker_model = None
            self._reranker_processor = None
            self._reranker_score_linear = None
            self._reranker_score_weight = None
            self._reranker_score_bias = None
            raise
        print(f"[MLXBackend] Loaded reranker ({self._quantization})")

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
            print(f"[MLXBackend] Failed to initialize reranker: {e}")
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
                print(f"[MLXBackend] Rerank error at doc {idx}: {e}")
                scores.append(0.5)

        return scores

    # =========================================================================
    # Query Expander (Torch Fallback)
    # =========================================================================

    def _load_expander(self):
        """Load expander using torch backend (MLX doesn't support this model)."""
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
        """Preload models and run a dummy embed pass to prime MLX compilation."""
        import time

        print(f"[MLXBackend] Warming up (mode={self._mode}, quant={self._quantization})...")
        start = time.time()

        self._load_embedder()
        self._warm_embed()
        t1 = time.time()
        print(f"[MLXBackend]   Embedder+compile: {t1 - start:.1f}s")

        last_checkpoint = t1

        if self.needs_reranker():
            self._load_reranker()
            t2 = time.time()
            print(f"[MLXBackend]   Reranker: {t2 - t1:.1f}s")
            last_checkpoint = t2

        if self.needs_expander():
            self._load_expander()
            t3 = time.time()
            print(f"[MLXBackend]   Expander: {t3 - last_checkpoint:.1f}s")

        print(f"[MLXBackend] Ready in {time.time() - start:.1f}s")

    def get_info(self) -> BackendInfo:
        """Return backend information."""
        mem = 0
        if self._embedder_model:
            mem += 2000 if self._quantization == "4bit" else 4000
        if self._reranker_model:
            mem += 2000 if self._quantization == "4bit" else 4000
        if self._expander:
            mem += 4000

        return BackendInfo(
            name="mlx",
            device="mps",
            dtype=self._quantization,
            embedder_loaded=self._embedder_model is not None,
            reranker_loaded=self._reranker_model is not None,
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
