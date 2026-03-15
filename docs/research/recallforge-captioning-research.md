# REC-129: Image/Video Captioning Research

**Date:** 2026-03-15  
**Status:** Resolved — root cause identified and fixed

## Problem Statement

REC-122 implemented ingest-time captioning to make images searchable via BM25, but captions were never actually generated. The feature was effectively dead code.

## Root Cause Analysis

### Why the Embedding Model Can't Caption

`Qwen3-VL-Embedding-2B` (`Qwen3VLForEmbedding`) is a **representation model** — it produces fixed-dimensional vectors (2048-d) from text/image inputs. It has no language model head for autoregressive text generation. Calling `mlx_vlm.generate()` on it fails with an architecture mismatch.

This is expected. Embedding models are fine-tuned to compress meaning into vectors, not to produce text.

### Why the Reranker Model Can't Caption

`Qwen3-VL-Reranker-2B` (`Qwen3VLForConditionalGeneration`) technically HAS a language model head — it's a full conditional generation model. However, it's fine-tuned specifically for binary relevance judgments (yes/no). When asked to caption:

```
Output: "2011, the world's first ever, the first ever, the first ever..."
```

Degenerate repetition. The model's generation distribution has collapsed to a narrow yes/no space during fine-tuning. It can process images through the vision encoder (we use this for VL reranking) but can't produce useful free-form text.

### Why the Original Code Always Failed Silently

```python
# Original code (removed)
from mlx_vlm import generate
output = generate(
    self._embedder_model,      # ← Embedding model, no LM head
    self._embedder_processor,
    prompt=prompt,             # ← Raw text string, no vision tokens
    image=image_path,
    max_tokens=50,
)
```

Two bugs stacked:

1. **Wrong model**: Used the embedder (no generation capability)
2. **Wrong prompt format**: Even if the model could generate, the raw text prompt lacks vision tokens

The `except Exception: pass` swallowed both errors, falling through to a descriptor-matching fallback that produced captions like "Image appears to show a screenshot of software, code, or UI" — too generic for BM25 to use meaningfully.

### The Critical Prompt Format Bug

This was the key finding. `mlx_vlm.generate()` requires the prompt to contain vision token placeholders:

```
<|vision_start|><|image_pad|><|vision_end|>
```

These are injected by `processor.apply_chat_template()` ONLY when the message uses the multimodal content format:

```python
# WORKS — vision tokens injected
messages = [{"role": "user", "content": [
    {"type": "image", "image": "file:///path/to/image.png"},
    {"type": "text", "text": "Describe this image."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# → "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"

# FAILS — no vision tokens
prompt = "Describe this image."
# → "Describe this image." (no <|vision_start|>...<|vision_end|>)
# generate() processes image pixels but prompt has 0 image token slots → crash
```

The error message is: `ValueError: Image features and image tokens do not match: tokens: 0, features 1064`

This same bug class would affect any Qwen3-VL usage where someone passes a raw string prompt instead of using the chat template with image content blocks.

## Solution

### Model Choice: Qwen3-VL-2B-Instruct-4bit

Same Qwen3-VL architecture family, fine-tuned for instruction following including image description. Apache 2.0 license.

| Property | Value |
|----------|-------|
| Model | `mlx-community/Qwen3-VL-2B-Instruct-4bit` |
| Size | ~1.5GB at 4-bit quantization |
| Architecture | `Qwen3VLForConditionalGeneration` |
| License | Apache 2.0 |
| Vision encoder | Same as Embedding/Reranker (shared architecture) |
| Generation quality | High — produces real descriptive captions |
| Latency | ~3.3s per image (60 tokens, M4 Mac Mini) |

### Example Caption Output

**Input:** `tests/uat/corpus/images/neural_network_diagram.png`  
**Caption:** "A wooden-framed chalkboard displays a vibrant, chaotic network of colorful, swirling lines and shapes drawn with chalk, with several chalk pieces and an eraser on the wooden ledge below."

This is a real, specific, BM25-searchable description. Compare to the previous descriptor fallback: "Image appears to show a screenshot of software, code, or UI."

### Memory Budget (16GB M4)

| Model | Purpose | Memory |
|-------|---------|--------|
| Qwen3-VL-Embedding-2B-4bit | Text/image embedding | ~1.2GB |
| Qwen3-VL-Reranker-2B-4bit | Cross-encoder reranking | ~1.2GB |
| Qwen3-VL-2B-Instruct-4bit | Captioning (lazy load) | ~1.5GB |
| **Total peak (during ingest)** | | **~3.9GB** |
| **Total (search only)** | | **~2.4GB** |

The captioner is loaded lazily on first caption request and can be unloaded after ingest completes via `_unload_captioner()`.

### Implementation

```python
def caption_image(self, image_path: str) -> str:
    self._load_captioner()  # Lazy load Qwen3-VL-2B-Instruct-4bit
    
    # Build prompt with vision tokens via chat template
    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{abs_path}"},
        {"type": "text", "text": "Describe this image in one concise sentence..."},
    ]}]
    prompt = self._captioner_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    output = vlm_generate(
        self._captioner_model, self._captioner_processor,
        prompt=prompt, image=[image_path], max_tokens=60,
    )
    return output.text[:512]
```

## Alternatives Considered

### 1. Use Qwen2.5-VL-3B-Instruct (previous gen)
- Slightly larger (3B vs 2B)
- Older architecture
- Would work but no advantage over Qwen3-VL-2B-Instruct
- **Rejected:** Same family, newer is better

### 2. Share vision encoder weights between models
- Qwen3-VL-Embedding, Reranker, and Instruct all use the same vision encoder architecture
- In theory, could load vision weights once and share
- In practice, `mlx_vlm.load()` loads the full model as a unit — splitting requires custom model loading code
- **Deferred:** Optimization for REC-130 (latency research)

### 3. Use a dedicated lightweight captioning model (BLIP-2, Florence-2)
- Smaller, faster, purpose-built for captioning
- But: different architecture, different dependencies, different quantization
- More complexity for marginal improvement
- **Rejected:** Qwen3-VL-2B-Instruct is already small enough and we already have the Qwen3-VL stack

### 4. Generate captions via external API (Claude, GPT-4V)
- Best quality but requires network, costs money, adds latency
- Defeats the purpose of local-first
- **Rejected:** RecallForge is a local tool

### 5. Use the embedding model's internal representations for captioning
- Extract attention patterns or intermediate features to derive descriptions
- Research-grade, not production-ready
- **Rejected:** Too experimental

## Key Takeaway

The Qwen3-VL family has three model types sharing the same architecture:
1. **Embedding** — produces vectors, no generation
2. **Reranker** — generates yes/no, not free text  
3. **Instruct** — full instruction-following generation

All three share the same vision encoder and can be loaded as MLX 4-bit. For RecallForge, we need all three: embedding for indexing/search, reranker for scoring, instruct for captioning. Total memory at 4-bit is ~3.9GB peak, well within 16GB budget.
