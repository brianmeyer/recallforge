# Vision-Language Embedding Models Survey for RecallForge
## Local Multimodal Retrieval on Apple Silicon (16GB M4) - 2025-2026

**Research Date:** March 15, 2026  
**Current Stack:** Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B  
**Target:** Find best embedder + reranker combo fitting in ~8GB total on 16GB Apple Silicon

---

## Executive Summary

For RecallForge's use case (local-first multimodal memory search on 16GB Apple Silicon), **Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B remains the best choice** as of March 2026. However, several alternatives exist with different tradeoffs:

| Model | Params | Embed Dim | License | MLX Ready | Est. 4-bit Memory |
|-------|--------|-----------|---------|-----------|-------------------|
| **Qwen3-VL-Embedding-2B** | 2B | 2048 | Apache 2.0 | ✅ Yes | ~1.2GB |
| **Qwen3-VL-Reranker-2B** | 2B | N/A | Apache 2.0 | ✅ Yes | ~1.2GB |
| jina-clip-v2 | 2B | 1024 | Apache 2.0 | ⚠️ Partial | ~1.2GB |
| nomic-embed-vision | 435M | 768 | Apache 2.0 | ✅ Yes | ~250MB |
| ColPali v1.3 | 3B (PaliGemma) | 128-d per patch | Gemma | ⚠️ Via PyTorch | ~2-3GB |
| SigLIP2 Base | 86M | 768 | Apache 2.0 | ✅ Yes | ~150MB |
| gme-Qwen2-VL-2B | 2B | 2048 | Apache 2.0 | ✅ Yes | ~1.2GB |

---

## 1. Qwen Family (Current + Alternatives)

### Qwen3-VL-Embedding-2B (CURRENT)
- **Parameters:** 2B
- **Embedding Dimension:** 2048
- **License:** Apache 2.0
- **Modalities:** Text, images, screenshots, videos, arbitrary multimodal combinations
- **Context Length:** 32K tokens
- **MTEB/MMEB-v2 Scores:**
  - MMEB-v2 (Retrieval) Avg: 73.4
  - MMEB-v2 Image: 74.8
  - MMEB-v2 Video: 53.6
  - MMEB-v2 VisDoc: 79.2
  - MMTEB (Retrieval): 68.1
  - JinaVDR: 71.0
  - ViDoRe v3: 52.9
- **MLX Support:** ✅ Full - `mlx-community/Qwen3-VL-Embedding-2B-4bit` available
- **Memory Estimate (4-bit):** ~1.2GB
- **Features:**
  - MRL (Matryoshka Representation Learning) support - can truncate to smaller dims
  - Instruction-aware (customizable for specific tasks)
  - Quantization support for output embeddings
  - 30+ languages

### Qwen3-VL-Embedding-8B
- **Parameters:** 8B
- **Embedding Dimension:** 4096
- **License:** Apache 2.0
- **MTEB Scores:** Higher than 2B across all benchmarks
- **Memory Estimate (4-bit):** ~4.5GB
- **Verdict:** Too large for 8GB embedder+reranker budget

### Qwen3-VL-Reranker-2B (CURRENT)
- **Parameters:** 2B
- **License:** Apache 2.0
- **Modalities:** Text, images, screenshots, videos, arbitrary multimodal combinations
- **Context Length:** 32K tokens
- **Performance:**
  - MMEB-v2 (Retrieval) Avg: 75.1 (vs 73.4 for embedder alone)
  - ViDoRe v3: 60.8 (vs 52.9 for embedder alone)
  - JinaVDR: 80.9 (vs 71.0 for embedder alone)
- **MLX Support:** ✅ Available via mlx-vlm
- **Memory Estimate (4-bit):** ~1.2GB

### Qwen3-VL-Reranker-8B
- **Parameters:** 8B
- **Performance:** Best scores (79.2 MMEB-v2 avg, 66.7 ViDoRe v3)
- **Memory Estimate (4-bit):** ~4.5GB
- **Verdict:** Too large for 8GB budget

### GME-Qwen2-VL Series (Alibaba-NLP)
- **Models:** gme-Qwen2-VL-2B-Instruct, gme-Qwen2-VL-7B-Instruct
- **Embedding Dimension:** 2048 (2B), 4096 (7B)
- **License:** Apache 2.0
- **Features:** Generalized Multimodal Embedding based on Qwen2-VL
- **MTEB Scores:** Competitive but slightly lower than Qwen3-VL-Embedding
- **Verdict:** Qwen3-VL supersedes these

---

## 2. NVIDIA Nemotron Family

### NV-Embed-v2
- **Parameters:** Not publicly specified (likely 7B+ class)
- **Embedding Dimension:** 4096
- **License:** NVIDIA Open Model License (commercial use allowed)
- **Modalities:** Text-only (NOT multimodal)
- **MTEB Score:** 72.3 (text retrieval)
- **Verdict:** ❌ Not suitable - text-only, not vision-language

**Note:** NVIDIA has not released a vision-language embedding model as of March 2026. Nemotron focuses on LLMs and text embeddings.

---

## 3. Jina AI Models

### jina-clip-v2
- **Parameters:** 2B total (ViT-L/14 + XLM-RoBERTa-large)
- **Embedding Dimension:** 1024
- **License:** Apache 2.0
- **Modalities:** Text, images
- **MTEB Scores:**
  - Image-to-Text: 56.5
  - Text-to-Image: 56.5
  - Strong multilingual performance (100+ languages)
- **MLX Support:** ⚠️ Partial - requires custom conversion
- **Memory Estimate (4-bit):** ~1.2GB
- **Features:**
  - True CLIP-style joint embedding space
  - Excellent multilingual support
  - Good for cross-lingual retrieval
- **Verdict:** Good alternative if multilingual is priority, but lower vision performance than Qwen3-VL

### jina-reranker-m0 (Multimodal)
- **Parameters:** 2.4B (based on Qwen2-VL-2B)
- **License:** Apache 2.0
- **Modalities:** Text, images, visual documents
- **Context Length:** 10,240 tokens
- **Architecture:** Decoder-only VLM with LoRA fine-tuning
- **MLX Support:** ⚠️ Requires custom implementation
- **Memory Estimate (4-bit):** ~1.4GB
- **Features:**
  - Dynamic image resolution (56x56 to 4K)
  - 29+ languages
  - Code search optimized
  - Can rerank both text and image documents
- **Verdict:** Strong alternative reranker, but Qwen3-VL-Reranker has better MLX support

---

## 4. Nomic AI

### nomic-embed-vision-v1.5
- **Parameters:** 435M
- **Embedding Dimension:** 768
- **License:** Apache 2.0
- **Modalities:** Images (paired with nomic-embed-text for multimodal)
- **MTEB/ViDoRe:** Not specifically benchmarked on standard multimodal retrieval
- **MLX Support:** ✅ Available
- **Memory Estimate (4-bit):** ~250MB
- **Features:**
  - Designed to be compatible with nomic-embed-text-v1.5
  - Shared embedding space with text model
- **Limitations:**
  - Requires separate text model for multimodal
  - Not a true unified VLM architecture
- **Verdict:** Good for size-constrained scenarios, but not a drop-in replacement

---

## 5. BAAI BGE Family

### BGE-M3
- **Parameters:** 568M (XLM-RoBERTa-large based)
- **Embedding Dimension:** 1024
- **License:** MIT
- **Modalities:** Text-only (NOT multimodal)
- **Features:**
  - Dense + Sparse + ColBERT multi-vector retrieval
  - 100+ languages
  - 8192 token context
- **Verdict:** ❌ Not suitable - text-only

### BGE-Visualized (bge-visualized)
- **Status:** Limited documentation, appears to be experimental
- **Verdict:** ❌ Not production-ready for multimodal retrieval

---

## 6. ColPali / ColQwen Family (Document Retrieval Specialists)

### ColPali v1.3
- **Base Model:** PaliGemma-3B (SigLIP + Gemma-2B)
- **Architecture:** Multi-vector (ColBERT-style late interaction)
- **License:** Gemma License (base), MIT (adapters)
- **Modalities:** Document images (PDF pages), text queries
- **ViDoRe Scores:** State-of-the-art for visual document retrieval
- **Memory Estimate:** ~2-3GB (requires full VLM)
- **Features:**
  - Generates multiple vectors per document (one per image patch)
  - Late interaction scoring (memory intensive at query time)
  - Optimized for PDF document retrieval
- **MLX Support:** ⚠️ Requires PyTorch, no native MLX
- **Verdict:** Excellent for PDF-heavy workloads, but different architecture than RecallForge's current setup

### ColQwen2 v1.0 / v0.2
- **Base Model:** Qwen2-VL-2B
- **Architecture:** Multi-vector ColBERT-style
- **License:** Apache 2.0
- **Modalities:** Document images, text queries
- **ViDoRe v2 Scores:** 87.2 (ColQwen2 v0.2)
- **Memory Estimate:** ~2-3GB
- **Features:**
  - Better than ColPali on most benchmarks
  - Uses Qwen2-VL vision encoder
- **Verdict:** Strong for document retrieval, but multi-vector architecture requires different indexing

### ColQwen2.5 v1.0 / v0.2
- **Base Model:** Qwen2.5-VL-2B
- **License:** Apache 2.0
- **Features:** Latest version with improved performance
- **Verdict:** Same architectural considerations as ColQwen2

**Important Note:** ColPali/ColQwen models use **multi-vector representations** (one vector per image patch), which requires:
- Different indexing strategy (not single vector per document)
- Late interaction scoring at query time (higher compute)
- Not directly compatible with RecallForge's current single-vector architecture

---

## 7. SigLIP / SigLIP2 Family

### SigLIP2 Base (patch16-224 / patch16-512)
- **Parameters:** 86M (Base)
- **Embedding Dimension:** 768
- **License:** Apache 2.0
- **Modalities:** Text, images
- **Features:**
  - Improved semantic understanding, localization, dense features
  - Decoder loss + global-local prediction
  - Aspect ratio and resolution adaptability
- **MLX Support:** ✅ Yes
- **Memory Estimate (4-bit):** ~150MB
- **Verdict:** Good lightweight option, but lower quality than Qwen3-VL

### SigLIP2 Large
- **Parameters:** 304M
- **Embedding Dimension:** 1024
- **License:** Apache 2.0
- **Memory Estimate (4-bit):** ~400MB
- **Verdict:** Better quality/size tradeoff than Base

### SigLIP2 So400M
- **Parameters:** 400M+ (largest variant)
- **Embedding Dimension:** 1152
- **License:** Apache 2.0
- **Memory Estimate (4-bit):** ~600MB
- **Verdict:** Best SigLIP2 variant, still smaller than Qwen3-VL

**SigLIP2 vs Qwen3-VL:**
- SigLIP2: Lighter, faster, good for simple image-text matching
- Qwen3-VL: Higher quality, better reasoning, video support, document understanding

---

## 8. OpenAI CLIP (Baseline)

### CLIP ViT-B/32, ViT-B/16, ViT-L/14
- **Parameters:** 149M-400M
- **Embedding Dimension:** 512-768
- **License:** MIT
- **Modalities:** Text, images
- **MTEB:** Baseline performance, surpassed by modern models
- **MLX Support:** ✅ Yes
- **Verdict:** Legacy option, newer models significantly better

---

## 9. Reranker Options Summary

| Model | Base | Params | Modalities | MLX | Memory (4-bit) |
|-------|------|--------|------------|-----|----------------|
| **Qwen3-VL-Reranker-2B** | Qwen3-VL | 2B | Text, Image, Video | ✅ | ~1.2GB |
| **Qwen3-VL-Reranker-8B** | Qwen3-VL | 8B | Text, Image, Video | ✅ | ~4.5GB |
| jina-reranker-m0 | Qwen2-VL | 2.4B | Text, Image, Documents | ⚠️ | ~1.4GB |
| jina-reranker-v2 | XLM-R | 278M | Text only | ✅ | ~300MB |
| bge-reranker-v2 | Various | 560M+ | Text only | ✅ | ~600MB |

**Key Finding:** Qwen3-VL-Reranker-2B is the only production-ready multimodal reranker with native MLX support as of March 2026.

---

## 10. MLX Conversion Status

### Fully Supported (Native MLX Community)
- ✅ Qwen3-VL-Embedding-2B-4bit
- ✅ Qwen3-VL-Reranker-2B (via mlx-vlm)
- ✅ SigLIP2 variants
- ✅ CLIP variants
- ✅ nomic-embed-vision

### Requires Custom Conversion
- ⚠️ jina-clip-v2
- ⚠️ jina-reranker-m0
- ⚠️ ColPali/ColQwen (PyTorch only)

### Conversion Feasibility
Most PyTorch models can be converted to MLX using:
```bash
mlx_lm.convert --hf-path <model> -q --upload-repo mlx-community/<model>-4bit
```

Vision-language models require special handling via `mlx-vlm` package.

---

## 11. Memory Footprint Analysis (16GB Apple Silicon)

### Target Budget: ~8GB for embedder + reranker

| Configuration | Est. Memory | Viable? |
|---------------|-------------|---------|
| Qwen3-VL-Embed-2B (4-bit) + Qwen3-VL-Reranker-2B (4-bit) | ~2.4GB | ✅ Yes |
| Qwen3-VL-Embed-8B (4-bit) + Qwen3-VL-Reranker-2B (4-bit) | ~5.7GB | ✅ Yes |
| Qwen3-VL-Embed-8B (4-bit) + Qwen3-VL-Reranker-8B (4-bit) | ~9GB | ⚠️ Tight |
| ColPali-3B + Qwen3-VL-Reranker-2B | ~4.5GB | ✅ Yes |
| SigLIP2-Large + jina-reranker-m0 | ~1.8GB | ✅ Yes |

**Recommendation:** Current stack (Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B) uses only ~2.4GB, leaving plenty of headroom for:
- Vector database (Chroma/Milvus)
- Application memory
- System overhead

---

## 12. Model Swappability Analysis

### Current RecallForge Architecture
```python
class ModelBackend(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        ...
    
    @abstractmethod
    def embed_image(self, image: Image.Image) -> np.ndarray:
        ...
    
    @abstractmethod
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        ...
```

### Swappability Assessment

| Model | Embed Dim | Compatible? | Effort |
|-------|-----------|-------------|--------|
| Qwen3-VL-Embedding-2B | 2048 | ✅ Native | None |
| jina-clip-v2 | 1024 | ⚠️ Requires adapter | Low |
| nomic-embed-vision | 768 | ⚠️ Requires adapter | Low |
| SigLIP2 | 768/1024/1152 | ⚠️ Requires adapter | Low |
| ColPali | Multi-vector | ❌ Architecture mismatch | High |

### Embedding Dimension Compatibility
- **Current:** 2048 (Qwen3-VL-Embedding-2B)
- **Alternative models:** Mostly 768-1024
- **Impact:** Would require re-indexing all existing vectors
- **Mitigation:** Use MRL (Matryoshka) to truncate Qwen3-VL to match other dims, or maintain separate indices

### Required Changes for Model Swap
1. **Same dimension (2048):** Minimal changes - swap model path, update processor
2. **Different dimension:**
   - Re-index all documents
   - Or add projection layer to map to 2048
   - Or store dimension in metadata and handle dynamically

---

## 13. Recommendations

### Best Overall for RecallForge (Current)
**Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B**
- ✅ Best quality/size tradeoff
- ✅ Native MLX support
- ✅ Apache 2.0 license
- ✅ Unified architecture (same base model)
- ✅ MRL support for flexible dimensions
- ✅ Video support
- ✅ 30+ languages

### Alternative: Size-Constrained
**SigLIP2-So400M + jina-reranker-m0**
- ~800MB total
- Lower quality but much smaller
- Good for multiple model instances

### Alternative: Document-Heavy Workloads
**ColQwen2.5 + Qwen3-VL-Reranker-2B**
- Best for PDF/visual document retrieval
- Requires architectural changes (multi-vector)
- Higher complexity

### Alternative: Maximum Quality (if memory allows)
**Qwen3-VL-Embedding-8B + Qwen3-VL-Reranker-8B**
- ~9GB total (tight on 16GB)
- Best benchmark scores
- Consider for 32GB+ systems

---

## 14. Action Items

1. **Stay with current stack** - Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B is optimal for 16GB
2. **Monitor for:**
   - Qwen4-VL series (likely late 2026)
   - MLX-native ColPali/ColQwen implementations
   - Newer SigLIP variants
3. **Consider MRL** - Use 1024 or 512-dim truncation to save storage/index memory
4. **Evaluate jina-reranker-m0** if document-heavy use cases emerge

---

## References

- Qwen3-VL-Embedding: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
- Qwen3-VL-Reranker: https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B
- jina-clip-v2: https://huggingface.co/jinaai/jina-clip-v2
- jina-reranker-m0: https://huggingface.co/jinaai/jina-reranker-m0
- ColPali: https://huggingface.co/vidore/colpali-v1.3
- ColQwen2: https://huggingface.co/vidore/colqwen2-v1.0
- SigLIP2: https://huggingface.co/google/siglip2-base-patch16-224
- MLX Community: https://huggingface.co/mlx-community
- Technical Report: https://arxiv.org/abs/2601.04720
