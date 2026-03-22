# NVIDIA NeMo Retriever Models Research for RecallForge

**Research Date:** March 15, 2026  
**Researcher:** Subagent for RecallForge  
**Purpose:** Evaluate NVIDIA vision-language embedding models as alternatives to Qwen3-VL-Embedding-2B for local-first multimodal memory MCP server

---

## Executive Summary

NVIDIA has released a family of **ColBERT-style late interaction embedding models** under the NeMo Retriever brand. These models are specifically designed for visual document retrieval (text queries → image documents) and achieve state-of-the-art results on ViDoRe benchmarks. However, **all models use non-commercial licenses**, which may limit adoption for RecallForge.

**Key Finding:** The models are based on Qwen3-VL and Llama 3.2 architectures, making them theoretically convertible to MLX, but no official MLX ports exist yet.

---

## 1. NVIDIA Vision-Language Embedding Models

### 1.1 nemotron-colembed-vl-8b-v2

| Attribute | Value |
|-----------|-------|
| **HuggingFace URL** | https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2 |
| **Parameters** | ~8.8B |
| **Architecture** | Qwen3-VL-8B-Instruct based encoder |
| **Vision Encoder** | SigLIP2-SO-400M |
| **License** | **CC BY-NC 4.0 (Non-Commercial)** |
| **Input Types** | Text + Image |
| **Embedding Dimension** | 4096 |
| **Max Sequence Length** | 10,240 tokens |
| **Corresponding Reranker** | No dedicated reranker (ColBERT-style late interaction) |
| **ViDoRe V3 Score** | **63.54** (Rank #1 as of Jan 26, 2026) |
| **Release Date** | January 26, 2026 |

**Key Features:**
- State-of-the-art on ViDoRe V3 benchmark
- Late interaction (ColBERT-style) multi-vector representations
- Supports up to 8 image tiles + 1 thumbnail per image
- Each image tile consumes 256 tokens

---

### 1.2 nemotron-colembed-vl-4b-v2

| Attribute | Value |
|-----------|-------|
| **HuggingFace URL** | https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2 |
| **Parameters** | ~4.8B |
| **Architecture** | Qwen3-VL-4B-Instruct based encoder |
| **Vision Encoder** | SigLIP-2 (large-patch16-256) |
| **License** | **CC BY-NC 4.0 (Non-Commercial)** |
| **Input Types** | Text + Image |
| **Embedding Dimension** | 2560 |
| **Max Sequence Length** | 10,240 tokens |
| **Corresponding Reranker** | No dedicated reranker |
| **ViDoRe V3 Score** | **61.42** (Rank #3 as of Jan 26, 2026) |
| **Release Date** | January 26, 2026 |

---

### 1.3 llama-nemotron-colembed-vl-3b-v2

| Attribute | Value |
|-----------|-------|
| **HuggingFace URL** | https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2 |
| **Parameters** | ~4.4B |
| **Architecture** | SigLIP2-giant-opt-patch16-384 + Llama-3.2-3B |
| **Vision Encoder** | google/siglip2-giant-opt-patch16-384 |
| **License** | **NVIDIA Non-Commercial License + Llama 3.2 Community License** |
| **Input Types** | Text + Image |
| **Embedding Dimension** | 3072 |
| **Max Sequence Length** | 10,240 tokens |
| **Corresponding Reranker** | No dedicated reranker |
| **ViDoRe V3 Score** | **59.70** |
| **Release Date** | January 26, 2026 |

---

### 1.4 llama-nemotron-embed-vl-1b-v2

| Attribute | Value |
|-----------|-------|
| **HuggingFace URL** | https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2 |
| **Parameters** | ~1B (estimated) |
| **Architecture** | Llama-3.2-1B based |
| **Vision Encoder** | SigLIP-based |
| **License** | **NVIDIA Non-Commercial License + Llama 3.2 Community License** |
| **Input Types** | Text + Image |
| **Embedding Dimension** | TBD |
| **Max Sequence Length** | TBD |
| **Corresponding Reranker** | llama-nemotron-rerank-vl-1b-v2 |
| **ViDoRe V3 Score** | Not yet benchmarked |
| **Release Date** | January 2026 |

**Corresponding Reranker:** https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2

---

### 1.5 llama-embed-nemotron-reasoning-3b

| Attribute | Value |
|-----------|-------|
| **HuggingFace URL** | https://huggingface.co/nvidia/llama-embed-nemotron-reasoning-3b |
| **Parameters** | ~3B |
| **Architecture** | Llama-based |
| **License** | **NVIDIA Non-Commercial License** |
| **Input Types** | Text only (embedding model) |
| **Embedding Dimension** | TBD |
| **Max Sequence Length** | TBD |
| **Corresponding Reranker** | Unknown |
| **ViDoRe V3 Score** | N/A (text-only model) |
| **Purpose** | Text embedding optimized for reasoning tasks |

**Note:** This is a text-only embedding model, not vision-language.

---

## 2. MLX Compatibility Analysis

### 2.1 Current MLX Availability

**Search Results:** No official MLX ports found in mlx-community for nemotron embedding models.

**Found in mlx-community:**
- `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` - This is a text generation model, not an embedding model
- `mlx-community/NVIDIA-Nemotron-Nano-9B-v2-6bit` - Also a text generation model

**Conclusion:** No NVIDIA embedding models are currently available in MLX format.

### 2.2 MLX Conversion Feasibility

| Model | Architecture | MLX Convertible? | Notes |
|-------|-------------|------------------|-------|
| nemotron-colembed-vl-8b-v2 | Qwen3-VL-8B | **Yes** | Qwen3-VL is supported by mlx-vlm |
| nemotron-colembed-vl-4b-v2 | Qwen3-VL-4B | **Yes** | Qwen3-VL is supported by mlx-vlm |
| llama-nemotron-colembed-vl-3b-v2 | Llama-3.2-3B | **Yes** | Llama architecture fully supported in MLX |
| llama-nemotron-embed-vl-1b-v2 | Llama-3.2-1B | **Yes** | Llama architecture fully supported in MLX |
| llama-embed-nemotron-reasoning-3b | Llama-3B | **Yes** | Standard transformer, should convert |

**Conversion Path:**
1. Use `mlx-lm` or `mlx-vlm` conversion tools
2. Models use standard transformer architectures (Qwen3-VL, Llama 3.2)
3. Custom model code may require adaptation for MLX

### 2.3 Memory Requirements (Estimated)

| Model | FP16 Size | 4-bit Quantized | 16GB Device? |
|-------|-----------|-----------------|--------------|
| nemotron-colembed-vl-8b-v2 | ~17.6 GB | ~4.4 GB | ⚠️ Marginal at FP16 |
| nemotron-colembed-vl-4b-v2 | ~9.6 GB | ~2.4 GB | ✅ Yes |
| llama-nemotron-colembed-vl-3b-v2 | ~8.8 GB | ~2.2 GB | ✅ Yes |
| llama-nemotron-embed-vl-1b-v2 | ~2 GB | ~0.5 GB | ✅ Yes |
| llama-embed-nemotron-reasoning-3b | ~6 GB | ~1.5 GB | ✅ Yes |

---

## 3. Comparison with Qwen3-VL-Embedding-2B

### 3.1 Current RecallForge Model

| Attribute | Qwen3-VL-Embedding-2B |
|-----------|----------------------|
| **HuggingFace URL** | https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B |
| **Parameters** | 2B |
| **Architecture** | Qwen3-VL |
| **License** | **Apache 2.0** ✅ |
| **Input Types** | Text + Image |
| **Embedding Dimension** | 2048 |
| **Max Sequence Length** | 32,768 tokens |
| **Corresponding Reranker** | Qwen3-VL-Reranker-2B |
| **MLX Available** | ✅ Yes (mlx-community) |

### 3.2 Side-by-Side Comparison

| Model | Params | License | ViDoRe V3 | Emb Dim | MLX Ready | Memory (4-bit) |
|-------|--------|---------|-----------|---------|-----------|----------------|
| Qwen3-VL-Embedding-2B | 2B | Apache 2.0 | ~55* | 2048 | ✅ | ~1 GB |
| nemotron-colembed-vl-4b-v2 | 4.8B | **CC BY-NC** | 61.42 | 2560 | ❌ | ~2.4 GB |
| llama-nemotron-colembed-vl-3b-v2 | 4.4B | **Non-Comm** | 59.70 | 3072 | ❌ | ~2.2 GB |
| llama-nemotron-embed-vl-1b-v2 | ~1B | **Non-Comm** | TBD | TBD | ❌ | ~0.5 GB |

*Estimated based on similar models

### 3.3 Quality vs Size Tradeoff

**For 16GB Apple Silicon Device:**

1. **Best Quality:** nemotron-colembed-vl-4b-v2 (ViDoRe 61.42)
   - But: Non-commercial license, requires conversion
   
2. **Best Balance:** llama-nemotron-colembed-vl-3b-v2 (ViDoRe 59.70)
   - But: Non-commercial license, requires conversion
   
3. **Best for Production:** Qwen3-VL-Embedding-2B
   - Apache 2.0 license
   - Already available in MLX
   - Smaller memory footprint
   - Proven in RecallForge

---

## 4. NeMo Retriever GitHub Repository Analysis

### 4.1 Repository Overview

**URL:** https://github.com/NVIDIA/NeMo-Retriever

**Primary Purpose:** Document content and metadata extraction microservice (nv-ingest)

**Key Capabilities:**
- PDF, image, video, audio extraction
- Text, table, chart, infographic extraction
- OCR and contextualization
- Embedding generation
- Vector database integration (Milvus)

### 4.2 Agentic Retrieval Components

**Location:** `retrieval-bench/` and related evaluation tools

**What's Available:**
- Benchmarking framework for retrieval accuracy
- Evaluation scripts for ViDoRe benchmarks
- MTEB integration for leaderboard submissions
- Recall@k evaluation metrics

**Agentic Loop Status:**
- The "agentic retrieval" appears to be **closed-source** or part of NVIDIA's commercial offerings
- No open-source agentic loop code found in the public repository
- The repository focuses on extraction and benchmarking, not agentic RAG

### 4.3 Dependencies

**Core Dependencies:**
- Python 3.12+
- transformers>=4.57.2 (for ColEmbed v2)
- flash-attn==2.6.3
- PyTorch with CUDA support
- Ray (for distributed processing)
- Milvus (vector database)

**NIM Microservices Required:**
- yolox-graphic-elements NIM
- nemotron-parse NIM (optional, for scanned PDFs)
- Various extraction NIMs

### 4.4 Adaptability for RecallForge

**Can We Adapt It?**

| Component | Adaptable? | Notes |
|-----------|-----------|-------|
| Extraction pipeline | ⚠️ Partial | Designed for NVIDIA GPUs, CUDA-dependent |
| Benchmarking tools | ✅ Yes | Python-based, can evaluate any model |
| MTEB evaluation | ✅ Yes | Standard MTEB library integration |
| Agentic loop | ❌ No | Not open-sourced |

**Challenges:**
1. Heavy NVIDIA ecosystem dependency (CUDA, NIMs)
2. Designed for server deployment, not local-first
3. No Apple Silicon/MLX support
4. Non-commercial license restrictions

---

## 5. Recommendations for RecallForge

### 5.1 Short Term (Keep Current Stack)

**Recommendation:** Continue using **Qwen3-VL-Embedding-2B**

**Rationale:**
- ✅ Apache 2.0 license (commercial use allowed)
- ✅ Already working with MLX
- ✅ Smaller memory footprint
- ✅ Proven integration
- ✅ No conversion needed

### 5.2 Medium Term (Evaluate Alternatives)

**Option A:** Port nemotron-colembed-vl-4b-v2 to MLX
- Pros: Better quality (61.42 vs ~55)
- Cons: Non-commercial license limits usage

**Option B:** Wait for Qwen3-VL-Embedding-4B or larger
- Pros: Same architecture, likely Apache 2.0
- Cons: Not yet released

**Option C:** Hybrid approach
- Use Qwen3-VL for production
- Experiment with nemotron models for research

### 5.3 License Considerations

**Critical Issue:** All NVIDIA embedding models use **non-commercial licenses**

| Model | License | Commercial Use? |
|-------|---------|-----------------|
| nemotron-colembed-vl-8b/4b-v2 | CC BY-NC 4.0 | ❌ No |
| llama-nemotron-colembed-vl-3b-v2 | NVIDIA Non-Commercial | ❌ No |
| llama-nemotron-embed-vl-1b-v2 | NVIDIA Non-Commercial | ❌ No |
| Qwen3-VL-Embedding-2B | Apache 2.0 | ✅ Yes |

**Conclusion:** NVIDIA models are unsuitable for commercial RecallForge deployments without license negotiation.

---

## 6. Technical Notes

### 6.1 ColBERT-Style Late Interaction

The NVIDIA models use **ColBERT-style late interaction**:
- Query and document encoded separately
- Multi-vector representations per token
- Late interaction scoring at retrieval time
- More accurate but computationally heavier than single-vector embeddings

**Implications for RecallForge:**
- Requires different indexing approach
- Higher memory usage during retrieval
- Better accuracy for document retrieval

### 6.2 MLX Conversion Steps (If Needed)

```bash
# Install mlx-vlm for vision-language models
pip install mlx-vlm

# Convert Qwen3-VL based models (8b, 4b)
# Note: Custom model code may need adaptation
python -m mlx_vlm.convert --hf-path nvidia/nemotron-colembed-vl-4b-v2

# Convert Llama-based models (3b, 1b)
# Use mlx-lm for text models
pip install mlx-lm
python -m mlx_lm.convert --hf-path nvidia/llama-nemotron-colembed-vl-3b-v2
```

**Challenges:**
- Custom `AutoModel` code in NVIDIA models
- ColBERT-style output requires custom handling
- May need to extract base model weights and reimplement forward pass

---

## 7. References

1. Nemotron ColEmbed V2 Paper: https://arxiv.org/abs/2602.03992
2. NVIDIA ColEmbed V2 Blog: https://huggingface.co/blog/nvidia/nemotron-colembed-v2
3. NeMo Retriever Docs: https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/
4. ViDoRe V3 Benchmark: https://huggingface.co/blog/QuentinJG/introducing-vidore-v3
5. Qwen3-VL Technical Report: https://arxiv.org/pdf/2511.21631

---

## 8. Summary Table

| Model | Params | Arch | License | ViDoRe V3 | MLX | 16GB OK? | Commercial? |
|-------|--------|------|---------|-----------|-----|----------|---------------|
| **Qwen3-VL-Embedding-2B** | 2B | Qwen3-VL | Apache 2.0 | ~55 | ✅ | ✅ | ✅ |
| nemotron-colembed-vl-8b-v2 | 8.8B | Qwen3-VL | CC BY-NC | 63.54 | ❌ | ⚠️ | ❌ |
| nemotron-colembed-vl-4b-v2 | 4.8B | Qwen3-VL | CC BY-NC | 61.42 | ❌ | ✅ | ❌ |
| llama-nemotron-colembed-vl-3b-v2 | 4.4B | Llama-3.2 | Non-Comm | 59.70 | ❌ | ✅ | ❌ |
| llama-nemotron-embed-vl-1b-v2 | ~1B | Llama-3.2 | Non-Comm | TBD | ❌ | ✅ | ❌ |
| llama-embed-nemotron-reasoning-3b | 3B | Llama | Non-Comm | N/A | ❌ | ✅ | ❌ |

---

*Report generated for RecallForge memory system evaluation*
