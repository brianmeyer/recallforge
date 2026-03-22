# ViDoRe v3 Benchmark Research for RecallForge

## Executive Summary

ViDoRe v3 is a comprehensive multimodal RAG benchmark designed for **document retrieval** on visually-rich enterprise documents. While valuable for document retrieval evaluation, it is **NOT ideal** for a memory MCP system because it focuses on static document corpora rather than personal/information memory retrieval. Alternative benchmarks like MRAG-Bench or MRAMG-Bench may be more relevant for multimodal memory systems.

---

## 1. What is ViDoRe v3?

### Overview
- **Paper**: arXiv:2601.08620 (January 2026)
- **Developers**: ILLUIN Technology with contributions from NVIDIA
- **Integration**: MTEB leaderboard (https://mteb-leaderboard.hf.space/?benchmark_name=ViDoRe%28v3%29)
- **License**: Commercially permissive

### Datasets (10 total, 8 public + 2 private)

| Dataset | Domain | Language | Main Modalities | Pages | Queries |
|---------|--------|----------|-----------------|-------|---------|
| French Public Company Annual Reports | Finance-FR | French | Text, Table, Charts | 2,384 | 320 |
| U.S. Public Company Annual Reports | Finance-EN | English | Text, Table | 2,942 | 309 |
| Computer Science Textbooks | Computer Science | English | Text, Infographic, Tables | 1,360 | 215 |
| HR Reports from EU | HR | English | Text, Table, Charts | 1,110 | 318 |
| French Governmental Energy Reports | Energy | French | Text, Charts | 2,229 | 308 |
| USAF Technical Orders | Industrial | English | Text, Tables, Infographics, Images | 5,244 | 283 |
| FDA Reports | Pharmaceuticals | English | Text, Charts, Images, Infographic, Tables | 2,313 | 364 |
| French Physics Lectures | Physics | French | Text, Images, Infographics | 1,674 | 302 |
| **Private Set 1** | Nuclear Energy | English | Unknown | Unknown | Unknown |
| **Private Set 2** | Telecom Standards | English | Unknown | Unknown | Unknown |

### Scale
- **Total pages**: ~26,000
- **Total queries**: 3,099 (human-verified)
- **Languages**: 6 (English, French, Spanish, German, Italian, Portuguese)
- **Human annotation effort**: 12,000 hours

### Evaluation Protocol

**Primary metric**: NDCG@10 (Normalized Discounted Cumulative Gain at rank 10)

**Query taxonomy** (7 types × 3 formats):
- **Query Types**: Open-ended, Compare-Contrast, Enumerative, Numerical, Boolean, Extractive, Multi-hop
- **Query Formats**: Question, Keyword, Instruction

**Ground truth includes**:
- Page-level relevance rankings (0-2 scale: Not Relevant, Critically Relevant, Fully Relevant)
- Bounding box annotations for evidence localization
- Written reference answers
- Modality labels (Text, Table, Chart, Infographic, Image, Mixed, Other)

---

## 2. ViDoRe v3 Pipeline Leaderboard (Top 10)

### English Evaluation Results (NDCG@10)

| Rank | Model | Size | Average | CS | Nuclear | Finance | Pharma | HR | Industrial | Telecom |
|------|-------|------|---------|-----|---------|---------|--------|-----|-----------|---------|
| 1 | **nemo-colembed-3b-v2** | 3B | **65.6%** | 77.1 | 50.7 | 64.2 | 66.0 | 62.3 | 51.7 | 69.7 |
| 2 | nemo-colembed-1b | 1B | 64.3% | 75.5 | 52.2 | 67.0 | 66.2 | 64.5 | 56.1 | 68.7 |
| 3 | jinav4 | 3B | 63.9% | 74.2 | 52.4 | 66.1 | 65.2 | 64.6 | 55.9 | 68.7 |
| 4 | colnomic-7b | 7B | 63.0% | 78.2 | 48.2 | 63.1 | 64.6 | 62.9 | 54.2 | 69.6 |
| 5 | colnomic-3b | 3B | 61.7% | 75.5 | 45.5 | 63.0 | 63.7 | 62.6 | 52.8 | 68.6 |
| 6 | colqwen2.5 | 3B | 59.2% | 75.2 | 42.9 | 61.2 | 60.9 | 59.2 | 49.4 | 65.3 |
| 7 | nomic-7b (dense) | 7B | 57.3% | 70.9 | 42.3 | 57.6 | 63.8 | 55.9 | 48.5 | 62.0 |
| 8 | colqwen2 | 2B | 56.3% | 73.5 | 44.1 | 50.9 | 58.1 | 54.7 | 49.8 | 63.2 |
| 9 | colpali-v1.3 | 7B | 53.0% | 72.5 | 38.1 | 43.3 | 57.7 | 53.3 | 47.0 | 59.2 |
| 10 | nomic-3b (dense) | 3B | 51.7% | 62.1 | 37.2 | 53.3 | 59.2 | 51.9 | 41.1 | 57.2 |

### Key Findings from Leaderboard

**Open Source vs Proprietary**:
- Top models are predominantly **open source** (ColEmbed, ColNomic, ColQwen, Jina)
- **nemo-colembed-3b-v2** leads (NVIDIA's ColEmbed, open weights)
- Proprietary models not prominently featured in paper's evaluation

**Model Architecture Insights**:
- **Late-interaction models** (ColEmbed, ColPali, ColQwen) outperform dense models
- **Visual retrievers** consistently beat text-only retrievers for visually-rich documents
- **Reranking** provides significant boost (+13.2 NDCG@10 for textual pipelines)
- Text rerankers (zerank-2) outperform visual rerankers (jina-reranker-m0)

---

## 3. Can We Run ViDoRe v3 Locally?

### Evaluation Code
- **Yes, open source**: Integrated into MTEB framework
- **Codebase**: `https://github.com/embeddings-benchmark/mteb`
- **Paper notes**: "Codebase coming soon" (but MTEB integration exists)

### Running via MTEB

```python
import mteb

# Get ViDoRe v3 benchmark
benchmark = mteb.get_benchmark("ViDoRe(v3)")

# Load a model
model = mteb.get_model("vidore/colqwen2.5-v0.2")

# Evaluate
results = mteb.evaluate(model=model, tasks=benchmark)
```

### Datasets Location
- **HuggingFace**: `https://huggingface.co/collections/vidore/vidore-benchmark-v3`
- Individual datasets: `vidore/vidore_v3_{domain}` (e.g., `vidore/vidore_v3_industrial`)
- Format: Parquet with image, text, PDF formats

### Data Size Estimates

| Dataset | Size (Downloads) |
|---------|------------------|
| vidore_v3_hr | ~13.4k downloads, ~4.4k likes |
| vidore_v3_finance_en | ~13.6k downloads, ~4.62k likes |
| vidore_v3_industrial | ~16.7k downloads, ~3.95k likes |
| vidore_v3_pharmaceuticals | ~14.9k downloads, ~3.9k likes |
| vidore_v3_computer_science | ~8.95k downloads, ~4.26k likes |
| vidore_v3_energy | ~10.7k downloads, ~756 likes |
| vidore_v3_physics | ~16.6k downloads, ~3.73k likes |
| vidore_v3_finance_fr | ~13.1k downloads, ~3.52k likes |

**Estimated total download**: 50-100GB for all public datasets (images are largest component)

### Apple Silicon (16GB) Feasibility

**Feasible with constraints**:

1. **Small models (< 3B params)**:
   - ColEmbed-1B, ColQwen2.5, Nomic-3B can run on 16GB
   - Requires quantization for comfortable memory

2. **Larger models (7B+)**:
   - ColNomic-7B, ColPali-v1.3 need 8-bit or 4-bit quantization
   - Apple Silicon unified memory helps

3. **Inference considerations**:
   - Page images must be loaded into memory
   - ~300 pages per dataset average
   - Batch processing recommended

**Recommendation**: Start with ColEmbed-1B or ColQwen2.5 (both under 3B). Use MTEB's caching to avoid re-downloading.

---

## 4. ViDoRe v3 vs RecallForge Use Case

### What ViDoRe v3 Evaluates
- **Document retrieval** from static, curated corpora
- **Multi-page synthesis** for answer generation
- **Visual grounding** (bounding boxes on document images)
- **Cross-lingual retrieval** (6 query languages, 2 document languages)

### Gap Analysis for Memory MCP

| Feature | ViDoRe v3 | Memory MCP Need |
|---------|-----------|-----------------|
| **Corpus Type** | Static documents | Dynamic personal memory |
| **Temporal Dimension** | Fixed snapshots | Evolving over time |
| **Query Intent** | Information finding | Memory recall/reconstruction |
| **Document Types** | PDFs, reports | Notes, conversations, images, links |
| **Personalization** | None required | Essential |
| **Privacy** | Public data | Private/sensitive data |
| **Scale** | 26K pages | Potentially millions of memories |

### Verdict

**ViDoRe v3 is NOT the right primary benchmark for RecallForge** because:

1. **Designed for document RAG, not memory systems**
   - Assumes curated, static corpora
   - No temporal aspect (memories change over time)

2. **No personalization dimension**
   - Memory retrieval is inherently personal
   - ViDoRe queries are generic, not user-specific

3. **Different relevance criteria**
   - Memory relevance is subjective to the user
   - ViDoRe has "ground truth" annotations

4. **Useful components**:
   - Visual grounding methodology (bounding boxes)
   - Query type taxonomy (open-ended, multi-hop, etc.)
   - Multi-hop queries are relevant for memory synthesis

---

## 5. Other Relevant Benchmarks

### Multimodal RAG Benchmarks

#### MRAG-Bench (Multimodal RAG Benchmark)
- **Focus**: Scenarios where visual knowledge > textual knowledge
- **Relevance**: Higher than ViDoRe for memory systems
- **URL**: https://mragbench.github.io/

#### MRAMG-Bench (Multimodal Retrieval-Augmented Multimodal Generation)
- **Paper**: arXiv:2502.04176 (April 2025)
- **Focus**: End-to-end multimodal RAG with multimodal output
- **Relevance**: Good for multimodal memory generation
- **Key feature**: Multimodal answer generation, not just retrieval

#### REAL-MM-RAG
- **Paper**: arXiv:2502.12342 (2025)
- **Focus**: Real-world multimodal retrieval scenarios
- **Relevance**: Good for practical deployment evaluation

### Reasoning-Intensive Retrieval

#### BRIGHT Benchmark
- **Paper**: arXiv:2407.12883
- **Focus**: Reasoning-intensive retrieval (beyond surface matching)
- **Scale**: 1,384 queries across economics, psychology, math, coding
- **Key finding**: Top MTEB models achieve only 18.3 nDCG@10 on BRIGHT vs 59.0 on MTEB
- **Relevance**: Good for testing memory reasoning (e.g., "What did I say about X last month?")
- **URL**: https://github.com/xlang-ai/bright

### Multimodal Understanding Benchmarks

#### MMMU (Massive Multi-discipline Multimodal Understanding)
- **Focus**: Expert-level multimodal reasoning (college-level tasks)
- **Key finding**: GPT-4V achieves only 56% accuracy
- **Relevance**: Less relevant for memory retrieval, more for reasoning

#### MTEB (Massive Text Embedding Benchmark) v2
- **Focus**: Broad embedding evaluation (1000+ languages, multimodal support)
- **ViDoRe v3 integration**: Already integrated as benchmark subset
- **URL**: https://github.com/embeddings-benchmark/mteb
- **Features**:
  - Supports multimodal input (text + images)
  - Retrieval, reranking, classification tasks
  - Local evaluation capability

### Agent Memory Benchmarks (Research Needed)
- **No widely-adopted benchmark specifically for agent memory MCPs**
- Closest analogues are conversational memory benchmarks, but these are typically text-only
- Potential gap in the field for a dedicated memory MCP benchmark

---

## 6. Recommendations for RecallForge

### Primary Benchmark Strategy

1. **Don't use ViDoRe v3 as primary benchmark**
   - Wrong domain (document retrieval vs memory recall)

2. **Consider these alternatives**:
   - **MRAMG-Bench**: Best fit for multimodal memory generation
   - **BRIGHT**: Good for reasoning-intensive memory queries
   - **Custom benchmark**: May need to create RecallForge-specific evaluation

### Hybrid Evaluation Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    RecallForge Evaluation                   │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Retrieval Accuracy                                 │
│   - BRIGHT (reasoning-intensive queries)                    │
│   - Custom memory-specific queries                          │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Multimodal Understanding                           │
│   - MRAMG-Bench (multimodal answer generation)              │
│   - Visual grounding accuracy (borrowed from ViDoRe)        │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Memory-Specific Metrics                            │
│   - Temporal recall accuracy                                │
│   - Cross-modal retrieval (text→image, image→text)          │
│   - Personalization score                                   │
│   - Memory deduplication/consolidation                       │
└─────────────────────────────────────────────────────────────┘
```

### What to Borrow from ViDoRe v3

1. **Query taxonomy**: 7 query types are relevant for memory queries
2. **Bounding box grounding**: Visual grounding methodology
3. **Evaluation protocol**: NDCG@10 for ranking, LLM-as-judge for answers
4. **MTEB integration**: Use the framework for RecallForge evaluation

### Running ViDoRe v3 Locally (If Desired)

```bash
# Install MTEB
pip install mteb

# Clone latest (ViDoRe v3 requires latest)
git clone https://github.com/embeddings-benchmark/mteb.git
cd mteb && pip install .

# Evaluate on ViDoRe v3
python -c "
import mteb
benchmark = mteb.get_benchmark('ViDoRe(v3)')
model = mteb.get_model('vidore/colqwen2.5-v0.2')
results = mteb.evaluate(model=model, tasks=benchmark)
"
```

---

## 7. Key Papers & Links

| Resource | URL |
|----------|-----|
| ViDoRe v3 Paper | https://arxiv.org/abs/2601.08620 |
| ViDoRe v3 Blog | https://huggingface.co/blog/QuentinJG/introducing-vidore-v3 |
| ViDoRe Datasets | https://huggingface.co/collections/vidore/vidore-benchmark-v3 |
| MTEB Leaderboard | https://huggingface.co/spaces/mteb/leaderboard |
| MTEB GitHub | https://github.com/embeddings-benchmark/mteb |
| BRIGHT Benchmark | https://arxiv.org/abs/2407.12883 |
| MRAMG-Bench | https://arxiv.org/abs/2502.04176 |
| MRAG-Bench | https://mragbench.github.io/ |
| REAL-MM-RAG | https://arxiv.org/abs/2502.12342 |

---

## 8. Summary Table

| Benchmark | Best For | RecallForge Fit | Local Run |
|-----------|----------|-----------------|-----------|
| ViDoRe v3 | Document retrieval, visual grounding | ❌ Wrong domain | ✅ MTEB |
| BRIGHT | Reasoning-intensive retrieval | ⚠️ Partial (text only) | ✅ MTEB |
| MRAMG-Bench | Multimodal answer generation | ✅ Good fit | ⚠️ Check availability |
| MRAG-Bench | Visual knowledge scenarios | ✅ Good fit | ⚠️ Check availability |
| MTEB v2 | General embedding evaluation | ✅ Framework choice | ✅ Local |
| MMMU | Expert reasoning | ❌ Not memory-focused | ✅ Available |

---

*Research completed: 2026-03-15*
*Author: Molly (Subagent)*