# QMD-VL Architecture

A **Vision-Language Memory Search System** built on LanceDB with Qwen3-VL embeddings.

## Overview

QMD-VL is a Python implementation of the QMD memory system, designed for storing, indexing, and retrieving multimodal content (text, images, video) using both lexical (BM25) and semantic (vector) search.

```
┌─────────────────────────────────────────────────────────────────┐
│                         QMD-VL Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
│   │   Document  │     │   Chunker   │     │   Qwen3-VL  │     │
│   │   Ingest    │ ──► │   (Smart)   │ ──► │  Embedder   │     │
│   └─────────────┘     └─────────────┘     └─────────────┘     │
│          │                   │                   │            │
│          ▼                   ▼                   ▼            │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │                     LanceDB Store                        │ │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │ │
│   │  │Documents │  │ Content  │  │Embeddings│  │  Cache  │ │ │
│   │  │ (meta)   │  │ (text)   │  │(vectors)│  │ (LLM)   │ │ │
│   │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │ │
│   └─────────────────────────────────────────────────────────┘ │
│                              │                                 │
│          ┌───────────────────┼───────────────────┐           │
│          ▼                   ▼                   ▼           │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
│   │   BM25      │     │   Vector    │     │   Hybrid    │     │
│   │   Search    │     │   Search    │     │   Search    │     │
│   │  (Tantivy)  │     │   (ANN)     │     │  (RRF)      │     │
│   └─────────────┘     └─────────────┘     └─────────────┘     │
│          │                   │                   │            │
│          └───────────────────┴───────────────────┘           │
│                              ▼                                │
│                     ┌─────────────┐                          │
│                     │   Results   │                          │
│                     │  (Scored)   │                          │
│                     └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Database Layer (`db.py`)

LanceDB tables with Apache Arrow schemas:

| Table | Purpose | Key Fields |
|-------|---------|------------|
| **embeddings** | Chunk embeddings | `hash_seq` (PK), `content_hash`, `text_body`, `vector` (2048-dim) |
| **documents** | Document registry | `id` (PK), `file_path`, `title`, `content_hash`, `active` |
| **content** | Full document text | `hash` (PK), `doc` (full text), `content_type` |
| **cache** | LLM response cache | `key` (PK), `value` (JSON), `created_at` |

### 2. Store Layer (`store.py`)

Core operations:
- `chunk_document(text)` → Smart chunking at natural break points
- `insert_document_with_embedding(path, text, collection, embed_func)` → Full pipeline
- `search_fts(query)` → BM25 lexical search via Tantivy
- `search_vec(embedding)` → Vector similarity search via LanceDB ANN

### 3. Embedding Layer (`embed.py`)

Qwen3-VL-Embedding-2B wrapper:
- 2048-dimensional embeddings
- MPS (Apple Silicon), CUDA, and CPU support
- Float16 precision for memory efficiency
- Supports text, image, and multimodal content

## Search Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                      Search Pipeline                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Query (text)                                                   │
│       │                                                          │
│       ├─────────────────────────────────────────────┐           │
│       │                                             │           │
│       ▼                                             ▼           │
│   ┌──────────────┐                         ┌──────────────┐   │
│   │   Embed      │                         │   Tokenize   │   │
│   │   Query      │                         │   Query      │   │
│   └──────────────┘                         └──────────────┘   │
│       │                                             │           │
│       ▼                                             ▼           │
│   ┌──────────────┐                         ┌──────────────┐   │
│   │   Vector     │                         │   BM25        │   │
│   │   ANN Search │                         │   FTS Search  │   │
│   │   (LanceDB)  │                         │   (Tantivy)   │   │
│   └──────────────┘                         └──────────────┘   │
│       │                                             │           │
│       │    Score = 1 - distance/2                  │           │
│       │    (cosine-like normalization)             │           │
│       │                                             │           │
│       │    Score = raw / max_score                 │           │
│       │    (normalize to [0, 1])                   │           │
│       │                                             │           │
│       ▼                                             ▼           │
│   ┌──────────────────────────────────────────────────────┐     │
│   │              Hybrid Search (Optional)                │     │
│   │   Reciprocal Rank Fusion (RRF):                      │     │
│   │   score = sum(1 / (k + rank)) for each list         │     │
│   │   k = 60 (constant)                                  │     │
│   └──────────────────────────────────────────────────────┘     │
│       │                                                          │
│       ▼                                                          │
│   ┌──────────────────────────────────────────────────────┐     │
│   │              Results (deduplicated by file_path)     │     │
│   │   - hash_seq, content_hash, file_path, title         │     │
│   │   - text_body (snippet), score, collection           │     │
│   └──────────────────────────────────────────────────────┘     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Chunking Strategy

Smart document chunking preserves semantic boundaries:

```
┌────────────────────────────────────────────────────────────────┐
│                    Chunking Pipeline                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Document (full text)                                         │
│       │                                                        │
│       ▼                                                        │
│   ┌────────────────────────────────────────────────────────┐  │
│   │   Scan for Natural Break Points                         │  │
│   │   - Headings (h1=100, h2=90, h3=80, ...)               │  │
│   │   - Code blocks (``` ... ```)                          │  │
│   │   - Horizontal rules (---, ***, ___)                   │  │
│   │   - Paragraph breaks (double newline)                  │  │
│   │   - List items (-, *, 1.)                              │  │
│   │   - Regular newlines                                    │  │
│   └────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼                                                        │
│   ┌────────────────────────────────────────────────────────┐  │
│   │   Find Best Cutoff Points                              │  │
│   │   - Target: 512 tokens (~2048 chars)                  │  │
│   │   - Window: 200 chars before target                   │  │
│   │   - Score: break_score * distance_multiplier          │  │
│   │   - Skip: inside code fences                          │  │
│   └────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼                                                        │
│   ┌────────────────────────────────────────────────────────┐  │
│   │   Create Overlapping Chunks                            │  │
│   │   - Overlap: 64 tokens (~256 chars)                   │  │
│   │   - Position tracking for citation                    │  │
│   │   - Sequence numbers for ordering                     │  │
│   └────────────────────────────────────────────────────────┘  │
│       │                                                        │
│       ▼                                                        │
│   Chunks: [{text, pos}, {text, pos}, ...]                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Document Ingestion

```python
# Full pipeline
content_hash = await insert_document_with_embedding(
    path="notes/meeting.md",
    text=document_content,
    collection="work",
    embed_func=my_embed_function,  # async (text) -> List[float]
    model="qwen3-vl-embedding-2b"
)
```

### Search Query

```python
# Hybrid search (recommended)
fts_results = await search_fts("graph memory agent")
vec_results = await search_vec(query_embedding)

# Combine with RRF or use separately
all_results = merge_results(fts_results, vec_results)
```

## Future Stages

| Stage | Feature | Status |
|-------|---------|--------|
| **Stage 1** | Foundation + Text Embedding + BM25 | ✅ Complete |
| **Stage 2** | Image/Video Embedding + Multimodal Search | 🔜 Planned |
| **Stage 3** | Graph Memory (Neo4j/NetworkX) | 🔜 Planned |
| **Stage 4** | LLM Integration + MCP Server | 🔜 Planned |
| **Stage 5** | GraphRAG + Temporal Memory | 🔜 Planned |

## Dependencies

- **lancedb**: Vector database with built-in FTS (Tantivy)
- **pyarrow**: Apache Arrow for columnar storage
- **torch**: Tensor operations for embeddings
- **transformers**: HuggingFace model loading
- **qwen-vl-utils**: Qwen3-VL utilities
- **pillow**: Image processing
- **numpy**: Numerical operations

## Performance Notes

- **Embedding dimension**: 2048 (Qwen3-VL-Embedding-2B)
- **Chunk size**: ~512 tokens (~2048 characters)
- **Chunk overlap**: ~64 tokens (~256 characters)
- **Vector search**: ANN via LanceDB IVF-PQ index
- **FTS**: Tantivy BM25 with inverted index
- **Deduplication**: By file_path in search results

## Storage Location

Default: `~/.qmd/store.lance/`

Override with `INDEX_PATH` environment variable or pass custom path to `initialize_database()`.