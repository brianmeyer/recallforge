# RecallForge Architecture

RecallForge is a cross-modal vision-language search engine built on LanceDB and Qwen3-VL-Embedding.
It is inspired by and builds upon [QMD](https://github.com/tobil/qmd) by [Tobi](https://github.com/tobil).

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RecallForge Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   Document   │───>│   Chunking   │───>│   ModelBackend           │   │
│  │   Ingestion  │    │   + Smart    │    │   (Torch or MLX)         │   │
│  │              │    │   Breaks     │    │   2048-dim vectors       │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│         │                   │                        │                   │
│         ▼                   ▼                        ▼                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   StorageBackend (LanceDB)                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │ embeddings  │  │  documents  │  │   content   │  │  cache  │  │   │
│  │  │ + FTS index │  │  registry   │  │   (bodies)  │  │         │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                  HybridSearcher Pipeline                          │   │
│  │                                                                   │   │
│  │   Query ──┬──> BM25 (Tantivy FTS) ──┐                           │   │
│  │           │                          │                            │   │
│  │           └──> Vector Search ───────┼──> RRF Fusion ──> Rerank   │   │
│  │                                       │                  │        │   │
│  │   [full mode: Query Expansion]        ▼                  ▼        │   │
│  │   Lex/Vec/HyDE expansions ──> Scored Results ──> Final Ranking   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture Layers

### 1. ModelBackend ABC (`src/recallforge/backends/base.py`)

Abstract base class for all inference backends. Backends are injectable and interchangeable.

```python
from recallforge.backends.base import ModelBackend

class MyBackend(ModelBackend):
    def embed_text(self, text: str) -> np.ndarray: ...
    def embed_texts(self, texts: List[str]) -> np.ndarray: ...
    def embed_image(self, image_path: str) -> np.ndarray: ...
    def embed_images(self, image_paths: List[str]) -> np.ndarray: ...
    def rerank(self, query: str, documents: List[Dict]) -> List[float]: ...
    def expand_query(self, query: str) -> Dict[str, str]: ...
    def warm_up(self) -> None: ...
    def get_info(self) -> BackendInfo: ...
```

#### Tiered Modes

| Mode | Models Loaded | Memory | Quality |
|------|---------------|--------|---------|
| `embed` | Embedder only | ~4 GB | Baseline |
| `hybrid` | Embedder + Reranker | ~8 GB | Better |
| `full` | Embedder + Reranker + Expander | ~12 GB | Best |

#### Concrete Backends

- **TorchBackend** (`torch_backend.py`): PyTorch — CUDA > MPS > CPU, float16
  - Embedder: `Qwen/Qwen3-VL-Embedding-2B`
  - Reranker: `Qwen/Qwen3-VL-Reranker-2B`
  - Expander: `tobil/qmd-query-expansion-qwen3.5-2B`
- **MLXBackend** (`mlx_backend.py`): Apple Silicon MLX
  - BF16: `arthurcollet/Qwen3-VL-Embedding-2B-mlx`
  - 4-bit: `arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit`
  - Expander: Torch fallback

### 2. StorageBackend ABC (`src/recallforge/storage/base.py`)

Abstract base class for all storage backends.

```python
from recallforge.storage.base import StorageBackend

class LanceDBBackend(StorageBackend):
    def initialize(self, store_path: str) -> None: ...
    def insert_document(self, ...) -> str: ...
    def find_document(self, ...) -> Optional[Document]: ...
    def insert_content(self, ...) -> None: ...
    def insert_embedding(self, ...) -> None: ...
    def search_fts(self, query, limit, ...) -> List[SearchResult]: ...
    def search_vec(self, vector, limit, ...) -> List[SearchResult]: ...
    def get_cached(self, key) -> Optional[str]: ...
    def set_cached(self, key, value) -> None: ...
```

#### LanceDB Tables

**embeddings** (main, with vector + FTS)
```
hash_seq       | content_hash | collection | file_path | content_type
title          | text_body    | seq        | pos        | model
embedded_at    | vector[2048]
```

**documents** (registry)
```
id | collection | file_path | title | content_hash | content_type | active
created_at | updated_at
```

**content** (bodies, content-addressed)
```
hash | doc | content_type | created_at
```

**cache** (key-value)
```
key | value | created_at
```

### 3. HybridSearcher (`src/recallforge/search.py`)

Takes injected `ModelBackend` and `StorageBackend`. Runs tiered pipeline:

```
Query
  │
  ├──[all modes]──> BM25 probe
  │
  ├──[full mode]──> Query expansion (lex/vec/hyde)
  │
  ├──[all modes]──> Parallel searches (ThreadPoolExecutor)
  │                  ├── BM25 (original + lex expansions)
  │                  └── Vector (original + vec + hyde)
  │
  ├──[all modes]──> RRF fusion (k=60, weighted)
  │
  ├──[hybrid/full]─> Cross-encoder reranking
  │
  └──[all modes]──> Score blending → top-K HybridResult
```

#### RRF Weights
- First 2 result lists: weight=2.0
- Additional expansion lists: weight=1.0

#### Score Blending
- RRF rank 1-3: 75% RRF + 25% reranker
- RRF rank 4-10: 60% RRF + 40% reranker
- RRF rank 11+: 40% RRF + 60% reranker

### 4. Auto Backend Selection (`src/recallforge/__init__.py`)

```python
import recallforge

# Auto: MLX on Apple Silicon if available, else Torch
backend = recallforge.get_backend()  # RECALLFORGE_BACKEND=auto

# Explicit
os.environ["RECALLFORGE_BACKEND"] = "mlx"
os.environ["RECALLFORGE_MODE"] = "hybrid"
os.environ["RECALLFORGE_MLX_QUANTIZE"] = "4bit"
backend = recallforge.get_backend()
```

### 5. MCP Server (`src/recallforge/server.py`)

```
Tools: search, search_fts, search_vec, index_document, index_image, status, rebuild_fts
Transport: stdio
Startup: backend.warm_up() for predictable latency
Signals: SIGTERM/SIGINT graceful shutdown
```

## Storage Layout

```
~/.recallforge/          (default, override with RECALLFORGE_STORE_PATH)
└── store.lance/
    ├── embeddings/
    │   ├── data/*.parquet
    │   └── _indices/text_body_fts/    # Tantivy FTS
    ├── documents/
    ├── content/
    └── cache/
```

## Data Flow: Indexing

```
path + text
     │
     ├──> hash_content()   ──> content_hash
     ├──> extract_title()  ──> title
     ├──> insert_content() ──> content table
     ├──> insert_document()──> documents table
     │
     └──> chunk_document() ──> [{text, pos}, ...]
               │
               ▼
         embed_func(chunk["text"])  ──> vector[2048]
               │
               ▼
         insert_embedding()  ──> embeddings table
               │
               ▼
         ensure_fts_index()  ──> Tantivy index rebuild
```

## Performance

| Metric | Value |
|--------|-------|
| Chunk size | 512 tokens (~2048 chars) |
| Chunk overlap | 64 tokens |
| Vector dims | 2048 (float32) |
| Smart breaks | H1-H6, code blocks, paragraphs, list items |
| ANN index | IVF_HNSW_SQ (large collections) |
| Parallel searches | ThreadPoolExecutor (8 workers default) |

### Search Optimization (P0)

- **N+1 Lookup Elimination**: Search result construction prefers `text_body` from the embeddings table row (already fetched via LanceDB query) over calling `get_content()` which requires a separate lookup to the content table. Falls back to `get_content()` only when `text_body` is empty.
- **Lazy Content Loading**: Full document body is only fetched when explicitly needed for final output. The reranker input path uses chunk text (`text_body`) for candidate scoring, avoiding unnecessary content lookups during the scoring phase.
- **Output Contract**: The MCP `search` tool output remains stable - results include all fields (`filepath`, `score`, `body`, etc.) with the same shape as before. Only internal lookup patterns have changed.

## Stage Roadmap

| Stage | Feature | Status |
|-------|---------|--------|
| 1 | Foundation + Text Embedding + BM25 | ✅ Complete |
| 2 | Image embedding + cross-modal search | ✅ Complete |
| 3 | MCP server + CLI | ✅ Complete |
| 4 | Model Registry + Parallelism | ✅ Complete |
| 5 | Multi-backend + Tiered modes + Benchmarks | ✅ Complete |

## Attribution

RecallForge is inspired by and builds upon [QMD](https://github.com/tobil/qmd) by [Tobi](https://github.com/tobil).
