# QMD-VL Architecture

QMD-VL is a Python vision-language memory search system built on LanceDB and Qwen3-VL-Embedding.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           QMD-VL Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   Document   │───>│   Chunking   │───>│   Embedding (Qwen3-VL)   │   │
│  │   Ingestion  │    │   + Smart    │    │   2048-dim vectors       │   │
│  │              │    │   Breaks     │    │   MPS / CUDA / CPU      │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│         │                   │                        │                   │
│         ▼                   ▼                        ▼                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     LanceDB Storage                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │ embeddings  │  │  documents  │  │   content   │  │  cache  │  │   │
│  │  │ + FTS index │  │  registry   │  │   (bodies)  │  │ (LLM)   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     Search Pipeline                               │   │
│  │                                                                   │   │
│  │   Query ──┬──> BM25 (Tantivy FTS) ──┐                           │   │
│  │           │                          │                            │   │
│  │           └──> Vector Search ───────┼──> RRF Fusion ──> Rerank   │   │
│  │                                       │                  │        │   │
│  │                                       ▼                  ▼        │   │
│  │                              Scored Results ──> Final Ranking    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Document Ingestion (`store.py`)

- **`insert_document()`**: Register document in `documents` table
- **`insert_content()`**: Store full text in `content` table (content-addressed by hash)
- **`insert_embedding()`**: Store chunk embedding in `embeddings` table
- **`index_document()`**: Full pipeline - hash, chunk, embed, store

### 2. Smart Chunking (`store.py`)

Breaks documents at natural boundaries, not arbitrary positions:

```
Document Text
     │
     ▼
┌────────────────────────────────────────┐
│          Break Point Detection          │
│  ┌──────────────────────────────────┐   │
│  │  H1-H6 headings (100-50 pts)    │   │
│  │  Code blocks (80 pts)            │   │
│  │  Horizontal rules (60 pts)       │   │
│  │  Blank lines (20 pts)            │   │
│  │  List items (5 pts)              │   │
│  │  Newlines (1 pt)                 │   │
│  └──────────────────────────────────┘   │
│                   │                      │
│                   ▼                      │
│  ┌──────────────────────────────────┐   │
│  │  Code fence detection            │   │
│  │  (avoid breaking inside ```...```) │ │
│  └──────────────────────────────────┘   │
│                   │                      │
│                   ▼                      │
│  ┌──────────────────────────────────┐   │
│  │  Best cutoff finder              │   │
│  │  - Window around target position │   │
│  │  - Distance decay multiplier     │   │
│  │  - Score × multiplier = priority │   │
│  └──────────────────────────────────┘   │
└────────────────────────────────────────┘
     │
     ▼
[{text, pos}, {text, pos}, ...] chunks
```

### 3. Embedding (`embed.py`)

Qwen3-VL-Embedding-2B wrapper:

```python
from src.embed import get_embedder, embed_text

# Initialize (lazy-loaded)
embedder = get_embedder()

# Embed single text
vector = embedder.embed_text("query text")  # -> np.ndarray[2048]

# Embed multiple texts
vectors = embedder.embed_texts(["text1", "text2"])  # -> np.ndarray[N, 2048]

# Embed images
vector = embedder.embed_image("path/to/image.jpg")

# Mixed content
vectors = embedder.embed_mixed([
    {"text": "query"},
    {"image": "url"},
    {"text": "caption", "image": "path"}
])
```

### 4. LanceDB Tables (`db.py`)

#### embeddings (main table)
```python
{
    "hash_seq": "{content_hash}_{seq}",  # PK
    "content_hash": "sha256...",
    "collection": "my-docs",
    "file_path": "notes/example.md",
    "content_type": "text",
    "title": "Example Document",
    "text_body": "chunk text for BM25...",
    "seq": 0,
    "pos": 0,
    "model": "Qwen/Qwen3-VL-Embedding-2B",
    "embedded_at": 1709847234567,
    "vector": [0.123, -0.456, ...],  # 2048 floats
}
```

#### documents (registry)
```python
{
    "id": "uuid",
    "collection": "my-docs",
    "file_path": "notes/example.md",
    "title": "Example Document",
    "content_hash": "sha256...",
    "content_type": "text",
    "active": 1,
    "created_at": 1709847234567,
    "updated_at": 1709847234567,
}
```

#### content (bodies)
```python
{
    "hash": "sha256...",
    "doc": "full document text...",
    "content_type": "text",
    "created_at": 1709847234567,
}
```

#### cache (LLM results)
```python
{
    "key": "sha256(url+body)",
    "value": "JSON result...",
    "created_at": 1709847234567,
}
```

### 5. Search Pipeline

#### BM25 (FTS)

```python
from src.store import search_fts

results = search_fts(
    query="graph database knowledge",
    limit=20,
    collection="my-docs"  # optional
)

# Returns: List[SearchResult]
# - filepath, display_path, title, body, score
# - source: 'fts'
```

#### Vector Search

```python
from src.store import search_vec
from src.embed import embed_text

query_vector = embed_text("how do AI agents remember things")
results = search_vec(
    vector=query_vector,
    limit=20,
    collection="my-docs"
)

# Score = 1 - distance/2 (for cosine-like distance)
```

#### Hybrid (BM25 + Vector)

Future Stage will implement:

```python
# Query expansion
expanded_queries = expand_query("how do AI agents remember things")
# -> [
#      {"type": "lex", "text": "AI agent memory systems"},
#      {"type": "vec", "text": "episodic memory knowledge graphs"},
#      {"type": "hyde", "text": "generated hypothetical answer..."}
#    ]

# Parallel retrieval
bm25_results = search_fts(original_query)
vec_results = search_vec(embedded_query)

# Reranking with Qwen3-VL-Reranker
final_results = rerank(query, candidates)
```

## Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Document Indexing Flow                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  path: "docs/architecture.md"                                    │
│  text: "## Architecture\n\nThe system uses..."                   │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐                                             │
│  │ hash_content()  │ ──> "abc123..."                             │
│  └─────────────────┘                                             │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐                                             │
│  │ extract_title() │ ──> "Architecture"                          │
│  └─────────────────┘                                             │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │ insert_content  │ ──> │ content table   │                    │
│  └─────────────────┘     └─────────────────┘                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │ insert_document │ ──> │ documents table │                    │
│  └─────────────────┘     └─────────────────┘                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐                                             │
│  │ chunk_document  │ ──> [{text, pos}, ...]                     │
│  │  - Break points │                                             │
│  │  - Overlap      │                                             │
│  └─────────────────┘                                             │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │ embed_text()    │ ──> │ [0.1, -0.2, ...] │ (2048 floats)     │
│  └─────────────────┘     └─────────────────┘                    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │ insert_embedding│ ──> │ embeddings table│                    │
│  └─────────────────┘     └─────────────────┘                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Storage Layout

```
~/.qmd/
└── store.lance/
    ├── embeddings/
    │   ├── data/
    │   │   └── *.parquet
    │   ├── _indices/
    │   │   └── text_body_fts/      # Tantivy FTS index
    │   └── _metadata/
    ├── documents/
    │   └── ...
    ├── content/
    │   └── ...
    └── cache/
        └── ...
```

## Performance Considerations

### Chunking
- Default: 512 tokens (~2048 chars) with 64 token overlap
- Smart breaks at headings, code blocks, paragraphs
- Avoids splitting inside code fences

### Embedding
- Qwen3-VL-Embedding-2B: 2048-dim vectors
- MPS (Apple Silicon): float16, eager attention (no flash_attention_2)
- CUDA: bfloat16, flash_attention_2
- Batch embedding for efficiency

### Search
- BM25: Tantivy FTS index on `text_body` column
- Vector: LanceDB ANN (IVF_HNSW_SQ for large collections)
- Hybrid: RRF fusion + Qwen3-VL-Reranker

## Stage Roadmap

| Stage | Feature | Status |
|-------|----------|--------|
| 1 | Foundation + Text Embedding + BM25 | ✅ Complete |
| 2 | Image embedding + multimodal search | 🔲 Planned |
| 3 | MCP server for IDE integration | 🔲 Planned |
| 4 | Query expansion (HyDE) | 🔲 Planned |
| 5 | Reranker integration | 🔲 Planned |
| 6 | Hybrid retrieval pipeline | 🔲 Planned |

## Usage

```python
from src import db, store, embed

# Initialize database
db.initialize_database("/path/to/store")

# Index a document
content_hash = store.index_document(
    path="docs/example.md",
    text=open("docs/example.md").read(),
    collection="my-docs",
    model="Qwen/Qwen3-VL-Embedding-2B",
    embed_func=embed.embed_text,
)

# BM25 search
results = store.search_fts("knowledge graph", limit=10)
for r in results:
    print(f"{r.title}: {r.score:.3f}")

# Vector search
query_vec = embed.embed_text("how to store relationships")
results = store.search_vec(query_vec, limit=10)
for r in results:
    print(f"{r.title}: {r.score:.3f}")

# Check if embeddings exist
has_vectors = store.has_vectors()
print(f"Index has embeddings: {has_vectors}")
```