# RecallForge

![CI](https://github.com/brianmeyer/recallforge/actions/workflows/ci.yml/badge.svg) ![PyPI](https://img.shields.io/badge/PyPI-coming_soon-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.12+-blue)

**Every modality, one search. Local first.**

Search text, images, documents, and video in one unified query. Type "whiteboard diagram from last meeting" to find the photo. Drop an image to find related notes. Index `pdf`/`docx`/`pptx` locally for agent memory. All embeddings stay on your machine.

## Quick Start (30 seconds)

```bash
# 1. Install (pick your platform)
pip install recallforge[mlx]     # Apple Silicon
pip install recallforge[cuda]    # NVIDIA GPU
pip install recallforge[torch]   # CPU / other

# 2. Index anything — text, images, documents, video
recallforge index ./photos ./docs
recallforge index ~/Movies/demo.mp4

# 3. Search any modality
recallforge search "whiteboard diagram from last meeting"
recallforge search --image ./photos/whiteboard.png
recallforge search --video ~/Movies/demo.mp4
```

That's it. RecallForge auto-detects MLX on Apple Silicon, PyTorch elsewhere.

## MCP Server

Run as a Model Context Protocol server for AI agents:

```bash
recallforge serve --mode embed --backend mlx --quantize 4bit
```

Exposes **17 tools** for agents: `ingest`, `search`, `search_fts`, `search_vec`, `index_document`, `index_image`, `memory_add`, `memory_update`, `memory_delete`, `index_folder`, `status`, `rebuild_fts`, `list_collections`, `list_namespaces`, `batch`, `get_config`, `set_config`.

Configure in Claude Desktop:

```json
{
  "mcpServers": {
    "recallforge": {
      "command": "recallforge",
      "args": ["serve", "--mode", "full"]
    }
  }
}
```

## What makes RecallForge different

- **Cross-modal search:** Text↔Text, Text↔Image, Image↔Text, Image↔Image, Video↔Text, Video↔Image, Video↔Video
- **Shared embedding space:** Qwen3-VL encodes images and text into the same 2048-dim vectors
- **3-stage pipeline:** Embedding → Reranking → Query expansion (all multimodal)
- **Runs anywhere:** MLX 4-bit on Apple Silicon (~1.7GB RAM), PyTorch on CUDA/MPS/CPU
- **Fast:** Cold start 7.6s, warm search 53ms p50 (MLX 4-bit, Mac mini M4)
- **Tiered modes:** embed (~1.7GB), hybrid (~3.4GB), full (~4.4GB) — pick your tradeoff
- **100% local:** Your data never leaves your machine

## Performance

Measured on Mac mini M4 16GB, MLX 4-bit, embed mode:

| Metric | MLX 4-bit | PyTorch fp16 |
|--------|-----------|--------------|
| Warm search p50 | 53ms | 599ms |
| Warm search p95 | 55ms | — |
| Cold start | 7.6s | ~20s |
| Peak RSS (embed) | 329MB | ~4GB |
| Text indexing | 5.0 docs/sec | — |

**COCO Retrieval (50 images, embed mode, MLX 4-bit):**

| Direction | R@1 | R@5 | R@10 |
|-----------|-----|-----|------|
| Text → Image | 23.6% | 36.8% | 46.4% |
| Image → Text | 30.0% | 42.0% | 54.0% |

## Installation

RecallForge requires a backend for inference. Choose the right one for your platform:

```bash
pip install recallforge[mlx]       # Apple Silicon (recommended, 4-bit quantization)
pip install recallforge[cuda]      # NVIDIA GPU
pip install recallforge[torch]     # CPU / other PyTorch targets
pip install recallforge[docs]      # add richer PDF extraction (optional)
```

> **Note:** `pip install recallforge` installs the core without a backend.
> You need at least one of `[mlx]`, `[cuda]`, or `[torch]` to run inference.

From source:

```bash
git clone https://github.com/brianmeyer/recallforge.git
cd recallforge
pip install -e ".[mlx]"
```

## Tiered Search Modes

| Mode | Memory | Models | Use Case |
|------|--------|--------|----------|
| `embed` | ~1.7GB | Embedder only | Memory-constrained, fast searches |
| `hybrid` | ~3.4GB | + Reranker | Balanced quality and memory |
| `full` | ~4.4GB | + Query Expander | Maximum retrieval quality |

```bash
recallforge serve --mode embed   # minimal
recallforge serve --mode hybrid  # balanced
recallforge serve --mode full    # best quality (default)
```

## CLI Usage

```bash
# Index files
recallforge index ~/Documents --collection docs
recallforge index ~/Movies/demo.mp4 --collection docs
recallforge index ~/Documents/roadmap.pptx ~/Documents/notes.docx

# Search
recallforge search "machine learning algorithms"
recallforge search --image ~/Pictures/diagram.png
recallforge search --video ~/Movies/demo.mp4

# Status
recallforge status

# Watch a folder for new files (auto-index on change)
recallforge watch start ~/Documents --collection docs
recallforge watch list
recallforge watch stop ~/Documents
```

## Python API

```python
from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

backend = get_backend()
storage = get_storage()
backend.warm_up()

# Index
storage.index_document(
    path="notes.md",
    text="My notes about AI...",
    collection="my_docs",
    model="Qwen3-VL-Embedding-2B",
    embed_func=backend.embed_text,
)

# Search
searcher = HybridSearcher(backend=backend, storage=storage, limit=10)
results = searcher.search("artificial intelligence")
for r in results:
    print(f"[{r.score:.3f}] {r.title}")
```

## MCP Tools

### `ingest` (recommended)

Unified entry point for all content types:

```json
{
  "file_path": "/path/to/file",
  "folder_path": "/path/to/folder",
  "collection": "default",
  "recursive": true
}
```

### `search`

Full hybrid search with all pipeline stages:

```json
{
  "query": "machine learning algorithms",
  "limit": 10,
  "collection": "docs"
}
```

### Other tools

- `search_fts` — BM25 full-text search only
- `search_vec` — Vector similarity search only
- `index_document` — Index a text document
- `index_image` — Index an image for cross-modal search
- `memory_add` / `memory_update` / `memory_delete` — Manage agent memory
- `index_folder` — Batch index a folder
- `status` — Server health check
- `rebuild_fts` — Rebuild full-text index
- `list_collections` — List all collections
- `list_namespaces` — List namespace combinations
- `batch` — Execute multiple operations in one call
- `get_config` / `set_config` — Inspect and adjust runtime config

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RECALLFORGE_BACKEND` | `auto` | `auto`, `mlx`, `torch` |
| `RECALLFORGE_MODE` | `full` | `embed`, `hybrid`, `full` |
| `RECALLFORGE_MLX_QUANTIZE` | `4bit` | `4bit`, `bf16` |
| `RECALLFORGE_STORE_PATH` | `~/.recallforge` | Storage directory |

## Architecture

```
src/recallforge/
├── backends/
│   ├── mlx_backend.py    # MLX 4-bit/bf16 (Apple Silicon)
│   └── torch_backend.py  # PyTorch (CUDA/MPS/CPU)
├── storage/
│   └── lancedb_backend.py # LanceDB + Tantivy FTS
├── cache.py              # LRU embedding cache
├── search.py             # Hybrid search pipeline (BM25 + vector + RRF)
├── server.py             # MCP server (17 tools)
├── documents.py          # PDF/DOCX/PPTX extraction
├── video.py              # Frame/transcript extraction
├── watch_folder.py       # Folder monitoring with dedup
└── cli.py                # CLI interface
```

**Search pipeline:** BM25 probe → Query expansion (full mode) → Parallel BM25 + Vector → RRF fusion → Reranking (hybrid/full) → Score blending

## Development

```bash
pytest tests/ -m "not live"    # Unit tests (no model download needed)
pytest tests/ -m live -v       # Integration tests (requires models)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Attribution

RecallForge is inspired by [QMD](https://github.com/tobil/qmd) by Tobi. QMD pioneered the multi-stage retrieval pipeline (embedding, reranking, query expansion). RecallForge extends this pattern to vision-language with cross-modal retrieval and multi-backend support.

## License

MIT License