# RecallForge

![CI](https://github.com/brianmeyer/recallforge/actions/workflows/ci.yml/badge.svg) ![PyPI](https://img.shields.io/badge/PyPI-coming_soon-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.12+-blue)

**Every modality, one search. Local first.**

Search text, images, documents, and video in one unified query. Type "whiteboard diagram from last meeting" to find the photo. Drop an image to find related notes. Index `pdf`/`docx`/`pptx` locally for agent memory. All embeddings stay on your machine.

## Quick Start (30 seconds)

```bash
# 1. Install
pip install recallforge

# 2. Index something
recallforge index ./photos ./docs

# 3. Search
recallforge search "whiteboard diagram from last meeting"
recallforge search --image ./photos/whiteboard.png
```

That's it. RecallForge auto-detects MLX on Apple Silicon, PyTorch elsewhere.

## MCP Server

Run as a Model Context Protocol server for AI agents:

```bash
recallforge serve --mode embed --backend mlx --quantize 4bit
```

Exposes **12 tools** for agents: `ingest`, `search`, `search_fts`, `search_vec`, `index_document`, `index_image`, `memory_add`, `memory_update`, `memory_delete`, `index_folder`, `status`, `rebuild_fts`.

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
- **Runs anywhere:** MLX 4-bit on Apple Silicon (~2GB), PyTorch on CUDA/MPS/CPU
- **Fast:** Cold start 3.8s, warm search 161ms (MLX 4-bit)
- **Tiered modes:** embed (~2GB), hybrid (~4GB), full (~8GB) — pick your tradeoff
- **100% local:** Your data never leaves your machine

## Performance

| Metric | MLX 4-bit | PyTorch fp16 |
|--------|-----------|--------------|
| Warm search | 161ms | 414ms |
| Cold start | 3.8s | ~36s |
| Memory (embed) | ~2GB | ~4-8GB |

## Installation

```bash
pip install recallforge            # auto-detects best backend
pip install recallforge[mlx]       # force MLX (Apple Silicon)
pip install recallforge[cuda]      # force CUDA (NVIDIA)
pip install recallforge[docs]      # richer PDF extraction
```

From source:

```bash
git clone https://github.com/brianmeyer/recallforge.git
cd recallforge
pip install -e ".[mlx]"
```

## Tiered Search Modes

| Mode | Memory | Models | Use Case |
|------|--------|--------|----------|
| `embed` | ~2GB | Embedder only | Memory-constrained, fast searches |
| `hybrid` | ~4GB | + Reranker | Balanced quality and memory |
| `full` | ~8GB | + Query Expander | Maximum retrieval quality |

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

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RECALLFORGE_BACKEND` | `auto` | `auto`, `mlx`, `torch` |
| `RECALLFORGE_MODE` | `full` | `embed`, `hybrid`, `full` |
| `RECALLFORGE_MLX_QUANTIZE` | `4bit` | `4bit`, `bf16` |
| `RECALLFORGE_STORE_PATH` | `~/.recallforge` | Storage directory |

## Architecture

```
RecallForge
├── backends/          # Model backends (MLX, PyTorch)
├── storage/          # LanceDB + Tantivy FTS
├── search.py         # Hybrid search pipeline
├── server.py         # MCP server (12 tools)
├── documents.py      # PDF/DOCX/PPTX extraction
├── video.py          # Frame/transcript extraction
└── cli.py            # CLI interface
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