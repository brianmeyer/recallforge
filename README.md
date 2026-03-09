# RecallForge

**Cross-Modal Vision-Language Search Engine**

RecallForge is a powerful semantic search system that combines BM25 full-text search, vector similarity search, query expansion, and cross-encoder reranking. It supports both text and image content, enabling truly cross-modal retrieval.

## Attribution

**RecallForge is inspired by and builds upon [QMD](https://github.com/tobil/qmd) by [Tobi](https://github.com/tobil).**

QMD (Query-Document Matching) pioneered the approach of combining Qwen3-VL embeddings with LanceDB for semantic search. RecallForge extends this foundation with:

- Multi-backend architecture (PyTorch + MLX)
- Tiered search modes (embed/hybrid/full)
- Cross-encoder reranking
- Query expansion
- Cross-modal image-text search
- MCP server integration

Huge thanks to Tobi for the original vision and implementation that made this project possible.

---

## Features

- **Cross-Modal Search**: Query text to find images, or images to find text
- **Hybrid Search**: Combines BM25 + vector search with RRF fusion
- **Query Expansion**: Generates lexical, vector, and HyDE variants
- **Cross-Encoder Reranking**: Refines results with joint query-document scoring
- **Multi-Backend**: PyTorch (CUDA/MPS/CPU) or MLX (Apple Silicon)
- **Tiered Modes**: Choose your memory/quality tradeoff
- **MCP Integration**: Use as a Model Context Protocol server

## Installation

### Basic (PyTorch)

```bash
pip install recallforge
```

### With MLX (Apple Silicon)

```bash
pip install recallforge[mlx]
```

### With CUDA

```bash
pip install recallforge[cuda]
```

### From Source

```bash
git clone https://github.com/brianmeyer/recallforge.git
cd recallforge
pip install -e .

# For MLX support on Apple Silicon
pip install -e ".[mlx]"
```

## Quick Start

### CLI Usage

```bash
# Index files
recallforge index ~/Documents --collection docs

# Search
recallforge search "machine learning algorithms"

# Start MCP server
recallforge serve --mode full

# Check status
recallforge status
```

### Python API

```python
from recallforge import get_backend, get_storage
from recallforge.search import HybridSearcher

# Initialize
backend = get_backend()  # Auto-detect best backend
storage = get_storage()  # Default: ~/.recallforge

# Warm up models
backend.warm_up()

# Index documents
storage.index_document(
    path="notes.md",
    text="My important notes about AI and machine learning...",
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

### MCP Server

```bash
# Start the MCP server
recallforge serve

# Or with specific mode
recallforge serve --mode hybrid --backend torch
```

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

## Tiered Search Modes

RecallForge supports three search modes with different memory/quality tradeoffs:

### `embed` Mode (~4GB VRAM)

- **Models**: Embedder only
- **Search**: BM25 + Vector
- **Best for**: Memory-constrained environments, fast searches

```bash
recallforge serve --mode embed
```

```python
os.environ["RECALLFORGE_MODE"] = "embed"
```

### `hybrid` Mode (~8GB VRAM)

- **Models**: Embedder + Reranker
- **Search**: BM25 + Vector + Reranking
- **Best for**: Balanced quality and memory usage

```bash
recallforge serve --mode hybrid
```

### `full` Mode (~12GB VRAM) [Default]

- **Models**: Embedder + Reranker + Query Expander
- **Search**: BM25 + Vector + Expansion + Reranking
- **Best for**: Maximum retrieval quality

```bash
recallforge serve --mode full
```

## Model Backends

### PyTorch Backend (Default)

Works on CUDA, MPS (Apple Silicon), and CPU.

```bash
export RECALLFORGE_BACKEND=torch
```

**Model IDs:**
- Embedder: `Qwen/Qwen3-VL-Embedding-2B`
- Reranker: `Qwen/Qwen3-VL-Reranker-2B`
- Expander: `tobil/qmd-query-expansion-qwen3.5-2B`

### MLX Backend (Apple Silicon)

Optimized for Apple Silicon with optional quantization.

```bash
export RECALLFORGE_BACKEND=mlx
export RECALLFORGE_MLX_QUANTIZE=bf16  # or 4bit
```

**Model IDs (BF16):**
- Embedder: `arthurcollet/Qwen3-VL-Embedding-2B-mlx`
- Reranker: `arthurcollet/Qwen3-VL-Reranker-2B-mlx`
- Expander: Uses PyTorch fallback

**Model IDs (4-bit):**
- Embedder: `arthurcollet/Qwen3-VL-Embedding-2B-mlx-4bit`
- Reranker: `arthurcollet/Qwen3-VL-Reranker-2B-mlx-4bit`
- Expander: Uses PyTorch fallback

### Auto-Detection

By default, RecallForge auto-detects the best backend:

```bash
export RECALLFORGE_BACKEND=auto  # Default
```

- Apple Silicon → MLX
- CUDA GPU → PyTorch/CUDA
- Otherwise → PyTorch/CPU

## Storage Backend

Currently supports **LanceDB** with vector search and Tantivy FTS:

```bash
export RECALLFORGE_STORAGE=lancedb  # Default
export RECALLFORGE_STORE_PATH=~/.recallforge  # Default
```

Future support planned for:
- ChromaDB
- Qdrant

## MCP Tools

When running as an MCP server, RecallForge exposes these tools:

### `search`

Full hybrid search with all pipeline stages.

```json
{
  "query": "machine learning algorithms",
  "limit": 10,
  "collection": "docs",
  "content_type": "text"
}
```

### `search_fts`

BM25 full-text search only.

```json
{
  "query": "neural networks",
  "limit": 20
}
```

### `search_vec`

Vector similarity search only.

```json
{
  "query": "document about AI",
  "limit": 20
}
```

### `index_document`

Index a text document.

```json
{
  "path": "notes.md",
  "text": "Document content here...",
  "collection": "my_docs"
}
```

### `index_image`

Index an image for cross-modal search.

```json
{
  "path": "/path/to/image.png",
  "collection": "images"
}
```

### `status`

Get server status.

```json
{}
```

### `rebuild_fts`

Rebuild the full-text search index.

```json
{}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RECALLFORGE_BACKEND` | `auto` | Model backend: `torch`, `mlx`, `auto` |
| `RECALLFORGE_MODE` | `full` | Search mode: `embed`, `hybrid`, `full` |
| `RECALLFORGE_MLX_QUANTIZE` | `bf16` | MLX quantization: `bf16`, `4bit` |
| `RECALLFORGE_STORAGE` | `lancedb` | Storage backend |
| `RECALLFORGE_STORE_PATH` | `~/.recallforge` | Storage directory |

### CLI Options

```bash
recallforge serve \
  --mode full \
  --backend auto \
  --quantize bf16 \
  --store-path ~/.recallforge
```

## Architecture

```
RecallForge
├── recallforge/
│   ├── backends/          # Model backends
│   │   ├── base.py        # ModelBackend ABC
│   │   ├── torch_backend.py
│   │   └── mlx_backend.py
│   ├── storage/           # Storage backends
│   │   ├── base.py        # StorageBackend ABC
│   │   └── lancedb_backend.py
│   ├── search.py          # Hybrid search pipeline
│   ├── server.py          # MCP server
│   └── cli.py             # CLI interface
└── tests/
    ├── test_live.py       # Live tests with real models
    └── benchmark.py       # Benchmark suite
```

### Search Pipeline

1. **BM25 Probe**: Initial retrieval to detect strong signal
2. **Query Expansion** (full mode): Generate lex/vec/hyde variants
3. **Parallel Searches**: BM25 + Vector searches via ThreadPoolExecutor
4. **RRF Fusion**: Reciprocal Rank Fusion to combine result lists
5. **Reranking** (hybrid/full): Cross-encoder relevance scoring
6. **Score Blending**: Weighted combination of RRF and rerank scores

## Benchmarks

Run the benchmark suite:

```bash
python tests/benchmark.py --backends torch --modes embed hybrid full
```

Results are saved to `benchmarks/results.json` and `benchmarks/RESULTS.md`.

## Development

### Running Tests

```bash
# Unit tests (with mocks)
pytest tests/ -m "not live"

# Live tests (real models, slow)
pytest tests/ -m live -v

# All tests
pytest tests/ -v
```

### Code Style

```bash
# Format
black recallforge tests

# Lint
ruff check recallforge tests
```

## License

MIT License

## Acknowledgments

- **Tobi** for [QMD](https://github.com/tobil/qmd) - the original inspiration
- **Qwen Team** for Qwen3-VL-Embedding and Qwen3-VL-Reranker
- **LanceDB** for the excellent vector database