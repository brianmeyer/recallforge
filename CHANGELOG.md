# Changelog

All notable changes to RecallForge will be documented in this file.

## [Unreleased]

*Nothing yet.*

## [0.1.0] — 2026-03-13

First public release. The local multimodal memory engine for AI agents.

### Core

- **Cross-modal search** across text, images, documents (PDF/DOCX/PPTX), and video in one unified query
- **Shared embedding space** via Qwen3-VL (2048-dim vectors for all modalities)
- **3-stage retrieval pipeline:** embedding → reranking → query expansion (all multimodal)
- **Tiered search modes:** embed (~1.7GB), hybrid (~3.4GB), full (~4.4GB) on MLX 4-bit

### Backends

- **MLX backend** for Apple Silicon with 4-bit and bf16 quantization
- **PyTorch backend** for CUDA, MPS, and CPU
- Auto-detection of best available backend
- First-run model download UX with progress reporting

### MCP Server

- 17 MCP tools: `search`, `search_fts`, `search_vec`, `ingest`, `index_document`, `index_image`, `index_folder`, `memory_add`, `memory_update`, `memory_delete`, `status`, `rebuild_fts`, `list_collections`, `list_namespaces`, `batch`, `get_config`, `set_config`
- Structured error responses with 4 standard codes
- Batch operation support (up to 20 ops per call)
- Runtime config inspection and adjustment

### CLI

- `recallforge index` — index files and folders
- `recallforge search` — text, image, and video search
- `recallforge serve` — MCP server mode
- `recallforge status` — health check
- `recallforge watch` — folder monitoring with start/stop/list/status

### Storage

- LanceDB + Tantivy FTS hybrid backend
- Schema migration for namespace columns on existing stores
- Content-addressable deduplication
- Collection and namespace support

### Performance

- Warm search p50: 53ms (MLX 4-bit, Mac mini M4)
- Warm search p95: 55ms
- Cold start: 7.6s
- Text indexing: 5.0 docs/sec
- FTS miss fallback optimization (no expensive BM25 fallback on empty results)
- Bulk mode context manager for deferred FTS rebuilds during batch ingest

### Quality

- COCO retrieval benchmark (50 images, MLX 4-bit embed mode):
  - Text → Image: R@1 23.6%, R@5 36.8%, R@10 46.4%
  - Image → Text: R@1 30.0%, R@5 42.0%, R@10 54.0%

### Developer

- 256+ unit tests, 35 SQL validation tests, comprehensive UAT suite
- GitHub Actions CI (Python 3.12/3.13 matrix)
- LRU embedding cache for repeat queries
- Watch folder with path-keyed dedup and flush-on-shutdown
- Configurable max file size on ingest
- MIT License
