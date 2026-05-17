# Changelog

All notable changes to RecallForge will be documented in this file.

## [Unreleased]

- Replaced the tiny UAT video clips with compact episodic-memory fixtures, richer transcript sidecars, related artifact metadata, and regression coverage for the video corpus.

## [0.2.1] — 2026-05-17

RecallForge 0.2.1 is a hardening release for the post-0.2.0 memory and MLX safety work. It keeps the local-first MCP surface intact while making Apple Silicon multimodal validation safer and memory behavior more predictable.

### MLX Safety

- Serialized heavy MLX multimodal operations by default to reduce overlapping unified-memory pressure
- Added conservative MLX video frame/pixel budgets and runtime knobs for advanced operators
- Made MLX raw video query embedding fail open to caption/transcript-first retrieval by default
- Kept native qwen-vl-utils video decoding off the default MLX path unless explicitly enabled
- Added a crash-safe MLX benchmark smoke profile with RSS telemetry and stop-limit support

### Memory and Retrieval

- Added Qwen-backed query expansion paths for text and media-query probes
- Persisted canonical parent memory summaries during ingest
- Added memory-summary and rollup provenance in search results
- Added file-based query routing and stronger guards around expansion/file query behavior
- Added derived media tags for root memories and hardened generated tag parsing
- Added explicit memory-level benchmark scoring separate from child-asset hits

### Reliability

- Added OCR fallback support for scanned PDF memories
- Fixed watch-folder startup race behavior and stabilized async watch tests
- Hardened batch tag ordering and rolled memory path canonicalization
- Expanded regression coverage for MLX guardrails, query expansion, storage, documents, watch folders, and benchmark output

## [0.2.0] — 2026-03-22

RecallForge 0.2.0 makes the project much more usable as a local multimodal memory MCP. This release improves cross-modal retrieval, adds a stronger MCP surface for agents, and lays the foundation for memory-level results instead of isolated file fragments.

**Note:** Python 3.12 and 3.13 supported. Python 3.14 is not yet supported due to the pinned `pyarrow` wheel range for this release.

### Memory MCP

- First-class `memory_id` support in storage and search
- Parent/child memory linkage for derived assets such as frames, transcripts, and document sections
- Memory-centric MCP capabilities including memory lookup and listing
- `memory://` resources for stable memory access
- Memory-level result rollup so related assets can surface as a single memory hit

### Search and Retrieval

- Hybrid search support for image and video retrieval paths
- Parallel `search_batch` MCP tool for multi-query workflows
- `explain_results` MCP tool for search transparency and score inspection
- Ingest-time image/video captioning to improve BM25 and keyword recall
- Optional vision-language query expansion and stronger cross-modal benchmark coverage
- PDF vision fallback so scanned or image-heavy documents remain searchable

### Runtime and Product

- HTTP transport for the MCP server with persistent model loading
- Configurable model selection via environment variables and MCP config
- Qwen3.5-0.8B captioning for lighter-weight media understanding
- Version-aware server reporting and improved health/config surfaces
- Collection management API for better multi-namespace organization

### Reliability and Performance

- Fast media retrieval defaults for better real-world latency
- Query-side media handling fixes so reranking is no longer blind to media queries
- Thread-safe model swaps plus explicit memory cleanup
- Video frame sampling and cleanup fixes to reduce OOMs and tensor retention
- Expanded UAT, observability, score audits, and release validation coverage
- Trusted publishing workflow for tag-based PyPI releases

## [0.1.0] — 2026-03-13

First public release. The local multimodal memory engine for AI agents.

**Note:** Python 3.12 and 3.13 supported. Python 3.14 is not yet supported due to pyarrow wheel availability.

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
