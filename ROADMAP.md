# RecallForge Roadmap

*Local multimodal memory for AI agents — where retrieval meets intelligence.*

**Current version:** v0.2.0  
**Status:** Active development (solo/small team)

This roadmap is honest about where RecallForge is, where it's going, and what's explicitly not planned. We prioritize depth over breadth, local-first over cloud-everything, and correctness over speed.

---

## Near-term (v0.2.x)

*Immediate gaps and quick wins that build on existing foundations.*

### MCP & Transport
- **HTTP MCP transport** — Move from stdio to a shared long-lived HTTP server to eliminate repeated model loading and enable persistent connections.
- **SSE streaming for MCP** — Add Server-Sent Events support for real-time result streaming and progress updates.

### Observability & Debugging
- **Explain mode for search** — Add `explain=True` flag that returns per-stage contribution scores (vector, keyword, rerank) so users understand *why* results rank where they do.
- **Debug logging** — Structured JSON logs at each pipeline stage for local tracing and performance analysis.

### Collections & Query
- **Multimodal query expansion** — When searching with image/video + text, expand the text portion through the query expander. Long-term: VLM-based expansion that captions visual content into searchable text.
- **Intent-aware query steering** — Allow optional `intent` parameter that pre-weights retrieval strategies (e.g., `intent="exact_lookup"` boosts BM25, `intent="semantic"` prioritizes vector).
- **Collection management API** — Add, remove, and rename collections without rebuilding the entire index.

### Distribution
- **Homebrew formula** — `brew install recallforge` for macOS users.
- **PyPI package polish** — Better metadata, type stubs, and documented API surface.
- **Claude plugin marketplace** — List RecallForge as an MCP plugin once HTTP transport is stable.

---

## Mid-term (v0.3.x - v0.5.x)

*Bigger features that expand what RecallForge can do and who can use it.*

### Language & Platform
- **JS/TS SDK** — Node.js/Bun-native client with full type safety. Brings RecallForge to the TypeScript agent ecosystem.
- **REST API** — Language-agnostic HTTP interface for polyglot environments.

### Multimodal Expansion
- **Audio modality** — Ingest and retrieve audio via embeddings ( Whisper-style or dedicated audio encoders).
- **OCR pipeline** — Native scanned document support using a lightweight OCR stage before embedding extraction.

### Data Model
- **Context tree for collections** — Hierarchical metadata attached to collections, returned with results. Enables agent memory with parent/child relationships.
- **Multi-get batch retrieval** — Retrieve multiple documents by glob pattern or doc ID in a single call.

### Agent Patterns
- **REC-111: Cross-modal search mode** — Skip reranker/BM25 for image results. Benchmark shows reranker drops text→image R@1 from 30% to 0%. Content-type-aware pipeline routing.
- **REC-112: `search_batch`** — Parallel multi-query with RRF merge. Single MCP call, multiple queries, fused results. Reduces round trips for agentic patterns.
- **REC-113: `search_iterative`** — Server-side multi-round retrieval. Strategies: refine (query refinement per round), expand (parallel variants), multi_query (decompose complex queries). Inspired by NVIDIA NeMo agentic retrieval (Apache 2.0).
- **REC-114: `explain_results`** — Per-document relevance explanations (BM25/vector/reranker score breakdown). Transparency for agentic retrieval.
- **REC-115: Semantic query cache** — Embedding-similarity cache (threshold 0.95) for cross-session acceleration.
- **REC-116: Model backend swappability** — Configurable embedding models with MRL dimension truncation, model registry, migration tooling. Current Qwen3-VL stack stays default.
- **Conversation history memory** — Built-in support for thread-aware retrieval with automatic turn-based indexing.
- **Entity extraction & memory** — Automatic named entity recognition and entity-centric retrieval paths.

### Scale (Pragmatic)
- **Index sharding** — Horizontal split of large collections across multiple SQLite files for local scale-out.
- **Incremental background indexing** — Queue-based ingestion that doesn't block queries.

---

## Long-term (v1.0+)

*Vision-level items that require significant R&D or architectural shifts.*

### Execution & Backends
- **ONNX Runtime backend** — Lighter alternative to PyTorch for environments where torch is too heavy.
- **Core ML backend** — Native Apple Neural Engine acceleration via Core ML alongside existing MLX support.

### Distribution & Ecosystem
- **Plugin system** — Third-party extensions for custom embedders, rerankers, and storage backends.
- **Prebuilt model packs** — Downloadable "knowledge bases" (legal, medical, code) with optimized embedders.

### Advanced Memory
- **Temporal memory** — Time-decayed retrieval weights and "forgetting" curves for long-lived agent memory.
- **Cross-session persistence patterns** — Standards for agent memory portability across restarts and devices.

---

## Not Planned

*Things we've considered and decided against, with reasons.*

### Cloud & Hosted
- **Managed cloud service** — RecallForge is explicitly local-first. We won't compete with Pinecone, Weaviate Cloud, or similar. Self-hosting is the point.
- **Multi-tenant SaaS** — Out of scope. The architecture assumes single-user, single-machine deployment.

### Scale (Extreme)
- **Distributed/clustered mode** — Billions of vectors is a non-goal. If you need that scale, use a cloud-native vector DB. RecallForge targets thousands to low-millions of documents.

### Specific Integrations

- **Native Windows support** — Runs on macOS and Linux. Windows via WSL.

### Feature Creep
- **Built-in LLM** — We embed and retrieve; we don't generate. Bring your own LLM.
- **Web UI** — Out of scope. RecallForge is a library and MCP server, not a standalone application.

---

## Contributing

This is a solo/small-team project with ambitious technical goals. Issues and PRs welcome, but check the roadmap first. We prioritize:

1. Correctness over speed
2. Local-first over cloud
3. API stability over feature count

Questions? Open an issue or reach out.

---

*Last updated: 2026-03-13*
