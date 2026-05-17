# RecallForge Memory MCP Roadmap

## Executive Summary

RecallForge should be built as a memory system first and a ranking system second.

The fast path is the product:
- ingest multimodal inputs into stable memory objects
- retrieve broadly and cheaply
- expose those memories cleanly through MCP tools and resources

The expensive path is a refinement:
- only run extra reasoning after cheap retrieval
- only on a strict top-K
- only when the cheap stage leaves real ambiguity

This document turns that principle into a staged roadmap we can ship and promote.

## Phase 1: Memory MCP Foundation

Goal:
- Make memories the primary unit, not files or chunks.

Current Linear fit:
- `REC-170`
- `REC-171`
- `REC-172`
- `REC-173`

What this phase delivers:
- stable `memory_id` values
- canonical parent memories for videos, documents, and text memories
- linked child assets like frames, transcript chunks, and sections
- `memory://` resources plus memory-centric MCP tools
- memory-level search rollups instead of duplicate child hits

Why this comes first:
- until search returns stable memory objects, later ranking and explanation work attaches to the wrong unit

## Phase 2: Cheap Broad Retrieval Core

Goal:
- Make the default path fast, recall-oriented, and available locally.

Current Linear fit:
- `REC-169`
- `REC-148`
- `REC-72`
- `REC-71`
- `REC-146`

Retrieval stack:
- vector search
- BM25 / FTS
- captions, transcripts, OCR, and derived text
- metadata and namespace filters
- reciprocal rank fusion

What this phase delivers:
- strong broad recall across text, image, video, audio, and document memories before any heavy reasoning runs

Why it comes before advanced ranking:
- expensive stages cannot recover candidates that cheap retrieval never found

## Phase 3: Memory-Aware Ranking and Scope

Goal:
- Make RecallForge feel like memory retrieval, not generic search.

Shipped Linear work:
- `REC-75`
- `REC-78`

What this phase delivers:
- scope policy across user, session, project, and profile
- recency, importance, TTL, and access-weighted ranking
- conversation memories and turn rollups
- background ingest consistency guarantees

Why it comes here:
- once recall is stable, policy signals can improve usefulness without hiding relevant memories

## Phase 4: Structured Memory Enrichment

Goal:
- Push more intelligence into ingest-time structure instead of query-time cost.

Shipped Linear work:
- `REC-76`

Likely follow-ons:
- related-memory linking
- timeline grouping
- participant and topic extraction

What this phase delivers:
- lightweight entities and relations
- richer memory-to-memory connections across modalities
- better “why did this memory match?” support without needing a big model for every query

Why it comes after foundation and scope:
- structured enrichment compounds complexity and works best once the parent-memory model is already stable

## Phase 5: Gated Expensive Intelligence

Goal:
- Reintroduce expensive reasoning only as a bounded late stage.

Current Linear fit:
- `REC-130`
- `REC-147`
- `REC-168`

Shipped Linear work:
- `REC-115`

What this phase should look like:
- retrieve first
- rerank only a strict top-K
- batch scoring instead of serial scoring
- cap visual token budgets
- cache safely with index-version invalidation
- skip expensive stages when the cheap stage already has a clear winner

Potential follow-on issues:
- confidence-gated rerank escalation policy
- memory explanation and evidence schema for late-stage scoring

Why this must stay late:
- the research is consistent: rerankers help when bounded, but they are the wrong default for broad retrieval

## Phase 6: Validation and Launch

Goal:
- Prove RecallForge as a memory MCP, not just a benchmark pipeline.

Shipped Linear work:
- `REC-153`
- `REC-61`
- `REC-33`

What this phase delivers:
- memory-level evaluation
- explanation quality checks
- latency and RSS budget enforcement
- real episodic corpora coverage
- MCP progress notifications for long-running search, ingest, batch, and rebuild workflows
- alpha and beta validation with real workflows
- explicit feature flags and local-only opt-in crash reports

Why this comes last:
- launch should validate the staged architecture in practice, not just synthetic retrieval quality

## Recommended Linear Shape

- Keep `Memory MCP Foundation` for `REC-170`, `REC-171`, `REC-172`, `REC-173`
- Keep `Retrieval and Ranking` for cheap broad retrieval work like `REC-169`, `REC-148`, `REC-72`, `REC-71`, `REC-146`
- Add a milestone such as `Memory Policy and Enrichment` for `REC-84`, `REC-83`, `REC-75`, `REC-76`, `REC-78`
- Keep `Research Queue` for gated expensive-stage work like `REC-130`, `REC-115`, `REC-147`, `REC-168`
- Keep `Benchmark Integrity` and `Launch and Distribution` for future public validation work

## Architecture Principle

- Cheap path is the product.
- Expensive path is the refinement.
- Structured reasoning belongs mostly at ingest time, or at the very end of search, not in the middle of broad candidate generation.

## References

- MCP server concepts and resources vs tools:
  [modelcontextprotocol/specification](https://github.com/modelcontextprotocol/specification/blob/main/docs/docs/learn/server-concepts.mdx)
- LanceDB tables, schema evolution, and retrieval patterns:
  [LanceDB docs](https://docs.lancedb.com/)
- Retrieve-then-rerank guidance:
  [SentenceTransformers retrieve & rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- Cross-encoder efficiency options:
  [SentenceTransformers CrossEncoder efficiency](https://sbert.net/docs/cross_encoder/usage/efficiency.html)
- Late-stage bounded ranking:
  [Vespa phased ranking](https://docs.vespa.ai/en/ranking/phased-ranking.html)
- Lightweight reranker option:
  [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
- Late interaction alternative:
  [ColBERT paper](https://arxiv.org/abs/2004.12832)
- Visual token budget controls:
  [Qwen2.5-VL docs](https://huggingface.co/docs/transformers/main/model_doc/qwen2_5_vl)

Last updated: 2026-03-21
