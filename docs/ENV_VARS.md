# RecallForge Environment Variables

This is the canonical reference for all `RECALLFORGE_*` environment variables used in the codebase.

## Runtime selection

- `RECALLFORGE_BACKEND`  
  Backend selector: `auto` (default), `torch`, `mlx`.

- `RECALLFORGE_MODE`  
  Search mode: `embed` or `hybrid`.

- `RECALLFORGE_MLX_QUANTIZE`  
  MLX quantization mode: `bf16` or `4bit`.

- `RECALLFORGE_DISABLE_MLX`  
  Disable MLX backend probing when set to `1`.

- `RECALLFORGE_STORAGE`  
  Storage backend selector (currently `lancedb`).

- `RECALLFORGE_STORE_PATH`  
  Path to the RecallForge data store.

## Search pipeline tuning

- `RECALLFORGE_OVERFETCH_FACTOR`  
  Candidate overfetch multiplier before final trim.

- `RECALLFORGE_MAX_CANDIDATES`  
  Hard cap for candidate pool size before reranking.

- `RECALLFORGE_RERANK_TOP_K`  
  Number of top RRF candidates to rerank.

## Server behavior

- `RECALLFORGE_TRACE`  
  Enables trace logging for MCP tools when set to `1`.

- `RECALLFORGE_MCP_MAX_CONCURRENCY`  
  Maximum number of blocking MCP tool operations run concurrently.

## Storage/FTS internals

- `RECALLFORGE_BM25_FALLBACK_MAX_ROWS`  
  Row limit used by BM25 fallback recovery paths.

- `RECALLFORGE_BULK_FLUSH_DOCS`  
  Batch flush threshold for document table writes.

- `RECALLFORGE_BULK_FLUSH_EMBEDDINGS`  
  Batch flush threshold for embedding table writes.
