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

## MLX safety knobs

- `RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY`
  Concurrency ceiling for the heaviest MLX multimodal operations. Default is `1` for local safety.

- `RECALLFORGE_MLX_VIDEO_SAMPLE_FPS`
  Sampling rate for MLX raw-video processing. Lower values reduce memory pressure.

- `RECALLFORGE_MLX_VIDEO_MAX_FRAMES`
  Frame cap for MLX raw-video processing. The shipped default is intentionally conservative for local-agent use.

- `RECALLFORGE_MLX_VIDEO_FALLBACK_MAX_FRAMES`
  Frame cap for the ffmpeg-based frame-averaging fallback used when native video embedding is unavailable or downgraded.

- `RECALLFORGE_MLX_MIN_PIXELS`
  Lower bound for MLX processor visual resolution budgeting.

- `RECALLFORGE_MLX_MAX_PIXELS`
  Upper bound for MLX processor visual resolution budgeting.

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

- `RECALLFORGE_ENABLE_MEDIA_RERANKING`
  Enable multimodal reranking for image/video-involved searches. Disabled by default.

- `RECALLFORGE_ENABLE_RAW_VIDEO_QUERY_EMBEDDING`
  Enable raw video query embedding. On MLX, RecallForge now defaults to safer caption/transcript-first retrieval unless you explicitly enable this.

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
