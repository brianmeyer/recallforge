# RecallForge MCP Tool Reference

## Overview
RecallForge exposes a local Model Context Protocol (MCP) server over stdio for multimodal retrieval and memory operations. It supports text, images, video, and document ingest/search, plus runtime configuration.

Start the server:

```bash
recallforge serve --mode hybrid
```

Example MCP client config (Claude Desktop):

```json
{
  "mcpServers": {
    "recallforge": {
      "command": "recallforge",
      "args": ["serve", "--mode", "hybrid"]
    }
  }
}
```

## Tool Categories

### Search
- `search`
- `search_fts`
- `search_vec`

### Ingest
- `ingest`
- `index_document`
- `index_image`

### Memory
- `memory_add`
- `memory_update`
- `memory_delete`

### Admin / Introspection
- `status`
- `rebuild_fts`
- `list_collections`
- `list_namespaces`
- `batch`
- `get_config`
- `set_config`

---

## search

**Description:** Full hybrid search combining BM25, vector search, and reranking (in `hybrid` mode).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Conditionally* | — | Text query |
| image_path | string | Conditionally* | — | Image query path |
| video_path | string | Conditionally* | — | Video query path |
| limit | integer | No | 10 | Max results |
| collection | string | No | server default collection | Collection filter |
| content_type | string (`text`\|`image`\|`video`) | No | — | Content type filter |
| user_id | string | No | — | User namespace filter |
| session_id | string | No | — | Session namespace filter |
| project_id | string | No | — | Project namespace filter |
| profile | string | No | — | Profile namespace filter |

\* Exactly one of `query`, `image_path`, or `video_path` must be provided.

**Example Request:**
```json
{
  "query": "whiteboard diagram from last meeting",
  "limit": 5,
  "collection": "default"
}
```

**Example Response:**
```json
{
  "query": "whiteboard diagram from last meeting",
  "image_path": null,
  "video_path": null,
  "mode": "hybrid",
  "count": 1,
  "results": [
    {
      "filepath": "/notes/meeting.md",
      "title": "meeting.md",
      "score": 0.8921,
      "rerank_score": 0.9334,
      "rrf_rank": 1,
      "source": "hybrid",
      "snippet": "...",
      "user_id": null,
      "session_id": null,
      "project_id": null,
      "profile": null
    }
  ]
}
```

**Errors:**
- `INVALID_INPUT`: when zero or multiple query inputs are provided.
- `INTERNAL_ERROR`: uncaught exceptions in dispatch/call path.

**Notes:** Best default search tool for agents. Output includes fused/reranked metrics.

---

## search_fts

**Description:** Full-text BM25 search using Tantivy.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Yes (runtime) | — | Search query |
| limit | integer | No | 20 | Max results |
| collection | string | No | server default collection | Collection filter |
| content_type | string (`text`\|`image`\|`video`) | No | — | Content type filter |
| user_id | string | No | — | User namespace filter |
| session_id | string | No | — | Session namespace filter |
| project_id | string | No | — | Project namespace filter |
| profile | string | No | — | Profile namespace filter |

**Example Request:**
```json
{
  "query": "deployment checklist",
  "limit": 10
}
```

**Example Response:**
```json
{
  "query": "deployment checklist",
  "count": 1,
  "results": [
    {
      "filepath": "/docs/runbook.md",
      "title": "runbook.md",
      "score": 13.2401,
      "source": "fts",
      "user_id": null,
      "session_id": null,
      "project_id": null,
      "profile": null
    }
  ]
}
```

**Errors:**
- `INVALID_INPUT`: when `query` is empty/missing.
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Fast lexical retrieval. Does not use embedding similarity.

---

## search_vec

**Description:** Vector similarity search using embeddings/ANN.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| query | string | Conditionally* | — | Text query |
| image_path | string | Conditionally* | — | Image query path |
| video_path | string | Conditionally* | — | Video query path |
| limit | integer | No | 20 | Max results |
| collection | string | No | server default collection | Collection filter |
| content_type | string (`text`\|`image`\|`video`) | No | — | Content type filter |
| user_id | string | No | — | User namespace filter |
| session_id | string | No | — | Session namespace filter |
| project_id | string | No | — | Project namespace filter |
| profile | string | No | — | Profile namespace filter |

\* Exactly one of `query`, `image_path`, or `video_path` must be provided.

**Example Request:**
```json
{
  "image_path": "/tmp/query.png",
  "limit": 8
}
```

**Example Response:**
```json
{
  "query": null,
  "image_path": "/tmp/query.png",
  "video_path": null,
  "count": 2,
  "results": [
    {
      "filepath": "/images/diagram.png",
      "title": "diagram.png",
      "score": 0.8123,
      "source": "vec",
      "user_id": null,
      "session_id": null,
      "project_id": null,
      "profile": null
    }
  ]
}
```

**Errors:**
- `INVALID_INPUT`: when query inputs are missing/ambiguous.
- `NOT_FOUND`: when backend lacks raw video query support (`video_path`).
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Uses only vector similarity, no BM25/reranking.

---

## ingest

**Description:** Unified ingestion for raw text, a single file, or a folder; auto-routes by modality.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| text | string | Conditionally* | — | Raw text content |
| path | string | No | — | Memory path key for raw text ingest |
| file_path | string | Conditionally* | — | Path to single file |
| folder_path | string | Conditionally* | — | Path to folder |
| recursive | boolean | No | true | Recurse subfolders |
| collection | string | No | server default collection | Collection name |
| content_types | array[string] | No | ["text","image","video","document"] | Allowed content types |
| include_globs | array[string] | No | — | Include globs |
| exclude_globs | array[string] | No | — | Exclude globs |
| max_file_size_mb | integer | No | server default (100) | Max file size in MB |
| user_id | string | No | — | User namespace |
| session_id | string | No | — | Session namespace |
| project_id | string | No | — | Project namespace |
| profile | string | No | — | Profile namespace |

\* Use exactly one primary input source: `text`, `file_path`, or `folder_path`.

**Example Request:**
```json
{
  "folder_path": "/Users/me/Documents",
  "recursive": true,
  "collection": "work",
  "content_types": ["text", "document"],
  "max_file_size_mb": 50
}
```

**Example Response:**
```json
{
  "success": true,
  "collection": "work",
  "indexed_text": 42,
  "indexed_images": 0,
  "indexed_videos": 0,
  "indexed_documents": 17,
  "skipped": 3
}
```

**Errors:**
- `INVALID_INPUT`: returned by storage/validation for invalid combinations/inputs.
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** This is the recommended ingest entry point. `set_config` can change default `collection`/`max_file_size_mb` used when omitted.

---

## index_document

**Description:** Index a text document directly.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| path | string | Yes | — | Document path key |
| text | string | Yes | — | Document text |
| collection | string | No | server default collection | Collection name |

**Example Request:**
```json
{
  "path": "notes/project-alpha.md",
  "text": "Decision log...",
  "collection": "work"
}
```

**Example Response:**
```json
{
  "success": true,
  "path": "notes/project-alpha.md",
  "collection": "work",
  "hash": "abc123..."
}
```

**Errors:**
- `INVALID_INPUT`: when `path` or `text` is missing/empty.
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Lower-level than `ingest`; good for direct programmatic writes.

---

## index_image

**Description:** Index an image for cross-modal retrieval.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| path | string | Yes | — | Image file path |
| collection | string | No | server default collection | Collection name |

**Example Request:**
```json
{
  "path": "/Users/me/Pictures/board.png",
  "collection": "default"
}
```

**Example Response:**
```json
{
  "success": true,
  "path": "/Users/me/Pictures/board.png",
  "collection": "default",
  "hash": "def456..."
}
```

**Errors:**
- `INVALID_INPUT`: when `path` is missing/empty.
- `NOT_FOUND`: when file does not exist.
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Requires a readable local file path on the server host.

---

## memory_add

**Description:** Add (or replace at same path) a memory entry with optional metadata.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| path | string | Yes | — | Memory key/path |
| text | string | Yes | — | Memory content |
| collection | string | No | server default collection | Collection name |
| user_id | string | No | — | User namespace |
| session_id | string | No | — | Session namespace |
| project_id | string | No | — | Project namespace |
| profile | string | No | — | Profile namespace |
| importance | number | No | — | 0.0 to 1.0 importance score |
| ttl_seconds | integer | No | — | TTL in seconds; `0`/`null` means no expiration |
| tags | array[string] | No | — | Tags |

**Example Request:**
```json
{
  "path": "agents/molly/preferences/tone",
  "text": "Direct, concise, no fluff.",
  "importance": 0.9,
  "tags": ["persona", "style"]
}
```

**Example Response:**
```json
{
  "success": true,
  "path": "agents/molly/preferences/tone",
  "collection": "default",
  "hash": "789abc...",
  "operation": "add",
  "user_id": null,
  "session_id": null,
  "project_id": null,
  "profile": null,
  "importance": 0.9,
  "ttl_seconds": null,
  "tags": ["persona", "style"]
}
```

**Errors:**
- `INVALID_INPUT`: when `path` or `text` is missing/empty.
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Internally uses upsert semantics.

---

## memory_update

**Description:** Update an existing memory entry (same upsert backend path as add).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| path | string | Yes | — | Memory key/path |
| text | string | Yes | — | Updated memory content |
| collection | string | No | server default collection | Collection name |
| user_id | string | No | — | User namespace |
| session_id | string | No | — | Session namespace |
| project_id | string | No | — | Project namespace |
| profile | string | No | — | Profile namespace |
| importance | number | No | — | 0.0 to 1.0 importance score |
| ttl_seconds | integer | No | — | TTL in seconds; `0`/`null` means no expiration |
| tags | array[string] | No | — | Tags |

**Example Request:**
```json
{
  "path": "agents/molly/preferences/tone",
  "text": "Direct, sharp, action-first.",
  "tags": ["persona", "style"]
}
```

**Example Response:**
```json
{
  "success": true,
  "path": "agents/molly/preferences/tone",
  "collection": "default",
  "hash": "fed321...",
  "operation": "update",
  "user_id": null,
  "session_id": null,
  "project_id": null,
  "profile": null,
  "importance": null,
  "ttl_seconds": null,
  "tags": ["persona", "style"]
}
```

**Errors:**
- `INVALID_INPUT`: when `path` or `text` is missing/empty.
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Same storage path as `memory_add`; difference is semantic intent (`operation` field).

---

## memory_delete

**Description:** Delete/deactivate a memory entry and remove embeddings.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| path | string | Yes | — | Memory key/path |
| collection | string | No | server default collection | Collection name |
| user_id | string | No | — | User namespace |
| session_id | string | No | — | Session namespace |
| project_id | string | No | — | Project namespace |
| profile | string | No | — | Profile namespace |

**Example Request:**
```json
{
  "path": "agents/molly/preferences/tone"
}
```

**Example Response:**
```json
{
  "success": true,
  "path": "agents/molly/preferences/tone",
  "removed_vectors": 1
}
```

**Errors:**
- `INVALID_INPUT`: when `path` is missing/empty.
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Namespace filters must match the original memory to delete it.

---

## status

**Description:** Return server/model/database status.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| — | — | No | — | No input params |

**Example Request:**
```json
{}
```

**Example Response:**
```json
{
  "version": "0.1.0",
  "models": {
    "backend": "mlx",
    "device": "mps",
    "dtype": "float16",
    "embedder_loaded": true,
    "reranker_loaded": true,
    "expander_loaded": true,
    "memory_gb": 1.9,
    "quantization": "4bit",
    "mode": "hybrid"
  },
  "database": {
    "embeddings_count": 1024,
    "documents_count": 256
  }
}
```

**Errors:**
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Useful heartbeat/health endpoint for agent startup checks.

---

## rebuild_fts

**Description:** Rebuild Tantivy full-text index.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| — | — | No | — | No input params |

**Example Request:**
```json
{}
```

**Example Response:**
```json
{
  "success": true,
  "message": "FTS index rebuilt"
}
```

**Errors:**
- `BACKEND_ERROR`: when FTS rebuild throws storage/backend exception.
- `INTERNAL_ERROR`: uncaught exceptions outside handler-level catch.

**Notes:** Run after large ingest operations if lexical results look stale.

---

## list_collections

**Description:** List unique collection names, optionally filtered by namespace.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| user_id | string | No | — | User namespace filter |
| session_id | string | No | — | Session namespace filter |
| project_id | string | No | — | Project namespace filter |
| profile | string | No | — | Profile namespace filter |

**Example Request:**
```json
{
  "user_id": "u123"
}
```

**Example Response:**
```json
{
  "count": 2,
  "collections": ["default", "work"]
}
```

**Errors:**
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Good for discovering tenant-visible collections.

---

## list_namespaces

**Description:** List unique namespace combinations (`user_id`, `session_id`, `project_id`, `profile`).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| collection | string | No | — | Optional collection filter |

**Example Request:**
```json
{
  "collection": "work"
}
```

**Example Response:**
```json
{
  "count": 1,
  "namespaces": [
    {
      "user_id": "u123",
      "session_id": "s456",
      "project_id": "p789",
      "profile": "prod"
    }
  ]
}
```

**Errors:**
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Useful for debugging multi-tenant memory partitioning.

---

## batch

**Description:** Execute multiple RecallForge tool calls in one request (max 20 operations).

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| operations | array[object] | Yes | — | Operations list (max 20) |

Operation object schema:
- `tool` (string, required)
- `arguments` (object, required)

**Example Request:**
```json
{
  "operations": [
    { "tool": "memory_add", "arguments": { "path": "x", "text": "hello" } },
    { "tool": "search", "arguments": { "query": "hello", "limit": 3 } }
  ]
}
```

**Example Response:**
```json
{
  "batch_results": [
    { "index": 0, "tool": "memory_add", "status": "success", "result": { "success": true } },
    { "index": 1, "tool": "search", "status": "success", "result": { "count": 1, "results": [] } }
  ],
  "total": 2,
  "succeeded": 2,
  "failed": 0
}
```

**Errors:**
- `INVALID_INPUT`: not emitted as structured `_error_response`; invalid list/type returns plain `{ "error": ... }` payload.
- `INTERNAL_ERROR`: uncaught exceptions in top-level `call_tool`.

**Notes:**
- Nested `batch` calls are explicitly rejected per item.
- If operations exceed 20, handler returns plain `{ "error": ... }` payload.

---

## get_config

**Description:** Return current effective server config.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| — | — | No | — | No input params |

**Example Request:**
```json
{}
```

**Example Response:**
```json
{
  "version": "0.1.0",
  "backend": "mlx",
  "mode": "hybrid",
  "quantize": "4bit",
  "data_dir": "/Users/me/.recallforge",
  "collection": "default",
  "max_file_size_mb": 100
}
```

**Errors:**
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** Reflects mutable runtime values (`mode`, `collection`, `max_file_size_mb`) when changed via `set_config`.

---

## set_config

**Description:** Update safe runtime config fields.

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| mode | string (`embed`\|`hybrid`) | No | current mode | Search mode |
| collection | string | No | current default collection | Default collection for tools that omit `collection` |
| max_file_size_mb | number | No | current max | Default ingest file-size limit (must be >= 1) |

**Example Request:**
```json
{
  "mode": "hybrid",
  "collection": "work",
  "max_file_size_mb": 64
}
```

**Example Response:**
```json
{
  "version": "0.1.0",
  "backend": "mlx",
  "mode": "hybrid",
  "quantize": "4bit",
  "data_dir": "/Users/me/.recallforge",
  "collection": "work",
  "max_file_size_mb": 64
}
```

**Errors:**
- `INVALID_INPUT`: when trying to set immutable fields (`backend`, `quantize`, `data_dir`), unknown fields, invalid mode, empty collection, or invalid `max_file_size_mb`.
- `INTERNAL_ERROR`: uncaught exceptions.

**Notes:** `backend`, `quantize`, and `data_dir` require server restart and are intentionally immutable at runtime.

---

## Recommended Agent Workflows

### 1) Index a folder then search it
1. Call `ingest` with `folder_path`, filters, and optional namespace.
2. Optionally call `rebuild_fts` after heavy ingest.
3. Query with `search` for best quality, or `search_fts` / `search_vec` for targeted behavior.

### 2) Add agent memory, search later
1. Store facts/preferences with `memory_add` (include `user_id`/`session_id`/`project_id`/`profile` when needed).
2. Update with `memory_update` as facts evolve.
3. Retrieve later with `search` + matching namespace filters.

### 3) Configure mode, then ingest
1. Inspect config using `get_config`.
2. Set desired runtime defaults with `set_config` (for mode, default collection, max file size).
3. Run `ingest` without repeating shared defaults.

## Error Code Reference
Structured errors returned via `_error_response(code, message, details)`:

- `INVALID_INPUT`
  - Input is missing, malformed, or violates validation rules.
- `NOT_FOUND`
  - Resource is missing (for example, image path does not exist) or capability is unavailable (e.g., unsupported raw video query backend).
- `BACKEND_ERROR`
  - Backend/storage operation failed in handler-managed exception path (currently used in `rebuild_fts`).
- `INTERNAL_ERROR`
  - Unhandled exception at top-level tool dispatch.

### Error payload shape
```json
{
  "error": true,
  "code": "INVALID_INPUT",
  "message": "...",
  "details": {}
}
```

### Note on non-standard errors
`batch` currently returns plain `{ "error": "..." }` strings for some validation failures (non-list `operations`, oversized batches), rather than structured `_error_response` codes.
