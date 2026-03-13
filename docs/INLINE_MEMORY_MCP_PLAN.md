# Inline Memory MCP Plan (QMD parity + agent write ergonomics)

Date: 2026-03-10

## Goal
Make RecallForge useful as inline agent memory, not just retrieval over pre-indexed files.

## What exists today
- Read/search tools: `search`, `search_fts`, `search_vec`
- Write/index tools: `index_document`, `index_image` (path/text oriented)
- Missing true memory CRUD ergonomics for agents.

## Phase 1 (this run)
1. **New MCP tools**
   - `memory_add`: add/append memory item with generated id/path
   - `memory_update`: replace memory by id/path
   - `memory_delete`: deactivate/delete memory by id/path
   - `index_folder`: bulk index a directory recursively with include/exclude filters
2. **Storage correctness upgrades**
   - Add "replace by path" behavior: clean old embeddings for path before re-index
   - Add delete/deactivate that also removes associated embeddings by content hash
3. **Tool contracts**
   - Structured JSON responses with counts/ids/errors
   - Idempotent updates where possible
4. **Tests**
   - UAT for CRUD tools + folder indexing
   - Regression test for repeated updates not duplicating vectors
5. **Docs**
   - README MCP tools section update
   - Quick examples for agent memory flows

## Phase 2 (next)
- `watch_folder` sidecar/daemon mode
- namespace/profile semantics (user/session/project)
- memory importance/ttl/tags
- multimodal memory item upsert (`memory_add` with image+text payload)

## Success criteria
- Agent can create/update/delete memories without manual file prep
- Agent can point to a folder and ingest it in one call
- Re-indexing same logical memory does not bloat embeddings
- Existing search quality unchanged
