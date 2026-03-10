# RecallForge Code Audit

## Scope
- Audited `src/recallforge/` end-to-end: CLI, backend selection, model backends, storage, search pipeline, and MCP server.
- Audited `Qwen3-VL-Embedding/src/models/` integration points used by RecallForge.
- Cross-checked behavior against unit tests, live tests, and UAT scripts in `tests/` and `tests/uat/`.
- Attempted to run tests for validation. `pytest -q tests/test_backends.py` aborts during import (fatal MLX crash in `mlx_backend.py:24`).

## High-Risk Findings Summary

| ID | Severity | Finding |
|---|---|---|
| F1 | Critical | Importing `recallforge` can hard-crash the Python process due to eager MLX import path (`src/recallforge/backends/__init__.py:11`, `src/recallforge/backends/mlx_backend.py:23-27`, `src/recallforge/__init__.py:108`). |
| F2 | Critical | CLI/runtime env overrides are mostly ineffective because config is cached at import time (`src/recallforge/__init__.py:30-33`, `:50-52`; `src/recallforge/cli.py:143-150`, `:233-238`). |
| F3 | High | Qwen model import pathing is inconsistent and fragile; reranker path fallback is incomplete (`src/recallforge/backends/torch_backend.py:86-93`, `:251-255`). |
| F4 | High | Search ranking weights are nondeterministic due to async result insertion order and order-dependent RRF weights (`src/recallforge/search.py:223-227`, `:243-247`). |
| F5 | High | Server “graceful shutdown” flag is never consumed (`src/recallforge/server.py:27`, `:30-35`), so signal handling is incomplete. |
| F6 | Medium | Re-indexing updated documents can leave stale embeddings; doc registry updates but old content vectors are not cleaned (`src/recallforge/storage/lancedb_backend.py:421-433`, `:890-905`). |

## End-to-End Entry Points and Call Chains

### CLI / module entry
- `python -m recallforge` -> `src/recallforge/__main__.py:10-13` -> `recallforge.cli.main()`.
- `recallforge` console script -> `pyproject.toml:47-48` -> `recallforge.cli:main`.

### `recallforge serve`
- `cli.main()` -> `cmd_serve()` (`src/recallforge/cli.py:140-154`)
- `cmd_serve()` sets env vars (`:143-150`) -> `server.run_server()` (`:152-153`)
- `run_server()` -> `asyncio.run(server.main())` (`src/recallforge/server.py:418-420`)
- `server.main()`:
  - reads mode/store env (`:388-390`)
  - `get_backend()` + `get_storage()` (`:392-393`)
  - `backend.warm_up()` (`:399`)
  - `create_server(..., mode=mode)` (`:406`)
  - MCP stdio loop `server.run(...)` (`:410-415`)

### `recallforge index`
- `cli.main()` -> `cmd_index()` (`src/recallforge/cli.py:157-220`)
- `get_storage()` + `get_backend()` (`:159-164`)
- `backend._load_embedder()` warm-up (`:166`)
- For each file:
  - text -> `storage.index_document(..., embed_func=backend.embed_text)` (`:182-188`, `:208-214`)
  - image -> `storage.index_image(..., embed_func=backend.embed_image)` (`:173-177`, `:199-203`)
- `index_document()` pipeline (`src/recallforge/storage/lancedb_backend.py:860-909`): hash -> content table -> document table -> chunk -> embed -> embeddings table -> FTS rebuild.
- `index_image()` pipeline (`:911-973`): path-hash -> content/doc rows -> image embedding -> embedding row -> FTS rebuild.

### `recallforge search`
- `cli.main()` -> `cmd_search()` (`src/recallforge/cli.py:229-263`)
- `get_storage()` + `get_backend()` (`:236-239`)
- `HybridSearcher(...).search(query)` (`:242-250`)
- Search pipeline (`src/recallforge/search.py:351-389`):
  - BM25 probe (`:369` -> `storage.search_fts`)
  - optional expansion (`:373-375` -> `backend.expand_query`)
  - parallel vec/BM25 expansion searches (`:377` -> `_run_parallel_searches`)
  - add original FTS (`:380`)
  - RRF fusion (`:383`)
  - rerank (`:386` -> `backend.rerank`)
  - blend + return (`:389`)

### `recallforge status`
- `cli.main()` -> `cmd_status()` (`src/recallforge/cli.py:265-296`)
- `backend.get_info()` + storage counts (`:273`, `:292-293`).

### MCP tool chains
- Server registered in `create_server()` (`src/recallforge/server.py:42-173`)
- Tool dispatch (`:150-171`):
  - `search` -> `_handle_search` (`:176-215`) -> `HybridSearcher.search`
  - `search_fts` -> `_handle_search_fts` (`:217-249`) -> `storage.search_fts`
  - `search_vec` -> `_handle_search_vec` (`:251-286`) -> `backend.embed_text` -> `storage.search_vec`
  - `index_document` -> `_handle_index_document` (`:288-313`) -> `storage.index_document`
  - `index_image` -> `_handle_index_image` (`:315-340`) -> `storage.index_image`
  - `status` -> `_handle_status` (`:342-370`)
  - `rebuild_fts` -> `_handle_rebuild_fts` (`:372-380`)

## Module-by-Module Audit

## `src/recallforge/__main__.py`
1. Entry points and call chain
- Sole entry point delegates module execution to CLI (`__main__.py:10-13`).

2. Import issues
- None directly.

3. Dead code / unreachable
- None.

4. Error handling gaps
- None (delegates to CLI).

5. Test expectation mismatches
- None specific.

## `src/recallforge/cli.py`
1. Entry points and call chain
- Main parser + 4 subcommands (`cli.py:19-137`).
- `serve`, `index`, `search`, `status` chains documented above.

2. Import issues
- Imports frozen config constants (`RECALLFORGE_*`) from package init (`cli.py:16`), which contributes to stale env behavior when args override env later.

3. Dead code / unreachable
- Unused `json` import (`cli.py:12`).
- Imported constants are not used directly except as static references in help text context.

4. Error handling gaps
- `cmd_index()` only catches per-file exceptions in directory traversal (`:216-217`), not single-file failures (`:170-189`).
- `cmd_index()` always returns `0` even when some files failed (`:220`).
- Text indexing assumes UTF-8 for all non-image files (`:180`, `:206`), no binary/encoding fallback.

5. Test expectation mismatches
- UAT scripts call `--mode embed/hybrid/full` expecting behavior differences, but CLI mode override is often ignored due frozen config values in `get_backend()` (see `__init__.py` section). Tests mostly check “command ran” instead of mode-specific behavior (`tests/uat/test_cli.sh:85-107`).

## `src/recallforge/__init__.py`
1. Entry points and call chain
- Factory entry points: `get_backend()` (`:36-79`) and `get_storage()` (`:82-105`).

2. Import issues
- Critical design issue: environment values are read once at import (`:30-33`) and reused forever (`:50-52`, `:96`).
- Eager convenience imports trigger heavy optional backend imports (`:108-110`), including MLX path.

3. Dead code / unreachable
- None, but convenience imports are risky side effects for a package init.

4. Error handling gaps
- `get_backend()` assumes optional backend imports are safe. If MLX import hard-fails (not ImportError), process aborts before fallback.

5. Test expectation mismatches
- Tests and CLI flows assume runtime env changes take effect; they do not if package was already imported.

## `src/recallforge/backends/__init__.py`
1. Entry points and call chain
- Re-exports backend classes (`backends/__init__.py:9-18`), imported transitively by `recallforge.__init__`.

2. Import issues
- Eagerly imports `MLXBackend` at module import (`:11`), making optional backend non-optional in practice.

3. Dead code / unreachable
- None.

4. Error handling gaps
- No guard around MLX import path here.

5. Test expectation mismatches
- Unit tests that only need base classes still import through package namespace and trigger MLX import.

## `src/recallforge/backends/base.py`
1. Entry points and call chain
- Defines `ModelBackend` ABC contract used by search, CLI, and server.

2. Import issues
- None.

3. Dead code / unreachable
- None significant.

4. Error handling gaps
- None in ABC.

5. Test expectation mismatches
- Tests align with API contract.

## `src/recallforge/backends/torch_backend.py`
1. Entry points and call chain
- Backend used by `get_backend()` for torch/auto fallback.
- `embed_*` -> `_load_embedder()` -> Qwen embedder.
- `rerank` -> `_load_reranker()` -> Qwen reranker.
- `expand_query` -> `_load_expander()` -> HF CausalLM generation.

2. Import issues (including `models/qwen3_vl_embedding`)
- `_add_qwen_repo_to_path()` computes wrong repo path (`torch_backend.py:89`): points to `<repo>/src/Qwen3-VL-Embedding`, not `<repo>/Qwen3-VL-Embedding`.
- Embedder import has 3-stage fallback and eventually computes correct path (`:161-175`).
- Reranker import does **not** have equivalent robust fallback (`:251-255`), so reranker can fail even when embedder works.
- Uses ambiguous top-level import `models.qwen3_vl_embedding` / `models.qwen3_vl_reranker` (`:162`, `:252`), vulnerable to collisions with unrelated `models` packages.

3. Dead code / unreachable
- `dataclass` and `Optional` imports unused.

4. Error handling gaps
- `_load_reranker()` import failures are not wrapped with actionable error messages.
- `warm_up()` can print timings but does not isolate partial model-load failures by stage.

5. Test expectation mismatches
- Tests assume full backend functionality in source tree and package installs, but path logic depends on repo layout and not packaging.

## `src/recallforge/backends/mlx_backend.py`
1. Entry points and call chain
- Used by `get_backend()` when selected or in auto mode on Apple Silicon.

2. Import issues
- Module-level `import mlx.core as mx` in a `try/except ImportError` block (`mlx_backend.py:23-27`) does not protect against non-ImportError hard failures. In this environment, this import aborts Python.
- `_add_qwen_repo_to_path()` has same wrong path pattern as torch backend (`:88-92`).

3. Dead code / unreachable
- `_add_qwen_repo_to_path()` appears unused by MLX runtime logic (MLX backend does not import local Qwen model modules).
- Unused import `apply_chat_template` in `_load_embedder()` (`:106`).

4. Error handling gaps
- Optional backend safety is insufficient; a bad MLX runtime can take down all RecallForge imports.

5. Test expectation mismatches
- Tests assume MLX is optional/skip-able, but eager import can fail before skip logic executes.

## `src/recallforge/storage/base.py`
1. Entry points and call chain
- Defines storage ABC used by CLI, server, and search.

2. Import issues
- None.

3. Dead code / unreachable
- None.

4. Error handling gaps
- None in interface layer.

5. Test expectation mismatches
- Tests align with declared interface.

## `src/recallforge/storage/lancedb_backend.py`
1. Entry points and call chain
- Called via `get_storage()` and by all indexing/search flows.
- Core high-level entry points: `index_document()` (`:860`), `index_image()` (`:911`), `search_fts()` (`:676`), `search_vec()` (`:732`).

2. Import issues
- None critical in this module.

3. Dead code / unreachable
- Unused imports: `numpy as np`, `dataclass` already used; `np` appears unused.

4. Error handling gaps
- Many broad `except Exception: pass/return` blocks hide root causes (`:434-435`, `:499-500`, `:519-520`, `:546-547`, `:709-710`, etc.).
- `search_vec()` does not guard vector-search builder failures with fallback (`:757-762`).
- Re-indexing updated document path does not delete old embeddings tied to previous `content_hash` (`insert_document` update `:421-433` + `index_document` write path `:890-905`), causing stale vector rows.

5. Test expectation mismatches
- Tests validate duplicate re-index for same content hash, but not changed-content updates, so stale-vector behavior is not covered.

## `src/recallforge/search.py`
1. Entry points and call chain
- Main entry: `HybridSearcher.search()` (`:351-389`) and helper `hybrid_query()` (`:392-430`).

2. Import issues
- None critical.

3. Dead code / unreachable
- `_vector_search()` is defined but never called (`:105-113`).
- Unused module imports `json` and `os` (`:14-15`).

4. Error handling gaps
- Batch embedding failure in `_run_parallel_searches()` only logs and silently drops all vector branches (`:154-159`).
- Task failures are printed but not surfaced to caller (`:228-230`).

5. Test expectation mismatches
- RRF list weighting is order-dependent (`:243-247`) while result-list insertion order is async completion order (`:223-227`), producing nondeterministic ranking behavior that tests do not lock down.

## `src/recallforge/server.py`
1. Entry points and call chain
- `run_server()` (`:418-420`) -> `main()` (`:383-416`) -> `create_server()` (`:42-173`) -> MCP handlers.

2. Import issues
- None separate from package import issues.

3. Dead code / unreachable
- `_shutdown_requested` is set by signal handler (`:27`, `:30-35`) but never checked.
- `_server` global is assigned (`:407`) but otherwise unused.

4. Error handling gaps
- Tool handler catches all exceptions and returns plain text JSON error (`:170-171`) without traceback/typed error metadata.
- Signal path does not actually stop stdio server loop.

5. Test expectation mismatches
- UAT claims graceful shutdown coverage, but test only verifies signal handlers are registered (`tests/uat/test_mcp_server.sh:196-205`). No behavior validates shutdown flag usage.

## `Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py`
1. Entry points and call chain
- RecallForge uses `Qwen3VLEmbedder.process()` via torch backend (`torch_backend.py:180-205`, `:230`).

2. Import issues
- Depends on `transformers.models.qwen3_vl.*` (`qwen3_vl_embedding.py:12-13`), which requires recent Transformers versions.
- RecallForge dependency floor is much lower (`pyproject.toml:25` is `transformers>=4.40`), while Qwen submodule expects newer (`Qwen3-VL-Embedding/pyproject.toml:20` is `>=4.57.3`).

3. Dead code / unreachable
- `PAD_TOKEN` constant unused (`qwen3_vl_embedding.py:38`).
- `_truncate_tokens()` appears unused in this file (`:206-223`).
- `check_model_inputs` import/fallback is not active because decorator is commented (`:19-23`, `:89`).

4. Error handling gaps
- `_preprocess_inputs()` catches broad exceptions and substitutes `NULL` prompt (`:339-353`), potentially masking malformed multimodal inputs.

5. Test expectation mismatches
- No RecallForge tests directly cover this module-level behavior or these fallback semantics.

## `Qwen3-VL-Embedding/src/models/qwen3_vl_reranker.py`
1. Entry points and call chain
- RecallForge uses `Qwen3VLReranker.process()` via torch backend rerank path (`torch_backend.py:260-297`).

2. Import issues
- Same Transformers/Qwen version coupling risk as embedding module.

3. Dead code / unreachable
- No obvious unreachable core path; mostly active utilities.

4. Error handling gaps
- `tokenize()` also swallows multimodal preprocessing errors and falls back to synthetic NULL input (`qwen3_vl_reranker.py:177-186`).

5. Test expectation mismatches
- No direct test coverage in RecallForge for reranker tokenization fallbacks.

## Integration-Specific Import Findings (`Qwen3-VL-Embedding/src/models`)

1. Path dependency is repo-layout-specific
- RecallForge assumes the Qwen submodule exists next to repo root and is available on `sys.path`. This is not guaranteed in packaged installs.

2. Packaging mismatch
- RecallForge only packages `src/` Python packages (`pyproject.toml:50-52`), but Qwen model code lives outside package tree in `Qwen3-VL-Embedding/`. Installed wheels won’t include it unless separately installed.

3. Namespace/import ambiguity
- Importing from `models.*` is unsafe in larger Python environments due name collisions.

4. Reranker import asymmetry
- Embedder loader has late fallback insertion of correct submodule path; reranker loader does not.

## Test vs Code Inconsistencies

1. Import stability vs optional MLX expectation
- Tests assume optional backends should be skippable.
- Actual: importing `recallforge` can abort process due eager MLX import.

2. Test path setup inconsistency
- `tests/test_live.py:19` and `tests/benchmark.py:27` comments say “Add src to path” but insert repo root instead of `src`, unlike other tests.

3. CLI mode/override behavior not truly validated
- UAT invokes `--mode` variants (`tests/uat/test_cli.sh:85-107`), but assertions only check that output contains “Results for”, not that mode-specific behavior changed.
- Actual code path uses frozen env constants (`src/recallforge/__init__.py:30-33`, `:50-52`), so runtime mode override can be ignored.

4. Graceful shutdown claim not tested end-to-end
- UAT labels graceful shutdown coverage, but only checks handler registration, while runtime flag is unused.

## Recommended Fix Order

1. Make package import safe
- Remove eager MLX import from default package import path.
- Use lazy backend imports in `get_backend()` and catch broad backend-import failures with clear fallbacks.

2. Replace frozen config globals with runtime env reads
- In `get_backend()` / `get_storage()`, read `os.environ` at call time.
- This also fixes CLI `--mode`, `--backend`, `--quantize` overrides.

3. Stabilize Qwen integration
- Use deterministic import path strategy with absolute path resolution once.
- Add robust reranker fallback path matching embedder behavior.
- Avoid top-level `models.*` imports; prefer explicit package/module names.
- Align dependency floors (`transformers`, `torch`) with Qwen requirements or gate features explicitly.

4. Fix ranking determinism
- Make RRF weights explicit by list name (`original_fts`, `original_vec`, etc.) instead of dict insertion order.

5. Tighten error handling and observability
- Replace broad silent `except` blocks with structured warnings/errors.
- Surface partial indexing failures in CLI exit codes.

6. Update tests to enforce behavior, not only command success
- Add assertions that mode flag changes rerank behavior and model-loading state.
- Add regression tests for stale-embedding cleanup on document updates.
