# Changelog

All notable changes to RecallForge will be documented in this file.

## [Unreleased]

### Performance

- **P0: Fix FTS miss fallback behavior** — Empty FTS results (no matches) no longer trigger expensive full-table BM25 fallback. BM25 fallback is now reserved for true FTS index errors only. This significantly reduces latency for queries that legitimately return no results.
- **P0: Improve FTS rebuild policy in bulk ingest** — Added `bulk_mode()` context manager to defer FTS index rebuilds until the end of bulk operations. `index_folder()` and `ingest()` now use bulk mode to avoid repeated mid-batch rebuilds, improving batch ingestion performance.

### Added

- `bulk_mode()` context manager on `LanceDBBackend` for explicit control over FTS rebuild timing during bulk operations.

### Changed

- `search_fts()` now returns empty list for queries with no matches instead of falling back to in-memory BM25.
- `index_folder()` and `ingest()` defer FTS rebuilds until batch completion.