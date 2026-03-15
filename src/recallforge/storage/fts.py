"""FTS helpers for LanceDB storage backend."""

import time

from .lancedb_shared import logger, trace_log


class _BulkModeContext:
    """Context manager for bulk mode FTS rebuild deferral."""

    def __init__(self, backend: "LanceDBBackend"):
        self._backend = backend
        self._was_in_bulk = False

    def __enter__(self):
        self._was_in_bulk = self._backend._bulk_mode
        self._backend._bulk_mode = True
        trace_log("bulk_mode_enter", was_in_bulk=self._was_in_bulk)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._backend._bulk_mode = self._was_in_bulk
        trace_log("bulk_mode_exit", needs_rebuild=self._backend._fts_needs_rebuild)

        # Trigger a single rebuild at the end of bulk mode if needed
        if not self._was_in_bulk and self._backend._fts_needs_rebuild:
            self._backend._do_fts_rebuild()

        return False  # Don't suppress exceptions


class LanceDBFTSMixin:
    def _schedule_fts_rebuild(self) -> None:
        """
        Schedule a debounced FTS rebuild.

        Instead of rebuilding on every write, we track pending writes
        and rebuild only when threshold is hit or on explicit request.

        In bulk mode, all rebuilds are deferred until bulk mode ends.
        """
        self._fts_rebuild_pending += 1
        self._fts_needs_rebuild = True

        trace_log("schedule_fts_rebuild", pending=self._fts_rebuild_pending, bulk_mode=self._bulk_mode)

        # In bulk mode, defer all rebuilds until bulk ends
        if self._bulk_mode:
            return

        # Immediate rebuild if threshold exceeded
        if self._fts_rebuild_pending >= self.FTS_REBUILD_PENDING_THRESHOLD:
            self._do_fts_rebuild()

    def _do_fts_rebuild(self) -> None:
        """
        Execute the actual FTS rebuild with rate limiting.
        """
        now = time.time()
        elapsed = now - self._fts_last_rebuild

        # Rate limit: don't rebuild more than once per min interval
        if elapsed < self.FTS_REBUILD_MIN_INTERVAL and self._fts_rebuild_pending < self.FTS_REBUILD_PENDING_THRESHOLD:
            trace_log("fts_rebuild_skipped", reason="rate_limited", elapsed=elapsed)
            return

        trace_log("fts_rebuild_executing", pending=self._fts_rebuild_pending)

        self._fts_last_rebuild = now
        self._fts_rebuild_pending = 0
        self._fts_needs_rebuild = False

        try:
            self.ensure_fts_index(force_rebuild=True)
        except Exception as e:
            logger.error(f"FTS rebuild failed: {e}")
            # Don't re-raise; FTS rebuild failure is non-fatal

    def bulk_mode(self):
        """
        Context manager to defer FTS rebuilds during bulk operations.

        Usage:
            with storage.bulk_mode():
                for doc in documents:
                    storage.upsert_memory(...)
            # FTS rebuild happens once at context exit

        Returns:
            Context manager that defers FTS rebuilds.
        """
        return _BulkModeContext(self)

    def ensure_fts_index(self, force_rebuild: bool = False) -> None:
        """Ensure the FTS index exists."""
        if self._embeddings_table is None:
            return

        try:
            row_count = self._embeddings_table.count_rows()
        except Exception as e:
            logger.error(f"ensure_fts_index: failed to count rows: {e}")
            return

        if row_count == 0:
            return

        if force_rebuild:
            try:
                self._embeddings_table.create_fts_index("text_body", replace=True)
                trace_log("fts_index_created", force=True, rows=row_count)
            except Exception as e:
                logger.error(f"ensure_fts_index: failed to create FTS index: {e}")
            return

        try:
            indices = self._embeddings_table.list_indices()
        except Exception as e:
            logger.warning(f"ensure_fts_index: failed to list indices: {e}")
            return

        has_fts = any(
            "text_body" in (i.columns or []) and "FTS" in str(i.index_type or i.type or "").upper()
            for i in indices
        )

        if not has_fts:
            try:
                self._embeddings_table.create_fts_index("text_body", replace=True)
                trace_log("fts_index_created", force=False, rows=row_count)
            except Exception as e:
                logger.error(f"ensure_fts_index: failed to create FTS index: {e}")

    def rebuild_fts_index(self) -> None:
        """Rebuild the FTS index."""
        if self._embeddings_table is None:
            return

        row_count = self._embeddings_table.count_rows()
        if row_count == 0:
            return

        indices = self._embeddings_table.list_indices()
        for idx in indices:
            if "text_body" in (idx.columns or []) and "FTS" in str(idx.index_type or idx.type or "").upper():
                self._embeddings_table.drop_index(idx.name)
                break

        self.ensure_fts_index()
