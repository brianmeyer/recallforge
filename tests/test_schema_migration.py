"""
test_schema_migration.py — REC-34

Verifies that LanceDBBackend transparently migrates stores created before the
namespace columns (user_id, session_id, project_id, profile) were added to the
schema.

Strategy
--------
1. Spin up a real LanceDB connection using the *old* schema (namespace columns
   omitted) and write one seed row into the embeddings and documents tables.
2. Close the connection so the tables are fully flushed to disk.
3. Open the same directory with the *new* LanceDBBackend (which calls
   _migrate_schema internally).
4. Assert that all four namespace columns now appear in both table schemas.
5. Write a new row that includes namespace values — must not raise.
6. Read the migrated row back and verify the namespace values round-trip.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from typing import List

import numpy as np
import pyarrow as pa
import lancedb

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

from recallforge.storage.lancedb_backend import LanceDBBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_DIM = LanceDBBackend.EMBED_DIM
NAMESPACE_COLS = ["user_id", "session_id", "project_id", "profile"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vector(seed: str) -> List[float]:
    """Deterministic 2048-dim unit vector seeded from a string."""
    h = hashlib.sha256(seed.encode()).digest()
    rng = np.random.default_rng(np.frombuffer(h, dtype=np.uint8))
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def mock_embed(text: str) -> List[float]:
    return _unit_vector(text)


# ---------------------------------------------------------------------------
# Old-schema builders (namespace columns intentionally omitted)
# ---------------------------------------------------------------------------

def _old_embeddings_schema() -> pa.Schema:
    """Embeddings schema as it existed before REC-34 (no namespace columns)."""
    return pa.schema([
        pa.field("hash_seq",      pa.string(),                             nullable=False),
        pa.field("content_hash",  pa.string(),                             nullable=False),
        pa.field("collection",    pa.string(),                             nullable=False),
        pa.field("file_path",     pa.string(),                             nullable=False),
        pa.field("content_type",  pa.string(),                             nullable=False),
        pa.field("title",         pa.string(),                             nullable=True),
        pa.field("text_body",     pa.string(),                             nullable=True),
        pa.field("seq",           pa.int32(),                              nullable=False),
        pa.field("pos",           pa.int32(),                              nullable=False),
        pa.field("model",         pa.string(),                             nullable=True),
        pa.field("embedded_at",   pa.int64(),                              nullable=False),
        pa.field("vector",        pa.list_(pa.float32(), list_size=EMBED_DIM), nullable=False),
        # ← user_id / session_id / project_id / profile deliberately absent
    ])


def _old_documents_schema() -> pa.Schema:
    """Documents schema as it existed before REC-34 (no namespace columns)."""
    return pa.schema([
        pa.field("id",            pa.string(), nullable=False),
        pa.field("collection",    pa.string(), nullable=False),
        pa.field("file_path",     pa.string(), nullable=False),
        pa.field("title",         pa.string(), nullable=True),
        pa.field("content_hash",  pa.string(), nullable=False),
        pa.field("content_type",  pa.string(), nullable=False),
        pa.field("active",        pa.int8(),   nullable=False),
        pa.field("created_at",    pa.int64(),  nullable=False),
        pa.field("updated_at",    pa.int64(),  nullable=False),
        # ← namespace columns deliberately absent
    ])


def _seed_old_store(lance_dir: str) -> None:
    """
    Create a LanceDB store using the old schema and write one seed row to each
    table, then close the connection.  Simulates a store created by an older
    version of RecallForge.
    """
    conn = lancedb.connect(lance_dir)

    # --- embeddings table ---
    emb_row = pa.table(
        {
            "hash_seq":     ["abc123_0"],
            "content_hash": ["abc123"],
            "collection":   ["test"],
            "file_path":    ["old_doc.md"],
            "content_type": ["text"],
            "title":        ["Old Doc"],
            "text_body":    ["Legacy content without namespace."],
            "seq":          pa.array([0], type=pa.int32()),
            "pos":          pa.array([0], type=pa.int32()),
            "model":        ["mock"],
            "embedded_at":  pa.array([1_700_000_000_000], type=pa.int64()),
            "vector":       [_unit_vector("old_doc")],
        },
        schema=_old_embeddings_schema(),
    )
    conn.create_table("embeddings", data=emb_row)

    # --- documents table ---
    doc_row = pa.table(
        {
            "id":           ["doc-001"],
            "collection":   ["test"],
            "file_path":    ["old_doc.md"],
            "title":        ["Old Doc"],
            "content_hash": ["abc123"],
            "content_type": ["text"],
            "active":       pa.array([1], type=pa.int8()),
            "created_at":   pa.array([1_700_000_000_000], type=pa.int64()),
            "updated_at":   pa.array([1_700_000_000_000], type=pa.int64()),
        },
        schema=_old_documents_schema(),
    )
    conn.create_table("documents", data=doc_row)

    # LanceDB sync is implicit; just let the conn go out of scope.


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

class TestSchemaMigration(unittest.TestCase):
    """REC-34: schema migration adds missing namespace columns to old stores."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="rf-rec34-migration-")
        self.lance_dir = os.path.join(self.temp_dir, "store.lance")

        # Build the legacy store *before* touching LanceDBBackend.
        _seed_old_store(self.lance_dir)

        # Now open with the new backend — _migrate_schema() runs automatically.
        self.backend = LanceDBBackend(self.temp_dir)
        self.backend.initialize(self.temp_dir)

    def tearDown(self):
        self.backend.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Schema presence
    # ------------------------------------------------------------------

    def test_embeddings_table_has_namespace_columns(self):
        """After migration, embeddings table must contain all namespace columns."""
        schema = self.backend._embeddings_table.schema
        col_names = {field.name for field in schema}
        for col in NAMESPACE_COLS:
            with self.subTest(column=col):
                self.assertIn(
                    col,
                    col_names,
                    msg=f"Column '{col}' missing from embeddings schema after migration",
                )

    def test_documents_table_has_namespace_columns(self):
        """After migration, documents table must contain all namespace columns."""
        schema = self.backend._documents_table.schema
        col_names = {field.name for field in schema}
        for col in NAMESPACE_COLS:
            with self.subTest(column=col):
                self.assertIn(
                    col,
                    col_names,
                    msg=f"Column '{col}' missing from documents schema after migration",
                )

    # ------------------------------------------------------------------
    # Write succeeds with namespace columns
    # ------------------------------------------------------------------

    def test_write_with_namespace_succeeds(self):
        """
        upsert_memory with explicit namespace values must not raise after migration.
        REC-34 regression: previously raised 'Field user_id not found'.
        """
        try:
            self.backend.upsert_memory(
                path="new_namespaced_doc.md",
                text="Content written after schema migration.",
                collection="test",
                embed_func=mock_embed,
                model="mock",
                user_id="user-42",
                session_id="sess-abc",
                project_id="proj-xyz",
                profile="default",
            )
        except Exception as exc:
            self.fail(
                f"upsert_memory raised after migration — REC-34 regression: {exc}"
            )

    # ------------------------------------------------------------------
    # Namespace values round-trip
    # ------------------------------------------------------------------

    def test_namespace_values_round_trip(self):
        """
        Values written with namespace fields must be retrievable via search.
        Confirms that migrated columns are writable and readable, not just present.
        """
        self.backend.upsert_memory(
            path="roundtrip.md",
            text="Namespace round-trip verification.",
            collection="test",
            embed_func=mock_embed,
            model="mock",
            user_id="alice",
            session_id="s1",
            project_id="p1",
            profile="prod",
        )

        # Vector search should return the newly written doc.
        query_vec = mock_embed("Namespace round-trip verification.")
        results = self.backend.search_vec(
            vector=query_vec,
            limit=5,
            collection="test",
            user_id="alice",
        )

        self.assertTrue(
            len(results) > 0,
            "Expected at least one result for the namespaced write.",
        )
        # The top result should carry the correct namespace.
        top = results[0]
        self.assertEqual(top.user_id, "alice")
        self.assertEqual(top.session_id, "s1")
        self.assertEqual(top.project_id, "p1")
        self.assertEqual(top.profile, "prod")

    # ------------------------------------------------------------------
    # Legacy (pre-migration) rows unaffected
    # ------------------------------------------------------------------

    def test_legacy_rows_still_readable(self):
        """
        Pre-migration rows (which have NULL namespace columns) must still be
        searchable after migration.  Migration must not corrupt existing data.
        """
        query_vec = mock_embed("Legacy content without namespace.")
        results = self.backend.search_vec(
            vector=query_vec,
            limit=5,
            collection="test",
        )
        paths = [r.display_path for r in results]
        self.assertTrue(
            any("old_doc" in p for p in paths),
            f"Legacy row 'old_doc.md' not found after migration. Got: {paths}",
        )

    # ------------------------------------------------------------------
    # Idempotency: running again must not raise
    # ------------------------------------------------------------------

    def test_migration_is_idempotent(self):
        """
        Calling _migrate_schema() on an already-migrated store must be a no-op.
        Covers the scenario where the backend is re-initialised against the same store.
        """
        try:
            self.backend._migrate_schema()
        except Exception as exc:
            self.fail(f"_migrate_schema() raised on already-migrated store: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
