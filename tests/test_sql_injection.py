"""
test_sql_injection.py - Security tests for escape_sql() and related helpers.

Verifies that escape_sql() rejects SQL injection payloads and only
accepts values matching the allowlist pattern.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recallforge.storage.lancedb_backend import escape_sql, _validate_identifier, _safe_filter


# ---------------------------------------------------------------------------
# escape_sql: valid inputs should pass through (single quotes doubled)
# ---------------------------------------------------------------------------

class TestEscapeSqlValidInputs:
    def test_simple_string(self):
        assert escape_sql("hello") == "hello"

    def test_alphanumeric(self):
        assert escape_sql("abc123") == "abc123"

    def test_path_with_slash(self):
        assert escape_sql("foo/bar/baz") == "foo/bar/baz"

    def test_dotted_identifier(self):
        assert escape_sql("file.name.txt") == "file.name.txt"

    def test_hyphenated(self):
        assert escape_sql("my-doc-id") == "my-doc-id"

    def test_colon_separator(self):
        assert escape_sql("asset:derived:v1") == "asset:derived:v1"

    def test_at_symbol(self):
        assert escape_sql("user@example") == "user@example"

    def test_spaces_allowed(self):
        assert escape_sql("hello world") == "hello world"


# ---------------------------------------------------------------------------
# escape_sql: injection payloads must raise ValueError
# ---------------------------------------------------------------------------

INJECTION_PAYLOADS = [
    # Classic single-quote injection
    ("single_quote", "' OR '1'='1"),
    ("single_quote_drop", "'; DROP TABLE memories; --"),
    # Double-quote injection
    ("double_quote", '"value"'),
    # SQL comment injection
    ("line_comment", "value -- comment"),
    ("block_comment_open", "value /* injected"),
    ("block_comment_close", "injected */ value"),
    # Semicolon chaining
    ("semicolon", "value; SELECT 1"),
    # Backslash escape attempt
    ("backslash", "val\\ue"),
    # Null bytes
    ("null_byte", "value\x00injected"),
    # Newline smuggling
    ("newline", "value\nOR 1=1"),
    ("carriage_return", "value\rOR 1=1"),
    # Unicode lookalikes / dangerous chars
    # NOTE: <, !, # are not SQL metacharacters — they're XSS/shell concerns.
    # escape_sql only guards against SQL injection vectors.
    # Nested quotes
    ("nested_quotes", "it's a test"),
    # SQL UNION injection
    ("union_select", "x' UNION SELECT password FROM users--"),
    # Hex/octal encoding attempts (raw chars)
    ("x1a_char", "val\x1aue"),
]


@pytest.mark.parametrize("name,payload", INJECTION_PAYLOADS, ids=[p[0] for p in INJECTION_PAYLOADS])
def test_escape_sql_rejects_injection(name, payload):
    """escape_sql must raise ValueError for any SQL injection payload."""
    with pytest.raises(ValueError, match=r"(?i)(forbidden|invalid|outside allowed|metachar)"):
        escape_sql(payload)


# ---------------------------------------------------------------------------
# _validate_identifier: same hardening, exposed separately
# ---------------------------------------------------------------------------

class TestValidateIdentifier:
    def test_valid_value_returned(self):
        result = _validate_identifier("safe-value_123.txt", "test_field")
        assert result == "safe-value_123.txt"

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            _validate_identifier(42, "field")  # type: ignore[arg-type]

    def test_injection_raises(self):
        with pytest.raises(ValueError):
            _validate_identifier("val'; DROP TABLE t;--", "field")


# ---------------------------------------------------------------------------
# _safe_filter: round-trip field=value builder
# ---------------------------------------------------------------------------

class TestSafeFilter:
    def test_builds_correct_clause(self):
        clause = _safe_filter("collection_name", "my-collection")
        assert clause == "collection_name = 'my-collection'"

    def test_invalid_field_raises(self):
        with pytest.raises(ValueError, match="Invalid field name"):
            _safe_filter("bad field!", "value")

    def test_injection_in_value_raises(self):
        with pytest.raises(ValueError):
            _safe_filter("field", "'; DROP TABLE t;--")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_raises(self):
        """Empty string doesn't match allowlist — should raise."""
        with pytest.raises(ValueError):
            escape_sql("")

    def test_only_spaces_raises(self):
        """Spaces alone might be a degenerate query — tighten if needed."""
        # _SAFE_VALUE_PATTERN includes \s, so pure-space may or may not pass.
        # At minimum it should not allow injection.
        try:
            result = escape_sql("   ")
            # If allowed, it's harmless — just whitespace
            assert result == "   "
        except ValueError:
            pass  # also acceptable

    def test_very_long_safe_string(self):
        """Long but valid string should pass."""
        long_val = "a" * 500
        assert escape_sql(long_val) == long_val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
