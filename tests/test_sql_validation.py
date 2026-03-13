"""Tests for SQL filter validation and path safety (REC-42)."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recallforge.storage.lancedb_backend import _validate_identifier, _safe_filter


class TestValidateIdentifier:
    """Test _validate_identifier with real-world file paths and edge cases."""

    # These MUST pass — real-world file paths
    @pytest.mark.parametrize("path", [
        "Photos (2024)/image.png",
        "file [v2].docx",
        "/Users/brian/Documents & Reports/notes.md",
        "project#42/README.md",
        "data,backup/file.txt",
        "résumé.pdf",
        "path/with=equals/file.txt",
        "sample_video.mp4::transcript:0",
        "sample_video.mp4::frame:001",
        "/Users/brian/Photos (2024)/vacation [best]/IMG_001.jpg",
        "file~backup.txt",
        "hello world.txt",
        "path/to/file@2x.png",
        "collection:default",
        "simple-path",
        "under_score",
        "dots.in.name.txt",
    ])
    def test_valid_paths_pass(self, path):
        assert _validate_identifier(path) == path

    # These MUST fail — SQL injection attempts
    @pytest.mark.parametrize("value,reason", [
        ("'; DROP TABLE embeddings; --", "single quote + SQL command"),
        ('"; DROP TABLE', "double quote + SQL command"),
        ("value\x00injection", "null byte"),
        ("value\ninjection", "newline"),
        ("value\rinjection", "carriage return"),
        ("path\\to\\file", "backslash"),
        ("value -- comment", "SQL comment"),
        ("value /* comment */", "SQL block comment"),
        ("it's a path", "embedded single quote"),
    ])
    def test_injection_attempts_blocked(self, value, reason):
        with pytest.raises(ValueError):
            _validate_identifier(value, "test_field")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            _validate_identifier("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError):
            _validate_identifier("   ")

    def test_non_string_rejected(self):
        with pytest.raises(ValueError):
            _validate_identifier(123)

    def test_none_rejected(self):
        with pytest.raises(ValueError):
            _validate_identifier(None)


class TestSafeFilter:
    """Test _safe_filter builds correct SQL clauses."""

    def test_basic_filter(self):
        result = _safe_filter("collection", "my_docs")
        assert result == "collection = 'my_docs'"

    def test_path_with_parens(self):
        result = _safe_filter("file_path", "Photos (2024)/image.png")
        assert result == "file_path = 'Photos (2024)/image.png'"

    def test_path_with_brackets(self):
        result = _safe_filter("file_path", "file [v2].docx")
        assert result == "file_path = 'file [v2].docx'"

    def test_invalid_field_name_rejected(self):
        with pytest.raises(ValueError):
            _safe_filter("bad field", "value")

    def test_injection_in_value_rejected(self):
        with pytest.raises(ValueError):
            _safe_filter("collection", "'; DROP TABLE --")
