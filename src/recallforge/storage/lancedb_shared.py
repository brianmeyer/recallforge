"""Shared helpers for LanceDB storage backend modules."""

import hashlib
import logging
import os
import re
import struct
from pathlib import Path

logger = logging.getLogger("recallforge.storage")

TRACE_ENABLED = os.environ.get("RECALLFORGE_TRACE", "0") == "1"


def trace_log(operation: str, **kwargs) -> None:
    """Structured trace logging for debugging."""
    if TRACE_ENABLED:
        logger.debug(f"[TRACE] {operation}: {kwargs}")


DEFAULT_INDEX_DIR = os.path.join(os.path.expanduser("~"), ".recallforge")

_SQL_METACHARACTERS = frozenset("'\";\\\n\r\x00\x1a")
_SQL_COMMENT_PATTERNS = ("--", "/*", "*/")


def _validate_identifier(value: str, field_name: str = "value") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")

    if not value.strip():
        raise ValueError(f"{field_name} is empty or whitespace-only")

    for pattern in _SQL_COMMENT_PATTERNS:
        if pattern in value:
            raise ValueError(f"{field_name} contains forbidden SQL pattern: {pattern}")

    if any(c in _SQL_METACHARACTERS for c in value):
        raise ValueError(f"{field_name} contains forbidden SQL metacharacters")

    return value


def _safe_filter(field: str, value: str) -> str:
    if not re.match(r"^[\w_]+$", field):
        raise ValueError(f"Invalid field name: {field}")

    validated = _validate_identifier(value, field)
    escaped = validated.replace("'", "''")
    return f"{field} = '{escaped}'"


def escape_sql(s: str) -> str:
    return _validate_identifier(s, "value").replace("'", "''")


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_file_bytes(file_path: str) -> str:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {file_path}")

    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except IOError as e:
        logger.error(f"hash_file_bytes: failed to read {file_path}: {e}")
        raise

    try:
        mtime = path.stat().st_mtime
        h.update(struct.pack(">d", mtime))
    except OSError as e:
        logger.warning(f"hash_file_bytes: failed to get mtime for {file_path}: {e}")

    return h.hexdigest()


def extract_title(content: str, filename: str) -> str:
    match = re.match(r"^##?\s+(.+)$", content, re.MULTILINE)
    if match:
        title = match.group(1).strip()
        if title not in ("📝 Notes", "Notes"):
            return title
        match = re.search(r"\n##\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

    title_prop = re.search(r"^#\+TITLE:\s*(.+)$", content, re.MULTILINE)
    if title_prop:
        return title_prop.group(1).strip()

    return os.path.splitext(os.path.basename(filename))[0]


def get_docid(hash_str: str) -> str:
    return hash_str[:6]
