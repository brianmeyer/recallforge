"""Lightweight entity and relation extraction for memory graph enrichment."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional


_MAX_ENTITIES_PER_TEXT = 24
_MAX_ENTITY_LEN = 80
_MAX_EVIDENCE_CHARS = 360

_STOP_ENTITIES = {
    "A", "An", "And", "Are", "As", "At", "By", "For", "From", "In", "Into", "Is",
    "It", "Of", "On", "Or", "The", "This", "To", "With",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}
_STOP_ENTITY_KEYS = {_entity.lower() for _entity in _STOP_ENTITIES}

_PROPER_NOUN_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&._-]{1,}(?:\s+[A-Z][A-Za-z0-9&._-]{1,}){0,4}\b"
)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)*\b")
_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_.-]{2,}")
_ISSUE_RE = re.compile(r"\b[A-Z]{2,10}-\d{1,6}\b")
_URL_RE = re.compile(r"\bhttps?://[^\s)>\]]+")
_DOMAIN_RE = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")


@dataclass(frozen=True)
class ExtractedEntity:
    """One normalized entity mention with source evidence."""

    name: str
    entity_key: str
    entity_type: str
    evidence: str


@dataclass(frozen=True)
class ExtractedRelation:
    """A lightweight relation edge between two entity mentions."""

    subject_key: str
    subject_name: str
    object_key: str
    object_name: str
    relation_type: str
    evidence: str


def _clean_entity(raw: str) -> str:
    text = re.sub(r"\s+", " ", str(raw or "").strip())
    return text.strip(".,;:!?()[]{}\"'`")


def normalize_entity_key(name: str) -> str:
    """Normalize an entity mention to a stable lookup key."""
    lowered = name.lower()
    lowered = re.sub(r"^@", "", lowered)
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered


def _classify_entity(name: str) -> str:
    if name.startswith("@"):
        return "person"
    if _ISSUE_RE.fullmatch(name):
        return "ticket"
    if _URL_RE.fullmatch(name) or _DOMAIN_RE.fullmatch(name):
        return "url"
    if _ACRONYM_RE.fullmatch(name):
        return "acronym"
    if any(token in name.lower().split() for token in ("project", "program", "initiative")):
        return "project"
    return "proper_noun"


def _evidence_for(text: str, start: int, end: int) -> str:
    left = max(0, start - 120)
    right = min(len(text), end + 180)
    snippet = re.sub(r"\s+", " ", text[left:right]).strip()
    if len(snippet) > _MAX_EVIDENCE_CHARS:
        snippet = snippet[: _MAX_EVIDENCE_CHARS - 3].rsplit(" ", 1)[0].strip() + "..."
    return snippet


def _iter_entity_matches(text: str):
    for pattern in (_URL_RE, _HANDLE_RE, _ISSUE_RE, _ACRONYM_RE, _PROPER_NOUN_RE, _DOMAIN_RE):
        yield from pattern.finditer(text)


def extract_entities(text: str, *, max_entities: int = _MAX_ENTITIES_PER_TEXT) -> list[ExtractedEntity]:
    """Extract deterministic entity mentions from text without external NLP deps."""
    if not isinstance(text, str) or not text.strip():
        return []

    found: dict[str, ExtractedEntity] = {}
    occupied_spans: list[tuple[int, int]] = []
    for match in sorted(_iter_entity_matches(text), key=lambda item: (item.start(), -(item.end() - item.start()))):
        if any(match.start() < end and match.end() > start for start, end in occupied_spans):
            continue
        name = _clean_entity(match.group(0))
        if not name or len(name) > _MAX_ENTITY_LEN or name in _STOP_ENTITIES:
            continue
        key = normalize_entity_key(name)
        if len(key) < 2 or key in _STOP_ENTITY_KEYS or key in found:
            continue
        found[key] = ExtractedEntity(
            name=name,
            entity_key=key,
            entity_type=_classify_entity(name),
            evidence=_evidence_for(text, match.start(), match.end()),
        )
        occupied_spans.append((match.start(), match.end()))
        if len(found) >= max_entities:
            break
    return list(found.values())


def extract_relations(
    entities: Iterable[ExtractedEntity],
    *,
    max_pairs: int = 48,
) -> list[ExtractedRelation]:
    """Create co-mention relation edges for entities found in the same evidence unit."""
    unique: dict[str, ExtractedEntity] = {}
    for entity in entities:
        unique.setdefault(entity.entity_key, entity)

    relations: list[ExtractedRelation] = []
    for left, right in combinations(list(unique.values())[:12], 2):
        evidence = left.evidence if len(left.evidence) >= len(right.evidence) else right.evidence
        relations.append(
            ExtractedRelation(
                subject_key=left.entity_key,
                subject_name=left.name,
                object_key=right.entity_key,
                object_name=right.name,
                relation_type="co_mentions",
                evidence=evidence,
            )
        )
        if len(relations) >= max_pairs:
            break
    return relations


def stable_graph_id(*parts: Optional[str]) -> str:
    """Build a stable hash ID for graph rows."""
    seed = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()
