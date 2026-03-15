"""Chunking helpers for LanceDB storage backend."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4
CHUNK_WINDOW_CHARS = 200


@dataclass
class BreakPoint:
    pos: int
    score: int
    type: str


@dataclass
class CodeFenceRegion:
    start: int
    end: int


BREAK_PATTERNS = [
    (r"\n#{1}(?!#)", 100, "h1"),
    (r"\n#{2}(?!#)", 90, "h2"),
    (r"\n#{3}(?!#)", 80, "h3"),
    (r"\n#{4}(?!#)", 70, "h4"),
    (r"\n#{5}(?!#)", 60, "h5"),
    (r"\n#{6}(?!#)", 50, "h6"),
    (r"\n```", 80, "codeblock"),
    (r"\n(?:---|\*\*\*|___)\s*\n", 60, "hr"),
    (r"\n\n+", 20, "blank"),
    (r"\n[-*]\s", 5, "list"),
    (r"\n\d+\.\s", 5, "numlist"),
    (r"\n", 1, "newline"),
]


def scan_break_points(text: str) -> List[BreakPoint]:
    """Find all potential break points in text."""
    seen: Dict[int, BreakPoint] = {}

    for pattern, score, btype in BREAK_PATTERNS:
        for match in re.finditer(pattern, text):
            pos = match.start()
            existing = seen.get(pos)
            if existing is None or score > existing.score:
                seen[pos] = BreakPoint(pos, score, btype)

    return sorted(seen.values(), key=lambda b: b.pos)


def find_code_fences(text: str) -> List[CodeFenceRegion]:
    """Find all code fence regions in text."""
    regions: List[CodeFenceRegion] = []
    in_fence = False
    fence_start = 0

    for match in re.finditer(r"\n```", text):
        if not in_fence:
            fence_start = match.start()
            in_fence = True
        else:
            regions.append(CodeFenceRegion(fence_start, match.end()))
            in_fence = False

    if in_fence:
        regions.append(CodeFenceRegion(fence_start, len(text)))

    return regions


def is_inside_code_fence(pos: int, fences: List[CodeFenceRegion]) -> bool:
    """Check if position is inside a code fence."""
    return any(f.start < pos < f.end for f in fences)


def find_best_cutoff(
    break_points: List[BreakPoint],
    target_pos: int,
    window_chars: int = CHUNK_WINDOW_CHARS,
    decay_factor: float = 0.7,
    code_fences: List[CodeFenceRegion] = None
) -> int:
    """Find the best break point near target position."""
    if code_fences is None:
        code_fences = []

    window_start = target_pos - window_chars
    best_score = -1
    best_pos = target_pos

    for bp in break_points:
        if bp.pos < window_start:
            continue
        if bp.pos > target_pos:
            break
        if is_inside_code_fence(bp.pos, code_fences):
            continue

        distance = target_pos - bp.pos
        normalized_dist = distance / window_chars
        multiplier = 1.0 - (normalized_dist * normalized_dist) * decay_factor
        final_score = bp.score * multiplier

        if final_score > best_score:
            best_score = final_score
            best_pos = bp.pos

    return best_pos


def chunk_document(
    content: str,
    max_chars: int = CHUNK_SIZE_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
    window_chars: int = CHUNK_WINDOW_CHARS
) -> List[Dict[str, Any]]:
    """Split document into overlapping chunks at natural break points."""
    if len(content) <= max_chars:
        return [{"text": content, "pos": 0}]

    break_points = scan_break_points(content)
    code_fences = find_code_fences(content)
    chunks: List[Dict[str, Any]] = []
    char_pos = 0

    while char_pos < len(content):
        target_end = min(char_pos + max_chars, len(content))
        end_pos = target_end

        if end_pos < len(content):
            best = find_best_cutoff(break_points, target_end, window_chars, 0.7, code_fences)
            if best > char_pos and best <= target_end:
                end_pos = best

        if end_pos <= char_pos:
            end_pos = min(char_pos + max_chars, len(content))

        chunks.append({"text": content[char_pos:end_pos], "pos": char_pos})

        if end_pos >= len(content):
            break

        char_pos = end_pos - overlap_chars
        last_chunk_pos = chunks[-1]["pos"]
        if char_pos <= last_chunk_pos:
            char_pos = end_pos

    return chunks
