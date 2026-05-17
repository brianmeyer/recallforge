"""Helpers for turning conversations into canonical RecallForge memories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


_DEFAULT_ROLE = "speaker"
_MAX_SUMMARY_CHARS = 4000
_MAX_TURN_CHARS = 6000


@dataclass(frozen=True)
class ConversationTurn:
    """Normalized representation of one conversation turn or message group."""

    role: str
    content: str
    speaker: Optional[str] = None
    timestamp: Optional[str] = None


def _compact_text(value: Any, *, max_chars: Optional[int] = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if max_chars is not None and len(text) > max_chars:
        truncated = text[: max_chars - 3].rsplit(" ", 1)[0].strip()
        return (truncated or text[: max_chars - 3]).rstrip() + "..."
    return text


def normalize_conversation_turns(raw_turns: Any) -> list[ConversationTurn]:
    """Validate and normalize MCP conversation turn payloads."""
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError("turns must be a non-empty array")

    turns: list[ConversationTurn] = []
    for index, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, dict):
            raise ValueError(f"turns[{index}] must be an object")

        raw_content = raw_turn.get("content", raw_turn.get("text"))
        content = _compact_text(raw_content, max_chars=_MAX_TURN_CHARS)
        if not content:
            raise ValueError(f"turns[{index}].content must be a non-empty string")

        speaker = _compact_text(raw_turn.get("speaker")) or None
        role = _compact_text(raw_turn.get("role") or speaker or _DEFAULT_ROLE).lower()
        if len(role) > 64:
            role = role[:64].strip()

        timestamp = _compact_text(raw_turn.get("timestamp")) or None
        turns.append(
            ConversationTurn(
                role=role or _DEFAULT_ROLE,
                speaker=speaker,
                timestamp=timestamp,
                content=content,
            )
        )

    return turns


def normalize_conversation_tags(
    turns: Iterable[ConversationTurn],
    extra_tags: Optional[list[str]] = None,
) -> list[str]:
    """Build compact retrieval tags for a conversation memory."""
    tags: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        tag = _compact_text(raw).lower()
        if not tag or tag in seen:
            return
        seen.add(tag)
        tags.append(tag)

    add("conversation")
    turns_list = list(turns)
    add(f"turns:{len(turns_list)}")
    for turn in turns_list:
        add(f"role:{turn.role}")
        if turn.speaker:
            add(f"participant:{turn.speaker.lower()}")

    for tag in extra_tags or []:
        add(str(tag))

    return tags[:16]


def conversation_turn_path(root_path: str, index: int) -> str:
    """Return the logical child path for a one-based turn index."""
    normalized_root = _compact_text(root_path)
    if not normalized_root:
        raise ValueError("path is required")
    return f"{normalized_root}::turn:{index:04d}"


def build_conversation_summary(
    *,
    title: str,
    turns: list[ConversationTurn],
    summary: Optional[str] = None,
) -> str:
    """Build deterministic parent text for a conversation root memory."""
    title_text = _compact_text(title) or "Conversation"
    lines = [f"# {title_text}", "", f"Conversation with {len(turns)} turns."]

    participants = sorted(
        {
            (turn.speaker or turn.role).strip()
            for turn in turns
            if (turn.speaker or turn.role).strip()
        },
        key=str.lower,
    )
    if participants:
        lines.append(f"Participants: {', '.join(participants[:12])}.")

    summary_text = _compact_text(summary, max_chars=1200) if summary else ""
    if summary_text:
        lines.extend(["", "Summary:", summary_text])

    excerpt_turns = turns[: min(len(turns), 8)]
    if excerpt_turns:
        lines.extend(["", "Turn excerpts:"])
        for index, turn in enumerate(excerpt_turns, start=1):
            label = turn.speaker or turn.role
            timestamp = f" [{turn.timestamp}]" if turn.timestamp else ""
            excerpt = _compact_text(turn.content, max_chars=360)
            lines.append(f"- Turn {index} {label}{timestamp}: {excerpt}")

    text = "\n".join(lines).strip()
    return text[:_MAX_SUMMARY_CHARS].strip()


def build_conversation_turn_text(
    *,
    title: str,
    turn: ConversationTurn,
    index: int,
    total: int,
) -> str:
    """Build searchable text for one child turn memory."""
    title_text = _compact_text(title) or "Conversation"
    label = turn.speaker or turn.role
    timestamp = f"\nTimestamp: {turn.timestamp}" if turn.timestamp else ""
    return (
        f"# {title_text} - turn {index} of {total}\n"
        f"Role: {turn.role}\n"
        f"Speaker: {label}{timestamp}\n\n"
        f"{turn.content}"
    ).strip()
