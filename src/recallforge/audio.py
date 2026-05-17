"""Audio ingestion helpers for RecallForge.

RecallForge's first audio path is transcript-first: audio files are indexed when
they have a nearby transcript sidecar. The parser intentionally reuses the
video sidecar format so temporal media can share one simple convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .video import TranscriptSegment, find_transcript_sidecar, load_transcript_segments


AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".weba",
    ".wma",
}


def is_audio_file(path: str | Path) -> bool:
    """Return True when *path* has a supported audio extension."""
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def find_audio_transcript_sidecar(audio_path: str | Path) -> Optional[Path]:
    """Find a transcript sidecar next to an audio file."""
    return find_transcript_sidecar(audio_path)


def load_audio_transcript_segments(
    audio_path: str | Path,
    logical_path: str,
) -> tuple[List[TranscriptSegment], Optional[str]]:
    """Load timestamped transcript segments for an audio file."""
    return load_transcript_segments(audio_path, logical_path)
