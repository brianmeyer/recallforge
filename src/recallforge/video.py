"""Video extraction helpers for RecallForge.

Frame extraction uses ffmpeg/ffprobe when available.
Transcript ingestion supports sidecar .srt, .vtt, and .txt files.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg",
}
TRANSCRIPT_EXTENSIONS = (".srt", ".vtt", ".txt")


@dataclass
class VideoFrame:
    image_path: str
    logical_path: str
    title: str
    timestamp_seconds: float


@dataclass
class TranscriptSegment:
    logical_path: str
    title: str
    text: str
    start_seconds: float
    end_seconds: Optional[float]


@dataclass
class VideoArtifacts:
    frames: List[VideoFrame]
    transcripts: List[TranscriptSegment]
    duration_seconds: Optional[float]
    transcript_path: Optional[str]
    ffmpeg_available: bool


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _parse_timestamp(raw: str) -> float:
    text = raw.strip().replace(",", ".")
    hours, minutes, seconds = text.split(":")
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


def _clean_caption_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _load_sidecar_text(sidecar_path: Path) -> List[TranscriptSegment]:
    suffix = sidecar_path.suffix.lower()
    raw = sidecar_path.read_text(encoding="utf-8", errors="replace")
    stem = sidecar_path.stem

    if suffix == ".txt":
        text = _clean_caption_text(raw)
        if not text:
            return []
        return [
            TranscriptSegment(
                logical_path="",
                title=f"{stem} transcript",
                text=text,
                start_seconds=0.0,
                end_seconds=None,
            )
        ]

    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n"))
    segments: List[TranscriptSegment] = []
    cue_index = 0
    ts_pattern = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,\.]\d{3})"
    )

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].upper() == "WEBVTT":
            continue
        if lines[0].startswith("NOTE"):
            continue

        ts_line = None
        text_lines: List[str] = []
        for line in lines:
            if ts_pattern.search(line):
                ts_line = line
                continue
            if ts_line is None and line.isdigit():
                continue
            if ts_line is not None:
                text_lines.append(line)

        if ts_line is None:
            continue

        match = ts_pattern.search(ts_line)
        if match is None:
            continue

        text = _clean_caption_text(" ".join(text_lines))
        if not text:
            continue

        cue_index += 1
        segments.append(
            TranscriptSegment(
                logical_path="",
                title=f"{stem} transcript {cue_index}",
                text=text,
                start_seconds=_parse_timestamp(match.group("start")),
                end_seconds=_parse_timestamp(match.group("end")),
            )
        )

    return segments


def find_transcript_sidecar(video_path: str | Path) -> Optional[Path]:
    path = Path(video_path)
    for suffix in TRANSCRIPT_EXTENSIONS:
        candidate = path.with_suffix(suffix)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_transcript_segments(video_path: str | Path, logical_path: str) -> tuple[List[TranscriptSegment], Optional[str]]:
    sidecar = find_transcript_sidecar(video_path)
    if sidecar is None:
        return [], None

    segments = _load_sidecar_text(sidecar)
    for index, segment in enumerate(segments, start=1):
        segment.logical_path = (
            f"{logical_path}::transcript:{index:04d}@{segment.start_seconds:.2f}s"
        )
    return segments, str(sidecar)


def probe_duration_seconds(video_path: str | Path) -> Optional[float]:
    if not ffmpeg_available():
        return None

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    duration = payload.get("format", {}).get("duration")
    if duration in (None, ""):
        return None
    return float(duration)


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    logical_path: str,
    frame_interval_seconds: float = 5.0,
    max_frames: int = 8,
) -> tuple[List[VideoFrame], Optional[float]]:
    if not ffmpeg_available():
        return [], None

    output_root = Path(output_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    duration_seconds = probe_duration_seconds(video_path)
    pattern = output_root / "frame_%04d.jpg"

    fps_value = 1.0 / max(frame_interval_seconds, 0.1)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps_value}",
            "-frames:v",
            str(max_frames),
            str(pattern),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    frame_paths = sorted(output_root.glob("frame_*.jpg"))
    if not frame_paths:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(output_root / "frame_0001.jpg"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        frame_paths = sorted(output_root.glob("frame_*.jpg"))

    video_name = Path(logical_path).name
    frames: List[VideoFrame] = []
    for index, frame_path in enumerate(frame_paths, start=1):
        timestamp = (index - 1) * max(frame_interval_seconds, 0.1)
        if duration_seconds is not None:
            timestamp = min(timestamp, max(duration_seconds - 0.001, 0.0))
        frames.append(
            VideoFrame(
                image_path=str(frame_path),
                logical_path=f"{logical_path}::frame:{index:04d}@{timestamp:.2f}s",
                title=f"{video_name} frame {index}",
                timestamp_seconds=timestamp,
            )
        )

    return frames, duration_seconds


def extract_video_artifacts(
    video_path: str | Path,
    output_dir: str | Path,
    logical_path: str,
    frame_interval_seconds: float = 5.0,
    max_frames: int = 8,
) -> VideoArtifacts:
    transcripts, transcript_path = load_transcript_segments(video_path, logical_path)
    frames, duration_seconds = extract_video_frames(
        video_path=video_path,
        output_dir=output_dir,
        logical_path=logical_path,
        frame_interval_seconds=frame_interval_seconds,
        max_frames=max_frames,
    )

    if not frames and not transcripts:
        raise RuntimeError(
            "Video ingest requires ffmpeg/ffprobe for frame extraction or a sidecar transcript (.srt/.vtt/.txt)."
        )

    return VideoArtifacts(
        frames=frames,
        transcripts=transcripts,
        duration_seconds=duration_seconds,
        transcript_path=transcript_path,
        ffmpeg_available=ffmpeg_available(),
    )
