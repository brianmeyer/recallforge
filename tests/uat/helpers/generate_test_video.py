#!/usr/bin/env python3
"""Generate a small synthetic test video and transcript sidecar."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "Usage: generate_test_video.py <output.mp4> <image1> <image2> <image3>",
            file=sys.stderr,
        )
        return 2

    output_path = Path(sys.argv[1]).expanduser().resolve()
    images = [Path(arg).expanduser().resolve() for arg in sys.argv[2:5]]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transcript_path = output_path.with_suffix(".srt")
    transcript_path.write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:01,000",
                "Pasta dish on a white plate.",
                "",
                "2",
                "00:00:01,000 --> 00:00:02,000",
                "Forest landscape with green trees.",
                "",
                "3",
                "00:00:02,000 --> 00:00:03,000",
                "Whiteboard architecture diagram from a meeting.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        output_path.write_bytes(b"placeholder-video")
        print(json.dumps({
            "video_path": str(output_path),
            "transcript_path": str(transcript_path),
            "ffmpeg_available": False,
        }))
        return 0

    subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-t",
            "1",
            "-i",
            str(images[0]),
            "-loop",
            "1",
            "-t",
            "1",
            "-i",
            str(images[1]),
            "-loop",
            "1",
            "-t",
            "1",
            "-i",
            str(images[2]),
            "-filter_complex",
            (
                "[0:v]scale=960:540,format=yuv420p[v0];"
                "[1:v]scale=960:540,format=yuv420p[v1];"
                "[2:v]scale=960:540,format=yuv420p[v2];"
                "[v0][v1][v2]concat=n=3:v=1:a=0,format=yuv420p[v]"
            ),
            "-map",
            "[v]",
            str(output_path),
        ],
        check=True,
    )

    print(json.dumps({
        "video_path": str(output_path),
        "transcript_path": str(transcript_path),
        "ffmpeg_available": True,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
