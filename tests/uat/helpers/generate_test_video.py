#!/usr/bin/env python3
"""Generate a small synthetic test video and transcript sidecar."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _write_video_with_pyav(output_path: Path, images: list[Path]) -> bool:
    """Create a tiny MP4 from still images using PyAV when ffmpeg CLI is absent."""
    try:
        import av
        import numpy as np
        from PIL import Image
    except Exception:
        return False

    try:
        with av.open(str(output_path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=1)
            stream.pix_fmt = "yuv420p"
            stream.width = 960
            stream.height = 540

            for image_path in images:
                with Image.open(image_path) as img:
                    frame_image = img.convert("RGB").resize((stream.width, stream.height))
                    frame = av.VideoFrame.from_ndarray(np.asarray(frame_image), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
        return True
    except Exception:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return False


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
    real_video_available = False
    if ffmpeg_path is None:
        real_video_available = _write_video_with_pyav(output_path, images)
        if not real_video_available:
            output_path.write_bytes(b"placeholder-video")
        print(json.dumps({
            "video_path": str(output_path),
            "transcript_path": str(transcript_path),
            "ffmpeg_available": False,
            "real_video_available": real_video_available,
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
        "real_video_available": True,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
