#!/usr/bin/env python3
"""Generate persistent UAT video corpus for RecallForge.

Creates 5 short MP4 test videos (3-5 seconds, 320x240, 15fps) plus
matching .transcript.json sidecar files in tests/uat/corpus/videos/.

Run from the repo root:
    python3 tests/uat/helpers/generate_video_corpus.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES_DIR = REPO_ROOT / "tests" / "uat" / "corpus" / "images"
OUTPUT_DIR = REPO_ROOT / "tests" / "uat" / "corpus" / "videos"

WIDTH, HEIGHT, FPS = 320, 240, 15

# ---------------------------------------------------------------------------
# Video definitions
# ---------------------------------------------------------------------------

VIDEOS: list[dict] = [
    {
        "name": "coding_demo.mp4",
        "images": ["code_editor_screenshot.png"],
        "duration": 4,
        "effects": "text_overlay",
        "overlay_text": "Python coding tutorial",
        "transcript": [
            {"start": 0.0, "end": 2.0, "text": "Code editor showing a Python script."},
            {"start": 2.0, "end": 4.0, "text": "Walkthrough of coding concepts and syntax."},
        ],
        "description": "Coding tutorial demo with text overlay on code editor screenshot.",
    },
    {
        "name": "nature_timelapse.mp4",
        "images": ["forest_landscape.png", "mountain_landscape.png"],
        "duration": 4,
        "effects": "pan_zoom_slide",
        "overlay_text": None,
        "transcript": [
            {"start": 0.0, "end": 2.0, "text": "Dense forest landscape with green canopy."},
            {"start": 2.0, "end": 4.0, "text": "Mountain range with peaks and valleys."},
        ],
        "description": "Nature timelapse panning over forest and mountain landscapes.",
    },
    {
        "name": "architecture_walkthrough.mp4",
        "images": ["whiteboard_architecture.png", "floor_plan_blueprint.png"],
        "duration": 4,
        "effects": "pan_zoom_slide",
        "overlay_text": None,
        "transcript": [
            {"start": 0.0, "end": 2.0, "text": "Whiteboard diagram showing system architecture components."},
            {"start": 2.0, "end": 4.0, "text": "Detailed floor plan blueprint with room layouts."},
        ],
        "description": "Architecture walkthrough sliding between diagrams and blueprints.",
    },
    {
        "name": "cooking_tutorial.mp4",
        "images": ["food_pasta_dish.png"],
        "duration": 3,
        "effects": "zoom_in",
        "overlay_text": "Pasta recipe",
        "transcript": [
            {"start": 0.0, "end": 1.5, "text": "Fresh pasta dish plated on a white plate."},
            {"start": 1.5, "end": 3.0, "text": "Close-up of the pasta with sauce and garnish."},
        ],
        "description": "Cooking tutorial zooming into a pasta dish with recipe text overlay.",
    },
    {
        "name": "whiteboard_session.mp4",
        "images": ["whiteboard_brainstorm.png", "handwritten_notes.png"],
        "duration": 4,
        "effects": "pan_zoom_slide",
        "overlay_text": None,
        "transcript": [
            {"start": 0.0, "end": 2.0, "text": "Brainstorming session on a whiteboard with ideas."},
            {"start": 2.0, "end": 4.0, "text": "Handwritten notes with key points and action items."},
        ],
        "description": "Whiteboard session panning over brainstorm board and handwritten notes.",
    },
]


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _scale_filter(w: int = WIDTH, h: int = HEIGHT) -> str:
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"


def _burn_text_onto_image(source: Path, text: str, tmp_dir: str) -> Path:
    """Use Pillow to draw a caption bar at the bottom of the image.

    Returns path to a new temporary PNG with the text baked in.
    ffmpeg's drawtext filter requires libfreetype which may not be compiled in;
    this PIL approach works regardless of ffmpeg build options.
    """
    tmp_path = Path(tmp_dir) / f"_{source.stem}_captioned.png"
    with Image.open(source) as img:
        img = img.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        # Try to load a system font; fall back to the built-in bitmap font
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Measure text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 6
        bar_h = text_h + padding * 2

        # Semi-transparent black bar at the bottom
        bar_top = HEIGHT - bar_h
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        bar_draw = ImageDraw.Draw(overlay)
        bar_draw.rectangle([(0, bar_top), (WIDTH, HEIGHT)], fill=(0, 0, 0, 160))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(img)
        x = (WIDTH - text_w) // 2
        y = bar_top + padding
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        img.save(tmp_path, "PNG")
    return tmp_path


def _build_single_image_cmd(
    ffmpeg: str,
    image: Path,
    output: Path,
    duration: int,
    overlay_text: str | None,
    zoom: bool = False,
    tmp_dir: str | None = None,
) -> list[str]:
    """Build ffmpeg command for a single-image video (with optional text/zoom)."""
    # Burn text via Pillow before handing off to ffmpeg (avoids drawtext dependency)
    if overlay_text and tmp_dir:
        image = _burn_text_onto_image(image, overlay_text, tmp_dir)

    scale = _scale_filter()

    if zoom:
        # Ken Burns zoom-in: scale up and slowly crop toward centre
        vf = (
            f"{scale},"
            f"zoompan=z='min(zoom+0.002,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={duration * FPS}:s={WIDTH}x{HEIGHT}:fps={FPS}"
        )
    else:
        vf = scale

    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-t", str(duration), "-i", str(image),
        "-vf", vf,
        "-r", str(FPS),
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        str(output),
    ]


def _build_two_image_cmd(
    ffmpeg: str,
    img1: Path,
    img2: Path,
    output: Path,
    duration: int,
) -> list[str]:
    """Build ffmpeg command for two-image slide/pan video."""
    half = duration // 2
    scale = _scale_filter()

    filter_complex = (
        f"[0:v]{scale}[v0];"
        f"[1:v]{scale}[v1];"
        f"[v0]trim=duration={half},setpts=PTS-STARTPTS[a];"
        f"[v1]trim=duration={duration - half},setpts=PTS-STARTPTS[b];"
        f"[a][b]concat=n=2:v=1:a=0[v]"
    )

    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-t", str(duration), "-i", str(img1),
        "-loop", "1", "-t", str(duration), "-i", str(img2),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        str(output),
    ]


def generate_video(ffmpeg: str, spec: dict) -> Path:
    """Generate one video and return its path."""
    output = OUTPUT_DIR / spec["name"]
    images = [IMAGES_DIR / img for img in spec["images"]]

    # Validate images exist
    for img in images:
        if not img.exists():
            raise FileNotFoundError(f"Image not found: {img}")

    duration = spec["duration"]
    overlay = spec["overlay_text"]
    effects = spec["effects"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        if len(images) == 1:
            zoom = effects == "zoom_in"
            cmd = _build_single_image_cmd(
                ffmpeg, images[0], output, duration, overlay, zoom=zoom, tmp_dir=tmp_dir
            )
        else:
            cmd = _build_two_image_cmd(ffmpeg, images[0], images[1], output, duration)

        print(f"  Generating {spec['name']} ({duration}s) ...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FAILED")
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"ffmpeg failed for {spec['name']}")
    print("OK")
    return output


def write_transcript(spec: dict) -> Path:
    """Write .transcript.json sidecar for the video."""
    output = OUTPUT_DIR / spec["name"]
    sidecar = output.with_suffix("").with_name(output.stem + ".transcript.json")

    payload = {
        "video": spec["name"],
        "description": spec["description"],
        "duration_seconds": spec["duration"],
        "resolution": f"{WIDTH}x{HEIGHT}",
        "fps": FPS,
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in spec["transcript"]
        ],
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ERROR: ffmpeg not found in PATH", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Images dir : {IMAGES_DIR}")
    print(f"ffmpeg     : {ffmpeg}")
    print()

    errors: list[str] = []
    for spec in VIDEOS:
        try:
            vid_path = generate_video(ffmpeg, spec)
            sidecar = write_transcript(spec)
            size_kb = vid_path.stat().st_size // 1024
            print(f"    -> {vid_path.name} ({size_kb} KB), sidecar: {sidecar.name}")
        except Exception as exc:
            errors.append(f"{spec['name']}: {exc}")
            print(f"    ERROR: {exc}")

    print()
    if errors:
        print(f"FAILED ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1

    # Summary
    total_bytes = sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.mp4"))
    print(f"All 5 videos generated. Total size: {total_bytes // 1024} KB")
    if total_bytes > 5 * 1024 * 1024:
        print("WARNING: corpus exceeds 5 MB git-friendliness target", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
