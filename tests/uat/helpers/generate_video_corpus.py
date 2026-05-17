#!/usr/bin/env python3
"""Generate persistent episodic UAT video corpus for RecallForge.

Creates 5 compact MP4 test videos (9-12 seconds, 480x270, 12fps) plus
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

WIDTH, HEIGHT, FPS = 480, 270, 12

# ---------------------------------------------------------------------------
# Video definitions
# ---------------------------------------------------------------------------

VIDEOS: list[dict] = [
    {
        "name": "coding_demo.mp4",
        "scenario": "Screen recording from a late-afternoon RecallForge debugging session.",
        "images": ["code_editor_screenshot.png", "whiteboard_architecture.png", "handwritten_notes.png"],
        "duration": 12,
        "scene_durations": [4, 4, 4],
        "effects": "episodic_sequence",
        "scene_captions": [
            "3:14 PM - failing video search test",
            "3:18 PM - trace vector and BM25 paths",
            "3:24 PM - action items for rerank gate",
        ],
        "transcript": [
            {"start": 0.0, "end": 3.0, "text": "The developer opens the RecallForge search pipeline test and points out a failing video-to-text case."},
            {"start": 3.0, "end": 6.5, "text": "They compare vector retrieval, BM25 fusion, and reranker candidates for the same query."},
            {"start": 6.5, "end": 9.5, "text": "The architecture whiteboard shows parent video memories rolling up transcript and frame children."},
            {"start": 9.5, "end": 12.0, "text": "Handwritten notes capture follow-up actions: cap media reranking, keep transcripts searchable, and rerun the benchmark."},
        ],
        "description": "Screen recording of a RecallForge debugging session with code, architecture notes, and action items.",
        "related_images": ["images/code_editor_screenshot.png", "images/whiteboard_architecture.png", "images/handwritten_notes.png"],
        "related_documents": ["documents/recallforge_spec.docx", "documents/operations_manual.pdf", "documents/edge_deployment_guide.pdf"],
        "notes": "Useful for queries about debugging, screen recordings, RecallForge implementation details, reranking gates, and transcript-heavy developer memories.",
    },
    {
        "name": "nature_timelapse.mp4",
        "scenario": "Phone clip from a weekend trail scouting trip with field notes.",
        "images": ["forest_landscape.png", "mountain_landscape.png", "ocean_beach.png"],
        "duration": 12,
        "scene_durations": [4, 4, 4],
        "effects": "episodic_sequence",
        "scene_captions": [
            "8:42 AM - trailhead canopy",
            "9:35 AM - ridge overlook",
            "10:10 AM - coastal turnout",
        ],
        "transcript": [
            {"start": 0.0, "end": 3.5, "text": "The walk begins under a dense forest canopy with notes about tree cover and the shaded trail."},
            {"start": 3.5, "end": 7.5, "text": "The camera pauses at a mountain overlook while the narrator mentions weather, elevation, and route planning."},
            {"start": 7.5, "end": 10.0, "text": "A short coastal stop captures waves and beach access for a possible return trip."},
            {"start": 10.0, "end": 12.0, "text": "The clip ends with a reminder to compare these views with national park and climate notes."},
        ],
        "description": "Episodic outdoor trip clip spanning forest, mountain, and coastal scenes with field-note narration.",
        "related_images": ["images/forest_landscape.png", "images/mountain_landscape.png", "images/ocean_beach.png"],
        "related_documents": ["text/nature_forests.md", "text/nature_mountains.md", "text/nature_oceans.md", "text/travel_national_parks.md"],
        "notes": "Useful for visual queries over outdoor scenes and transcript queries about route planning, parks, weather, and environmental memories.",
    },
    {
        "name": "architecture_walkthrough.mp4",
        "scenario": "Hybrid office walkthrough connecting a floor plan to system architecture decisions.",
        "images": ["floor_plan_blueprint.png", "whiteboard_architecture.png", "neural_network_diagram.png"],
        "duration": 12,
        "scene_durations": [4, 4, 4],
        "effects": "episodic_sequence",
        "scene_captions": [
            "Room 214 - floor plan review",
            "War room - service diagram",
            "Model lab - embedding architecture",
        ],
        "transcript": [
            {"start": 0.0, "end": 3.0, "text": "The walkthrough starts on a floor plan, marking the meeting room and hallway where the demo will be installed."},
            {"start": 3.0, "end": 6.0, "text": "A whiteboard maps the API gateway, vector store, full-text index, and worker services."},
            {"start": 6.0, "end": 9.0, "text": "The presenter explains how embeddings, document sections, and media frames roll up into parent memories."},
            {"start": 9.0, "end": 12.0, "text": "The final note ties the physical walkthrough to the architecture deck and project milestone review."},
        ],
        "description": "Office and system-architecture walkthrough with floor plan, whiteboard, model diagram, and milestone narration.",
        "related_images": ["images/floor_plan_blueprint.png", "images/whiteboard_architecture.png", "images/neural_network_diagram.png"],
        "related_documents": ["documents/ai_architecture_deck.pptx", "documents/project_status_q1.docx", "documents/ai_strategy_report.docx"],
        "notes": "Useful for queries about architecture walkthroughs, floor plans, system design, model diagrams, and project planning documents.",
    },
    {
        "name": "cooking_tutorial.mp4",
        "scenario": "Kitchen memory from a weeknight family recipe session.",
        "images": ["food_pasta_dish.png", "handwritten_notes.png", "food_pasta_dish.png"],
        "duration": 9,
        "scene_durations": [3, 3, 3],
        "effects": "episodic_sequence",
        "scene_captions": [
            "6:02 PM - sauce check",
            "6:08 PM - recipe tweak notes",
            "6:16 PM - plated pasta",
        ],
        "transcript": [
            {"start": 0.0, "end": 2.5, "text": "The cook checks the pasta sauce and mentions basil, tomato, olive oil, and a lower-salt variation."},
            {"start": 2.5, "end": 5.5, "text": "A handwritten recipe note records timing changes and a reminder to try chili flakes next time."},
            {"start": 5.5, "end": 7.5, "text": "The plated pasta is compared with earlier cooking notes about fresh dough and sauce texture."},
            {"start": 7.5, "end": 9.0, "text": "The clip ends with a spoken tag for family dinner, recipe recall, and grocery planning."},
        ],
        "description": "Kitchen recipe memory with plated pasta, spoken substitutions, and handwritten cooking notes.",
        "related_images": ["images/food_pasta_dish.png", "images/handwritten_notes.png"],
        "related_documents": ["text/cooking_pasta.md", "text/cooking_spices.md", "text/cooking_asian_cuisine.md"],
        "notes": "Useful for queries about pasta, family dinner, recipe substitutions, grocery planning, and handwritten cooking notes.",
    },
    {
        "name": "whiteboard_session.mp4",
        "scenario": "Product planning meeting with whiteboard decisions and next-step notes.",
        "images": ["whiteboard_brainstorm.png", "whiteboard_architecture.png", "handwritten_notes.png"],
        "duration": 12,
        "scene_durations": [4, 4, 4],
        "effects": "episodic_sequence",
        "scene_captions": [
            "Sprint planning - memory UX",
            "Decision - parent/child rollups",
            "Owner notes - docs and release",
        ],
        "transcript": [
            {"start": 0.0, "end": 3.0, "text": "The team brainstorms how a user should ask for the last whiteboard from a meeting."},
            {"start": 3.0, "end": 6.0, "text": "The architecture sketch shows root memories with child frames, transcripts, OCR pages, and document sections."},
            {"start": 6.0, "end": 9.0, "text": "A decision is made to score parent memories separately from raw child assets in the benchmark."},
            {"start": 9.0, "end": 12.0, "text": "Handwritten action items assign documentation updates, benchmark reruns, and release checklist cleanup."},
        ],
        "description": "Planning-meeting recording with whiteboard brainstorms, memory rollup decisions, and handwritten action items.",
        "related_images": ["images/whiteboard_brainstorm.png", "images/whiteboard_architecture.png", "images/handwritten_notes.png"],
        "related_documents": ["documents/ai_strategy_report.docx", "documents/project_status_q1.docx", "documents/quarterly_review.pptx", "documents/recallforge_spec.docx"],
        "notes": "Useful for meeting-memory queries, whiteboard recall, product planning, memory rollups, benchmark scoring, and release documentation.",
    },
]


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _scale_filter(w: int = WIDTH, h: int = HEIGHT) -> str:
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"


def _wrap_caption(text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont, max_width: int) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    for word in words:
        candidate = " ".join(current + [word])
        bbox = measure.textbbox((0, 0), candidate, font=font)
        if current and (bbox[2] - bbox[0]) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


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

        text = _wrap_caption(text, font, WIDTH - 24)

        # Measure text bounding box
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3, align="center")
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
        draw.multiline_text((x, y), text, font=font, fill=(255, 255, 255), spacing=3, align="center")

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


def _build_image_sequence_cmd(
    ffmpeg: str,
    images: list[Path],
    output: Path,
    durations: list[float],
) -> list[str]:
    """Build ffmpeg command for an image-sequence video."""
    if len(images) != len(durations):
        raise ValueError("images and durations must have matching lengths")
    if not images:
        raise ValueError("at least one image is required")

    scale = _scale_filter()
    inputs: list[str] = []
    filters: list[str] = []
    concat_labels: list[str] = []

    for index, (image, duration) in enumerate(zip(images, durations)):
        inputs.extend(["-loop", "1", "-t", f"{duration:g}", "-i", str(image)])
        label = f"v{index}"
        filters.append(
            f"[{index}:v]{scale},trim=duration={duration:g},setpts=PTS-STARTPTS[{label}]"
        )
        concat_labels.append(f"[{label}]")

    filters.append(f"{''.join(concat_labels)}concat=n={len(images)}:v=1:a=0[v]")
    filter_complex = ";".join(filters)

    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "32",
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
    effects = spec["effects"]
    captions = spec.get("scene_captions", [])
    durations = [float(value) for value in spec.get("scene_durations", [])]
    if not durations:
        segment = float(duration) / float(len(images))
        durations = [segment for _ in images]

    with tempfile.TemporaryDirectory() as tmp_dir:
        prepared_images: list[Path] = []
        for index, image in enumerate(images):
            caption = captions[index] if index < len(captions) else ""
            prepared_images.append(
                _burn_text_onto_image(image, caption, tmp_dir)
                if caption
                else image
            )

        if len(images) == 1:
            zoom = effects == "zoom_in"
            cmd = _build_single_image_cmd(
                ffmpeg,
                prepared_images[0],
                output,
                duration,
                None,
                zoom=zoom,
                tmp_dir=tmp_dir,
            )
        else:
            cmd = _build_image_sequence_cmd(ffmpeg, prepared_images, output, durations)

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
        "memory_type": "episodic_video_fixture",
        "scenario": spec["scenario"],
        "description": spec["description"],
        "notes": spec["notes"],
        "related_images": spec.get("related_images", []),
        "related_documents": spec.get("related_documents", []),
        "duration_seconds": spec["duration"],
        "resolution": f"{WIDTH}x{HEIGHT}",
        "fps": FPS,
        "text": " ".join(
            [spec["description"], spec["scenario"], spec["notes"]]
            + [seg["text"] for seg in spec["transcript"]]
        ),
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
    print(f"All {len(VIDEOS)} videos generated. Total size: {total_bytes // 1024} KB")
    if total_bytes > 12 * 1024 * 1024:
        print("WARNING: corpus exceeds 12 MB git-friendliness target", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
