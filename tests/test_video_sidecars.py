import json
import tempfile
import unittest
from pathlib import Path

from recallforge.video import find_transcript_sidecar, load_transcript_segments


class TestVideoTranscriptSidecars(unittest.TestCase):
    def test_find_and_load_transcript_json_sidecar_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "architecture_walkthrough.mp4"
            sidecar_path = root / "architecture_walkthrough.transcript.json"
            video_path.write_bytes(b"fake video")
            sidecar_path.write_text(
                json.dumps(
                    {
                        "video": "architecture_walkthrough.mp4",
                        "description": "Architecture walkthrough sliding between diagrams and blueprints.",
                        "segments": [
                            {"start": 0.0, "end": 2.0, "text": "Whiteboard diagram showing system architecture."},
                            {"start": 2.0, "end": 4.0, "text": "Detailed floor plan blueprint with room layouts."},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(find_transcript_sidecar(video_path), sidecar_path)

            segments, transcript_path = load_transcript_segments(
                video_path,
                "videos/architecture_walkthrough.mp4",
            )

            self.assertEqual(transcript_path, str(sidecar_path))
            self.assertEqual(len(segments), 2)
            self.assertEqual(
                segments[0].logical_path,
                "videos/architecture_walkthrough.mp4::transcript:0001@0.00s",
            )
            self.assertEqual(segments[0].text, "Whiteboard diagram showing system architecture.")
            self.assertEqual(
                segments[1].logical_path,
                "videos/architecture_walkthrough.mp4::transcript:0002@2.00s",
            )
            self.assertEqual(segments[1].end_seconds, 4.0)

    def test_transcript_json_description_falls_back_when_segments_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_path = root / "coding_demo.mp4"
            sidecar_path = root / "coding_demo.transcript.json"
            video_path.write_bytes(b"fake video")
            sidecar_path.write_text(
                json.dumps(
                    {
                        "video": "coding_demo.mp4",
                        "description": "Coding tutorial demo with text overlay on code editor screenshot.",
                        "segments": [],
                    }
                ),
                encoding="utf-8",
            )

            segments, transcript_path = load_transcript_segments(
                video_path,
                "videos/coding_demo.mp4",
            )

            self.assertEqual(transcript_path, str(sidecar_path))
            self.assertEqual(len(segments), 1)
            self.assertEqual(
                segments[0].logical_path,
                "videos/coding_demo.mp4::transcript:0001@0.00s",
            )
            self.assertEqual(
                segments[0].text,
                "Coding tutorial demo with text overlay on code editor screenshot.",
            )


if __name__ == "__main__":
    unittest.main()
