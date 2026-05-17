"""Regression tests for the committed episodic video corpus."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_PATH = REPO_ROOT / "tests" / "uat" / "helpers" / "generate_video_corpus.py"
VIDEOS_DIR = REPO_ROOT / "tests" / "uat" / "corpus" / "videos"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_video_corpus", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestEpisodicVideoCorpus(unittest.TestCase):
    def test_generator_specs_are_rich_episodic_fixtures(self):
        module = _load_generator()

        self.assertEqual(len(module.VIDEOS), 5)
        for spec in module.VIDEOS:
            with self.subTest(video=spec["name"]):
                self.assertGreaterEqual(spec["duration"], 9)
                self.assertGreaterEqual(len(spec["images"]), 2)
                self.assertGreaterEqual(len(spec["transcript"]), 3)
                self.assertTrue(spec["scenario"])
                self.assertTrue(spec["notes"])
                self.assertTrue(spec["related_images"])
                self.assertTrue(spec["related_documents"])

    def test_committed_sidecars_include_searchable_transcript_text(self):
        sidecars = sorted(VIDEOS_DIR.glob("*.transcript.json"))

        self.assertEqual(len(sidecars), 5)
        for sidecar in sidecars:
            with self.subTest(sidecar=sidecar.name):
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                self.assertEqual(payload["memory_type"], "episodic_video_fixture")
                self.assertTrue(payload["scenario"])
                self.assertTrue(payload["description"])
                self.assertTrue(payload["notes"])
                self.assertTrue(payload["text"])
                self.assertGreaterEqual(len(payload["segments"]), 3)
                self.assertTrue(payload["related_images"])
                self.assertTrue(payload["related_documents"])


if __name__ == "__main__":
    unittest.main()
