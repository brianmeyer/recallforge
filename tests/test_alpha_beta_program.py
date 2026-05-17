"""
test_alpha_beta_program.py - Launch-program utilities for REC-33.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from recallforge.cli import main
from recallforge.diagnostics import collect_crash_report, sanitized_recallforge_env, write_crash_report
from recallforge.feature_flags import list_feature_flags


class TestFeatureFlags(unittest.TestCase):
    def test_feature_flag_registry_includes_beta_flags(self):
        flags = list_feature_flags(
            {
                "RECALLFORGE_ENABLE_MEDIA_RERANKING": "1",
                "RECALLFORGE_TRACE": "0",
            }
        )
        names = {flag["name"] for flag in flags}
        self.assertIn("RECALLFORGE_ENABLE_MEDIA_RERANKING", names)
        self.assertIn("RECALLFORGE_ENABLE_RAW_VIDEO_QUERY_EMBEDDING", names)
        media_flag = next(flag for flag in flags if flag["name"] == "RECALLFORGE_ENABLE_MEDIA_RERANKING")
        self.assertTrue(media_flag["enabled"])
        concurrency_flag = next(flag for flag in flags if flag["name"] == "RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY")
        self.assertIsNone(concurrency_flag["enabled"])

    def test_flags_cli_outputs_json(self):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["recallforge", "flags", "--json"]), redirect_stdout(stdout):
            code = main()
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("feature_flags", payload)
        self.assertTrue(payload["feature_flags"])


class TestCrashReports(unittest.TestCase):
    def test_sanitized_env_allowlists_and_redacts_home_path(self):
        home_store = str(Path.home() / ".recallforge")
        env = {
            "RECALLFORGE_STORE_PATH": home_store,
            "RECALLFORGE_MODE": "embed",
            "SECRET_TOKEN": "do-not-include",
        }
        sanitized = sanitized_recallforge_env(env)
        self.assertEqual(sanitized["RECALLFORGE_STORE_PATH"], "~/.recallforge")
        self.assertEqual(sanitized["RECALLFORGE_MODE"], "embed")
        self.assertNotIn("SECRET_TOKEN", sanitized)

    def test_collect_crash_report_is_local_only(self):
        report = collect_crash_report(
            message="search crashed after video query",
            include_env=True,
            environ={"RECALLFORGE_TRACE": "1"},
        )
        self.assertEqual(report["privacy"]["network_sent"], False)
        self.assertEqual(report["privacy"]["sharing"], "manual")
        self.assertEqual(report["environment"]["RECALLFORGE_TRACE"], "1")
        self.assertIn("search crashed", report["user_message"])

    def test_crash_report_cli_writes_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "crash.json"
            stdout = io.StringIO()
            with patch.object(
                sys,
                "argv",
                ["recallforge", "crash-report", "--output", str(output), "--message", "boom"],
            ), redirect_stdout(stdout):
                code = main()

            self.assertEqual(code, 0)
            self.assertTrue(output.exists())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["user_message"], "boom")
            self.assertFalse(payload["privacy"]["network_sent"])

    def test_write_crash_report_returns_resolved_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "report.json"
            written = write_crash_report(output, message="local failure")
            self.assertTrue(written.exists())
            payload = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(payload["user_message"], "local failure")


if __name__ == "__main__":
    unittest.main()
