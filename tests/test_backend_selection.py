"""
Tests for backend selection behavior.

These are non-live tests that validate safe fallback behavior
without loading any model weights.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import recallforge
from recallforge.backends.torch_backend import TorchBackend


class TestBackendSelection(unittest.TestCase):
    """Backend selection should be deterministic and crash-safe."""

    def test_explicit_torch_backend(self):
        with patch.dict(
            os.environ,
            {
                "RECALLFORGE_BACKEND": "torch",
                "RECALLFORGE_MODE": "embed",
            },
            clear=False,
        ):
            backend = recallforge.get_backend()
            self.assertIsInstance(backend, TorchBackend)

    def test_auto_backend_falls_back_to_torch_when_mlx_unavailable(self):
        with patch.dict(
            os.environ,
            {
                "RECALLFORGE_BACKEND": "auto",
                "RECALLFORGE_MODE": "embed",
            },
            clear=False,
        ):
            with patch("platform.system", return_value="Darwin"), patch(
                "platform.machine", return_value="arm64"
            ), patch("recallforge.backends.MLX_AVAILABLE", False):
                backend = recallforge.get_backend()
                self.assertIsInstance(backend, TorchBackend)


if __name__ == "__main__":
    unittest.main()
