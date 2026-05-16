#!/usr/bin/env python3
"""
Functional tests for camera calibration persistence.
The tests cover NPZ save/load behavior without requiring a camera or OpenCV window.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.core.calibration import load_calibration, save_calibration


class TestCalibration(unittest.TestCase):
    """Test calibration file save and load helpers."""

    def test_save_load_round_trip(self) -> None:
        """Save and load a synthetic calibration without numeric changes."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.npz"
            K = np.eye(3, dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)
            save_calibration(K, dist, path)
            loaded_K, loaded_dist = load_calibration(path)
            np.testing.assert_array_equal(loaded_K, K)
            np.testing.assert_array_equal(loaded_dist, dist)

    def test_missing_file_returns_none_tuple(self) -> None:
        """Return (None, None) when the calibration file does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            loaded_K, loaded_dist = load_calibration(Path(tmp) / "missing.npz")
            self.assertIsNone(loaded_K)
            self.assertIsNone(loaded_dist)


if __name__ == "__main__":
    unittest.main()
