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

import main
from src.core.calibration import load_calibration, load_calibration_image_size, save_calibration


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

    def test_image_size_round_trip_and_legacy_files(self) -> None:
        """Store the solved resolution, and report None for files saved without it."""
        with tempfile.TemporaryDirectory() as tmp:
            sized, legacy = Path(tmp) / "sized.npz", Path(tmp) / "legacy.npz"
            save_calibration(np.eye(3), np.zeros(5), sized, image_size=(1280, 720))
            save_calibration(np.eye(3), np.zeros(5), legacy)
            self.assertEqual(load_calibration_image_size(sized), (1280, 720))
            self.assertIsNone(load_calibration_image_size(legacy))

    def test_pipeline_rescales_calibrated_intrinsics_to_frame_size(self) -> None:
        """Calibrated K solved at 1280x720 is halved for 640x360 processed frames."""
        K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.npz"
            save_calibration(K, np.zeros(5), path, image_size=(1280, 720))
            scaled, source = main.resolve_intrinsics(str(path), 640, 360)
            np.testing.assert_allclose(scaled, [[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]])
            self.assertTrue(source.startswith("calibrated"))
            with self.assertRaises(FileNotFoundError):
                main.resolve_intrinsics(str(Path(tmp) / "missing.npz"), 640, 360)


if __name__ == "__main__":
    unittest.main()
