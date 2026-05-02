#!/usr/bin/env python3
"""
Functional tests for video input helpers.
These tests use a temporary synthetic video file and avoid requiring a real webcam.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.video_input import open_capture, read_frame, release_capture


class TestVideoInput(unittest.TestCase):
    """Test recorded-video input behavior."""

    def test_video_file_loading(self) -> None:
        """Open a generated video file and read one frame."""
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (160, 120),
            )
            for i in range(5):
                frame = np.full((120, 160, 3), i * 30, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            cap = open_capture("video", path=video_path)
            ok, frame = read_frame(cap)
            release_capture(cap)

            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape[:2], (120, 160))

    def test_bad_video_path_raises(self) -> None:
        """Fail clearly when a video file path does not exist."""
        with self.assertRaises(FileNotFoundError):
            open_capture("video", path="missing_video.mp4")


if __name__ == "__main__":
    unittest.main()
