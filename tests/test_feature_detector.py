#!/usr/bin/env python3
"""
Functional tests for ORB feature detection.
Synthetic high-contrast shapes are used so the test is deterministic and does not require sample assets.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.feature_detector import create_orb, detect_features, draw_keypoints


def synthetic_feature_frame() -> np.ndarray:
    """Create a synthetic image with repeatable corners and blobs."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(30, 220, 40):
        for x in range(30, 300, 40):
            cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
            cv2.rectangle(frame, (x - 12, y - 12), (x + 12, y + 12), (80, 180, 255), 2)
    return frame


class TestFeatureDetector(unittest.TestCase):
    """Test ORB detection and visualization."""

    def test_detect_features(self) -> None:
        """Detect ORB keypoints and descriptors on a synthetic frame."""
        detector = create_orb(1000)
        frame = synthetic_feature_frame()
        _, keypoints, descriptors = detect_features(frame, detector)

        self.assertGreater(len(keypoints), 20)
        self.assertIsNotNone(descriptors)

    def test_draw_keypoints(self) -> None:
        """Draw keypoints without changing image dimensions."""
        detector = create_orb(1000)
        frame = synthetic_feature_frame()
        _, keypoints, _ = detect_features(frame, detector)
        drawn = draw_keypoints(frame, keypoints)

        self.assertEqual(drawn.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
