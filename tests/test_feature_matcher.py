#!/usr/bin/env python3
"""
Functional tests for ORB feature matching.
The tests match two slightly shifted synthetic frames to verify descriptor filtering and visualization.
"""

from __future__ import annotations

import unittest

import cv2

from src.feature_detector import create_orb, detect_features
from src.feature_matcher import draw_matches, match_features, matched_points
from tests.test_feature_detector import synthetic_feature_frame


class TestFeatureMatcher(unittest.TestCase):
    """Test descriptor matching between consecutive frames."""

    def test_match_features(self) -> None:
        """Match features between two translated synthetic frames."""
        detector = create_orb(1000)
        frame_a = synthetic_feature_frame()
        matrix = cv2.getRotationMatrix2D((160, 120), 0.0, 1.0)
        matrix[0, 2] = 6.0
        matrix[1, 2] = 3.0
        frame_b = cv2.warpAffine(frame_a, matrix, (320, 240))

        _, kp_a, desc_a = detect_features(frame_a, detector)
        _, kp_b, desc_b = detect_features(frame_b, detector)
        matches = match_features(desc_a, desc_b, ratio_test=0.85, max_matches=200)
        pts_a, pts_b = matched_points(kp_a, kp_b, matches)
        vis = draw_matches(frame_a, kp_a, frame_b, kp_b, matches)

        self.assertGreater(len(matches), 10)
        self.assertEqual(pts_a.shape[1], 2)
        self.assertEqual(pts_b.shape[1], 2)
        self.assertEqual(vis.shape[0], frame_a.shape[0])


if __name__ == "__main__":
    unittest.main()
