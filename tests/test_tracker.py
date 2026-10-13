#!/usr/bin/env python3
"""
Functional tests for reference-frame tracking and relative scale propagation.
Synthetic keypoints with unique descriptors stand in for ORB so the tests isolate the geometry.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.pose_estimator import camera_center_from_tcw, make_intrinsics
from src.tracker import MonocularTracker


class SyntheticScene:
    """Project fixed 3D points into cameras translated along the world x axis."""

    def __init__(self, n_points: int = 150) -> None:
        rng = np.random.default_rng(7)
        self.points = rng.uniform([-2.0, -1.0, 4.0], [2.0, 1.0, 8.0], (n_points, 3))
        self.descriptors = rng.integers(0, 256, (n_points, 32), dtype=np.uint8)
        self.K = make_intrinsics(640, 480)

    def frame(self, center_x: float) -> tuple[tuple, np.ndarray]:
        """Return keypoints and descriptors seen from a camera at (center_x, 0, 0)."""
        cam = self.points - np.array([center_x, 0.0, 0.0])
        pix = (self.K @ cam.T).T
        pix = pix[:, :2] / pix[:, 2:3]
        keypoints = tuple(cv2.KeyPoint(float(u), float(v), 7.0) for u, v in pix)
        return keypoints, self.descriptors.copy()


def run_tracker(centers: list[float]) -> tuple[list[float], np.ndarray]:
    """Track a camera path and return GT and estimated centers for frames that produced a pose."""
    cv2.setRNGSeed(3)
    scene = SyntheticScene()
    tracker = MonocularTracker()
    tracker.initialize(*scene.frame(centers[0]))
    gt, est = [centers[0]], [camera_center_from_tcw(tracker.T_cw)]
    for center_x in centers[1:]:
        result = tracker.track(*scene.frame(center_x), scene.K)
        if result.success:
            gt.append(center_x)
            est.append(camera_center_from_tcw(result.T_cw))
    return gt, np.asarray(est)


class TestMonocularTracker(unittest.TestCase):
    """Check that step lengths keep their ratios and slow motion accumulates."""

    def assert_matches_up_to_scale(self, gt: list[float], est: np.ndarray) -> None:
        gt_xyz = np.column_stack([gt, np.zeros(len(gt)), np.zeros(len(gt))])
        scale = np.linalg.norm(est[-1]) / np.linalg.norm(gt_xyz[-1])
        np.testing.assert_allclose(est, scale * gt_xyz, atol=0.02 * scale * abs(gt[-1]))

    def test_uneven_steps_keep_relative_scale(self) -> None:
        centers = list(np.cumsum([0.0, 0.2, 0.4, 0.1, 0.3, 0.25]))
        gt, est = run_tracker(centers)
        self.assertEqual(len(gt), len(centers), "every step has enough parallax")
        self.assert_matches_up_to_scale(gt, est)

    def test_slow_motion_accumulates_against_reference(self) -> None:
        # Each 0.01 step is ~0.1 degrees of parallax, below the 0.5 degree threshold.
        centers = [0.01 * i for i in range(40)]
        gt, est = run_tracker(centers)
        self.assertGreaterEqual(len(gt), 3, "held reference should eventually gain parallax")
        self.assertLess(len(gt), len(centers), "small steps alone should not produce poses")
        self.assert_matches_up_to_scale(gt, est)


if __name__ == "__main__":
    unittest.main()
