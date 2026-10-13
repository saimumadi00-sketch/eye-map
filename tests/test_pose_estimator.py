#!/usr/bin/env python3
"""
Functional tests for relative pose estimation.
The test projects synthetic 3D points into two camera views so pose recovery can be checked without a dataset.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.pose_estimator import estimate_relative_pose_from_points, make_intrinsics, update_pose


def project_points(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D points into a camera."""
    cam = (R @ points_3d.T + t.reshape(3, 1)).T
    pix = (K @ cam.T).T
    return pix[:, :2] / pix[:, 2:3]


class TestPoseEstimator(unittest.TestCase):
    """Test essential-matrix based relative pose estimation."""

    def test_estimate_relative_pose_from_points(self) -> None:
        """Recover relative pose from synthetic point correspondences."""
        rng = np.random.default_rng(4)
        points = np.column_stack(
            [
                rng.uniform(-1.5, 1.5, 120),
                rng.uniform(-0.7, 0.7, 120),
                rng.uniform(4.0, 8.0, 120),
            ]
        )
        K = make_intrinsics(640, 480)
        R1 = np.eye(3, dtype=np.float64)
        t1 = np.zeros(3, dtype=np.float64)
        R2, _ = cv2.Rodrigues(np.array([0.0, 0.03, 0.0], dtype=np.float64))
        t2 = np.array([0.25, 0.0, 0.0], dtype=np.float64)

        pts1 = project_points(points, R1, t1, K).astype(np.float32)
        pts2 = project_points(points, R2, t2, K).astype(np.float32)
        result = estimate_relative_pose_from_points(pts1, pts2, K, ransac_thresh=1.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["R"].shape, (3, 3))
        self.assertEqual(result["t"].shape, (3,))
        self.assertEqual(update_pose(np.eye(4), result["R"], result["t"]).shape, (4, 4))
        self.assertGreaterEqual(np.count_nonzero(result["inlier_mask"]), 10)
        np.testing.assert_allclose(result["R"], R2, atol=0.01)
        np.testing.assert_allclose(result["t"], t2 / np.linalg.norm(t2), atol=0.01)

    def test_stationary_correspondences_reject_motion(self) -> None:
        rng = np.random.default_rng(4)
        points = rng.uniform([0, 0], [640, 480], (120, 2)).astype(np.float32)
        cv2.setRNGSeed(4)
        result = estimate_relative_pose_from_points(points, points, make_intrinsics(640, 480))
        self.assertFalse(result["success"])

    def test_pure_rotation_rejects_translation(self) -> None:
        rng = np.random.default_rng(8)
        points = rng.uniform([-1.5, -0.7, 4.0], [1.5, 0.7, 8.0], (120, 3))
        K = make_intrinsics(640, 480)
        rotation, _ = cv2.Rodrigues(np.array([0.0, 0.08, 0.0]))
        pts1 = project_points(points, np.eye(3), np.zeros(3), K).astype(np.float32)
        pts2 = project_points(points, rotation, np.zeros(3), K).astype(np.float32)
        result = estimate_relative_pose_from_points(pts1, pts2, K)
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
