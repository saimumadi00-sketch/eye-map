#!/usr/bin/env python3
"""
Camera motion estimation helpers for monocular visual odometry.
The estimated translation direction is useful for a relative trajectory, but monocular video cannot recover metric scale without extra information.
"""

from __future__ import annotations

import cv2
import numpy as np


def make_intrinsics(width: int, height: int) -> np.ndarray:
    """Estimate a simple camera intrinsic matrix when calibration is unavailable."""
    focal = 0.9 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_center_from_tcw(T_cw: np.ndarray) -> np.ndarray:
    """Compute camera center in world coordinates from a world-to-camera pose."""
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    return -R_cw.T @ t_cw


def estimate_relative_pose_from_points(
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 1.0,
) -> dict:
    """Estimate relative pose from matched 2D points."""
    if len(pts_prev) < 8 or len(pts_curr) < 8:
        return {"success": False, "reason": "not enough matched points"}

    E, inlier_mask = cv2.findEssentialMat(
        pts_prev,
        pts_curr,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=float(ransac_thresh),
    )
    if E is None or inlier_mask is None:
        return {"success": False, "reason": "essential matrix failed"}

    _, R, t, pose_mask = cv2.recoverPose(E, pts_prev, pts_curr, K)

    # Monocular translation has unknown metric scale; normalize for stable relative plotting.
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-9:
        t = t / t_norm

    inliers = inlier_mask.ravel().astype(bool)
    pose_inliers = pose_mask.ravel().astype(bool)
    return {
        "success": True,
        "R": R,
        "t": t.reshape(3),
        "inlier_mask": inliers & pose_inliers,
    }


def update_pose(T_cw: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Update cumulative world-to-camera pose using the latest relative motion."""
    T_rel = np.eye(4, dtype=np.float64)
    T_rel[:3, :3] = R
    T_rel[:3, 3] = t.reshape(3)
    return T_rel @ T_cw
