#!/usr/bin/env python3
"""
Sparse triangulation utilities for matched feature points.
The output is a rough point cloud suitable for demonstrating structure, not a dense GIS-grade terrain surface.
"""

from __future__ import annotations

import cv2
import numpy as np


def triangulate_relative(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
) -> np.ndarray:
    """Triangulate matched points in the previous camera coordinate frame."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1), dtype=np.float64)])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    points_h = cv2.triangulatePoints(P1, P2, pts_prev.T, pts_curr.T).T
    return points_h[:, :3] / points_h[:, 3:4]


def filter_triangulated_points(
    points_prev: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    max_norm: float = 150.0,
) -> np.ndarray:
    """Keep finite points with positive depth in both camera views."""
    if len(points_prev) == 0:
        return points_prev

    points_curr = (R @ points_prev.T + t.reshape(3, 1)).T
    valid = (
        np.isfinite(points_prev).all(axis=1)
        & (points_prev[:, 2] > 0.0)
        & (points_curr[:, 2] > 0.0)
        & (np.linalg.norm(points_prev, axis=1) < float(max_norm))
    )
    return points_prev[valid]


def transform_points(T_wc: np.ndarray, points_camera: np.ndarray) -> np.ndarray:
    """Transform 3D points from camera coordinates to world coordinates."""
    if len(points_camera) == 0:
        return points_camera
    points_h = np.hstack([points_camera, np.ones((len(points_camera), 1), dtype=np.float64)])
    return (T_wc @ points_h.T).T[:, :3]
