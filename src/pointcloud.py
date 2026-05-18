#!/usr/bin/env python3
"""
Point cloud export and visualization helpers.
The PLY writer uses a lightweight ASCII format, while Open3D visualization is optional and skipped gracefully if unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.core.utils import save_ply_xyz

logger = logging.getLogger(__name__)


def save_pointcloud(path: str | Path, points_xyz: np.ndarray, max_points: int = 120000) -> Path:
    """Save a sparse 3D point cloud as PLY."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if len(points) > int(max_points):
        idx = np.random.choice(len(points), size=int(max_points), replace=False)
        points = points[idx]
    ply_path = Path(path)
    save_ply_xyz(ply_path, points)
    return ply_path


def visualize_pointcloud(path: str | Path) -> bool:
    """Open a PLY point cloud in Open3D if the package is available."""
    try:
        import open3d as o3d
    except Exception:
        logger.warning("Open3D is unavailable. Point cloud was saved but not displayed.")
        return False

    ply_path = Path(path)
    cloud = o3d.io.read_point_cloud(str(ply_path))
    if cloud.is_empty():
        logger.warning("Point cloud is empty: %s", ply_path)
        return False
    o3d.visualization.draw_geometries([cloud], window_name="EyeMap Sparse Point Cloud")
    return True
