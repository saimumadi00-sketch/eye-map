#!/usr/bin/env python3
"""
Simple Open3D viewer for sparse map point clouds.
Usage:
  python src/mvp/view_cloud.py --ply outputs/<run_id>/sparse_map.ply
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View sparse point cloud with Open3D.")
    parser.add_argument("--ply", type=str, required=True, help="Path to sparse_map.ply")
    parser.add_argument("--point-size", type=float, default=2.0, help="Render point size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ply_path = Path(args.ply)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "Open3D is not installed. Install with: pip install open3d"
        ) from exc

    cloud = o3d.io.read_point_cloud(str(ply_path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty: {ply_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Sparse Map Viewer", width=1280, height=720)
    vis.add_geometry(cloud)
    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = [0.04, 0.04, 0.04]
    vis.run()
    vis.destroy_window()

    logger.info("Visualized: %s", ply_path)
    logger.debug("Points: %s", len(cloud.points))


if __name__ == "__main__":
    main()
