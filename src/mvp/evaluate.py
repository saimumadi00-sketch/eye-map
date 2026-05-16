#!/usr/bin/env python3
"""
Evaluation utilities for trajectory accuracy, point cloud quality, and depth comparison against VKITTI ground truth.
The functions here are intentionally lightweight so final-year project reports can reproduce numeric results without adding new dependencies.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    from src.core.utils import load_ply_xyz
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.core.utils import load_ply_xyz


def _require_file(path: str | Path) -> Path:
    """Return a Path if it exists, otherwise raise FileNotFoundError."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")
    return file_path


def _load_trajectory_csv(path: str | Path) -> dict[int, np.ndarray]:
    """Load frame_id,x,y,z rows into a frame-indexed dictionary."""
    csv_path = _require_file(path)
    trajectory: dict[int, np.ndarray] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row["frame_id"])
            trajectory[frame_id] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])],
                dtype=np.float64,
            )
    return trajectory


def _trajectory_points_in_order(path: str | Path) -> np.ndarray:
    """Load trajectory points sorted by frame_id as an Nx3 array."""
    trajectory = _load_trajectory_csv(path)
    if not trajectory:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([trajectory[k] for k in sorted(trajectory)], dtype=np.float64)


def compute_trajectory_length(trajectory_csv: str | Path) -> float:
    """Return the total arc length of a trajectory CSV."""
    points = _trajectory_points_in_order(trajectory_csv)
    if len(points) < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.sum(steps))


def compute_drift_ratio(trajectory_csv: str | Path) -> float:
    """Return end-to-start distance divided by trajectory arc length."""
    points = _trajectory_points_in_order(trajectory_csv)
    if len(points) < 2:
        return 0.0
    total_length = compute_trajectory_length(trajectory_csv)
    if total_length <= 0.0:
        return 0.0
    drift = np.linalg.norm(points[-1] - points[0])
    return float(drift / total_length)


def point_cloud_stats(ply_path: str | Path) -> dict:
    """Return point count, bounding box, and approximate point density for a PLY cloud."""
    points = load_ply_xyz(_require_file(ply_path))
    if len(points) == 0:
        return {
            "point_count": 0,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "approx_density": 0.0,
        }

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    extent = bbox_max - bbox_min
    volume = float(np.prod(extent))
    density = 0.0 if volume <= 0.0 else float(len(points) / volume)
    return {
        "point_count": int(len(points)),
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "approx_density": density,
    }


def compare_trajectories(est_csv: str | Path, gt_csv: str | Path) -> dict:
    """Compare estimated and ground-truth trajectories after frame_id alignment."""
    estimated = _load_trajectory_csv(est_csv)
    ground_truth = _load_trajectory_csv(gt_csv)
    common_ids = sorted(set(estimated) & set(ground_truth))
    if len(common_ids) < 2:
        raise ValueError("Need at least 2 aligned frame_id rows to compare trajectories.")

    errors = np.asarray(
        [np.linalg.norm(estimated[frame_id] - ground_truth[frame_id]) for frame_id in common_ids],
        dtype=np.float64,
    )
    return {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "max_error": float(np.max(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "aligned_frames": int(len(common_ids)),
    }


def _zero_depth_stats() -> dict:
    """Return an all-zero depth error statistics dictionary."""
    return {"mean_abs_err_cm": 0.0, "median_abs_err_cm": 0.0, "valid_point_count": 0}


def depth_error_stats(sequence, sparse_ply: str | Path, max_frames: int = 50) -> dict:
    """Compare sparse map point depths against VKITTI depth maps in centimeters."""
    points_world = load_ply_xyz(_require_file(sparse_ply))
    if len(points_world) == 0 or len(sequence) == 0:
        return _zero_depth_stats()

    sample_count = len(sequence) if int(max_frames) <= 0 else min(len(sequence), int(max_frames))
    indices = np.linspace(0, len(sequence) - 1, num=sample_count, dtype=np.int64)
    errors_cm: list[float] = []

    for sequence_index in indices:
        record = sequence[int(sequence_index)]
        rgb = record["rgb"]
        depth_cm = np.asarray(record["depth_cm"])
        K = np.asarray(record["K"], dtype=np.float64)
        T_cw = np.asarray(record["T_cw"], dtype=np.float64)
        h, w = depth_cm.shape
        _ = rgb

        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]
        points_cam = (R_cw @ points_world.T + t_cw.reshape(3, 1)).T
        positive_depth = points_cam[:, 2] > 0.0
        if not np.any(positive_depth):
            continue

        points_cam = points_cam[positive_depth]
        pixels = (K @ points_cam.T).T
        u = pixels[:, 0] / pixels[:, 2]
        v = pixels[:, 1] / pixels[:, 2]
        in_bounds = (u >= 0.0) & (u < float(w)) & (v >= 0.0) & (v < float(h))
        if not np.any(in_bounds):
            continue

        points_cam = points_cam[in_bounds]
        u_int = u[in_bounds].astype(np.int64)
        v_int = v[in_bounds].astype(np.int64)
        gt_depth_cm = depth_cm[v_int, u_int].astype(np.float64)
        has_depth = gt_depth_cm > 0.0
        if not np.any(has_depth):
            continue

        projected_depth_cm = points_cam[:, 2] * 100.0
        frame_errors = np.abs(projected_depth_cm[has_depth] - gt_depth_cm[has_depth])
        errors_cm.extend(float(value) for value in frame_errors)

    if not errors_cm:
        return _zero_depth_stats()

    errors = np.asarray(errors_cm, dtype=np.float64)
    return {
        "mean_abs_err_cm": float(np.mean(errors)),
        "median_abs_err_cm": float(np.median(errors)),
        "valid_point_count": int(len(errors)),
    }


def print_report(traj_stats=None, depth_stats=None, cloud_stats=None) -> None:
    """Print formatted evaluation sections for any provided statistics."""
    sections = [
        ("Trajectory Evaluation", traj_stats),
        ("Depth Error", depth_stats),
        ("Point Cloud Quality", cloud_stats),
    ]
    print("EyeMap Evaluation Report")
    print("=" * 24)
    for title, stats in sections:
        if not stats:
            continue
        print()
        print(title)
        print("-" * len(title))
        for key, value in stats.items():
            if isinstance(value, float):
                formatted = f"{value:.6f}"
            else:
                formatted = str(value)
            print(f"{key:24s} {formatted}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for direct report generation."""
    parser = argparse.ArgumentParser(description="Evaluate EyeMap trajectory and sparse point cloud outputs.")
    parser.add_argument("--est-csv", required=True, help="Path to estimated trajectory.csv.")
    parser.add_argument("--gt-csv", default=None, help="Optional path to gt_trajectory.csv.")
    parser.add_argument("--ply", default=None, help="Optional path to sparse_map.ply.")
    return parser.parse_args()


def main() -> None:
    """Run the evaluation CLI and print a terminal report."""
    args = parse_args()
    traj_stats = {
        "trajectory_length": compute_trajectory_length(args.est_csv),
        "drift_ratio": compute_drift_ratio(args.est_csv),
    }
    if args.gt_csv is not None:
        traj_stats.update(compare_trajectories(args.est_csv, args.gt_csv))

    cloud_stats = point_cloud_stats(args.ply) if args.ply is not None else None
    print_report(traj_stats=traj_stats, cloud_stats=cloud_stats)


if __name__ == "__main__":
    main()
