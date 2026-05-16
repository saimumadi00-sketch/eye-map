#!/usr/bin/env python3
"""
Functional tests for EyeMap evaluation helpers.
The tests use temporary CSV and PLY files so they do not depend on real output runs or datasets.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.core.utils import save_ply_xyz
from src.mvp.evaluate import (
    compare_trajectories,
    compute_drift_ratio,
    compute_trajectory_length,
    point_cloud_stats,
)


def write_trajectory_csv(path: Path, rows: list[list[float]]) -> None:
    """Write frame_id,x,y,z rows to a temporary trajectory CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "x", "y", "z"])
        writer.writerows(rows)


class TestEvaluate(unittest.TestCase):
    """Test trajectory and point cloud evaluation helpers."""

    def test_compute_trajectory_length_straight_line(self) -> None:
        """Compute length for a known three-point straight trajectory."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.csv"
            write_trajectory_csv(path, [[0, 0, 0, 0], [1, 1, 0, 0], [2, 2, 0, 0]])
            self.assertAlmostEqual(compute_trajectory_length(path), 2.0)

    def test_compute_drift_ratio_closed_loop(self) -> None:
        """Return zero drift for a trajectory that ends at its start."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.csv"
            write_trajectory_csv(path, [[0, 0, 0, 0], [1, 1, 0, 0], [2, 0, 0, 0]])
            self.assertAlmostEqual(compute_drift_ratio(path), 0.0)

    def test_compare_trajectories_requires_two_aligned_frames(self) -> None:
        """Raise ValueError when fewer than two frame IDs align."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            est = tmp_path / "trajectory.csv"
            gt = tmp_path / "gt_trajectory.csv"
            write_trajectory_csv(est, [[0, 0, 0, 0], [1, 1, 0, 0]])
            write_trajectory_csv(gt, [[0, 0, 0, 0], [2, 2, 0, 0]])
            with self.assertRaises(ValueError):
                compare_trajectories(est, gt)

    def test_point_cloud_stats_count(self) -> None:
        """Return the expected point count for a known PLY."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sparse_map.ply"
            points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            save_ply_xyz(path, points)
            self.assertEqual(point_cloud_stats(path)["point_count"], 4)


if __name__ == "__main__":
    unittest.main()
