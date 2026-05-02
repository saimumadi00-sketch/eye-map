#!/usr/bin/env python3
"""
Functional tests for output artifact generation.
Temporary directories are used to verify CSV, plot, and match-image exports without touching real output runs.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.exporter import save_trajectory_csv, save_trajectory_plot


class TestExporter(unittest.TestCase):
    """Test report artifact exporters."""

    def test_save_trajectory_outputs(self) -> None:
        """Write trajectory CSV and PNG files."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trajectory_rows = [[0, 0.0, 0.0, 0.0], [1, 1.0, 0.0, 0.5]]
            trajectory = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.5])]

            csv_path = save_trajectory_csv(tmp_path / "trajectory.csv", trajectory_rows)
            plot_path = save_trajectory_plot(tmp_path / "trajectory.png", trajectory)

            self.assertTrue(csv_path.exists())
            self.assertTrue(plot_path.exists())
            self.assertGreater(csv_path.stat().st_size, 0)
            self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
