#!/usr/bin/env python3
"""
Export helpers for report-ready artifacts.
This module writes trajectory CSV files, trajectory plot PNG files, match screenshots, and lightweight run logs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.utils import append_log, ensure_dir
from src.visualizer import draw_trajectory, save_image


def save_trajectory_csv(path: str | Path, rows: list[list[float]]) -> Path:
    """Save trajectory rows as frame_id,x,y,z CSV."""
    csv_path = Path(path)
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "x", "y", "z"])
        writer.writerows(rows)
    return csv_path


def save_trajectory_plot(path: str | Path, trajectory_xyz: list[np.ndarray], scale: float = 80.0) -> Path:
    """Save an OpenCV-rendered trajectory plot PNG."""
    plot = draw_trajectory(trajectory_xyz, scale=scale)
    return save_image(path, plot)


def save_match_screenshot(path: str | Path, image: np.ndarray) -> Path:
    """Save a feature-match visualization image."""
    return save_image(path, image)


def save_log(path: str | Path, message: str) -> None:
    """Append one line to the run log."""
    append_log(path, message)
