#!/usr/bin/env python3
"""
Shared utility helpers for the EyeMap terrain mapping project.
The functions here keep repeated YAML, timestamp, and simple ASCII PLY logic in one place so MVP scripts stay focused on computer vision steps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


# Create a compact timestamp identifier for output folders.
def timestamp_id() -> str:
    stamp = str(np.datetime64("now", "s"))
    return stamp.replace("-", "").replace(":", "").replace("T", "_")


# Load a YAML configuration file as a Python dictionary.
def load_yaml_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


# Load xyz points from a simple ASCII PLY file.
def load_ply_xyz(path: str | Path) -> np.ndarray:
    ply_path = Path(path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    lines = ply_path.read_text(encoding="utf-8").splitlines()
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            end_idx = i
            break

    if end_idx is None:
        raise ValueError(f"Invalid PLY file, missing end_header: {ply_path}")

    points = []
    for line in lines[end_idx + 1 :]:
        chunks = line.strip().split()
        if len(chunks) < 3:
            continue
        points.append([float(chunks[0]), float(chunks[1]), float(chunks[2])])

    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


# Save xyz points to a simple ASCII PLY file.
def save_ply_xyz(path: str | Path, points_xyz: np.ndarray) -> None:
    ply_path = Path(path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]

    with ply_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
