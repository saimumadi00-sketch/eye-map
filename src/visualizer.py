#!/usr/bin/env python3
"""
OpenCV visualization helpers for live frames, feature overlays, matches, and trajectory plots.
The trajectory plot is intentionally simple so report figures can be generated without extra plotting dependencies.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import ensure_dir


def draw_status(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw status text lines on a frame copy."""
    output = frame.copy()
    for i, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (15, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def draw_trajectory(trajectory_xyz: list[np.ndarray], scale: float = 80.0) -> np.ndarray:
    """Draw an x-z camera trajectory canvas."""
    canvas = np.full((700, 700, 3), 20, dtype=np.uint8)
    origin = np.array([350, 500], dtype=np.int32)

    cv2.putText(
        canvas,
        "Camera trajectory (x-z, relative scale)",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    if len(trajectory_xyz) < 2:
        return canvas

    points_2d = []
    for p in trajectory_xyz:
        points_2d.append(
            np.array([int(origin[0] + p[0] * scale), int(origin[1] - p[2] * scale)], dtype=np.int32)
        )

    for i in range(1, len(points_2d)):
        cv2.line(canvas, tuple(points_2d[i - 1]), tuple(points_2d[i]), (70, 220, 70), 2)
    cv2.circle(canvas, tuple(points_2d[-1]), 5, (40, 120, 255), -1)
    return canvas


def show_window(name: str, image: np.ndarray, enabled: bool = True) -> None:
    """Display an OpenCV window when visualization is enabled."""
    if enabled:
        cv2.imshow(name, image)


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Save an image to disk and return its path."""
    image_path = Path(path)
    ensure_dir(image_path.parent)
    cv2.imwrite(str(image_path), image)
    return image_path
