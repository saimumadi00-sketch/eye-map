#!/usr/bin/env python3
"""
Video input helpers for webcam and recorded-file processing.
The module wraps OpenCV capture setup so the main pipeline can read frames consistently from either source.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def open_capture(source: str, path: str | Path | None = None, camera_index: int = 0) -> cv2.VideoCapture:
    """Open a webcam or video file and return an OpenCV capture object."""
    if source == "webcam":
        cap = cv2.VideoCapture(int(camera_index))
    elif source == "video":
        if path is None:
            raise ValueError("Video source requires --path.")
        video_path = Path(path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
    else:
        raise ValueError(f"Unsupported source: {source}")

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source} input.")
    return cap


def read_frame(cap: cv2.VideoCapture, resize_width: int = 0) -> tuple[bool, np.ndarray | None]:
    """Read one frame and optionally resize it to a target width."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None
    if resize_width and resize_width > 0:
        frame = resize_frame(frame, resize_width)
    return True, frame


def resize_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    """Resize a frame while preserving aspect ratio."""
    height, width = frame.shape[:2]
    if width == resize_width:
        return frame
    scale = resize_width / float(width)
    new_size = (resize_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def release_capture(cap: cv2.VideoCapture) -> None:
    """Release an OpenCV capture object."""
    cap.release()
