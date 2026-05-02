#!/usr/bin/env python3
"""
Keyframe saving and metadata management for the EyeMap MVP.
Saved keyframes can be reused later for COLMAP or other offline photogrammetry workflows.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.utils import ensure_dir


class KeyframeManager:
    """Save fixed-interval keyframes and write their pose metadata."""

    # Initialize keyframe directory, interval, and metadata list.
    def __init__(self, keyframe_dir: str | Path, interval: int = 20) -> None:
        """Create a keyframe manager for one output run."""
        self.keyframe_dir = ensure_dir(keyframe_dir)
        self.interval = max(1, int(interval))
        self.metadata: list[dict] = []

    # Decide whether a frame should be saved as a keyframe.
    def should_save(self, frame_id: int) -> bool:
        """Return True when the frame should be saved as a keyframe."""
        return frame_id == 0 or frame_id % self.interval == 0

    # Save a frame image and append pose metadata.
    def save(self, frame_id: int, frame: np.ndarray, T_cw: np.ndarray) -> Path:
        """Save one keyframe image and record its metadata."""
        image_name = f"frame_{frame_id:06d}.png"
        image_path = self.keyframe_dir / image_name
        cv2.imwrite(str(image_path), frame)
        self.metadata.append(
            {
                "frame_id": int(frame_id),
                "image": image_name,
                "T_cw": T_cw.reshape(-1).tolist(),
            }
        )
        return image_path

    # Save keyframe metadata JSON to disk.
    def write_metadata(self, path: str | Path) -> None:
        """Write keyframe metadata as JSON."""
        meta_path = Path(path)
        ensure_dir(meta_path.parent)
        meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
