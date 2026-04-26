#!/usr/bin/env python3
"""
Virtual KITTI 2 dataset loader for EyeMap evaluation runs.
This module reads RGB frames, depth maps, camera intrinsics, and ground-truth world-to-camera poses using the official VKITTI folder layout.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VKITTISequence:
    """Represent one Virtual KITTI 2 scene/variant/camera sequence."""

    # Build paths, frame lists, and metadata tables for one VKITTI sequence.
    def __init__(
        self,
        rgb_dir: str | Path,
        depth_dir: str | Path,
        text_dir: str | Path,
        scene: str,
        variant: str,
        camera: int = 0,
    ) -> None:
        """Initialize paths, metadata tables, and valid frame index for one sequence."""
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)
        self.text_dir = Path(text_dir)
        self.scene = scene
        self.variant = variant
        self.camera = int(camera)

        camera_name = f"Camera_{self.camera}"
        self.rgb_frame_dir = self.rgb_dir / scene / variant / "frames" / "rgb" / camera_name
        self.depth_frame_dir = self.depth_dir / scene / variant / "frames" / "depth" / camera_name
        self.intrinsic_path = self.text_dir / scene / variant / "intrinsic.txt"
        self.extrinsic_path = self.text_dir / scene / variant / "extrinsic.txt"

        self.intrinsics = parse_intrinsics_file(self.intrinsic_path, camera_id=self.camera)
        self.extrinsics = parse_extrinsics_file(self.extrinsic_path, camera_id=self.camera)
        self.frames = self._build_frame_index()

    # Return the number of usable frames with RGB, depth, K, and T_cw available.
    def __len__(self) -> int:
        """Return the number of valid frames in the sequence."""
        return len(self.frames)

    # Load one indexed frame and its calibration/pose metadata.
    def __getitem__(self, i: int) -> dict:
        """Return one frame record with image, depth, intrinsics, and pose."""
        frame = self.frames[i]
        frame_id = frame["frame_id"]
        return {
            "frame_id": frame_id,
            "rgb": load_rgb(frame["rgb_path"]),
            "depth_cm": load_depth_cm(frame["depth_path"]),
            "K": self.intrinsics[frame_id],
            "T_cw": self.extrinsics[frame_id],
        }

    # Build a sorted list of frame records that have all required files and metadata.
    def _build_frame_index(self) -> list[dict]:
        """Build the valid frame index from RGB, depth, intrinsic, and extrinsic availability."""
        if not self.rgb_frame_dir.exists():
            raise FileNotFoundError(f"RGB frame directory not found: {self.rgb_frame_dir}")
        if not self.depth_frame_dir.exists():
            raise FileNotFoundError(f"Depth frame directory not found: {self.depth_frame_dir}")

        rgb_by_frame = {}
        for path in sorted(self.rgb_frame_dir.glob("rgb_*.jpg")):
            frame_id = _frame_id_from_path(path, prefix="rgb")
            rgb_by_frame[frame_id] = path

        depth_by_frame = {}
        for path in sorted(self.depth_frame_dir.glob("depth_*.png")):
            frame_id = _frame_id_from_path(path, prefix="depth")
            depth_by_frame[frame_id] = path

        valid_frames = []
        for frame_id in sorted(rgb_by_frame):
            if frame_id not in depth_by_frame:
                print(f"[WARN] Skipping frame {frame_id}: missing depth PNG.")
                continue
            if frame_id not in self.intrinsics:
                print(f"[WARN] Skipping frame {frame_id}: missing intrinsics.")
                continue
            if frame_id not in self.extrinsics:
                print(f"[WARN] Skipping frame {frame_id}: missing extrinsics.")
                continue
            valid_frames.append(
                {
                    "frame_id": frame_id,
                    "rgb_path": rgb_by_frame[frame_id],
                    "depth_path": depth_by_frame[frame_id],
                }
            )

        for frame_id in sorted(set(depth_by_frame) - set(rgb_by_frame)):
            print(f"[WARN] Skipping frame {frame_id}: missing RGB JPG.")

        if not valid_frames:
            raise ValueError(
                "No valid VKITTI frames found after matching RGB, depth, intrinsics, and extrinsics."
            )
        return valid_frames


# Load one RGB image as OpenCV BGR.
def load_rgb(path: str | Path) -> np.ndarray:
    """Load a VKITTI RGB frame as a BGR numpy array."""
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load RGB image: {image_path}")
    return image


# Load one 16-bit depth image in centimeters.
def load_depth_cm(path: str | Path) -> np.ndarray:
    """Load a VKITTI depth frame as a uint16 centimeter map."""
    depth_path = Path(path)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Could not load depth image: {depth_path}")
    if depth.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth image, got {depth.dtype}: {depth_path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


# Convert depth from centimeters to meters.
def depth_cm_to_meters(depth_cm: np.ndarray) -> np.ndarray:
    """Convert a uint16 centimeter depth map to float32 meters."""
    return depth_cm.astype(np.float32) / 100.0


# Parse VKITTI intrinsic.txt into frame_id -> K.
def parse_intrinsics_file(path: str | Path, camera_id: int = 0) -> dict[int, np.ndarray]:
    """Parse VKITTI camera intrinsics for one camera into 3x3 matrices."""
    intrinsic_path = Path(path)
    if not intrinsic_path.exists():
        raise FileNotFoundError(f"Intrinsic file not found: {intrinsic_path}")

    intrinsics = {}
    for line_no, raw_line in enumerate(intrinsic_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or _is_header_line(line):
            continue
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Invalid intrinsic row at line {line_no}: {raw_line}")
        try:
            frame_id = int(parts[0])
            row_camera_id = int(parts[1])
            fx, fy, cx, cy = [float(value) for value in parts[2:6]]
        except ValueError as exc:
            raise ValueError(f"Invalid intrinsic row at line {line_no}: {raw_line}") from exc

        if row_camera_id != int(camera_id):
            continue

        intrinsics[frame_id] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    if not intrinsics:
        raise ValueError(f"No intrinsics found for camera {camera_id}: {intrinsic_path}")
    return intrinsics


# Parse VKITTI extrinsic.txt into frame_id -> T_cw.
def parse_extrinsics_file(path: str | Path, camera_id: int = 0) -> dict[int, np.ndarray]:
    """Parse VKITTI world-to-camera extrinsics for one camera into 4x4 matrices."""
    extrinsic_path = Path(path)
    if not extrinsic_path.exists():
        raise FileNotFoundError(f"Extrinsic file not found: {extrinsic_path}")

    extrinsics = {}
    for line_no, raw_line in enumerate(extrinsic_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or _is_header_line(line):
            continue
        parts = line.split()
        if len(parts) < 18:
            raise ValueError(f"Invalid extrinsic row at line {line_no}: {raw_line}")
        try:
            frame_id = int(parts[0])
            row_camera_id = int(parts[1])
            matrix_values = [float(value) for value in parts[2:18]]
        except ValueError as exc:
            raise ValueError(f"Invalid extrinsic row at line {line_no}: {raw_line}") from exc

        if row_camera_id != int(camera_id):
            continue

        extrinsics[frame_id] = np.asarray(matrix_values, dtype=np.float64).reshape(4, 4)

    if not extrinsics:
        raise ValueError(f"No extrinsics found for camera {camera_id}: {extrinsic_path}")
    return extrinsics


# Extract numeric frame id from VKITTI rgb_XXXXX/depth_XXXXX filenames.
def _frame_id_from_path(path: Path, prefix: str) -> int:
    """Extract the integer frame id from a VKITTI frame filename."""
    expected_prefix = f"{prefix}_"
    stem = path.stem
    if not stem.startswith(expected_prefix):
        raise ValueError(f"Unexpected VKITTI filename: {path}")
    try:
        return int(stem[len(expected_prefix) :])
    except ValueError as exc:
        raise ValueError(f"Unexpected VKITTI frame id in filename: {path}") from exc


# Detect text-file header rows such as "frame cameraID ...".
def _is_header_line(line: str) -> bool:
    """Return True when a metadata row looks like a header."""
    first = line.split()[0].lower()
    return first in {"frame", "frameid", "#"}
