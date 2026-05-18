# ============================================================
# CONTRIBUTOR STUB
# This file defines the interface for Intel RealSense D435i.
# To implement:
#   1. pip install pyrealsense2
#   2. Replace each NotImplementedError with real implementation
#   3. See docs/drone_deployment.md for wiring and setup
#   4. See docs/architecture.md for how this fits the pipeline
# ============================================================
"""Intel RealSense D435i camera interface for EyeMap.
CONTRIBUTOR STUB — implement this to enable live metric-scale mapping."""

from __future__ import annotations

import numpy as np

from src.sensors.camera_interface import CameraInterface

GUIDANCE_MESSAGE = (
    "RealSenseCapture.open() is not implemented. "
    "Install pyrealsense2 and implement this method to enable "
    "live depth-aided mapping. See docs/drone_deployment.md."
)


class RealSenseCapture(CameraInterface):
    """Define the RealSense capture contract for future contributors."""

    def __init__(self, width: int = 848, height: int = 480, fps: float = 30) -> None:
        """Store RealSense stream configuration without opening hardware."""
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)

    def open(self) -> None:
        """Open the RealSense stream once implemented by a contributor."""
        raise NotImplementedError(GUIDANCE_MESSAGE)

    def read(self) -> tuple[bool, np.ndarray]:
        """Return one BGR frame once the RealSense adapter is implemented."""
        raise NotImplementedError(GUIDANCE_MESSAGE)

    def get_intrinsics(self) -> np.ndarray:
        """Return calibrated RealSense color-camera intrinsics once implemented."""
        raise NotImplementedError(GUIDANCE_MESSAGE)

    def get_depth_frame(self) -> np.ndarray:
        """Return metric depth in metres as a float32 `(H, W)` array."""
        raise NotImplementedError(GUIDANCE_MESSAGE)

    def release(self) -> None:
        """Release RealSense resources when an implementation adds them."""
        pass

    @property
    def width(self) -> int:
        """Return the configured RealSense frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Return the configured RealSense frame height."""
        return self._height

    @property
    def fps(self) -> float:
        """Return the configured RealSense frame rate."""
        return self._fps
