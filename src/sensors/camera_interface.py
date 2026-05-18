"""Abstract camera interfaces for EyeMap input sources.
New input sources such as webcams, RealSense devices, and drone cameras implement this contract so the rest of the pipeline can consume frames consistently."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from src.core.utils import make_intrinsics


class CameraInterface(ABC):
    """Define the shared contract for all EyeMap camera sources."""

    @abstractmethod
    def open(self) -> None:
        """Open the camera or data source."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray]:
        """Return `(success, bgr_frame)` like `cv2.VideoCapture.read()`."""

    @abstractmethod
    def get_intrinsics(self) -> np.ndarray:
        """Return the camera intrinsic matrix `K` as a float64 `(3, 3)` array."""

    @abstractmethod
    def release(self) -> None:
        """Release the camera or data source."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Return the frame width in pixels."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Return the frame height in pixels."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Return the capture frame rate in frames per second."""


class OpenCVCamera(CameraInterface):
    """Wrap `cv2.VideoCapture` behind the shared camera interface."""

    def __init__(self, source: int | str, K: np.ndarray | None = None) -> None:
        """Store the OpenCV source and optional calibrated intrinsics."""
        self.source = source
        self._capture: cv2.VideoCapture | None = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._intrinsics = None if K is None else np.asarray(K, dtype=np.float64)
        if self._intrinsics is not None and self._intrinsics.shape != (3, 3):
            raise ValueError("K must have shape (3, 3)")

    def open(self) -> None:
        """Open the OpenCV capture and cache available stream metadata."""
        if self._capture is not None and self._capture.isOpened():
            return
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source}")
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if self._intrinsics is None and self._width > 0 and self._height > 0:
            self._intrinsics = make_intrinsics(self._width, self._height)

    def read(self) -> tuple[bool, np.ndarray]:
        """Read one BGR frame and infer missing metadata from the first frame."""
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("Camera source is not open.")
        success, frame = self._capture.read()
        if success and frame is not None:
            self._height, self._width = frame.shape[:2]
            if self._intrinsics is None:
                self._intrinsics = make_intrinsics(self._width, self._height)
        return success, frame

    def get_intrinsics(self) -> np.ndarray:
        """Return calibrated or frame-size-estimated camera intrinsics."""
        if self._intrinsics is None:
            if self._width <= 0 or self._height <= 0:
                raise RuntimeError("Camera intrinsics are unavailable until a frame size is known.")
            self._intrinsics = make_intrinsics(self._width, self._height)
        return self._intrinsics

    def release(self) -> None:
        """Release the OpenCV capture when it is open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def width(self) -> int:
        """Return the most recently observed frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Return the most recently observed frame height."""
        return self._height

    @property
    def fps(self) -> float:
        """Return the reported capture frame rate."""
        return self._fps
