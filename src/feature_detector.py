#!/usr/bin/env python3
"""
ORB feature detection for the EyeMap MVP.
ORB is used because it is free, fast on CPU, and available directly in OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np


def create_orb(nfeatures: int = 2000) -> cv2.ORB:
    """Create an ORB detector with the requested feature count."""
    return cv2.ORB_create(nfeatures=int(nfeatures))


def detect_features(frame: np.ndarray, detector: cv2.ORB) -> tuple[np.ndarray, tuple, np.ndarray | None]:
    """Convert a frame to grayscale and detect ORB keypoints/descriptors."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return gray, keypoints, descriptors


def draw_keypoints(frame: np.ndarray, keypoints: tuple) -> np.ndarray:
    """Draw detected keypoints on a copy of the frame."""
    return cv2.drawKeypoints(
        frame,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )
