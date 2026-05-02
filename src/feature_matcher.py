#!/usr/bin/env python3
"""
Feature matching helpers for consecutive video frames.
The module uses Hamming-distance brute force matching with Lowe's ratio test for ORB descriptors.
"""

from __future__ import annotations

import cv2
import numpy as np


def match_features(
    desc_prev: np.ndarray | None,
    desc_curr: np.ndarray | None,
    ratio_test: float = 0.75,
    max_matches: int = 500,
) -> list[cv2.DMatch]:
    """Match two ORB descriptor sets and return filtered matches."""
    if desc_prev is None or desc_curr is None:
        return []
    if len(desc_prev) < 2 or len(desc_curr) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(desc_prev, desc_curr, k=2)
    good_matches = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_test * second.distance:
            good_matches.append(first)

    good_matches.sort(key=lambda m: m.distance)
    return good_matches[: int(max_matches)]


def matched_points(kp_prev: tuple, kp_curr: tuple, matches: list[cv2.DMatch]) -> tuple[np.ndarray, np.ndarray]:
    """Convert matched keypoints into two Nx2 float arrays."""
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
    pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])
    return pts_prev, pts_curr


def draw_matches(
    frame_prev: np.ndarray,
    kp_prev: tuple,
    frame_curr: np.ndarray,
    kp_curr: tuple,
    matches: list[cv2.DMatch],
    max_draw: int = 80,
) -> np.ndarray:
    """Draw a match visualization image for reports and debugging."""
    return cv2.drawMatches(
        frame_prev,
        kp_prev,
        frame_curr,
        kp_curr,
        matches[: int(max_draw)],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
