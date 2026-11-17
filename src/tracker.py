#!/usr/bin/env python3
"""
Reference-frame monocular tracking with relative scale propagation.
Each frame is matched against the last frame that produced a pose, so slow motion accumulates parallax instead of being discarded pair by pair.
Translation scale is carried from one pair to the next through points triangulated in both, so step lengths stay consistent up to one global scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.feature_matcher import match_features
from src.pose_estimator import estimate_relative_pose_from_points, update_pose
from src.triangulation import transform_points, triangulate_relative, valid_triangulation_mask

NOT_ENOUGH_MATCHES = "not enough matches"


@dataclass
class TrackResult:
    """Outcome of tracking one frame against the current reference frame."""

    success: bool
    reason: str
    T_cw: np.ndarray
    matches: list[cv2.DMatch]
    ref_keypoints: tuple
    ref_image: np.ndarray | None
    points_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    scale: float = 0.0


class MonocularTracker:
    """Track camera motion against a reference frame and keep a consistent relative scale."""

    def __init__(
        self,
        ratio_test: float = 0.75,
        max_matches: int = 500,
        min_matches: int = 60,
        ransac_thresh: float = 1.0,
        max_point_norm: float = 150.0,
        min_scale_points: int = 8,
    ) -> None:
        """Store matching, pose, and triangulation thresholds."""
        self.ratio_test = float(ratio_test)
        self.max_matches = int(max_matches)
        self.min_matches = int(min_matches)
        self.ransac_thresh = float(ransac_thresh)
        self.max_point_norm = float(max_point_norm)
        self.min_scale_points = int(min_scale_points)

        self.T_cw = np.eye(4, dtype=np.float64)
        self._step_scale = 1.0
        self._ref_keypoints: tuple = ()
        self._ref_descriptors: np.ndarray | None = None
        self._ref_image: np.ndarray | None = None
        self._ref_T_cw = self.T_cw.copy()
        # Reference keypoint index -> 3D point in the reference camera frame, at the running scale.
        self._ref_points: dict[int, np.ndarray] = {}

    def initialize(self, keypoints: tuple, descriptors: np.ndarray | None, image: np.ndarray | None = None) -> None:
        """Use the first frame as the reference at the world origin."""
        self._set_reference(keypoints, descriptors, image, {})

    def track(
        self,
        keypoints: tuple,
        descriptors: np.ndarray | None,
        K: np.ndarray,
        image: np.ndarray | None = None,
    ) -> TrackResult:
        """Estimate the current pose from matches against the reference frame."""
        ref_keypoints, ref_image = self._ref_keypoints, self._ref_image
        matches = match_features(self._ref_descriptors, descriptors, self.ratio_test, self.max_matches)

        if len(matches) < self.min_matches:
            # Tracking lost: restart from this frame; the scale link to old points is gone.
            self._set_reference(keypoints, descriptors, image, {})
            return TrackResult(False, NOT_ENOUGH_MATCHES, self.T_cw.copy(), matches, ref_keypoints, ref_image)

        ref_idx = np.array([m.queryIdx for m in matches], dtype=np.int64)
        curr_idx = np.array([m.trainIdx for m in matches], dtype=np.int64)
        pts_ref = np.float32([ref_keypoints[i].pt for i in ref_idx])
        pts_curr = np.float32([keypoints[i].pt for i in curr_idx])

        pose = estimate_relative_pose_from_points(pts_ref, pts_curr, K, ransac_thresh=self.ransac_thresh)
        if not pose["success"]:
            # The reference still matches well, so keep it: small baselines fail here
            # (low parallax, or recoverPose rejecting distant points) until motion accumulates.
            return TrackResult(False, pose["reason"], self.T_cw.copy(), matches, ref_keypoints, ref_image)

        R, t = pose["R"], pose["t"]
        inliers = pose["inlier_mask"]
        points = np.zeros((0, 3), dtype=np.float64)
        point_ref_idx = np.zeros(0, dtype=np.int64)
        point_curr_idx = np.zeros(0, dtype=np.int64)
        if np.count_nonzero(inliers) >= 10:
            points = triangulate_relative(K, R, t, pts_ref[inliers], pts_curr[inliers])
            valid = valid_triangulation_mask(points, R, t, self.max_point_norm)
            points = points[valid]
            point_ref_idx = ref_idx[inliers][valid]
            point_curr_idx = curr_idx[inliers][valid]

        # Triangulation is linear in the baseline, so one factor rescales both translation and points.
        scale = self._relative_scale(points, point_ref_idx)
        t = t * scale
        points = points * scale

        self.T_cw = update_pose(self._ref_T_cw, R, t)
        points_world = transform_points(np.linalg.inv(self._ref_T_cw), points)
        points_curr = (R @ points.T + t.reshape(3, 1)).T
        self._step_scale = scale
        self._set_reference(keypoints, descriptors, image, dict(zip(point_curr_idx.tolist(), points_curr)))
        return TrackResult(True, "ok", self.T_cw.copy(), matches, ref_keypoints, ref_image, points_world, scale)

    def _relative_scale(self, points: np.ndarray, point_ref_idx: np.ndarray) -> float:
        """Scale a unit-baseline pair so shared points keep the distance the previous pair gave them."""
        ratios = []
        for i, point in zip(point_ref_idx.tolist(), points):
            norm = np.linalg.norm(point)
            if i in self._ref_points and norm > 1e-12:
                ratios.append(np.linalg.norm(self._ref_points[i]) / norm)
        if len(ratios) >= self.min_scale_points:
            scale = float(np.median(ratios))
            if np.isfinite(scale) and scale > 0.0:
                return scale
        # No usable overlap: assume the step length is unchanged.
        return self._step_scale

    def _set_reference(
        self,
        keypoints: tuple,
        descriptors: np.ndarray | None,
        image: np.ndarray | None,
        points: dict[int, np.ndarray],
    ) -> None:
        """Make a frame the reference for the next match, anchored at the current pose."""
        self._ref_keypoints = keypoints
        self._ref_descriptors = descriptors
        self._ref_image = image
        self._ref_T_cw = self.T_cw.copy()
        self._ref_points = points
