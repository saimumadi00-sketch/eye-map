#!/usr/bin/env python3
"""
Directions from the camera to objects, and how tracked objects move.
Angles use the camera's own axes (OpenCV: x right, y down, z forward): azimuth is degrees to the right of straight ahead, elevation is degrees above it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.objects.detector import Detection



def pixel_bearing(u: float, v: float, K: np.ndarray) -> tuple[float, float]:
    """Return (azimuth_deg, elevation_deg) of the ray through pixel (u, v)."""
    x = (float(u) - K[0, 2]) / K[0, 0]
    y = (float(v) - K[1, 2]) / K[1, 1]
    return math.degrees(math.atan2(x, 1.0)), math.degrees(math.atan2(-y, math.hypot(x, 1.0)))


def point_bearing(point_cam: np.ndarray) -> tuple[float, float, float]:
    """Return (azimuth_deg, elevation_deg, distance) of a 3D point given in the camera frame.

    Azimuth covers the full circle (-180, 180], so points behind the camera are handled.
    """
    x, y, z = (float(c) for c in point_cam)
    distance = math.sqrt(x * x + y * y + z * z)
    if distance <= 0.0:
        return 0.0, 0.0, 0.0
    return math.degrees(math.atan2(x, z)), math.degrees(math.asin(max(-1.0, min(1.0, -y / distance)))), distance


def direction_label(azimuth_deg: float) -> str:
    """Name a direction relative to the camera's forward axis.

    Within 5 degrees is "ahead" and up to 22.5 degrees "slightly left/right", so objects inside
    a normal field of view get distinct names; beyond that, 45-degree sectors
    ("ahead-right", "right", "behind-right", "behind", ...).
    """
    azimuth = (float(azimuth_deg) + 180.0) % 360.0 - 180.0
    side = "right" if azimuth > 0 else "left"
    magnitude = abs(azimuth)
    if magnitude <= 5.0:
        return "ahead"
    if magnitude <= 22.5:
        return f"slightly {side}"
    if magnitude <= 67.5:
        return f"ahead-{side}"
    if magnitude <= 112.5:
        return side
    if magnitude <= 157.5:
        return f"behind-{side}"
    return "behind"


def elevation_label(elevation_deg: float, level_band_deg: float = 10.0) -> str:
    """Return "above", "level" or "below" relative to the optical axis."""
    if elevation_deg > level_band_deg:
        return "above"
    if elevation_deg < -level_band_deg:
        return "below"
    return "level"


def world_to_camera(points_world: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
    """Transform Nx3 world points into the camera frame of pose T_cw."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    return points @ T_cw[:3, :3].T + T_cw[:3, 3]


def camera_to_world(point_cam: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
    """Transform one camera-frame point into the world frame."""
    return T_cw[:3, :3].T @ (np.asarray(point_cam, dtype=np.float64) - T_cw[:3, 3])


def locate_in_box(
    box: tuple[float, float, float, float],
    points_cam: np.ndarray,
    K: np.ndarray,
    min_points: int = 3,
    shrink: float = 0.8,
) -> tuple[np.ndarray | None, int]:
    """Estimate an object's 3D position (camera frame) from sparse map points that project into its box.

    The box is shrunk around its center so background seen at the edges counts less, and the
    nearest quartile of depths is used because a box usually contains the object in front of
    farther background. Returns (position or None, number of supporting points).
    """
    if len(points_cam) == 0:
        return None, 0
    x1, y1, x2, y2 = box
    cu, cv_ = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = shrink * (x2 - x1) / 2.0, shrink * (y2 - y1) / 2.0
    front = points_cam[points_cam[:, 2] > 1e-9]
    if len(front) == 0:
        return None, 0
    u = K[0, 0] * front[:, 0] / front[:, 2] + K[0, 2]
    v = K[1, 1] * front[:, 1] / front[:, 2] + K[1, 2]
    inside = (np.abs(u - cu) <= half_w) & (np.abs(v - cv_) <= half_h)
    count = int(np.count_nonzero(inside))
    if count < min_points:
        return None, count
    depth = float(np.percentile(front[inside, 2], 25))
    ray = np.array([(cu - K[0, 2]) / K[0, 0], (cv_ - K[1, 2]) / K[1, 1], 1.0])
    return ray * depth, count


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Return intersection-over-union of two x1,y1,x2,y2 boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class Motion:
    """How a tracked object is moving.

    `apparent` is motion across the image and includes the camera's own motion: a parked car
    drifts left as the camera pans right. `world` uses map positions, so it is independent of
    camera motion, but is only available when the object has depth from the sparse map.
    """

    apparent: str = "unknown"  # left, right, up, down, still, unknown
    apparent_px_per_frame: tuple[float, float] = (0.0, 0.0)
    range_trend: str = "unknown"  # approaching, receding, steady, unknown (from box size)
    world: str | None = None  # toward, away, left, right, stationary (relative to the camera's view)
    world_velocity: tuple[float, float, float] | None = None  # map units per frame


@dataclass
class Track:
    """One object followed across frames."""

    track_id: int
    label: str
    box: tuple[float, float, float, float]
    first_frame: int
    last_frame: int
    confidence: float
    hits: int = 1
    # (frame_id, center_u, center_v, box_area, world position or None)
    history: list[tuple[int, float, float, float, np.ndarray | None]] = field(default_factory=list)


class ObjectTracker:
    """Give detections stable ids by greedy same-class IoU matching, and describe their motion."""

    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 15, window: int = 8) -> None:
        """Configure matching. A track unseen for more than `max_missed_frames` frames is retired."""
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.window = max(2, int(window))
        self.tracks: dict[int, Track] = {}  # every track ever created, by id
        self._active: list[int] = []
        self._next_id = 1

    def update(
        self,
        frame_id: int,
        detections: list[Detection],
        world_positions: list[np.ndarray | None] | None = None,
    ) -> list[Track]:
        """Match one frame's detections to tracks; return the track for each detection, in order."""
        if world_positions is None:
            world_positions = [None] * len(detections)
        self._active = [i for i in self._active if frame_id - self.tracks[i].last_frame <= self.max_missed_frames]

        pairs = []
        for d_index, detection in enumerate(detections):
            for track_id in self._active:
                track = self.tracks[track_id]
                if track.label != detection.label:
                    continue
                iou = box_iou(track.box, detection.box)
                if iou >= self.iou_threshold:
                    pairs.append((iou, d_index, track_id))
        pairs.sort(reverse=True)

        assigned: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, d_index, track_id in pairs:
            if d_index not in assigned and track_id not in used_tracks:
                assigned[d_index] = track_id
                used_tracks.add(track_id)

        result = []
        for d_index, detection in enumerate(detections):
            track_id = assigned.get(d_index)
            if track_id is None:
                track = Track(self._next_id, detection.label, detection.box, frame_id, frame_id, detection.confidence, hits=0)
                self.tracks[track.track_id] = track
                self._active.append(track.track_id)
                self._next_id += 1
            else:
                track = self.tracks[track_id]
            track.box = detection.box
            track.last_frame = frame_id
            track.confidence = detection.confidence
            track.hits += 1
            u, v = detection.center
            track.history.append((frame_id, u, v, detection.area, world_positions[d_index]))
            result.append(track)
        return result

    def motion(self, track: Track, T_cw: np.ndarray | None = None) -> Motion:
        """Describe a track's recent motion over the last `window` observations."""
        recent = track.history[-self.window :]
        if len(recent) < 2:
            return Motion()
        frames = recent[-1][0] - recent[0][0]
        if frames <= 0:
            return Motion()

        du = (recent[-1][1] - recent[0][1]) / frames
        dv = (recent[-1][2] - recent[0][2]) / frames
        box_size = math.sqrt(max(recent[-1][3], 1.0))
        still_px = max(0.5, 0.01 * box_size)  # per frame; scales with how large the object appears
        if math.hypot(du, dv) < still_px:
            apparent = "still"
        elif abs(du) >= abs(dv):
            apparent = "right" if du > 0 else "left"
        else:
            apparent = "down" if dv > 0 else "up"

        growth = recent[-1][3] / max(recent[0][3], 1.0)
        range_trend = "approaching" if growth > 1.15 else "receding" if growth < 1.0 / 1.15 else "steady"

        world = None
        world_velocity = None
        located = [(f, p) for f, _, _, _, p in recent if p is not None]
        if len(located) >= 2 and located[-1][0] > located[0][0]:
            velocity = (located[-1][1] - located[0][1]) / (located[-1][0] - located[0][0])
            world_velocity = tuple(float(c) for c in velocity)
            if T_cw is not None:
                distance = float(np.linalg.norm(world_to_camera(located[-1][1], T_cw)[0]))
                v_cam = T_cw[:3, :3] @ velocity
                # Sparse depth is noisy, so demand 1% of the range per frame before calling it motion.
                if float(np.linalg.norm(velocity)) < 0.01 * max(distance, 1e-9):
                    world = "stationary"
                elif abs(v_cam[0]) >= abs(v_cam[2]):
                    world = "right" if v_cam[0] > 0 else "left"
                else:
                    world = "away" if v_cam[2] > 0 else "toward"
        return Motion(apparent, (float(du), float(dv)), range_trend, world, world_velocity)
