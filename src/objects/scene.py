#!/usr/bin/env python3
"""
Scene analysis for a tracking run: where objects are relative to the camera, how they move, and the overall layout.
`SceneAnalyzer` combines an optional detector, the object tracker, sparse-map depth, map clustering, and the exports.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.objects.clusters import MapCluster, cluster_map_points
from src.objects.detector import OnnxDetector
from src.objects.directions import (
    Motion,
    ObjectTracker,
    camera_to_world,
    direction_label,
    elevation_label,
    locate_in_box,
    pixel_bearing,
    point_bearing,
    world_to_camera,
)
from src.objects.layout import OBJECT_COLORS, draw_layout


@dataclass
class ObjectObservation:
    """One detected object in one frame, with its direction from the camera."""

    frame_id: int
    track_id: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    azimuth_deg: float  # degrees right of straight ahead
    elevation_deg: float  # degrees above straight ahead
    direction: str
    elevation: str
    distance: float | None  # map units; None when no map points fall on the object
    position_world: np.ndarray | None
    support_points: int
    motion: Motion


def _rounded(values, digits: int = 4) -> list[float] | None:
    return None if values is None else [round(float(v), digits) for v in values]


class SceneAnalyzer:
    """Collect object directions, motion, map clusters and the layout for one run."""

    def __init__(
        self,
        detector: OnnxDetector | None = None,
        detect_interval: int = 1,
        recent_chunks: int = 40,
        min_hits: int = 2,
        tracker: ObjectTracker | None = None,
    ) -> None:
        """Configure the analyzer.

        detector: None disables detection; clusters and the layout still work.
        detect_interval: run the detector every Nth frame to save time.
        recent_chunks: how many recent triangulation batches supply depth for objects.
        min_hits: detections needed before a track appears in the summaries.
        """
        self.detector = detector
        self.detect_interval = max(1, int(detect_interval))
        self.recent_chunks = max(1, int(recent_chunks))
        self.min_hits = max(1, int(min_hits))
        self.tracker = tracker or ObjectTracker()
        self.observations: list[ObjectObservation] = []
        self.latest: list[ObjectObservation] = []

    def process(self, frame_id: int, index: int, image: np.ndarray, K: np.ndarray, T_cw: np.ndarray, map_chunks: list[np.ndarray]) -> list[ObjectObservation]:
        """Detect and locate objects in one frame; returns this frame's observations.

        On frames skipped by `detect_interval` the previous observations are returned unchanged.
        """
        if self.detector is None:
            return []
        if index % self.detect_interval != 0:
            return self.latest
        detections = self.detector.detect(image)

        points_cam = np.zeros((0, 3), dtype=np.float64)
        if detections and map_chunks:
            points_cam = world_to_camera(np.vstack(map_chunks[-self.recent_chunks :]), T_cw)
        located = [locate_in_box(d.box, points_cam, K) for d in detections]
        world_positions = [None if p is None else camera_to_world(p, T_cw) for p, _ in located]
        tracks = self.tracker.update(frame_id, detections, world_positions)

        self.latest = []
        for detection, (position_cam, support), world, track in zip(detections, located, world_positions, tracks):
            azimuth, elevation = pixel_bearing(*detection.center, K)
            self.latest.append(
                ObjectObservation(
                    frame_id=int(frame_id),
                    track_id=track.track_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    box=detection.box,
                    azimuth_deg=azimuth,
                    elevation_deg=elevation,
                    direction=direction_label(azimuth),
                    elevation=elevation_label(elevation),
                    distance=None if position_cam is None else float(np.linalg.norm(position_cam)),
                    position_world=world,
                    support_points=support,
                    motion=self.tracker.motion(track, T_cw),
                )
            )
        self.observations.extend(self.latest)
        return self.latest

    @staticmethod
    def annotate(image: np.ndarray, observations: list[ObjectObservation]) -> np.ndarray:
        """Draw boxes with label, direction, distance and motion on a copy of the frame."""
        output = image.copy()
        for obs in observations:
            color = OBJECT_COLORS[obs.track_id % len(OBJECT_COLORS)]
            x1, y1, x2, y2 = (int(round(v)) for v in obs.box)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{obs.label} #{obs.track_id} {obs.direction} {obs.azimuth_deg:+.0f}deg"
            if obs.distance is not None:
                text += f" d={obs.distance:.1f}"
            motion = obs.motion.world or obs.motion.apparent
            if motion not in ("unknown", "still", "stationary"):
                text += f" moving {motion}"
            # Keep the label inside the frame and readable on busy backgrounds.
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            left = max(0, min(x1, output.shape[1] - text_w - 4))
            top = max(text_h + 6, y1)
            cv2.rectangle(output, (left, top - text_h - 6), (left + text_w + 4, top), color, -1)
            cv2.putText(output, text, (left + 2, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return output

    def layout_objects(self, T_cw: np.ndarray | None = None) -> list[dict]:
        """Return located tracks in the form `draw_layout` expects.

        A motion arrow ("velocity") is included only for tracks judged to be moving in the
        world, which needs the current pose `T_cw`.
        """
        objects = []
        for track in self.tracker.tracks.values():
            if track.hits < self.min_hits:
                continue
            located = [p for *_, p in track.history if p is not None]
            if not located:
                continue
            motion = self.tracker.motion(track, T_cw)
            moving = motion.world not in (None, "stationary")
            objects.append(
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "position": np.median(np.asarray(located[-self.tracker.window :]), axis=0),
                    "velocity": motion.world_velocity if moving else None,
                }
            )
        return objects

    def object_summaries(self, T_cw: np.ndarray, scale_m: float | None = None) -> list[dict]:
        """Summarize each track: last-seen direction, plus direction from the pose `T_cw` when located."""
        last_seen = {obs.track_id: obs for obs in self.observations}
        positions = {obj["track_id"]: obj["position"] for obj in self.layout_objects()}
        summaries = []
        for track in self.tracker.tracks.values():
            if track.hits < self.min_hits or track.track_id not in last_seen:
                continue
            obs = last_seen[track.track_id]
            motion = self.tracker.motion(track, T_cw)
            summary = {
                "track_id": track.track_id,
                "label": track.label,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "detections": track.hits,
                "last_seen": {
                    "azimuth_deg": round(obs.azimuth_deg, 2),
                    "elevation_deg": round(obs.elevation_deg, 2),
                    "direction": obs.direction,
                    "elevation": obs.elevation,
                    "distance": None if obs.distance is None else round(obs.distance, 4),
                },
                "position_world": _rounded(positions.get(track.track_id)),
                "from_final_camera": None,
                "motion": {
                    "apparent": motion.apparent,
                    "apparent_px_per_frame": _rounded(motion.apparent_px_per_frame, 2),
                    "range_trend": motion.range_trend,
                    "world": motion.world,
                    "world_velocity_per_frame": _rounded(motion.world_velocity),
                },
            }
            if track.track_id in positions:
                azimuth, elevation, distance = point_bearing(world_to_camera(positions[track.track_id], T_cw)[0])
                summary["from_final_camera"] = {
                    "azimuth_deg": round(azimuth, 2),
                    "elevation_deg": round(elevation, 2),
                    "direction": direction_label(azimuth),
                    "distance": round(distance, 4),
                }
                if scale_m is not None:
                    summary["from_final_camera"]["distance_m"] = round(distance * scale_m, 3)
            summaries.append(summary)
        return summaries

    @staticmethod
    def cluster_summaries(clusters: list[MapCluster], T_cw: np.ndarray, scale_m: float | None = None) -> list[dict]:
        """Describe each map cluster's direction and distance from the pose `T_cw`."""
        summaries = []
        for cluster in clusters:
            azimuth, elevation, distance = point_bearing(world_to_camera(cluster.centroid, T_cw)[0])
            summary = {
                "cluster_id": cluster.cluster_id,
                "points": cluster.count,
                "centroid_world": _rounded(cluster.centroid),
                "size_world": _rounded(cluster.maximum - cluster.minimum),
                "azimuth_deg": round(azimuth, 2),
                "elevation_deg": round(elevation, 2),
                "direction": direction_label(azimuth),
                "elevation": elevation_label(elevation),
                "distance": round(distance, 4),
            }
            if scale_m is not None:
                summary["distance_m"] = round(distance * scale_m, 3)
            summaries.append(summary)
        return summaries

    def export(
        self,
        trajectory: list[np.ndarray],
        map_points: np.ndarray,
        T_cw: np.ndarray,
        objects_dir: str | Path,
        layout_path: str | Path,
        scale_m: float | None = None,
    ) -> dict[str, Path]:
        """Write layout.png, scene.json, clusters.csv and (with a detector) objects.csv.

        Directions in scene.json are measured from the final camera pose `T_cw`. Distances are
        in map units; `scale_m` (metres per unit, from GPS) adds metric distances.
        """
        out_dir = Path(objects_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clusters = cluster_map_points(map_points)
        cluster_rows = self.cluster_summaries(clusters, T_cw, scale_m)
        object_rows = self.object_summaries(T_cw, scale_m)
        paths: dict[str, Path] = {}

        layout = draw_layout(
            trajectory, map_points, T_cw, clusters, self.layout_objects(T_cw),
            unit="m" if scale_m is not None else "units", unit_scale=scale_m or 1.0,
        )
        paths["layout"] = Path(layout_path)
        paths["layout"].parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(paths["layout"]), layout)

        paths["scene"] = out_dir / "scene.json"
        scene = {
            "frame": "world = first camera frame (x right, y down, z forward); azimuth is degrees right of the camera's forward axis",
            "distance_unit": "map units (relative scale)" if scale_m is None else "map units; distance_m in metres",
            "scale_m_per_unit": scale_m,
            "detector_enabled": self.detector is not None,
            "objects": object_rows,
            "clusters": cluster_rows,
        }
        paths["scene"].write_text(json.dumps(scene, indent=2), encoding="utf-8")

        paths["clusters"] = out_dir / "clusters.csv"
        with paths["clusters"].open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster_id", "points", "x", "y", "z", "azimuth_deg", "elevation_deg", "direction", "distance"])
            for row in cluster_rows:
                writer.writerow([
                    row["cluster_id"], row["points"], *row["centroid_world"],
                    row["azimuth_deg"], row["elevation_deg"], row["direction"], row["distance"],
                ])

        if self.detector is not None:
            paths["objects"] = out_dir / "objects.csv"
            with paths["objects"].open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_id", "track_id", "label", "confidence", "x1", "y1", "x2", "y2", "azimuth_deg",
                    "elevation_deg", "direction", "elevation", "distance", "support_points",
                    "apparent_motion", "range_trend", "world_motion",
                ])
                for obs in self.observations:
                    writer.writerow([
                        obs.frame_id, obs.track_id, obs.label, f"{obs.confidence:.3f}",
                        *(f"{v:.1f}" for v in obs.box), f"{obs.azimuth_deg:.2f}", f"{obs.elevation_deg:.2f}",
                        obs.direction, obs.elevation, "" if obs.distance is None else f"{obs.distance:.4f}",
                        obs.support_points, obs.motion.apparent, obs.motion.range_trend, obs.motion.world or "",
                    ])
        return paths
