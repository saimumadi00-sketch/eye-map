#!/usr/bin/env python3
"""
Top-down layout of a run: sparse map, camera path and heading, map clusters, and located objects.
The view looks down the world y axis: world x runs right and world z (the first camera's forward) runs up.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.objects.clusters import MapCluster

BACKGROUND = (24, 24, 24)
POINT_COLOR = (110, 110, 110)
PATH_COLOR = (70, 220, 70)
CAMERA_COLOR = (40, 120, 255)
CLUSTER_COLOR = (200, 170, 60)
OBJECT_COLORS = ((80, 80, 255), (255, 160, 60), (90, 220, 220), (220, 90, 220), (120, 255, 120), (255, 255, 120))


def _nice_length(target: float) -> float:
    """Round a length down to 1, 2 or 5 times a power of ten."""
    if not target > 0.0 or not math.isfinite(target):
        return 1.0
    power = 10.0 ** math.floor(math.log10(target))
    for factor in (5.0, 2.0, 1.0):
        if factor * power <= target:
            return factor * power
    return power


def draw_layout(
    trajectory: list[np.ndarray] | np.ndarray,
    map_points: np.ndarray,
    T_cw: np.ndarray | None = None,
    clusters: list[MapCluster] | None = None,
    objects: list[dict] | None = None,
    size: int = 800,
    max_points: int = 6000,
    unit: str = "units",
    unit_scale: float = 1.0,
) -> np.ndarray:
    """Render the layout as a BGR image.

    objects: dicts with "position" (world xyz), "label", "track_id" and optional
        "velocity" (world xyz per frame) for a motion arrow.
    unit / unit_scale: label and factor for the scale bar (e.g. "m" and metres per map unit).
    """
    clusters = clusters or []
    objects = [o for o in (objects or []) if o.get("position") is not None]
    path = np.asarray(trajectory, dtype=np.float64).reshape(-1, 3)
    path = path[np.isfinite(path).all(axis=1) & (np.abs(path).max(axis=1, initial=0.0) < 1e12)]
    points = np.asarray(map_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points).astype(np.int64)]

    # Fit the view to the path, clusters, objects and the bulk of the map (outliers are clipped).
    extents = [path[:, [0, 2]]] if len(path) else []
    if len(points):
        extents.append(np.percentile(points[:, [0, 2]], [2, 98], axis=0))
    for cluster in clusters:
        extents.append(np.array([cluster.minimum[[0, 2]], cluster.maximum[[0, 2]]]))
    if objects:
        extents.append(np.array([np.asarray(o["position"], dtype=np.float64)[[0, 2]] for o in objects]))
    canvas = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(canvas, "Layout (top-down, x right, z up)", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    if not extents:
        return canvas
    bounds = np.vstack(extents)
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    span = max(float(np.max(high - low)), 1e-9)
    center = (low + high) / 2.0
    margin = 60
    pixels_per_unit = (size - 2 * margin) / span

    def to_px(xz: np.ndarray) -> np.ndarray:
        xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        px = np.column_stack(
            [size / 2.0 + (xz[:, 0] - center[0]) * pixels_per_unit, size / 2.0 - (xz[:, 1] - center[1]) * pixels_per_unit]
        )
        return np.clip(np.nan_to_num(px), -1e6, 1e6).astype(np.int32)

    if len(points):
        px = to_px(points[:, [0, 2]])
        visible = (px[:, 0] >= 0) & (px[:, 0] < size) & (px[:, 1] >= 0) & (px[:, 1] < size)
        canvas[px[visible, 1], px[visible, 0]] = POINT_COLOR

    for cluster in clusters:
        (x1, y1), (x2, y2) = to_px(np.array([cluster.minimum[[0, 2]], cluster.maximum[[0, 2]]]))
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), CLUSTER_COLOR, 1)
        cv2.putText(canvas, f"C{cluster.cluster_id}", (int(min(x1, x2)), int(min(y1, y2)) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLUSTER_COLOR, 1, cv2.LINE_AA)

    if len(path) >= 2:
        cv2.polylines(canvas, [to_px(path[:, [0, 2]]).reshape(-1, 1, 2)], False, PATH_COLOR, 2)

    for obj in objects:
        color = OBJECT_COLORS[int(obj.get("track_id", 0)) % len(OBJECT_COLORS)]
        position = np.asarray(obj["position"], dtype=np.float64)
        px = to_px(position[[0, 2]])[0]
        cv2.circle(canvas, (int(px[0]), int(px[1])), 6, color, -1)
        velocity = obj.get("velocity")
        if velocity is not None and np.linalg.norm(np.asarray(velocity)[[0, 2]]) > 0.0:
            heading = np.asarray(velocity, dtype=np.float64)[[0, 2]]
            heading = heading / np.linalg.norm(heading) * 28.0
            tip = (int(px[0] + heading[0]), int(px[1] - heading[1]))
            cv2.arrowedLine(canvas, (int(px[0]), int(px[1])), tip, color, 2, tipLength=0.35)
        text = f"{obj.get('label', 'object')} #{obj.get('track_id', '?')}"
        cv2.putText(canvas, text, (int(px[0]) + 9, int(px[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if T_cw is not None:
        R_wc = T_cw[:3, :3].T
        camera = -R_wc @ T_cw[:3, 3]
        forward = (R_wc @ np.array([0.0, 0.0, 1.0]))[[0, 2]]
        px = to_px(camera[[0, 2]])[0]
        cv2.circle(canvas, (int(px[0]), int(px[1])), 7, CAMERA_COLOR, -1)
        if np.linalg.norm(forward) > 1e-9:
            forward = forward / np.linalg.norm(forward) * 34.0
            tip = (int(px[0] + forward[0]), int(px[1] - forward[1]))
            cv2.arrowedLine(canvas, (int(px[0]), int(px[1])), tip, CAMERA_COLOR, 2, tipLength=0.3)
        cv2.putText(canvas, "camera", (int(px[0]) + 10, int(px[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CAMERA_COLOR, 1, cv2.LINE_AA)

    bar_units = _nice_length(0.25 * span * unit_scale)
    bar_px = int(bar_units / unit_scale * pixels_per_unit)
    y = size - 24
    cv2.line(canvas, (16, y), (16 + bar_px, y), (230, 230, 230), 2)
    cv2.putText(canvas, f"{bar_units:g} {unit}", (16, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas
