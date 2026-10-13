#!/usr/bin/env python3
"""
Virtual KITTI adapter for running the live MVP pipeline on dataset frames.
It preserves the Phase 1 ORB, essential-matrix, triangulation, visualization, and export flow while replacing webcam capture with VKITTI sequence records.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.utils import save_ply_xyz, timestamp_id
from src.pose_estimator import camera_center_from_tcw
from src.tracker import NOT_ENOUGH_MATCHES, MonocularTracker
from src.visualizer import draw_trajectory

logger = logging.getLogger(__name__)


MIN_MATCHES = 60
RATIO_TEST = 0.75
MAX_MATCHES = 500
RANSAC_THRESH = 1.0
NFEATURES = 2000
MAX_POINT_NORM = 150.0
MAX_EXPORT_POINTS = 120000
DISPLAY_SCALE = 80.0


def _make_run_dirs(sequence, output_root: str | Path) -> tuple[Path, Path]:
    """Create the VKITTI run directory and keyframe directory."""
    run_name = f"{sequence.scene}_{sequence.variant}_{timestamp_id()}"
    run_dir = Path(output_root) / run_name
    keyframe_dir = run_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, keyframe_dir


def _write_trajectory_csv(path: str | Path, rows: list[list[float]]) -> None:
    """Write frame_id,x,y,z trajectory rows to a CSV file."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "x", "y", "z"])
        writer.writerows(rows)


def _append_ground_truth_row(rows: list[list[float]], record: dict) -> None:
    """Append one ground-truth camera center row from a VKITTI frame record."""
    gt_center = camera_center_from_tcw(np.asarray(record["T_cw"], dtype=np.float64))
    rows.append([int(record["frame_id"]), float(gt_center[0]), float(gt_center[1]), float(gt_center[2])])


def adapt_sequence_to_live_mvp(
    sequence,
    max_frames: int = 0,
    save_every: int = 20,
    output_root: str | Path = "outputs",
    display_enabled: bool = True,
) -> Path:
    """Run the Phase 1 live MVP pipeline on a VKITTISequence and return the output run directory."""
    if len(sequence) == 0:
        raise ValueError("VKITTI sequence contains no frames.")

    frame_limit = len(sequence) if int(max_frames) <= 0 else min(len(sequence), int(max_frames))
    save_interval = max(1, int(save_every))
    run_dir, keyframe_dir = _make_run_dirs(sequence, output_root)

    record_prev = sequence[0]
    frame_prev = record_prev["rgb"]
    first_frame_id = int(record_prev["frame_id"])

    orb = cv2.ORB_create(nfeatures=NFEATURES)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    kp_prev, desc_prev = orb.detectAndCompute(gray_prev, None)
    tracker = MonocularTracker(
        ratio_test=RATIO_TEST,
        max_matches=MAX_MATCHES,
        min_matches=MIN_MATCHES,
        ransac_thresh=RANSAC_THRESH,
        max_point_norm=MAX_POINT_NORM,
    )
    tracker.initialize(kp_prev, desc_prev, frame_prev)

    trajectory: list[np.ndarray] = [camera_center_from_tcw(tracker.T_cw)]
    map_points_world: list[np.ndarray] = []

    trajectory_rows: list[list[float]] = [[first_frame_id, 0.0, 0.0, 0.0]]
    gt_trajectory_rows: list[list[float]] = []
    _append_ground_truth_row(gt_trajectory_rows, record_prev)

    keyframe_meta: list[dict[str, object]] = []
    first_keyframe_name = f"frame_{first_frame_id:06d}.png"
    cv2.imwrite(str(keyframe_dir / first_keyframe_name), frame_prev)
    keyframe_meta.append(
        {
            "frame_id": first_frame_id,
            "image": first_keyframe_name,
            "T_cw": tracker.T_cw.reshape(-1).tolist(),
        }
    )

    for sequence_index in range(1, frame_limit):
        record_curr = sequence[sequence_index]
        frame_curr = record_curr["rgb"]
        frame_id = int(record_curr["frame_id"])
        K = np.asarray(record_curr["K"], dtype=np.float64)
        _append_ground_truth_row(gt_trajectory_rows, record_curr)

        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        kp_curr, desc_curr = orb.detectAndCompute(gray_curr, None)

        result = tracker.track(kp_curr, desc_curr, K, image=frame_curr)
        status_text = "tracking"
        if result.success:
            cam_center = camera_center_from_tcw(result.T_cw)
            trajectory.append(cam_center)
            trajectory_rows.append(
                [frame_id, float(cam_center[0]), float(cam_center[1]), float(cam_center[2])]
            )
            if len(result.points_world) > 0:
                map_points_world.append(result.points_world)
        elif result.reason == NOT_ENOUGH_MATCHES:
            status_text = f"few matches ({len(result.matches)})"
        else:
            status_text = result.reason

        if sequence_index % save_interval == 0:
            name = f"frame_{frame_id:06d}.png"
            cv2.imwrite(str(keyframe_dir / name), frame_curr)
            keyframe_meta.append(
                {
                    "frame_id": frame_id,
                    "image": name,
                    "T_cw": tracker.T_cw.reshape(-1).tolist(),
                }
            )

        traj_img = draw_trajectory(trajectory, scale=DISPLAY_SCALE)
        overlay = frame_curr.copy()
        cv2.putText(
            overlay,
            f"frame={frame_id} status={status_text}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"traj_len={len(trajectory)} map_chunks={len(map_points_world)}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if display_enabled:
            cv2.imshow("Live Frame", overlay)
            cv2.imshow("Trajectory", traj_img)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    if display_enabled:
        cv2.destroyAllWindows()

    _write_trajectory_csv(run_dir / "trajectory.csv", trajectory_rows)
    _write_trajectory_csv(run_dir / "gt_trajectory.csv", gt_trajectory_rows)
    (run_dir / "keyframes.json").write_text(json.dumps(keyframe_meta, indent=2), encoding="utf-8")

    ply_path = run_dir / "sparse_map.ply"
    if map_points_world:
        points = np.vstack(map_points_world)
        if len(points) > MAX_EXPORT_POINTS:
            idx = np.random.choice(len(points), size=MAX_EXPORT_POINTS, replace=False)
            points = points[idx]
        save_ply_xyz(ply_path, points)
        sparse_count = len(points)
    else:
        sparse_count = 0

    logger.info("Run directory: %s", run_dir)
    logger.debug("Trajectory rows: %s", len(trajectory_rows))
    logger.debug("Ground-truth trajectory rows: %s", len(gt_trajectory_rows))
    logger.debug("Saved keyframes: %s", len(keyframe_meta))
    if sparse_count > 0:
        logger.debug("Sparse points exported: %s", sparse_count)
        logger.info("Sparse map path: %s", ply_path)
    else:
        logger.debug("Sparse points exported: 0")

    return run_dir
