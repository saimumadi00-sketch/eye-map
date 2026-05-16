#!/usr/bin/env python3
"""
Virtual KITTI adapter for running the live MVP pipeline on dataset frames.
It preserves the Phase 1 ORB, essential-matrix, triangulation, visualization, and export flow while replacing webcam capture with VKITTI sequence records.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from src.core.utils import timestamp_id
    from src.mvp.live_mvp import (
        camera_center_from_tcw,
        draw_trajectory,
        match_descriptors,
        save_ply_xyz,
        triangulate_in_prev_camera,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.core.utils import timestamp_id
    from src.mvp.live_mvp import (
        camera_center_from_tcw,
        draw_trajectory,
        match_descriptors,
        save_ply_xyz,
        triangulate_in_prev_camera,
    )


MIN_MATCHES = 60
RATIO_TEST = 0.75
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

    T_cw = np.eye(4, dtype=np.float64)
    trajectory: list[np.ndarray] = [camera_center_from_tcw(T_cw)]
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
            "T_cw": T_cw.reshape(-1).tolist(),
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

        status_text = "tracking"
        if (
            desc_prev is None
            or desc_curr is None
            or len(kp_prev) < MIN_MATCHES
            or len(kp_curr) < MIN_MATCHES
        ):
            status_text = "insufficient keypoints"
        else:
            good_matches = match_descriptors(desc_prev, desc_curr, RATIO_TEST)
            if len(good_matches) >= MIN_MATCHES:
                pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good_matches])
                pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in good_matches])

                E, inlier_mask = cv2.findEssentialMat(
                    pts_prev,
                    pts_curr,
                    K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=RANSAC_THRESH,
                )

                if E is not None and inlier_mask is not None:
                    inlier_mask = inlier_mask.ravel().astype(bool)
                    _, R, t, pose_mask = cv2.recoverPose(E, pts_prev, pts_curr, K)
                    pose_mask = pose_mask.ravel().astype(bool)

                    t_norm = np.linalg.norm(t)
                    if t_norm > 1e-9:
                        t = t / t_norm

                    T_cw_prev = T_cw.copy()
                    T_rel = np.eye(4, dtype=np.float64)
                    T_rel[:3, :3] = R
                    T_rel[:3, 3] = t.reshape(3)
                    T_cw = T_rel @ T_cw

                    cam_center = camera_center_from_tcw(T_cw)
                    trajectory.append(cam_center)
                    trajectory_rows.append(
                        [frame_id, float(cam_center[0]), float(cam_center[1]), float(cam_center[2])]
                    )

                    tri_mask = inlier_mask & pose_mask
                    if np.count_nonzero(tri_mask) >= 10:
                        pts_prev_in = pts_prev[tri_mask]
                        pts_curr_in = pts_curr[tri_mask]
                        points_prev = triangulate_in_prev_camera(K, R, t, pts_prev_in, pts_curr_in)

                        points_curr = (R @ points_prev.T + t.reshape(3, 1)).T
                        valid = (
                            np.isfinite(points_prev).all(axis=1)
                            & (points_prev[:, 2] > 0.0)
                            & (points_curr[:, 2] > 0.0)
                            & (np.linalg.norm(points_prev, axis=1) < MAX_POINT_NORM)
                        )
                        points_prev = points_prev[valid]

                        if len(points_prev) > 0:
                            T_wc_prev = np.linalg.inv(T_cw_prev)
                            points_prev_h = np.hstack(
                                [points_prev, np.ones((len(points_prev), 1), dtype=np.float64)]
                            )
                            points_world = (T_wc_prev @ points_prev_h.T).T[:, :3]
                            map_points_world.append(points_world)
                else:
                    status_text = "essential matrix failed"
            else:
                status_text = f"few matches ({len(good_matches)})"

        if sequence_index % save_interval == 0:
            name = f"frame_{frame_id:06d}.png"
            cv2.imwrite(str(keyframe_dir / name), frame_curr)
            keyframe_meta.append(
                {
                    "frame_id": frame_id,
                    "image": name,
                    "T_cw": T_cw.reshape(-1).tolist(),
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

        cv2.imshow("Live Frame", overlay)
        cv2.imshow("Trajectory", traj_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        gray_prev = gray_curr
        kp_prev = kp_curr
        desc_prev = desc_curr

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

    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Trajectory rows: {len(trajectory_rows)}")
    print(f"[INFO] Ground-truth trajectory rows: {len(gt_trajectory_rows)}")
    print(f"[INFO] Saved keyframes: {len(keyframe_meta)}")
    if sparse_count > 0:
        print(f"[INFO] Sparse points exported: {sparse_count}")
        print(f"[INFO] Sparse map path: {ply_path}")
    else:
        print("[INFO] Sparse points exported: 0")

    return run_dir
