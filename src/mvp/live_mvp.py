#!/usr/bin/env python3
"""
Phase-1 live MVP:
- Reads webcam/video
- Tracks ORB features
- Estimates relative pose with Essential matrix
- Triangulates sparse 3D points
- Shows live frame + 2D trajectory
- Exports keyframes, trajectory, and sparse point cloud
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.core.utils import save_ply_xyz, timestamp_id
from src.tracker import NOT_ENOUGH_MATCHES, MonocularTracker

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live terrain mapping MVP (monocular).")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g., 0) or video file path.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV windows.")
    parser.add_argument("--save-every", type=int, default=20, help="Save every Nth frame.")
    parser.add_argument("--min-matches", type=int, default=60, help="Minimum matches for pose.")
    parser.add_argument("--ratio-test", type=float, default=0.75, help="Lowe ratio threshold.")
    parser.add_argument("--ransac-thresh", type=float, default=1.0, help="RANSAC pixel threshold.")
    parser.add_argument("--nfeatures", type=int, default=2000, help="ORB feature count.")
    parser.add_argument(
        "--max-export-points",
        type=int,
        default=120000,
        help="Randomly subsample point cloud if above this count.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=80.0,
        help="Scale factor for trajectory display.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Directory root where run outputs are stored.",
    )
    parser.add_argument(
        "--show-pointcloud",
        action="store_true",
        help="Open exported sparse cloud in Open3D viewer at the end of the run.",
    )
    return parser.parse_args()


def ensure_run_dirs(output_root: Path) -> tuple[Path, Path]:
    run_dir = output_root / timestamp_id()
    keyframe_dir = run_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, keyframe_dir


def make_intrinsics(width: int, height: int) -> np.ndarray:
    # Student-friendly default when camera intrinsics are not calibrated.
    focal = 0.9 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_center_from_tcw(T_cw: np.ndarray) -> np.ndarray:
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    return -R_cw.T @ t_cw


def draw_trajectory(trajectory_xyz: list[np.ndarray], scale: float = 80.0) -> np.ndarray:
    canvas = np.full((700, 700, 3), 20, dtype=np.uint8)
    origin = np.array([350, 500], dtype=np.int32)

    if len(trajectory_xyz) < 2:
        cv2.putText(
            canvas,
            "Trajectory (x-z)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return canvas

    points_2d: list[np.ndarray] = []
    for p in trajectory_xyz:
        x = int(origin[0] + p[0] * scale)
        z = int(origin[1] - p[2] * scale)
        points_2d.append(np.array([x, z], dtype=np.int32))

    for i in range(1, len(points_2d)):
        cv2.line(
            canvas,
            tuple(points_2d[i - 1]),
            tuple(points_2d[i]),
            (70, 220, 70),
            2,
            cv2.LINE_AA,
        )

    cv2.circle(canvas, tuple(points_2d[-1]), 4, (40, 120, 255), -1)
    cv2.putText(
        canvas,
        "Trajectory (x-z)",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return canvas


def source_to_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def try_show_pointcloud_open3d(ply_path: Path) -> None:
    try:
        import open3d as o3d
    except Exception:
        logger.warning("Open3D unavailable. Skipping point cloud viewer.")
        return

    cloud = o3d.io.read_point_cloud(str(ply_path))
    if cloud.is_empty():
        logger.warning("Exported PLY is empty. Skipping point cloud viewer.")
        return
    o3d.visualization.draw_geometries([cloud], window_name="Sparse Map (Open3D)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_dir, keyframe_dir = ensure_run_dirs(Path(args.output_root))

    cap = source_to_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    ok, frame_prev = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from source.")

    height, width = frame_prev.shape[:2]
    K = make_intrinsics(width, height)

    orb = cv2.ORB_create(nfeatures=args.nfeatures)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    kp_prev, desc_prev = orb.detectAndCompute(gray_prev, None)
    tracker = MonocularTracker(
        ratio_test=args.ratio_test,
        min_matches=args.min_matches,
        ransac_thresh=args.ransac_thresh,
    )
    tracker.initialize(kp_prev, desc_prev)

    trajectory: list[np.ndarray] = [camera_center_from_tcw(tracker.T_cw)]
    map_points_world: list[np.ndarray] = []

    trajectory_rows: list[list[float]] = [[0, 0.0, 0.0, 0.0]]
    keyframe_meta: list[dict[str, object]] = []

    first_keyframe_name = "frame_000000.png"
    cv2.imwrite(str(keyframe_dir / first_keyframe_name), frame_prev)
    keyframe_meta.append(
        {
            "frame_id": 0,
            "image": first_keyframe_name,
            "T_cw": tracker.T_cw.reshape(-1).tolist(),
        }
    )

    frame_id = 1
    while True:
        ok, frame_curr = cap.read()
        if not ok:
            break

        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        kp_curr, desc_curr = orb.detectAndCompute(gray_curr, None)

        result = tracker.track(kp_curr, desc_curr, K)
        status_text = "tracking"
        if result.success:
            cam_center = camera_center_from_tcw(result.T_cw)
            trajectory.append(cam_center)
            trajectory_rows.append([frame_id, cam_center[0], cam_center[1], cam_center[2]])
            if len(result.points_world) > 0:
                map_points_world.append(result.points_world)
        elif result.reason == NOT_ENOUGH_MATCHES:
            status_text = f"few matches ({len(result.matches)})"
        else:
            status_text = result.reason

        if frame_id % args.save_every == 0:
            name = f"frame_{frame_id:06d}.png"
            cv2.imwrite(str(keyframe_dir / name), frame_curr)
            keyframe_meta.append(
                {
                    "frame_id": frame_id,
                    "image": name,
                    "T_cw": tracker.T_cw.reshape(-1).tolist(),
                }
            )

        traj_img = draw_trajectory(trajectory, scale=args.display_scale)
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

        if not args.no_display:
            cv2.imshow("Live Frame", overlay)
            cv2.imshow("Trajectory", traj_img)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        frame_id += 1

        if args.max_frames > 0 and frame_id > args.max_frames:
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # Save trajectory CSV.
    traj_path = run_dir / "trajectory.csv"
    with traj_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "x", "y", "z"])
        writer.writerows(trajectory_rows)

    # Save keyframe metadata.
    meta_path = run_dir / "keyframes.json"
    meta_path.write_text(json.dumps(keyframe_meta, indent=2), encoding="utf-8")

    # Export sparse map point cloud.
    ply_path = run_dir / "sparse_map.ply"
    if map_points_world:
        points = np.vstack(map_points_world)
        if len(points) > args.max_export_points:
            idx = np.random.choice(len(points), size=args.max_export_points, replace=False)
            points = points[idx]
        save_ply_xyz(ply_path, points)

    logger.info("Run directory: %s", run_dir)
    logger.debug("Trajectory rows: %s", len(trajectory_rows))
    logger.debug("Saved keyframes: %s", len(keyframe_meta))
    if map_points_world:
        logger.debug("Sparse points exported: %s", len(points))
        logger.info("Sparse map path: %s", ply_path)
    else:
        logger.debug("Sparse points exported: 0")

    if args.show_pointcloud and map_points_world:
        try_show_pointcloud_open3d(ply_path)


if __name__ == "__main__":
    main()
