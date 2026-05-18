#!/usr/bin/env python3
"""
Command-line entry point for the modular EyeMap MVP.
It reads webcam/video frames, detects and matches ORB features, estimates relative motion, saves keyframes, exports trajectory artifacts, and optionally writes a sparse point cloud.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from src.exporter import save_log, save_match_screenshot, save_trajectory_csv, save_trajectory_plot
from src.feature_detector import create_orb, detect_features, draw_keypoints
from src.feature_matcher import draw_matches, match_features, matched_points
from src.keyframe_manager import KeyframeManager
from src.pointcloud import save_pointcloud, visualize_pointcloud
from src.pose_estimator import camera_center_from_tcw, estimate_relative_pose_from_points, make_intrinsics, update_pose
from src.triangulation import filter_triangulated_points, transform_points, triangulate_relative
from src.utils import ensure_dir, load_config, make_run_dir
from src.video_input import open_capture, read_frame, release_capture
from src.visualizer import draw_status, draw_trajectory, show_window

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MVP runner."""
    parser = argparse.ArgumentParser(description="EyeMap modular terrain mapping MVP.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--source", choices=["webcam", "video"], default="webcam", help="Input source type.")
    parser.add_argument("--path", default=None, help="Video file path when --source video is used.")
    parser.add_argument(
        "--mode",
        choices=["display", "features", "matches", "trajectory", "pointcloud"],
        default="trajectory",
        help="Visualization/output mode.",
    )
    parser.add_argument("--save-keyframes", action="store_true", help="Force keyframe saving.")
    parser.add_argument("--save-matches", action="store_true", help="Force match screenshot saving.")
    parser.add_argument("--export-pointcloud", action="store_true", help="Force sparse point cloud export.")
    parser.add_argument("--show-pointcloud", action="store_true", help="Open exported cloud with Open3D.")
    parser.add_argument("--max-frames", type=int, default=None, help="Override max frame count.")
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV windows.")
    return parser.parse_args()


def cfg(config: dict, key: str, default):
    """Read one top-level config value with a default."""
    return config.get(key, default)


def output_dirs(config: dict, run_dir: Path) -> dict[str, Path]:
    """Create and return output subdirectories for one run."""
    names = config.get("output", {})
    dirs = {
        "keyframes": ensure_dir(run_dir / names.get("keyframes", "keyframes")),
        "matches": ensure_dir(run_dir / names.get("matches", "matches")),
        "trajectory": ensure_dir(run_dir / names.get("trajectory", "trajectory")),
        "pointclouds": ensure_dir(run_dir / names.get("pointclouds", "pointclouds")),
        "plots": ensure_dir(run_dir / names.get("plots", "plots")),
        "logs": ensure_dir(run_dir / names.get("logs", "logs")),
    }
    return dirs


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the full webcam/video MVP pipeline."""
    config = load_config(args.config)
    output_root = config.get("output", {}).get("root", "outputs")
    run_dir = make_run_dir(output_root)
    dirs = output_dirs(config, run_dir)
    log_path = dirs["logs"] / "run.log"

    save_log(log_path, f"run_dir={run_dir}")
    cap = open_capture(args.source, path=args.path, camera_index=int(cfg(config, "camera_index", 0)))

    max_frames = int(args.max_frames if args.max_frames is not None else cfg(config, "max_frames", 0))
    resize_width = int(cfg(config, "resize_width", 0))
    display_enabled = not args.no_display

    ok, frame_prev = read_frame(cap, resize_width=resize_width)
    if not ok or frame_prev is None:
        release_capture(cap)
        logger.error("Could not read the first frame.")
        return 1

    height, width = frame_prev.shape[:2]
    K = make_intrinsics(width, height)
    detector = create_orb(int(cfg(config, "orb_features", 2000)))
    _, kp_prev, desc_prev = detect_features(frame_prev, detector)

    T_cw = np.eye(4, dtype=np.float64)
    trajectory = [camera_center_from_tcw(T_cw)]
    trajectory_rows: list[list[float]] = [[0, 0.0, 0.0, 0.0]]
    map_chunks: list[np.ndarray] = []

    save_keyframes = bool(cfg(config, "save_keyframes", True)) or args.save_keyframes
    save_matches = bool(cfg(config, "save_matches", True)) or args.save_matches
    export_cloud = bool(cfg(config, "export_pointcloud", True)) or args.export_pointcloud or args.mode == "pointcloud"
    show_cloud = bool(cfg(config, "show_pointcloud", False)) or args.show_pointcloud

    keyframes = KeyframeManager(dirs["keyframes"], interval=int(cfg(config, "keyframe_interval", 20)))
    if save_keyframes:
        keyframes.save(0, frame_prev, T_cw)

    frame_id = 1
    while True:
        ok, frame_curr = read_frame(cap, resize_width=resize_width)
        if not ok or frame_curr is None:
            break

        _, kp_curr, desc_curr = detect_features(frame_curr, detector)
        matches = match_features(
            desc_prev,
            desc_curr,
            ratio_test=float(cfg(config, "ratio_test", 0.75)),
            max_matches=int(cfg(config, "max_matches", 500)),
        )

        status = f"frame={frame_id} matches={len(matches)}"
        match_image = None
        if len(matches) >= int(cfg(config, "minimum_matches", 60)):
            pts_prev, pts_curr = matched_points(kp_prev, kp_curr, matches)
            pose = estimate_relative_pose_from_points(
                pts_prev,
                pts_curr,
                K,
                ransac_thresh=float(cfg(config, "ransac_thresh", 1.0)),
            )
            if pose["success"]:
                T_cw_prev = T_cw.copy()
                R = pose["R"]
                t = pose["t"]
                T_cw = update_pose(T_cw, R, t)
                center = camera_center_from_tcw(T_cw)
                trajectory.append(center)
                trajectory_rows.append([frame_id, center[0], center[1], center[2]])

                inlier_mask = pose["inlier_mask"]
                if np.count_nonzero(inlier_mask) >= 10:
                    points_prev = triangulate_relative(K, R, t, pts_prev[inlier_mask], pts_curr[inlier_mask])
                    points_prev = filter_triangulated_points(
                        points_prev,
                        R,
                        t,
                        max_norm=float(cfg(config, "max_point_norm", 150.0)),
                    )
                    if len(points_prev) > 0:
                        map_chunks.append(transform_points(np.linalg.inv(T_cw_prev), points_prev))
                status += f" pose=ok map_chunks={len(map_chunks)}"
            else:
                status += f" pose=fail:{pose['reason']}"
        else:
            status += " pose=skip:not_enough_matches"

        if save_keyframes and keyframes.should_save(frame_id):
            keyframes.save(frame_id, frame_curr, T_cw)

        if (save_matches or args.mode == "matches") and len(matches) > 0:
            match_image = draw_matches(frame_prev, kp_prev, frame_curr, kp_curr, matches)
            if frame_id % int(cfg(config, "match_save_interval", 20)) == 0:
                save_match_screenshot(dirs["matches"] / f"matches_{frame_id:06d}.png", match_image)

        display = draw_status(frame_curr, [status, "press q to quit"])
        if args.mode == "features":
            display = draw_keypoints(frame_curr, kp_curr)
        elif args.mode == "matches" and match_image is not None:
            display = match_image

        show_window("EyeMap", display, enabled=display_enabled)
        if args.mode in {"trajectory", "pointcloud"}:
            show_window(
                "Trajectory",
                draw_trajectory(trajectory, scale=float(cfg(config, "trajectory_scale", 80.0))),
                enabled=display_enabled,
            )

        if display_enabled and (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        frame_prev = frame_curr
        kp_prev = kp_curr
        desc_prev = desc_curr
        frame_id += 1
        if max_frames > 0 and frame_id > max_frames:
            break

    release_capture(cap)
    if display_enabled:
        cv2.destroyAllWindows()

    if save_keyframes:
        keyframes.write_metadata(run_dir / "keyframes.json")

    trajectory_csv = save_trajectory_csv(dirs["trajectory"] / "trajectory.csv", trajectory_rows)
    trajectory_plot = save_trajectory_plot(
        dirs["plots"] / "trajectory.png",
        trajectory,
        scale=float(cfg(config, "trajectory_scale", 80.0)),
    )

    cloud_path = None
    if export_cloud and map_chunks:
        cloud_path = save_pointcloud(
            dirs["pointclouds"] / "sparse_map.ply",
            np.vstack(map_chunks),
            max_points=int(cfg(config, "max_export_points", 120000)),
        )
        if show_cloud:
            visualize_pointcloud(cloud_path)

    save_log(log_path, f"frames_processed={frame_id}")
    save_log(log_path, f"trajectory_csv={trajectory_csv}")
    save_log(log_path, f"trajectory_plot={trajectory_plot}")
    if cloud_path is not None:
        save_log(log_path, f"pointcloud={cloud_path}")

    logger.info("Run directory: %s", run_dir)
    logger.info("Trajectory CSV: %s", trajectory_csv)
    logger.info("Trajectory plot: %s", trajectory_plot)
    logger.debug("Keyframes saved: %s", len(keyframes.metadata) if save_keyframes else 0)
    if cloud_path is not None:
        logger.info("Sparse point cloud: %s", cloud_path)
    else:
        logger.info("Sparse point cloud: not generated")
    return 0


def main() -> None:
    """Run the command-line application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    raise SystemExit(run_pipeline(parse_args()))


if __name__ == "__main__":
    main()
