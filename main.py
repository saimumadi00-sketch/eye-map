#!/usr/bin/env python3
"""
Command-line entry point for the modular EyeMap MVP.
It reads webcam/video frames, detects and matches ORB features, estimates relative motion, saves keyframes, exports trajectory artifacts, and optionally writes a sparse point cloud.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.core.calibration import load_calibration, load_calibration_image_size
from src.exporter import save_log, save_match_screenshot, save_trajectory_csv, save_trajectory_plot
from src.feature_detector import draw_keypoints
from src.feature_matcher import draw_matches
from src.keyframe_manager import KeyframeManager
from src.pipeline import FrameStep, NoFramesError, capture_frames, default_render, track_sequence
from src.pointcloud import save_pointcloud, visualize_pointcloud
from src.pose_estimator import make_intrinsics
from src.tracker import MonocularTracker
from src.utils import ensure_dir, load_config, make_run_dir
from src.video_input import open_capture, release_capture

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MVP runner."""
    parser = argparse.ArgumentParser(description="EyeMap modular terrain mapping MVP.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config" / "config.yaml"),
        help="Path to YAML config (defaults to the packaged configuration).",
    )
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
    parser.add_argument(
        "--calib",
        default=None,
        help="Calibration NPZ from src.core.calibration (overrides config calibration_file).",
    )
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV windows.")
    return parser.parse_args()


def cfg(config: dict, key: str, default):
    """Read one top-level config value with a default."""
    return config.get(key, default)


def resolve_intrinsics(calib_path: str | None, width: int, height: int) -> tuple[np.ndarray, str]:
    """Return calibrated K scaled to the processed frame size, or the estimated fallback."""
    if not calib_path:
        return make_intrinsics(width, height), "estimated"

    K_calib, _ = load_calibration(calib_path)
    if K_calib is None:
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    K = np.asarray(K_calib, dtype=np.float64).copy()
    calib_size = load_calibration_image_size(calib_path)
    if calib_size is not None and calib_size != (width, height):
        # Calibration was solved at another resolution (e.g. resize_width is set): rescale K.
        K[0, :] *= width / calib_size[0]
        K[1, :] *= height / calib_size[1]
    elif calib_size is None:
        logger.warning("Calibration has no stored image size; using K as-is for %dx%d frames.", width, height)
    return K, f"calibrated:{calib_path}"


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

    max_frames = int(args.max_frames if args.max_frames is not None else cfg(config, "max_frames", 0))
    trajectory_scale = float(cfg(config, "trajectory_scale", 80.0))
    calib_path = getattr(args, "calib", None) or cfg(config, "calibration_file", None)
    save_keyframes = bool(cfg(config, "save_keyframes", True)) or args.save_keyframes
    save_matches = bool(cfg(config, "save_matches", True)) or args.save_matches or args.mode == "matches"
    match_interval = max(1, int(cfg(config, "match_save_interval", 20)))
    export_cloud = bool(cfg(config, "export_pointcloud", True)) or args.export_pointcloud or args.mode == "pointcloud"
    show_cloud = bool(cfg(config, "show_pointcloud", False)) or args.show_pointcloud

    def intrinsics(width: int, height: int) -> np.ndarray:
        K, k_source = resolve_intrinsics(calib_path, width, height)
        save_log(log_path, f"intrinsics={k_source}")
        return K

    def render(step: FrameStep) -> np.ndarray:
        result = step.result
        match_image = None
        if save_matches and len(result.matches) > 0:
            match_image = draw_matches(
                result.ref_image, result.ref_keypoints, step.image, step.keypoints, result.matches
            )
            if step.frame_id % match_interval == 0:
                save_match_screenshot(dirs["matches"] / f"matches_{step.frame_id:06d}.png", match_image)
        if args.mode == "features":
            return draw_keypoints(step.image, step.keypoints)
        if args.mode == "matches" and match_image is not None:
            return match_image
        return default_render(step)

    tracker = MonocularTracker(
        ratio_test=float(cfg(config, "ratio_test", 0.75)),
        max_matches=int(cfg(config, "max_matches", 500)),
        min_matches=int(cfg(config, "minimum_matches", 60)),
        ransac_thresh=float(cfg(config, "ransac_thresh", 1.0)),
        max_point_norm=float(cfg(config, "max_point_norm", 150.0)),
    )
    keyframes = None
    if save_keyframes:
        keyframes = KeyframeManager(dirs["keyframes"], interval=int(cfg(config, "keyframe_interval", 20)))

    cap = open_capture(args.source, path=args.path, camera_index=int(cfg(config, "camera_index", 0)))
    try:
        run = track_sequence(
            capture_frames(cap, intrinsics, resize_width=int(cfg(config, "resize_width", 0))),
            tracker,
            keyframes=keyframes,
            nfeatures=int(cfg(config, "orb_features", 2000)),
            max_frames=max_frames,
            display_enabled=not args.no_display,
            show_trajectory=args.mode in {"trajectory", "pointcloud"},
            trajectory_scale=trajectory_scale,
            render=render,
        )
    except NoFramesError as exc:
        logger.error("%s", exc)
        return 1
    finally:
        release_capture(cap)
    if run.interrupted:
        logger.warning("Interrupted; saving outputs collected so far.")

    if keyframes is not None:
        keyframes.write_metadata(run_dir / "keyframes.json")
    trajectory_csv = save_trajectory_csv(dirs["trajectory"] / "trajectory.csv", run.trajectory_rows)
    trajectory_plot = save_trajectory_plot(dirs["plots"] / "trajectory.png", run.trajectory, scale=trajectory_scale)

    cloud_path = None
    if export_cloud and run.map_chunks:
        cloud_path = save_pointcloud(
            dirs["pointclouds"] / "sparse_map.ply",
            run.map_points(),
            max_points=int(cfg(config, "max_export_points", 120000)),
        )
        if show_cloud:
            visualize_pointcloud(cloud_path)

    save_log(log_path, f"frames_processed={run.frames_processed}")
    save_log(log_path, f"trajectory_csv={trajectory_csv}")
    save_log(log_path, f"trajectory_plot={trajectory_plot}")
    if cloud_path is not None:
        save_log(log_path, f"pointcloud={cloud_path}")

    logger.info("Run directory: %s", run_dir)
    logger.info("Trajectory CSV: %s", trajectory_csv)
    logger.info("Trajectory plot: %s", trajectory_plot)
    logger.debug("Keyframes saved: %s", len(keyframes.metadata) if keyframes is not None else 0)
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
