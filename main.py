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
from src.integration.gps_fusion import GPSFusion
from src.keyframe_manager import KeyframeManager
from src.objects.detector import OnnxDetector, load_class_names
from src.objects.layout import draw_layout
from src.objects.scene import SceneAnalyzer
from src.pipeline import FrameStep, NoFramesError, capture_frames, default_render, track_sequence
from src.pointcloud import save_pointcloud, visualize_pointcloud
from src.pose_estimator import make_intrinsics
from src.profiler import FeatureAutoTuner
from src.tracker import MonocularTracker
from src.utils import ensure_dir, load_config, make_run_dir
from src.video_input import open_capture, release_capture
from src.visualizer import show_window

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
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Adapt the ORB feature count to hold the target frame rate (config: orb_auto_tune).",
    )
    parser.add_argument("--target-fps", type=float, default=None, help="Frame rate goal for --auto-tune.")
    parser.add_argument(
        "--gps-csv",
        default=None,
        help="CSV of GPS fixes (frame_id,lat,lon[,alt]) used to recover metric scale (config: gps_csv).",
    )
    parser.add_argument(
        "--detect-model",
        default=None,
        help="YOLO-style ONNX model for object detection and directions (config: detect_model).",
    )
    parser.add_argument(
        "--detect-classes",
        default=None,
        help="Text file of class names, one per line (default: the 80 COCO classes; config: detect_classes).",
    )
    parser.add_argument("--detect-conf", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--detect-interval", type=int, default=None, help="Run the detector every Nth frame.")
    parser.add_argument("--show-layout", action="store_true", help="Show the live top-down layout window.")
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
        "objects": ensure_dir(run_dir / names.get("objects", "objects")),
    }
    return dirs


def export_metric_outputs(gps_fusion: GPSFusion, run, dirs: dict[str, Path], log_path: Path) -> Path | None:
    """Recover metric scale from GPS and write the trajectory in local East-North-Up metres.

    Returns the metric CSV path, or None when the scale could not be estimated.
    """
    for frame_id, x, y, z in run.trajectory_rows:
        gps_fusion.add_pose(int(frame_id), x, y, z)
    scale = gps_fusion.estimate_scale()
    if scale is None:
        logger.warning(
            "GPS scale not recovered: need at least %d frames with both a GPS fix and a pose, "
            "and %.0f m of GPS travel.",
            gps_fusion.min_samples,
            gps_fusion.min_baseline_m,
        )
        save_log(log_path, "gps_scale=unavailable")
        return None

    centers = np.asarray([row[1:] for row in run.trajectory_rows], dtype=np.float64)
    enu = gps_fusion.to_enu(centers)
    rows = [[int(row[0]), *map(float, point)] for row, point in zip(run.trajectory_rows, enu)]
    metric_csv = save_trajectory_csv(dirs["trajectory"] / "trajectory_metric_enu.csv", rows)
    lat, lon, alt = gps_fusion.origin
    save_log(log_path, f"gps_scale_m_per_unit={scale:.6f}")
    save_log(log_path, f"enu_origin_lat_lon_alt={lat:.8f},{lon:.8f},{alt:.3f}")
    save_log(log_path, f"trajectory_metric_enu_csv={metric_csv}")
    logger.info("GPS scale: %.4f m per unit; metric trajectory: %s", scale, metric_csv)
    return metric_csv


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

    detector = None
    detect_model = getattr(args, "detect_model", None) or cfg(config, "detect_model", None)
    if detect_model:
        classes_file = getattr(args, "detect_classes", None) or cfg(config, "detect_classes", None)
        detect_conf = getattr(args, "detect_conf", None)
        detector = OnnxDetector(
            detect_model,
            class_names=load_class_names(classes_file) if classes_file else None,
            input_size=int(cfg(config, "detect_input_size", 640)),
            conf_threshold=float(detect_conf if detect_conf is not None else cfg(config, "detect_conf", 0.35)),
        )
        save_log(log_path, f"detect_model={detect_model}")
    detect_interval = getattr(args, "detect_interval", None)
    analyzer = SceneAnalyzer(
        detector,
        detect_interval=int(detect_interval if detect_interval is not None else cfg(config, "detect_interval", 1)),
    )
    export_layout = bool(cfg(config, "export_layout", True))
    show_layout = getattr(args, "show_layout", False) and not args.no_display

    def render(step: FrameStep) -> np.ndarray:
        display = render_frame(step)
        if analyzer.detector is not None and step.K is not None:
            # Timed separately so the detector's cost is visible; it is also part of "render".
            with step.run.profiler.stage("objects", in_frame=False):
                observations = analyzer.process(
                    step.frame_id, step.index, step.image, step.K, step.result.T_cw, step.run.map_chunks
                )
            display = analyzer.annotate(display, observations)
            if observations and step.frame_id % match_interval == 0:
                save_match_screenshot(dirs["objects"] / f"objects_{step.frame_id:06d}.png", display)
        if show_layout:
            layout = draw_layout(
                step.run.trajectory, step.run.map_points(), step.result.T_cw, objects=analyzer.layout_objects(step.result.T_cw)
            )
            show_window("Layout", layout)
        return display

    def render_frame(step: FrameStep) -> np.ndarray:
        result = step.result
        match_image = None
        # Drawing matches is costly, so only do it for frames that are saved or shown.
        save_now = save_matches and step.frame_id % match_interval == 0
        if (save_now or args.mode == "matches") and len(result.matches) > 0:
            match_image = draw_matches(
                result.ref_image, result.ref_keypoints, step.image, step.keypoints, result.matches
            )
            if save_now:
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

    tuner = None
    if getattr(args, "auto_tune", False) or bool(cfg(config, "orb_auto_tune", False)):
        target_fps = getattr(args, "target_fps", None)
        tuner = FeatureAutoTuner(
            target_fps=float(target_fps if target_fps is not None else cfg(config, "target_fps", 15.0)),
            initial=int(cfg(config, "orb_features", 2000)),
            min_features=int(cfg(config, "orb_features_min", 500)),
            max_features=int(cfg(config, "orb_features_max", 4000)),
            min_matches=int(cfg(config, "minimum_matches", 60)),
        )

    gps_fusion = None
    gps_csv = getattr(args, "gps_csv", None) or cfg(config, "gps_csv", None)
    if gps_csv:
        gps_fusion = GPSFusion()
        fixes = gps_fusion.load_gps_csv(gps_csv)
        save_log(log_path, f"gps_csv={gps_csv} fixes={fixes}")

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
            tuner=tuner,
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

    if gps_fusion is not None:
        export_metric_outputs(gps_fusion, run, dirs, log_path)

    if export_layout and (run.map_chunks or analyzer.observations):
        scene_paths = analyzer.export(
            run.trajectory,
            run.map_points(),
            tracker.T_cw,
            dirs["objects"],
            dirs["plots"] / "layout.png",
            scale_m=gps_fusion.scale_factor if gps_fusion is not None else None,
        )
        for name, path in scene_paths.items():
            save_log(log_path, f"{name}={path}")
        logger.info("Layout: %s", scene_paths["layout"])
        logger.info("Scene description: %s", scene_paths["scene"])

    profile_report = run.profiler.report()
    save_log(log_path, f"frames_processed={run.frames_processed}")
    save_log(log_path, f"orb_features_final={run.nfeatures}")
    save_log(log_path, "profile:\n" + profile_report)
    logger.info("Performance:\n%s", profile_report)
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
