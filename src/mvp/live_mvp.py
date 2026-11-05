#!/usr/bin/env python3
"""
Phase-1 live MVP: a flag-driven, config-free front end to the shared EyeMap pipeline.
- Reads webcam/video
- Tracks ORB features, estimates relative pose, triangulates sparse 3D points
- Shows live frame + 2D trajectory
- Exports trajectory.csv, keyframes.json, keyframes/, and sparse_map.ply flat in the run directory
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from src.core.utils import timestamp_id
from src.keyframe_manager import KeyframeManager
from src.pipeline import capture_frames, export_flat_run, track_sequence
from src.pointcloud import visualize_pointcloud
from src.pose_estimator import make_intrinsics
from src.tracker import MonocularTracker

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


def source_to_capture(source: str) -> cv2.VideoCapture:
    """Open a webcam index such as "0" or a video file path."""
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_dir = Path(args.output_root) / timestamp_id()

    cap = source_to_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    tracker = MonocularTracker(
        ratio_test=args.ratio_test,
        min_matches=args.min_matches,
        ransac_thresh=args.ransac_thresh,
    )
    keyframes = KeyframeManager(run_dir / "keyframes", interval=args.save_every)
    try:
        run = track_sequence(
            capture_frames(cap, make_intrinsics),
            tracker,
            keyframes=keyframes,
            nfeatures=args.nfeatures,
            max_frames=args.max_frames,
            display_enabled=not args.no_display,
            trajectory_scale=args.display_scale,
        )
    finally:
        cap.release()

    ply_path = export_flat_run(run, run_dir, keyframes, args.max_export_points)
    logger.info("Run directory: %s", run_dir)
    logger.debug("Trajectory rows: %s", len(run.trajectory_rows))
    logger.debug("Saved keyframes: %s", len(keyframes.metadata))
    if ply_path is not None:
        logger.info("Sparse map path: %s", ply_path)
        if args.show_pointcloud:
            visualize_pointcloud(ply_path)


if __name__ == "__main__":
    main()
