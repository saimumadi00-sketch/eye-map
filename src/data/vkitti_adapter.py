#!/usr/bin/env python3
"""
Virtual KITTI adapter for running the live MVP pipeline on dataset frames.
It feeds VKITTI records (with their per-frame intrinsics) through the shared EyeMap pipeline and also writes the ground-truth trajectory for evaluation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.core.utils import timestamp_id
from src.exporter import save_trajectory_csv
from src.keyframe_manager import KeyframeManager
from src.pipeline import Frame, export_flat_run, track_sequence
from src.pose_estimator import camera_center_from_tcw
from src.tracker import MonocularTracker

logger = logging.getLogger(__name__)


MIN_MATCHES = 60
RATIO_TEST = 0.75
MAX_MATCHES = 500
RANSAC_THRESH = 1.0
NFEATURES = 2000
MAX_POINT_NORM = 150.0
MAX_EXPORT_POINTS = 120000
DISPLAY_SCALE = 80.0


def _ground_truth_row(record: dict) -> list[float]:
    """Return one ground-truth camera center row from a VKITTI frame record."""
    gt_center = camera_center_from_tcw(np.asarray(record["T_cw"], dtype=np.float64))
    return [int(record["frame_id"]), float(gt_center[0]), float(gt_center[1]), float(gt_center[2])]


def _sequence_frames(sequence, frame_limit: int, gt_rows: list[list[float]]) -> Iterator[Frame]:
    """Yield pipeline frames, recording each frame's ground-truth center as it is consumed."""
    for index in range(frame_limit):
        record = sequence[index]
        gt_rows.append(_ground_truth_row(record))
        yield int(record["frame_id"]), record["rgb"], np.asarray(record["K"], dtype=np.float64)


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
    run_dir = Path(output_root) / f"{sequence.scene}_{sequence.variant}_{timestamp_id()}"
    keyframes = KeyframeManager(run_dir / "keyframes", interval=save_every)
    tracker = MonocularTracker(
        ratio_test=RATIO_TEST,
        max_matches=MAX_MATCHES,
        min_matches=MIN_MATCHES,
        ransac_thresh=RANSAC_THRESH,
        max_point_norm=MAX_POINT_NORM,
    )

    gt_rows: list[list[float]] = []
    run = track_sequence(
        _sequence_frames(sequence, frame_limit, gt_rows),
        tracker,
        keyframes=keyframes,
        nfeatures=NFEATURES,
        display_enabled=display_enabled,
        trajectory_scale=DISPLAY_SCALE,
    )

    ply_path = export_flat_run(run, run_dir, keyframes, MAX_EXPORT_POINTS)
    save_trajectory_csv(run_dir / "gt_trajectory.csv", gt_rows)

    logger.info("Run directory: %s", run_dir)
    logger.debug("Trajectory rows: %s", len(run.trajectory_rows))
    logger.debug("Ground-truth trajectory rows: %s", len(gt_rows))
    logger.debug("Saved keyframes: %s", len(keyframes.metadata))
    if ply_path is not None:
        logger.info("Sparse map path: %s", ply_path)
    return run_dir
