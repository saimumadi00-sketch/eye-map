#!/usr/bin/env python3
"""
Shared frame-tracking loop used by every EyeMap entry point.
`main.py`, `src.mvp.live_mvp`, and the VKITTI adapter differ only in where frames and intrinsics come from and where outputs are written; the detect -> track -> keyframe -> display loop lives here once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.exporter import save_trajectory_csv
from src.feature_detector import create_orb, detect_features
from src.keyframe_manager import KeyframeManager
from src.pointcloud import save_pointcloud
from src.pose_estimator import camera_center_from_tcw
from src.profiler import FeatureAutoTuner, StageProfiler
from src.tracker import NOT_ENOUGH_MATCHES, MonocularTracker, TrackResult
from src.video_input import read_frame
from src.visualizer import draw_status, draw_trajectory, show_window

# One input frame: (frame_id, BGR image, 3x3 intrinsics).
Frame = tuple[int, np.ndarray, np.ndarray]


class NoFramesError(RuntimeError):
    """Raised when the input yields no frame at all."""


@dataclass
class TrackingRun:
    """Everything a run accumulates; filled in place so an interrupt keeps partial results."""

    trajectory: list[np.ndarray] = field(default_factory=list)
    trajectory_rows: list[list[float]] = field(default_factory=list)
    map_chunks: list[np.ndarray] = field(default_factory=list)
    frames_processed: int = 0
    interrupted: bool = False
    profiler: StageProfiler = field(default_factory=StageProfiler)
    # ORB feature budget after the last frame; changes only when auto-tuning is on.
    nfeatures: int = 0

    def map_points(self) -> np.ndarray:
        """Return all triangulated world points as one Nx3 array."""
        if not self.map_chunks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack(self.map_chunks)


@dataclass
class FrameStep:
    """Per-frame state handed to a custom renderer."""

    index: int
    frame_id: int
    image: np.ndarray
    keypoints: tuple
    result: TrackResult
    status: str
    run: TrackingRun


def capture_frames(
    cap: cv2.VideoCapture,
    intrinsics: Callable[[int, int], np.ndarray],
    resize_width: int = 0,
) -> Iterator[Frame]:
    """Yield numbered frames from an OpenCV capture; K is built once from the first frame size."""
    K = None
    frame_id = 0
    while True:
        ok, frame = read_frame(cap, resize_width=resize_width)
        if not ok or frame is None:
            return
        if K is None:
            height, width = frame.shape[:2]
            K = intrinsics(width, height)
        yield frame_id, frame, K
        frame_id += 1


def status_text(frame_id: int, result: TrackResult, run: TrackingRun) -> str:
    """Summarize one tracking result for the on-screen overlay."""
    status = f"frame={frame_id} matches={len(result.matches)}"
    if result.success:
        return status + f" pose=ok scale={result.scale:.3f} map_chunks={len(run.map_chunks)}"
    if result.reason == NOT_ENOUGH_MATCHES:
        return status + " pose=skip:not_enough_matches"
    return status + f" pose=fail:{result.reason}"


def default_render(step: FrameStep) -> np.ndarray:
    """Draw the status and performance lines on the current frame."""
    perf = f"fps={step.run.profiler.fps:.1f} features={step.run.nfeatures}"
    return draw_status(step.image, [step.status, perf, "press q to quit"])


def track_sequence(
    frames: Iterable[Frame],
    tracker: MonocularTracker,
    keyframes: KeyframeManager | None = None,
    nfeatures: int = 2000,
    max_frames: int = 0,
    display_enabled: bool = True,
    show_trajectory: bool = True,
    trajectory_scale: float = 80.0,
    render: Callable[[FrameStep], np.ndarray] = default_render,
    tuner: FeatureAutoTuner | None = None,
) -> TrackingRun:
    """Track every frame against the tracker's reference and collect trajectory, map, and keyframes.

    The first frame initializes the tracker at the world origin. Stops when frames run out,
    `max_frames` frames (if > 0) have been processed, q is pressed, or on Ctrl+C.
    Keyframes are saved for the first frame and every `keyframes.interval`-th frame after it.

    Per-stage timings accumulate in `run.profiler`. Time spent waiting for the next frame is
    reported as the "read" stage but excluded from frame time, so fps measures processing speed
    rather than the camera's frame rate. With a `tuner`, the ORB feature budget follows it
    (starting from the tuner's initial value, not `nfeatures`).
    """
    if tuner is not None:
        nfeatures = tuner.nfeatures
    detector = create_orb(nfeatures)
    run = TrackingRun(nfeatures=int(nfeatures))
    profiler = run.profiler
    iterator = iter(frames)
    try:
        first = next(iterator, None)
        if first is None:
            raise NoFramesError("Could not read the first frame.")
        frame_id, image, _ = first
        _, keypoints, descriptors = detect_features(image, detector)
        tracker.initialize(keypoints, descriptors, image)
        run.trajectory.append(camera_center_from_tcw(tracker.T_cw))
        run.trajectory_rows.append([frame_id, 0.0, 0.0, 0.0])
        run.frames_processed = 1
        if keyframes is not None:
            keyframes.save(frame_id, image, tracker.T_cw)

        index = 0
        while max_frames <= 0 or index < max_frames - 1:
            with profiler.stage("read", in_frame=False):
                item = next(iterator, None)
            if item is None:
                break
            index += 1
            frame_id, image, K = item

            profiler.start_frame()
            with profiler.stage("detect"):
                _, keypoints, descriptors = detect_features(image, detector)
            with profiler.stage("track"):
                result = tracker.track(keypoints, descriptors, K, image=image)
            if result.success:
                center = camera_center_from_tcw(result.T_cw)
                run.trajectory.append(center)
                run.trajectory_rows.append([frame_id, float(center[0]), float(center[1]), float(center[2])])
                if len(result.points_world) > 0:
                    run.map_chunks.append(result.points_world)
            run.frames_processed = index + 1

            if keyframes is not None and keyframes.should_save(index):
                with profiler.stage("keyframe"):
                    keyframes.save(frame_id, image, tracker.T_cw)

            with profiler.stage("render"):
                step = FrameStep(index, frame_id, image, keypoints, result, status_text(frame_id, result, run), run)
                display = render(step)
            quit_requested = False
            with profiler.stage("display"):
                show_window("EyeMap", display, enabled=display_enabled)
                if show_trajectory and display_enabled:
                    show_window("Trajectory", draw_trajectory(run.trajectory, scale=trajectory_scale))
                if display_enabled and (cv2.waitKey(1) & 0xFF) == ord("q"):
                    quit_requested = True
            frame_s = profiler.end_frame()

            if tuner is not None:
                budget = tuner.update(frame_s, len(result.matches))
                if budget != run.nfeatures:
                    detector.setMaxFeatures(budget)
                    run.nfeatures = budget
            if quit_requested:
                break
    except KeyboardInterrupt:
        # Ctrl+C is the only way to stop a headless webcam run; keep what was mapped so far.
        run.interrupted = True
    finally:
        if display_enabled:
            cv2.destroyAllWindows()
    return run


def export_flat_run(run: TrackingRun, run_dir: Path, keyframes: KeyframeManager, max_export_points: int) -> Path | None:
    """Write trajectory.csv, keyframes.json, and sparse_map.ply directly in run_dir.

    Returns the PLY path, or None when no points were triangulated.
    """
    save_trajectory_csv(run_dir / "trajectory.csv", run.trajectory_rows)
    keyframes.write_metadata(run_dir / "keyframes.json")
    if not run.map_chunks:
        return None
    return save_pointcloud(run_dir / "sparse_map.ply", run.map_points(), max_points=max_export_points)
