#!/usr/bin/env python3
"""
Checkerboard camera calibration for the EyeMap pipeline.
Calibrated intrinsics improve tracking accuracy over the estimated fallback K used when no camera calibration is available.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _parse_source(source: str):
    """Convert numeric webcam sources to int and leave file paths as strings."""
    if str(source).isdigit():
        return int(source)
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Required calibration source not found: {source_path}")
    return str(source_path)


def _parse_board_size(value: str) -> tuple[int, int]:
    """Parse a board-size string such as 9x6 into a tuple."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid board size, expected WIDTHxHEIGHT: {value}")
    return int(parts[0]), int(parts[1])


def collect_calib_frames(
    source,
    n_frames: int = 30,
    board_size: tuple[int, int] = (9, 6),
    display_enabled: bool = True,
    min_interval: int = 15,
) -> list:
    """Capture frames until the requested number of chessboard detections is collected.

    Detections closer than `min_interval` frames to the previous capture are skipped, so the
    board has time to move; consecutive near-identical views make calibration ill-conditioned.
    """
    cap = cv2.VideoCapture(_parse_source(str(source)))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open calibration source: {source}")

    collected = []
    frame_index = -1
    last_capture = None
    try:
        while len(collected) < int(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_index += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()
            if last_capture is not None and frame_index - last_capture < int(min_interval):
                found = False
            else:
                found, corners = cv2.findChessboardCorners(gray, board_size, None)
            if found:
                last_capture = frame_index
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                collected.append((gray.copy(), refined.copy()))
                cv2.drawChessboardCorners(display, board_size, refined, found)
            cv2.putText(
                display,
                f"captured={len(collected)}/{int(n_frames)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if display_enabled:
                cv2.imshow("Calibration Capture", display)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        cap.release()
        if display_enabled:
            cv2.destroyAllWindows()

    return collected


def calibrate_from_frames(
    frames: list,
    board_size: tuple[int, int] = (9, 6),
    square_mm: float = 25.0,
) -> tuple:
    """Run OpenCV camera calibration from detected chessboard frames."""
    if not frames:
        raise ValueError("At least one calibration frame is required.")

    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    grid = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_mm)

    object_points = [objp.copy() for _ in frames]
    image_points = [corners for _, corners in frames]
    image_size = (frames[0][0].shape[1], frames[0][0].shape[0])
    ret, K, dist_coeffs, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    return K, dist_coeffs, float(ret)


def save_calibration(K, dist_coeffs, path, image_size: tuple[int, int] | None = None) -> None:
    """Save camera matrix, distortion coefficients, and optional (width, height) to an NPZ file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if image_size is None else {"image_size": np.asarray(image_size, dtype=np.int64)}
    np.savez(str(out_path), K=np.asarray(K), dist_coeffs=np.asarray(dist_coeffs), **extra)


def load_calibration(path) -> tuple:
    """Load camera matrix and distortion coefficients from an NPZ file."""
    calib_path = Path(path)
    if not calib_path.exists():
        logger.warning("Calibration file not found: %s", calib_path)
        return None, None
    data = np.load(str(calib_path))
    return data["K"], data["dist_coeffs"]


def load_calibration_image_size(path) -> tuple[int, int] | None:
    """Return the (width, height) a calibration was solved at, if it was recorded."""
    calib_path = Path(path)
    if not calib_path.exists():
        return None
    data = np.load(str(calib_path))
    if "image_size" not in data:
        return None
    width, height = (int(v) for v in data["image_size"])
    return width, height


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for calibration capture."""
    parser = argparse.ArgumentParser(description="Capture checkerboard frames and estimate camera calibration.")
    parser.add_argument("--source", required=True, help="Webcam index or video path.")
    parser.add_argument("--n-frames", type=int, default=30, help="Number of chessboard detections to collect.")
    parser.add_argument("--board-size", default="9x6", help="Chessboard inner corners as WIDTHxHEIGHT.")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Checkerboard square size in millimeters.")
    parser.add_argument("--out", default="configs/calib.npz", help="Output NPZ path.")
    parser.add_argument(
        "--min-interval", type=int, default=15, help="Minimum frames between captured views."
    )
    parser.add_argument("--no-display", action="store_true", help="Capture without OpenCV windows.")
    return parser.parse_args()


def main() -> None:
    """Run calibration capture, solve intrinsics, and save the result."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    board_size = _parse_board_size(args.board_size)
    frames = collect_calib_frames(
        args.source,
        n_frames=args.n_frames,
        board_size=board_size,
        display_enabled=not args.no_display,
        min_interval=args.min_interval,
    )
    if len(frames) < 3:
        raise SystemExit(
            f"Only {len(frames)} chessboard view(s) captured; need at least 3 (ideally {args.n_frames})."
        )
    if len(frames) < args.n_frames:
        logger.warning("Source ended after %d of %d chessboard views.", len(frames), args.n_frames)
    K, dist_coeffs, reprojection_error = calibrate_from_frames(
        frames,
        board_size=board_size,
        square_mm=args.square_mm,
    )
    image_size = (frames[0][0].shape[1], frames[0][0].shape[0])
    save_calibration(K, dist_coeffs, args.out, image_size=image_size)
    logger.info("Saved calibration: %s", Path(args.out))
    logger.info("Reprojection error: %.6f", reprojection_error)


if __name__ == "__main__":
    main()
