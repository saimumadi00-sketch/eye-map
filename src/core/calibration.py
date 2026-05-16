#!/usr/bin/env python3
"""
Checkerboard camera calibration for the EyeMap pipeline.
Calibrated intrinsics improve tracking accuracy over the estimated fallback K used when no camera calibration is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


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


def collect_calib_frames(source, n_frames: int = 30, board_size: tuple[int, int] = (9, 6)) -> list:
    """Capture frames until the requested number of chessboard detections is collected."""
    cap = cv2.VideoCapture(_parse_source(str(source)))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open calibration source: {source}")

    collected = []
    try:
        while len(collected) < int(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_size, None)
            display = frame.copy()
            if found:
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
            cv2.imshow("Calibration Capture", display)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
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


def save_calibration(K, dist_coeffs, path) -> None:
    """Save camera matrix and distortion coefficients to an NPZ file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), K=np.asarray(K), dist_coeffs=np.asarray(dist_coeffs))


def load_calibration(path) -> tuple:
    """Load camera matrix and distortion coefficients from an NPZ file."""
    calib_path = Path(path)
    if not calib_path.exists():
        print(f"[WARN] Calibration file not found: {calib_path}")
        return None, None
    data = np.load(str(calib_path))
    return data["K"], data["dist_coeffs"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for calibration capture."""
    parser = argparse.ArgumentParser(description="Capture checkerboard frames and estimate camera calibration.")
    parser.add_argument("--source", required=True, help="Webcam index or video path.")
    parser.add_argument("--n-frames", type=int, default=30, help="Number of chessboard detections to collect.")
    parser.add_argument("--board-size", default="9x6", help="Chessboard inner corners as WIDTHxHEIGHT.")
    parser.add_argument("--square-mm", type=float, default=25.0, help="Checkerboard square size in millimeters.")
    parser.add_argument("--out", default="configs/calib.npz", help="Output NPZ path.")
    return parser.parse_args()


def main() -> None:
    """Run calibration capture, solve intrinsics, and save the result."""
    args = parse_args()
    board_size = _parse_board_size(args.board_size)
    frames = collect_calib_frames(args.source, n_frames=args.n_frames, board_size=board_size)
    K, dist_coeffs, reprojection_error = calibrate_from_frames(
        frames,
        board_size=board_size,
        square_mm=args.square_mm,
    )
    save_calibration(K, dist_coeffs, args.out)
    print(f"[INFO] Saved calibration: {Path(args.out)}")
    print(f"[INFO] Reprojection error: {reprojection_error:.6f}")


if __name__ == "__main__":
    main()
