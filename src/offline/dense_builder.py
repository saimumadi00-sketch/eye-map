#!/usr/bin/env python3
"""
Dense reconstruction from COLMAP sparse output via patch_match_stereo and stereo_fusion.
The dense path is optional because it usually requires a CUDA-capable NVIDIA GPU and a complete COLMAP workspace.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from src.offline.colmap_runner import check_colmap_available

logger = logging.getLogger(__name__)


def _has_cuda() -> bool:
    """Return True when nvidia-smi reports an available NVIDIA GPU."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, check=False)
    except OSError:
        return False
    return result.returncode == 0


def _find_sparse_model_dir(colmap_root: Path) -> Path:
    """Find the COLMAP sparse model directory from a root or model path."""
    candidates = [colmap_root, colmap_root / "sparse" / "0", colmap_root / "0"]
    for candidate in candidates:
        if (candidate / "cameras.bin").exists() or (candidate / "cameras.txt").exists():
            return candidate
    raise FileNotFoundError(f"Required COLMAP sparse model not found under: {colmap_root}")


def _find_sparse_ply(colmap_root: Path) -> Path:
    """Find a sparse PLY file near a COLMAP output directory."""
    candidates = [
        colmap_root / "sparse_map.ply",
        colmap_root.parent / "sparse_map.ply",
        colmap_root.parent.parent / "sparse_map.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required sparse PLY not found near: {colmap_root}")


def _find_image_dir(colmap_root: Path) -> Path:
    """Find the keyframe image directory associated with a COLMAP output."""
    candidates = [
        colmap_root / "keyframes",
        colmap_root.parent / "keyframes",
        colmap_root.parent.parent / "keyframes",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required keyframes directory not found near: {colmap_root}")


def _run_command(command: list[str]) -> bool:
    """Run one COLMAP command and print stderr on failure."""
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return True
    logger.error("Command failed: %s", " ".join(command))
    if result.stderr:
        logger.error(result.stderr.strip())
    return False


def run_dense(
    colmap_sparse_dir,
    output_dir,
    colmap_bin: str = "colmap",
    cpu_fallback: bool = True,
) -> Path:
    """Run COLMAP dense reconstruction or copy the sparse PLY when CUDA is unavailable."""
    sparse_root = Path(colmap_sparse_dir)
    if not sparse_root.exists():
        raise FileNotFoundError(f"Required COLMAP sparse directory not found: {sparse_root}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fused_path = out_dir / "fused.ply"

    if cpu_fallback and not _has_cuda():
        sparse_ply = _find_sparse_ply(sparse_root)
        shutil.copyfile(sparse_ply, fused_path)
        logger.warning("CUDA not available; copied sparse PLY as dense fallback.")
        return fused_path

    if not check_colmap_available(colmap_bin):
        return fused_path

    model_dir = _find_sparse_model_dir(sparse_root)
    image_dir = _find_image_dir(sparse_root)
    undistorted_dir = out_dir / "dense"
    commands = [
        [
            colmap_bin,
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(model_dir),
            "--output_path",
            str(undistorted_dir),
        ],
        [
            colmap_bin,
            "patch_match_stereo",
            "--workspace_path",
            str(undistorted_dir),
        ],
        [
            colmap_bin,
            "stereo_fusion",
            "--workspace_path",
            str(undistorted_dir),
            "--output_path",
            str(fused_path),
        ],
    ]

    for command in commands:
        if not _run_command(command):
            return fused_path
    return fused_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dense COLMAP reconstruction."""
    parser = argparse.ArgumentParser(description="Run COLMAP dense reconstruction from sparse output.")
    parser.add_argument("--sparse-dir", required=True, help="COLMAP sparse output root or model directory.")
    parser.add_argument("--out", required=True, help="Output directory for dense files.")
    parser.add_argument("--colmap-bin", default="colmap", help="COLMAP executable name or path.")
    parser.add_argument("--no-cpu-fallback", action="store_true", help="Do not copy sparse PLY when CUDA is missing.")
    return parser.parse_args()


def main() -> None:
    """Run the dense reconstruction CLI."""
    args = parse_args()
    fused = run_dense(
        args.sparse_dir,
        args.out,
        colmap_bin=args.colmap_bin,
        cpu_fallback=not args.no_cpu_fallback,
    )
    logger.info("Fused PLY: %s", fused)


if __name__ == "__main__":
    main()
