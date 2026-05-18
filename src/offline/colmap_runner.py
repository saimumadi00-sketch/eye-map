#!/usr/bin/env python3
"""
Subprocess wrapper for COLMAP sparse SfM reconstruction from saved keyframe images.
It keeps the optional offline reconstruction path separate from the live OpenCV MVP and fails cleanly when COLMAP is unavailable.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def check_colmap_available(colmap_bin: str = "colmap") -> bool:
    """Return True when the COLMAP executable is available on PATH."""
    if shutil.which(colmap_bin):
        return True
    logger.warning("COLMAP executable not found: %s", colmap_bin)
    logger.warning("Install COLMAP and make sure the 'colmap' command is available on PATH.")
    return False


def _run_command(command: list[str]) -> bool:
    """Run one subprocess command and print stderr on failure."""
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return True
    logger.error("Command failed: %s", " ".join(command))
    if result.stderr:
        logger.error(result.stderr.strip())
    return False


def run_sparse(keyframe_dir, output_dir, colmap_bin: str = "colmap") -> Path | None:
    """Run COLMAP sparse reconstruction and export the model to PLY."""
    keyframes = Path(keyframe_dir)
    if not keyframes.exists():
        raise FileNotFoundError(f"Required keyframe directory not found: {keyframes}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = out_dir / "sparse"
    model_dir = sparse_dir / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not check_colmap_available(colmap_bin):
        return None

    database_path = out_dir / "database.db"
    commands = [
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(keyframes),
        ],
        [
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
        ],
        [
            colmap_bin,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(keyframes),
            "--output_path",
            str(sparse_dir),
        ],
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(out_dir / "sparse_map.ply"),
            "--output_type",
            "PLY",
        ],
    ]

    for command in commands:
        if not _run_command(command):
            return None

    return out_dir / "sparse_map.ply"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sparse COLMAP reconstruction."""
    parser = argparse.ArgumentParser(description="Run COLMAP sparse reconstruction from EyeMap keyframes.")
    parser.add_argument("--keyframes", required=True, help="Path to keyframes directory.")
    parser.add_argument("--out", required=True, help="Output directory for COLMAP files.")
    parser.add_argument("--colmap-bin", default="colmap", help="COLMAP executable name or path.")
    return parser.parse_args()


def main() -> None:
    """Run the sparse reconstruction CLI."""
    args = parse_args()
    ply_path = run_sparse(args.keyframes, args.out, colmap_bin=args.colmap_bin)
    if ply_path is not None:
        logger.info("Sparse PLY: %s", ply_path)


if __name__ == "__main__":
    main()
