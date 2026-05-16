#!/usr/bin/env python3
"""
End-to-end VKITTI runner that processes one scene/variant and prints an evaluation report.
It connects the dataset loader, VKITTI live-MVP adapter, and report utilities for repeatable final-year project experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.data.vkitti_adapter import adapt_sequence_to_live_mvp
    from src.data.vkitti_loader import VKITTISequence
    from src.mvp.evaluate import compare_trajectories, point_cloud_stats, print_report
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.data.vkitti_adapter import adapt_sequence_to_live_mvp
    from src.data.vkitti_loader import VKITTISequence
    from src.mvp.evaluate import compare_trajectories, point_cloud_stats, print_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one VKITTI evaluation run."""
    parser = argparse.ArgumentParser(description="Run EyeMap on a Virtual KITTI 2 sequence.")
    parser.add_argument("--rgb-dir", required=True, help="Path to vkitti_2.0.3_rgb/")
    parser.add_argument("--depth-dir", required=True, help="Path to vkitti_2.0.3_depth/")
    parser.add_argument("--text-dir", required=True, help="Path to vkitti_2.0.3_textgt/")
    parser.add_argument("--scene", default="Scene01", help="VKITTI scene name.")
    parser.add_argument("--variant", default="clone", help="VKITTI variant name.")
    parser.add_argument("--camera", type=int, default=0, help="VKITTI camera index.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames, or 0 for all frames.")
    parser.add_argument("--save-every", type=int, default=20, help="Keyframe save interval.")
    parser.add_argument("--output-root", default="outputs", help="Output directory root.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation report generation.")
    return parser.parse_args()


def main() -> None:
    """Process one VKITTI sequence and optionally print evaluation metrics."""
    args = parse_args()
    sequence = VKITTISequence(
        args.rgb_dir,
        args.depth_dir,
        args.text_dir,
        args.scene,
        args.variant,
        camera=args.camera,
    )
    run_dir = adapt_sequence_to_live_mvp(
        sequence,
        max_frames=args.max_frames,
        save_every=args.save_every,
        output_root=args.output_root,
    )

    if not args.skip_eval:
        traj_stats = compare_trajectories(run_dir / "trajectory.csv", run_dir / "gt_trajectory.csv")
        sparse_ply = run_dir / "sparse_map.ply"
        cloud_stats = point_cloud_stats(sparse_ply) if sparse_ply.exists() else None
        print_report(traj_stats=traj_stats, cloud_stats=cloud_stats)

    print(f"[INFO] run_dir={run_dir}")


if __name__ == "__main__":
    main()
