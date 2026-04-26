#!/usr/bin/env python3
"""
Build a simple terrain/elevation grid from an exported sparse PLY point cloud.
Grid axes:
- x,z: ground plane
- y: elevation
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    from src.core.utils import load_ply_xyz
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.core.utils import load_ply_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert sparse map PLY to terrain grid CSV.")
    parser.add_argument("--ply", type=str, required=True, help="Input sparse_map.ply path.")
    parser.add_argument("--cell", type=float, default=0.25, help="Grid cell size in relative units.")
    parser.add_argument(
        "--stat",
        type=str,
        default="median",
        choices=["min", "mean", "median"],
        help="Elevation statistic per cell.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="terrain_grid.csv",
        help="Output CSV path for (gx, gz, elevation, count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_ply_xyz(Path(args.ply))
    if len(points) == 0:
        raise RuntimeError("No points found in PLY.")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    gx = np.floor(x / args.cell).astype(np.int64)
    gz = np.floor(z / args.cell).astype(np.int64)

    buckets: dict[tuple[int, int], list[float]] = {}
    for i in range(len(points)):
        key = (int(gx[i]), int(gz[i]))
        buckets.setdefault(key, []).append(float(y[i]))

    rows: list[list[float | int]] = []
    for (ix, iz), ys in buckets.items():
        ys_np = np.asarray(ys, dtype=np.float64)
        if args.stat == "min":
            elev = float(np.min(ys_np))
        elif args.stat == "mean":
            elev = float(np.mean(ys_np))
        else:
            elev = float(np.median(ys_np))
        rows.append([ix, iz, elev, len(ys)])

    rows.sort(key=lambda r: (r[1], r[0]))

    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_x", "grid_z", "elevation_y", "point_count"])
        writer.writerows(rows)

    print(f"[INFO] Input points: {len(points)}")
    print(f"[INFO] Output cells: {len(rows)}")
    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
