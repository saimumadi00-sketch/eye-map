#!/usr/bin/env python3
"""
Group the sparse map into clusters (obstacles, structures) without any detector.
Points are binned on the ground plane (x-z); neighbouring occupied cells form one cluster.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class MapCluster:
    """A connected group of sparse map points."""

    cluster_id: int
    centroid: np.ndarray  # world xyz (median of member points)
    minimum: np.ndarray  # world xyz bounding-box corner
    maximum: np.ndarray
    count: int


def default_cell_size(points: np.ndarray) -> float:
    """Pick a ground-plane cell of 2% of the map's robust x-z extent."""
    low = np.percentile(points[:, [0, 2]], 2, axis=0)
    high = np.percentile(points[:, [0, 2]], 98, axis=0)
    return max(float(np.linalg.norm(high - low)) * 0.02, 1e-6)


def cluster_map_points(
    points_world: np.ndarray,
    cell: float | None = None,
    min_cell_points: int = 3,
    min_cluster_points: int = 30,
    max_clusters: int = 20,
) -> list[MapCluster]:
    """Cluster Nx3 map points; return up to `max_clusters`, largest first, ids starting at 1.

    Cells with fewer than `min_cell_points` points are treated as noise, and clusters with
    fewer than `min_cluster_points` points are dropped.
    """
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < min_cluster_points:
        return []
    if cell is None:
        cell = default_cell_size(points)
    if not cell > 0.0:
        raise ValueError("cell must be positive")

    grid = np.floor(points[:, [0, 2]] / cell).astype(np.int64)
    cells, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    dense = {(int(cx), int(cz)): index for index, ((cx, cz), n) in enumerate(zip(cells, counts)) if n >= min_cell_points}

    component = np.full(len(cells), -1, dtype=np.int64)
    n_components = 0
    for start, start_index in dense.items():
        if component[start_index] >= 0:
            continue
        component[start_index] = n_components
        queue = deque([start])
        while queue:
            cx, cz = queue.popleft()
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbour = dense.get((cx + dx, cz + dz))
                    if neighbour is not None and component[neighbour] < 0:
                        component[neighbour] = n_components
                        queue.append((cx + dx, cz + dz))
        n_components += 1

    point_component = component[inverse]
    clusters = []
    for label in range(n_components):
        members = points[point_component == label]
        if len(members) >= min_cluster_points:
            clusters.append((len(members), members))
    clusters.sort(key=lambda item: item[0], reverse=True)
    return [
        MapCluster(index + 1, np.median(members, axis=0), members.min(axis=0), members.max(axis=0), count)
        for index, (count, members) in enumerate(clusters[:max_clusters])
    ]
