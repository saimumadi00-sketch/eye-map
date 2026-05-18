# ============================================================
# CONTRIBUTOR STUB
# This file defines the interface for GPS scale fusion.
# To implement:
#   1. Align GPS and visual-trajectory arc lengths
#   2. Replace each NotImplementedError with real implementation
#   3. See docs/drone_deployment.md for the scale algorithm
#   4. See docs/architecture.md for how this fits the pipeline
# ============================================================
"""GPS scale fusion for metric terrain reconstruction.
CONTRIBUTOR STUB — implement to recover real-world scale from GPS readings."""

from __future__ import annotations

import numpy as np

GUIDANCE_MESSAGE = (
    "GPSFusion.estimate_scale() is not implemented. "
    "Implement Haversine distance alignment between GPS arc length "
    "and trajectory arc length to recover metric scale. "
    "See docs/drone_deployment.md for the algorithm."
)


class GPSFusion:
    """Collect GPS and trajectory samples for future metric-scale recovery."""

    def __init__(self) -> None:
        """Initialize uncalibrated GPS and trajectory reading buffers."""
        self.scale_factor = None  # metres per relative unit, None until calibrated
        self.gps_readings: list[tuple[int, float, float, float]] = []  # list of (frame_id, lat, lon, alt_m)
        self.traj_readings: list[tuple[int, float, float, float]] = []  # list of (frame_id, x, y, z)

    def add_gps(self, frame_id: int, lat: float, lon: float, alt_m: float) -> None:
        """Append one GPS sample keyed by frame id."""
        self.gps_readings.append((frame_id, lat, lon, alt_m))

    def add_pose(self, frame_id: int, x: float, y: float, z: float) -> None:
        """Append one relative trajectory sample keyed by frame id."""
        self.traj_readings.append((frame_id, x, y, z))

    def estimate_scale(self) -> float | None:
        """Estimate metres per relative unit once implemented by a contributor."""
        raise NotImplementedError(GUIDANCE_MESSAGE)

    def apply_scale(self, points: np.ndarray) -> np.ndarray:
        """Scale relative points into metres once calibration is implemented."""
        raise NotImplementedError(GUIDANCE_MESSAGE)

    def is_calibrated(self) -> bool:
        """Return whether a metric scale factor has been estimated."""
        return self.scale_factor is not None
