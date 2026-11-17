"""GPS scale fusion for metric terrain reconstruction.

Monocular visual odometry recovers the trajectory only up to one global scale (and an
arbitrary rotation). `GPSFusion` pairs GPS fixes with visual camera centers by frame id and
recovers that scale, plus the rotation and translation that georeference the map into a
local East-North-Up (ENU) frame anchored at the first paired GPS fix.

Two estimators are available:

- "similarity" (default): least-squares similarity fit (Umeyama) between the visual
  trajectory and the ENU GPS track. GPS noise averages out, and it also yields the
  rotation/translation needed by `to_enu()`.
- "arc_length": ratio of GPS path length to visual path length. GPS jitter would inflate a
  path summed over every fix, so both paths are measured over the same chords, each at
  least `min_segment_m` long in GPS space.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from src.mvp.evaluate import fit_similarity

EARTH_RADIUS_M = 6_371_008.8  # IUGG mean Earth radius
METHODS = ("similarity", "arc_length")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two WGS84 lat/lon points (degrees)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def gps_to_enu(
    lat: float,
    lon: float,
    alt_m: float,
    ref_lat: float,
    ref_lon: float,
    ref_alt_m: float,
) -> np.ndarray:
    """Convert a GPS fix to local East-North-Up metres around a reference fix.

    Uses a local tangent-plane approximation, accurate to well under GPS noise for the
    few-kilometre extents a mapping flight covers.
    """
    dlon = (lon - ref_lon + 180.0) % 360.0 - 180.0  # shortest way round the antimeridian
    east = math.radians(dlon) * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    north = math.radians(lat - ref_lat) * EARTH_RADIUS_M
    return np.array([east, north, alt_m - ref_alt_m], dtype=np.float64)


class GPSFusion:
    """Recover metric scale and a local ENU georeference from paired GPS and visual poses."""

    def __init__(
        self,
        method: str = "similarity",
        min_baseline_m: float = 5.0,
        min_segment_m: float = 2.0,
        min_samples: int = 3,
    ) -> None:
        """Configure the estimator.

        min_baseline_m: GPS travel (metres) required before a scale is trusted; below this,
            receiver noise dominates the measurement and `estimate_scale()` returns None.
        min_segment_m: minimum GPS chord length used by the "arc_length" method.
        min_samples: minimum number of frames that have both a GPS fix and a visual pose.
        """
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")
        self.method = method
        self.min_baseline_m = float(min_baseline_m)
        self.min_segment_m = float(min_segment_m)
        self.min_samples = max(2, int(min_samples))

        self.scale_factor: float | None = None  # metres per relative unit, None until calibrated
        self.gps_readings: list[tuple[int, float, float, float]] = []  # (frame_id, lat, lon, alt_m)
        self.traj_readings: list[tuple[int, float, float, float]] = []  # (frame_id, x, y, z)
        self.origin: tuple[float, float, float] | None = None  # (lat, lon, alt_m) of the ENU origin
        self._rotation: np.ndarray | None = None
        self._translation: np.ndarray | None = None

    def add_gps(self, frame_id: int, lat: float, lon: float, alt_m: float = 0.0) -> None:
        """Append one GPS sample keyed by frame id; non-finite or out-of-range fixes are rejected."""
        values = (float(lat), float(lon), float(alt_m))
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"GPS fix for frame {frame_id} is not finite: {values}")
        if not (-90.0 <= values[0] <= 90.0 and -180.0 <= values[1] <= 180.0):
            raise ValueError(f"GPS fix for frame {frame_id} is out of range: lat={lat}, lon={lon}")
        self.gps_readings.append((int(frame_id), *values))

    def add_pose(self, frame_id: int, x: float, y: float, z: float) -> None:
        """Append one relative trajectory sample (camera center) keyed by frame id."""
        self.traj_readings.append((int(frame_id), float(x), float(y), float(z)))

    def load_gps_csv(self, path: str | Path) -> int:
        """Load GPS fixes from a CSV with `frame_id,lat,lon` and optional `alt` columns.

        Returns the number of fixes added.
        """
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"GPS CSV not found: {csv_path}")
        count = 0
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = {"frame_id", "lat", "lon"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"GPS CSV {csv_path} is missing column(s): {sorted(missing)}")
            for row in reader:
                alt = row.get("alt") or row.get("alt_m") or 0.0
                self.add_gps(int(row["frame_id"]), float(row["lat"]), float(row["lon"]), float(alt))
                count += 1
        return count

    def paired_samples(self) -> tuple[list[int], np.ndarray, np.ndarray]:
        """Return (frame_ids, visual Nx3, GPS ENU Nx3) for frames that have both readings.

        A frame id added more than once keeps its latest reading. ENU is anchored at the
        first paired GPS fix.
        """
        gps = {frame_id: (lat, lon, alt) for frame_id, lat, lon, alt in self.gps_readings}
        traj = {frame_id: (x, y, z) for frame_id, x, y, z in self.traj_readings}
        frame_ids = sorted(set(gps) & set(traj))
        if not frame_ids:
            empty = np.zeros((0, 3), dtype=np.float64)
            return [], empty, empty
        ref = gps[frame_ids[0]]
        visual = np.asarray([traj[i] for i in frame_ids], dtype=np.float64)
        enu = np.asarray([gps_to_enu(*gps[i], *ref) for i in frame_ids], dtype=np.float64)
        return frame_ids, visual, enu

    def estimate_scale(self) -> float | None:
        """Estimate metres per relative unit from the paired samples.

        Returns None (and leaves the fusion uncalibrated) when there are too few paired
        frames, the GPS track is shorter than `min_baseline_m`, or the visual trajectory
        has no motion. Safe to call repeatedly as more samples arrive.
        """
        self.scale_factor = None
        self._rotation = None
        self._translation = None

        frame_ids, visual, enu = self.paired_samples()
        if len(frame_ids) < self.min_samples:
            return None
        finite = np.isfinite(visual).all(axis=1)
        visual, enu = visual[finite], enu[finite]
        if len(visual) < self.min_samples:
            return None
        # Farthest fix from the start: a noise-tolerant measure of how far the drone travelled.
        if float(np.max(np.linalg.norm(enu - enu[0], axis=1))) < self.min_baseline_m:
            return None

        try:
            similarity_scale, rotation, translation = fit_similarity(visual, enu)
        except ValueError:
            return None

        scale = similarity_scale
        if self.method == "arc_length":
            arc_scale = self._arc_length_scale(visual, enu)
            if arc_scale is None:
                return None
            scale = arc_scale
            # Keep the georeference consistent with the chosen scale.
            translation = enu.mean(axis=0) - scale * rotation @ visual.mean(axis=0)
        if not math.isfinite(scale) or scale <= 0.0:
            return None

        gps = {frame_id: (lat, lon, alt) for frame_id, lat, lon, alt in self.gps_readings}
        self.origin = gps[frame_ids[0]]
        self.scale_factor = float(scale)
        self._rotation = rotation
        self._translation = translation
        return self.scale_factor

    def _arc_length_scale(self, visual: np.ndarray, enu: np.ndarray) -> float | None:
        """Return GPS path length over visual path length, both measured on shared chords."""
        gps_length = 0.0
        visual_length = 0.0
        anchor = 0
        for index in range(1, len(enu)):
            chord = float(np.linalg.norm(enu[index] - enu[anchor]))
            if chord < self.min_segment_m:
                continue
            gps_length += chord
            visual_length += float(np.linalg.norm(visual[index] - visual[anchor]))
            anchor = index
        if gps_length < self.min_baseline_m or visual_length <= 1e-12:
            return None
        return gps_length / visual_length

    def apply_scale(self, points: np.ndarray) -> np.ndarray:
        """Scale relative points into metres, keeping EyeMap's own axes and origin."""
        if self.scale_factor is None:
            raise RuntimeError("GPSFusion is not calibrated; call estimate_scale() first.")
        return np.asarray(points, dtype=np.float64) * self.scale_factor

    def to_enu(self, points: np.ndarray) -> np.ndarray:
        """Map relative Nx3 points into local East-North-Up metres around `origin`.

        Heading comes from the shape of the flight path, so a straight-line flight fixes
        scale and position but leaves the roll about the flight direction undetermined.
        """
        if self.scale_factor is None or self._rotation is None or self._translation is None:
            raise RuntimeError("GPSFusion is not calibrated; call estimate_scale() first.")
        pts = np.asarray(points, dtype=np.float64)
        return self.scale_factor * (pts @ self._rotation.T) + self._translation

    def is_calibrated(self) -> bool:
        """Return whether a metric scale factor has been estimated."""
        return self.scale_factor is not None
