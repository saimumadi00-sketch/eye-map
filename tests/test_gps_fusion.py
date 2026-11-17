"""Functional tests for GPS metric-scale recovery and ENU georeferencing."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.integration.gps_fusion import EARTH_RADIUS_M, GPSFusion, gps_to_enu, haversine_m

REF = (23.8151, 90.4255, 12.0)  # Dhaka


def enu_to_gps(enu: np.ndarray) -> tuple[float, float, float]:
    """Invert the local tangent-plane approximation around REF."""
    lat = REF[0] + math.degrees(enu[1] / EARTH_RADIUS_M)
    lon = REF[1] + math.degrees(enu[0] / (EARTH_RADIUS_M * math.cos(math.radians(REF[0]))))
    return lat, lon, REF[2] + enu[2]


def rotation_about_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def l_shaped_flight(n: int = 60) -> np.ndarray:
    """Metric ENU path: 40 m east, then 30 m north while climbing 5 m. Starts at the origin."""
    leg1 = np.column_stack([np.linspace(0, 40, n // 2), np.zeros(n // 2), np.zeros(n // 2)])
    leg2 = np.column_stack(
        [np.full(n // 2, 40.0), np.linspace(1, 30, n // 2), np.linspace(0.2, 5, n // 2)]
    )
    return np.vstack([leg1, leg2])


def make_fusion(enu: np.ndarray, true_scale: float, noise_m: float = 0.0, **kwargs) -> tuple[GPSFusion, np.ndarray]:
    """Feed a fusion with GPS fixes for `enu` and a rotated, down-scaled visual trajectory."""
    rng = np.random.default_rng(11)
    rotation = rotation_about_z(37.0)
    visual = (enu @ rotation.T) / true_scale + np.array([0.3, -0.1, 0.2])
    fusion = GPSFusion(**kwargs)
    for frame_id, (gps_point, visual_point) in enumerate(zip(enu, visual)):
        noisy = gps_point + rng.normal(0.0, noise_m, 3) if noise_m else gps_point
        fusion.add_gps(frame_id, *enu_to_gps(noisy))
        fusion.add_pose(frame_id, *visual_point)
    return fusion, visual


class TestGeodesy(unittest.TestCase):
    def test_haversine_one_degree_of_latitude(self) -> None:
        self.assertAlmostEqual(haversine_m(0.0, 0.0, 1.0, 0.0), 111_195.0, delta=5.0)
        self.assertEqual(haversine_m(23.8, 90.4, 23.8, 90.4), 0.0)

    def test_enu_agrees_with_haversine_locally(self) -> None:
        lat, lon, alt = enu_to_gps(np.array([120.0, -80.0, 3.0]))
        enu = gps_to_enu(lat, lon, alt, *REF)
        np.testing.assert_allclose(enu, [120.0, -80.0, 3.0], atol=1e-6)
        self.assertAlmostEqual(float(np.hypot(enu[0], enu[1])), haversine_m(REF[0], REF[1], lat, lon), delta=0.01)

    def test_enu_crosses_antimeridian(self) -> None:
        enu = gps_to_enu(0.0, -179.9995, 0.0, 0.0, 179.9995, 0.0)
        self.assertAlmostEqual(enu[0], haversine_m(0.0, 179.9995, 0.0, -179.9995), delta=0.01)


class TestGPSFusion(unittest.TestCase):
    def test_recovers_scale_and_georeference(self) -> None:
        enu = l_shaped_flight()
        fusion, visual = make_fusion(enu, true_scale=7.5)
        self.assertFalse(fusion.is_calibrated())
        scale = fusion.estimate_scale()
        self.assertAlmostEqual(scale, 7.5, places=4)
        self.assertTrue(fusion.is_calibrated())
        np.testing.assert_allclose(fusion.apply_scale(visual[:2]), visual[:2] * 7.5, rtol=1e-4)
        np.testing.assert_allclose(fusion.to_enu(visual), enu, atol=1e-3)
        np.testing.assert_allclose(fusion.origin, REF, atol=1e-9)

    def test_scale_survives_gps_noise(self) -> None:
        fusion, _ = make_fusion(l_shaped_flight(200), true_scale=4.0, noise_m=1.5)
        self.assertAlmostEqual(fusion.estimate_scale(), 4.0, delta=0.08)

    def test_arc_length_method_resists_jitter(self) -> None:
        enu = l_shaped_flight(400)
        fusion, _ = make_fusion(enu, true_scale=4.0, noise_m=0.5, method="arc_length", min_segment_m=8.0)
        self.assertAlmostEqual(fusion.estimate_scale(), 4.0, delta=0.2)
        # Summing every noisy fix instead would overestimate the GPS path several times over.
        naive, _ = make_fusion(enu, true_scale=4.0, noise_m=0.5, method="arc_length", min_segment_m=0.0)
        self.assertGreater(naive.estimate_scale(), 6.0)

    def test_straight_line_flight_still_gives_scale(self) -> None:
        enu = np.column_stack([np.linspace(0, 50, 30), np.zeros(30), np.zeros(30)])
        for method in ("similarity", "arc_length"):
            fusion, _ = make_fusion(enu, true_scale=3.0, method=method)
            self.assertAlmostEqual(fusion.estimate_scale(), 3.0, places=4, msg=method)

    def test_uncalibrated_cases_return_none(self) -> None:
        hover = np.zeros((20, 3))
        fusion, _ = make_fusion(hover, true_scale=2.0, noise_m=0.5)
        self.assertIsNone(fusion.estimate_scale(), "hover: GPS baseline below threshold")

        empty = GPSFusion()
        self.assertIsNone(empty.estimate_scale())
        with self.assertRaises(RuntimeError):
            empty.apply_scale(np.zeros((1, 3)))
        with self.assertRaises(RuntimeError):
            empty.to_enu(np.zeros((1, 3)))

        static_camera = GPSFusion()
        for frame_id, point in enumerate(l_shaped_flight()):
            static_camera.add_gps(frame_id, *enu_to_gps(point))
            static_camera.add_pose(frame_id, 0.0, 0.0, 0.0)
        self.assertIsNone(static_camera.estimate_scale(), "visual trajectory has no motion")

    def test_only_shared_frames_are_used_and_recalibration_resets(self) -> None:
        enu = l_shaped_flight()
        fusion, _ = make_fusion(enu, true_scale=5.0)
        fusion.add_gps(9999, 10.0, 10.0, 0.0)  # GPS fix with no visual pose: must be ignored
        fusion.add_pose(8888, 1e6, 1e6, 1e6)  # pose with no GPS fix: must be ignored
        self.assertAlmostEqual(fusion.estimate_scale(), 5.0, places=4)

        fusion.gps_readings.clear()
        self.assertIsNone(fusion.estimate_scale())
        self.assertFalse(fusion.is_calibrated(), "a failed re-estimate must not keep a stale scale")

    def test_rejects_bad_input(self) -> None:
        fusion = GPSFusion()
        with self.assertRaises(ValueError):
            fusion.add_gps(0, 91.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            fusion.add_gps(0, float("nan"), 0.0, 0.0)
        with self.assertRaises(ValueError):
            GPSFusion(method="magic")

    def test_load_gps_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gps.csv"
            path.write_text("frame_id,lat,lon,alt\n0,23.8,90.4,10\n5,23.8001,90.4001,\n", encoding="utf-8")
            fusion = GPSFusion()
            self.assertEqual(fusion.load_gps_csv(path), 2)
            self.assertEqual(fusion.gps_readings[1], (5, 23.8001, 90.4001, 0.0))

            bad = Path(tmp) / "bad.csv"
            bad.write_text("frame,latitude\n0,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                fusion.load_gps_csv(bad)
            with self.assertRaises(FileNotFoundError):
                fusion.load_gps_csv(Path(tmp) / "missing.csv")


if __name__ == "__main__":
    unittest.main()
