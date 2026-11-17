"""Exercise mapping outputs while forbidding all OpenCV window calls."""

import argparse
import csv
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import main
from src.core.calibration import collect_calib_frames
from src.data.vkitti_adapter import adapt_sequence_to_live_mvp
from src.mvp import live_mvp
from src.integration.gps_fusion import haversine_m
from src.pose_estimator import make_intrinsics
from tests.synthetic import HEIGHT, WIDTH, moving_scene_frames


class SyntheticSequence:
    scene = "synthetic"
    variant = "stationary"

    def __init__(self):
        rng = np.random.default_rng(42)
        self.frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {
            "frame_id": index, "rgb": self.frame.copy(),
            "K": make_intrinsics(320, 240), "T_cw": np.eye(4),
        }


class TestRunners(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for function in ("imshow", "waitKey", "destroyAllWindows"):
            self.stack.enter_context(patch.object(cv2, function, side_effect=AssertionError("GUI called")))

    def assert_stationary_outputs(self, run_dir, trajectory_path):
        with trajectory_path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1, "stationary input must not accumulate motion")
        metadata = json.loads((run_dir / "keyframes.json").read_text())
        self.assertEqual(len(metadata), 3)
        for item in metadata:
            np.testing.assert_allclose(np.array(item["T_cw"]).reshape(4, 4), np.eye(4))

    def make_video(self):
        path = self.root / "stationary.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
        self.assertTrue(writer.isOpened())
        frame = SyntheticSequence().frame
        try:
            for _ in range(3):
                writer.write(frame)
        finally:
            writer.release()
        return path

    def test_vkitti_headless_rejects_stationary_motion(self):
        run = adapt_sequence_to_live_mvp(
            SyntheticSequence(), save_every=1, output_root=self.root, display_enabled=False,
        )
        self.assert_stationary_outputs(run, run / "trajectory.csv")

    def test_calibration_headless_reads_video_without_windows(self):
        frames = collect_calib_frames(str(self.make_video()), n_frames=1, display_enabled=False)
        self.assertEqual(frames, [])

    def test_main_headless_rejects_stationary_motion(self):
        config = self.root / "config.yaml"
        config.write_text(f'keyframe_interval: 1\noutput:\n  root: "{self.root.as_posix()}/main"\n')
        args = argparse.Namespace(
            config=str(config), source="video", path=str(self.make_video()), no_display=True,
            max_frames=0, save_keyframes=False, save_matches=False, export_pointcloud=False,
            mode="trajectory", show_pointcloud=False,
        )
        self.assertEqual(main.run_pipeline(args), 0)
        run = next((self.root / "main").iterdir())
        self.assert_stationary_outputs(run, run / "trajectory" / "trajectory.csv")

    def test_main_recovers_metric_scale_from_gps_and_reports_performance(self):
        frames = moving_scene_frames(60)
        video = self.root / "moving.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (WIDTH, HEIGHT))
        self.assertTrue(writer.isOpened())
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

        # The drone flies due east at 0.5 m per frame from a fix in Dhaka.
        lat0, lon0, metres_per_frame = 23.8151, 90.4255, 0.5
        metres_per_deg_lon = haversine_m(lat0, lon0, lat0, lon0 + 1.0)
        gps = self.root / "gps.csv"
        with gps.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "lat", "lon", "alt"])
            for frame_id in range(len(frames)):
                writer.writerow([frame_id, lat0, lon0 + frame_id * metres_per_frame / metres_per_deg_lon, 30.0])

        config = self.root / "config.yaml"
        config.write_text(f'output:\n  root: "{self.root.as_posix()}/gps"\n')
        args = argparse.Namespace(
            config=str(config), source="video", path=str(video), no_display=True,
            max_frames=0, save_keyframes=False, save_matches=False, export_pointcloud=False,
            mode="trajectory", show_pointcloud=False, gps_csv=str(gps), auto_tune=True, target_fps=1e-3,
        )
        cv2.setRNGSeed(1)
        self.assertEqual(main.run_pipeline(args), 0)
        run = next((self.root / "gps").iterdir())

        with (run / "trajectory" / "trajectory_metric_enu.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 10)
        for row in rows:  # east = distance flown, north/up stay near zero
            expected_east = int(row["frame_id"]) * metres_per_frame
            self.assertAlmostEqual(float(row["x"]), expected_east, delta=3.0)
            self.assertAlmostEqual(float(row["y"]), 0.0, delta=3.0)
            self.assertAlmostEqual(float(row["z"]), 0.0, delta=3.0)

        log = (run / "logs" / "run.log").read_text()
        self.assertIn("gps_scale_m_per_unit=", log)
        final_budget = int(log.split("orb_features_final=")[1].split()[0])
        self.assertGreater(final_budget, 2000, "a trivially met frame-rate target grows the ORB budget")
        self.assertIn("profile:", log)
        for stage in ("detect", "track", "render"):
            self.assertIn(stage, log)

    def test_main_reports_when_gps_scale_is_unavailable(self):
        gps = self.root / "gps.csv"
        gps.write_text("frame_id,lat,lon\n0,23.8,90.4\n1,23.8,90.4\n2,23.8,90.4\n")
        config = self.root / "config.yaml"
        config.write_text(f'output:\n  root: "{self.root.as_posix()}/nogps"\ngps_csv: "{gps.as_posix()}"\n')
        args = argparse.Namespace(
            config=str(config), source="video", path=str(self.make_video()), no_display=True,
            max_frames=0, save_keyframes=False, save_matches=False, export_pointcloud=False,
            mode="trajectory", show_pointcloud=False,
        )
        with self.assertLogs("main", level="WARNING"):
            self.assertEqual(main.run_pipeline(args), 0)
        run = next((self.root / "nogps").iterdir())
        self.assertIn("gps_scale=unavailable", (run / "logs" / "run.log").read_text())
        self.assertFalse((run / "trajectory" / "trajectory_metric_enu.csv").exists())

    def test_legacy_headless_rejects_stationary_motion(self):
        with patch("sys.argv", [
            "live_mvp", "--source", str(self.make_video()), "--no-display", "--save-every", "1",
            "--output-root", str(self.root / "legacy"),
        ]):
            live_mvp.main()
        run = next((self.root / "legacy").iterdir())
        self.assert_stationary_outputs(run, run / "trajectory.csv")
