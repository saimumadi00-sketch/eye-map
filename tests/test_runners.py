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
from src.objects.detector import Detection
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

    def make_moving_video(self, n_frames=60):
        video = self.root / "moving.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (WIDTH, HEIGHT))
        self.assertTrue(writer.isOpened())
        try:
            for frame in moving_scene_frames(n_frames):
                writer.write(frame)
        finally:
            writer.release()
        return video

    def run_main(self, name, video, **overrides):
        config = self.root / f"{name}.yaml"
        config.write_text(f'match_save_interval: 10\noutput:\n  root: "{self.root.as_posix()}/{name}"\n')
        options = dict(
            config=str(config), source="video", path=str(video), no_display=True,
            max_frames=0, save_keyframes=False, save_matches=False, export_pointcloud=False,
            mode="trajectory", show_pointcloud=False,
        )
        options.update(overrides)
        cv2.setRNGSeed(1)
        self.assertEqual(main.run_pipeline(argparse.Namespace(**options)), 0)
        return next((self.root / name).iterdir())

    def test_main_exports_layout_and_map_clusters_without_a_detector(self):
        run = self.run_main("layout", self.make_moving_video())
        layout = cv2.imread(str(run / "plots" / "layout.png"))
        self.assertEqual(layout.shape, (800, 800, 3))
        scene = json.loads((run / "objects" / "scene.json").read_text())
        self.assertFalse(scene["detector_enabled"])
        self.assertEqual(scene["objects"], [])
        self.assertGreaterEqual(len(scene["clusters"]), 1)
        for cluster in scene["clusters"]:  # the camera looks at the scene, so everything is in front
            self.assertLess(abs(cluster["azimuth_deg"]), 90.0)
            self.assertGreater(cluster["distance"], 0.0)
        self.assertTrue((run / "objects" / "clusters.csv").exists())
        self.assertFalse((run / "objects" / "objects.csv").exists())
        self.assertIn("layout=", (run / "logs" / "run.log").read_text())

    def test_main_reports_object_directions_with_a_detector(self):
        created = {}

        class FixedDetector:
            """Reports one object fixed in the image, as if it travelled with the camera."""

            def __init__(self, model_path, class_names=None, input_size=640, conf_threshold=0.35):
                created.update(model=model_path, names=class_names, conf=conf_threshold)

            def detect(self, image):
                return [Detection("tree", 0, 0.8, (400.0, 150.0, 520.0, 330.0))]

        names = self.root / "classes.txt"
        names.write_text("tree\n")
        with patch.object(main, "OnnxDetector", FixedDetector):
            run = self.run_main(
                "objects", self.make_moving_video(), detect_model="model.onnx", detect_classes=str(names),
                detect_conf=0.5, detect_interval=2,
            )
        self.assertEqual(created, {"model": "model.onnx", "names": ["tree"], "conf": 0.5})

        with (run / "objects" / "objects.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 29, "60 frames, detector on every 2nd tracked frame")
        for row in rows:  # box center (460, 240): 140 px right of center at f = 576 px
            self.assertAlmostEqual(float(row["azimuth_deg"]), 13.66, delta=0.01)
            self.assertEqual((row["direction"], row["elevation"], row["track_id"]), ("slightly right", "level", "1"))
        self.assertEqual(rows[-1]["apparent_motion"], "still")
        located = [float(r["distance"]) for r in rows if r["distance"]]
        self.assertGreater(len(located), 20, "the box covers mapped texture, so it gets a distance")

        scene = json.loads((run / "objects" / "scene.json").read_text())
        (tree,) = scene["objects"]
        self.assertEqual((tree["label"], tree["detections"]), ("tree", 29))
        self.assertEqual(tree["from_final_camera"]["direction"], "slightly right")
        self.assertTrue(list((run / "objects").glob("objects_*.png")), "annotated frames are saved")
        self.assertIn("objects", (run / "logs" / "run.log").read_text().split("profile:")[1])

    def test_main_recovers_metric_scale_from_gps_and_reports_performance(self):
        frames = moving_scene_frames(60)
        video = self.make_moving_video()

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
