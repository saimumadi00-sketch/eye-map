"""Functional tests for object detection decoding, directions, motion, map clusters, layout and exports."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.objects.clusters import cluster_map_points
from src.objects.detector import COCO_CLASSES, Detection, OnnxDetector, decode_yolo_output, letterbox, load_class_names
from src.objects.directions import (
    ObjectTracker,
    box_iou,
    direction_label,
    elevation_label,
    locate_in_box,
    pixel_bearing,
    point_bearing,
)
from src.objects.layout import BACKGROUND, CAMERA_COLOR, draw_layout
from src.objects.scene import SceneAnalyzer
from src.pose_estimator import make_intrinsics

K = make_intrinsics(640, 480)  # fx = fy = 576, principal point (320, 240)
NAMES = ["person", "car", "dog"]


def yolo_rows(rows: list[tuple[float, float, float, float, int, float]], n_classes: int = 3) -> np.ndarray:
    """Build (N, 4 + classes) rows from (cx, cy, w, h, class_id, score)."""
    out = np.zeros((len(rows), 4 + n_classes), dtype=np.float32)
    for i, (cx, cy, w, h, class_id, score) in enumerate(rows):
        out[i, :4] = (cx, cy, w, h)
        out[i, 4 + class_id] = score
    return out


def project(point_cam: np.ndarray) -> tuple[float, float]:
    return (
        float(K[0, 0] * point_cam[0] / point_cam[2] + K[0, 2]),
        float(K[1, 1] * point_cam[1] / point_cam[2] + K[1, 2]),
    )


def box_around(u: float, v: float, half: float = 30.0) -> tuple[float, float, float, float]:
    return (u - half, v - half, u + half, v + half)


def pose(center: np.ndarray) -> np.ndarray:
    """World-to-camera pose of an unrotated camera at `center`."""
    T = np.eye(4)
    T[:3, 3] = -np.asarray(center, dtype=np.float64)
    return T


class FakeNet:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.blob = None

    def setInput(self, blob) -> None:
        self.blob = blob

    def forward(self) -> np.ndarray:
        return self.output


class ScriptedDetector:
    """Stand-in for OnnxDetector: returns the detections scripted for each call."""

    def __init__(self, per_call: list[list[Detection]]) -> None:
        self.per_call = per_call
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        detections = self.per_call[min(self.calls, len(self.per_call) - 1)]
        self.calls += 1
        return detections


class TestDetectorDecoding(unittest.TestCase):
    def test_letterbox_keeps_aspect_and_centers(self) -> None:
        padded, scale, pad = letterbox(np.zeros((480, 640, 3), dtype=np.uint8), 640)
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertEqual((scale, pad), (1.0, (0, 80)))
        self.assertTrue((padded[:80] == 114).all() and (padded[80:560] == 0).all())

    def test_decodes_yolov8_layout_with_nms_and_letterbox_undo(self) -> None:
        # Boxes are in 640x640 letterboxed pixels of a 1280x960 frame: scale 0.5, pad (0, 80).
        rows = yolo_rows([
            (320, 320, 100, 200, 0, 0.90),  # person
            (324, 322, 100, 200, 0, 0.60),  # same person, suppressed by NMS
            (100, 200, 60, 40, 1, 0.80),  # car
            (500, 500, 50, 50, 2, 0.10),  # below the confidence threshold
        ])
        output = rows.T[np.newaxis]  # (1, 4 + classes, N) as exported by YOLOv8/YOLO11
        detections = decode_yolo_output(output, NAMES, 0.5, (0, 80), (1280, 960))
        self.assertEqual([(d.label, round(d.confidence, 2)) for d in detections], [("person", 0.9), ("car", 0.8)])
        np.testing.assert_allclose(detections[0].box, (540, 280, 740, 680), atol=1e-3)
        np.testing.assert_allclose(detections[0].center, (640, 480), atol=1e-3)
        self.assertAlmostEqual(detections[0].area, 200 * 400, delta=1.0)

    def test_overlapping_boxes_of_different_classes_both_survive(self) -> None:
        rows = yolo_rows([(320, 320, 100, 100, 0, 0.9), (320, 320, 100, 100, 2, 0.8)])
        detections = decode_yolo_output(rows, NAMES, 1.0, (0, 0), (640, 640))
        self.assertEqual([d.label for d in detections], ["person", "dog"])

    def test_decodes_yolov5_layout_with_objectness(self) -> None:
        rows = np.zeros((2, 5 + 3), dtype=np.float32)
        rows[0] = (320, 320, 100, 100, 0.9, 0.1, 0.8, 0.0)  # car: 0.9 * 0.8 = 0.72
        rows[1] = (100, 100, 50, 50, 0.3, 0.9, 0.0, 0.0)  # person: 0.27, below threshold
        detections = decode_yolo_output(rows[np.newaxis], NAMES, 1.0, (0, 0), (640, 640))
        self.assertEqual([(d.label, round(d.confidence, 2)) for d in detections], [("car", 0.72)])

    def test_boxes_are_clipped_and_empty_output_is_fine(self) -> None:
        rows = yolo_rows([(10, 10, 100, 100, 0, 0.9)])
        (detection,) = decode_yolo_output(rows, NAMES, 1.0, (0, 0), (640, 480))
        np.testing.assert_allclose(detection.box, (0, 0, 60, 60), atol=1e-3)
        self.assertEqual(decode_yolo_output(yolo_rows([]), NAMES, 1.0, (0, 0), (640, 480)), [])

    def test_class_count_mismatch_is_explained(self) -> None:
        with self.assertRaisesRegex(ValueError, "--detect-classes"):
            decode_yolo_output(np.zeros((1, 84, 100), dtype=np.float32), NAMES, 1.0, (0, 0), (640, 480))

    def test_detector_runs_net_on_letterboxed_blob(self) -> None:
        rows = yolo_rows([(320, 320, 100, 200, 0, 0.9)], n_classes=80)
        net = FakeNet(rows.T[np.newaxis])
        detector = OnnxDetector("unused.onnx", net=net)
        (detection,) = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(net.blob.shape, (1, 3, 640, 640))
        self.assertEqual(detection.label, COCO_CLASSES[0])
        np.testing.assert_allclose(detection.box, (270, 140, 370, 340), atol=1e-3)  # pad_y 80 removed

    def test_missing_files_raise(self) -> None:
        with self.assertRaises(FileNotFoundError):
            OnnxDetector("does/not/exist.onnx")
        with tempfile.TemporaryDirectory() as tmp:
            names = Path(tmp) / "classes.txt"
            names.write_text("drone\n\n tree \n", encoding="utf-8")
            self.assertEqual(load_class_names(names), ["drone", "tree"])
            with self.assertRaises(FileNotFoundError):
                load_class_names(Path(tmp) / "missing.txt")


class TestDirections(unittest.TestCase):
    def test_pixel_bearing(self) -> None:
        self.assertEqual(pixel_bearing(320, 240, K), (0.0, 0.0))
        azimuth, elevation = pixel_bearing(320 + 576, 240, K)  # one focal length right of center
        self.assertAlmostEqual(azimuth, 45.0)
        self.assertAlmostEqual(elevation, 0.0)
        azimuth, elevation = pixel_bearing(320, 240 - 576, K)  # one focal length above center
        self.assertAlmostEqual(azimuth, 0.0)
        self.assertAlmostEqual(elevation, 45.0)
        self.assertLess(pixel_bearing(0, 240, K)[0], 0.0, "left of center is negative azimuth")

    def test_point_bearing_covers_full_circle(self) -> None:
        cases = {
            (0, 0, 5): (0.0, 0.0, 5.0),
            (5, 0, 0): (90.0, 0.0, 5.0),
            (-5, 0, 0): (-90.0, 0.0, 5.0),
            (0, 0, -5): (180.0, 0.0, 5.0),
            (0, -5, 0): (0.0, 90.0, 5.0),  # y is down, so -y is above
        }
        for point, expected in cases.items():
            np.testing.assert_allclose(point_bearing(np.array(point, dtype=float)), expected, atol=1e-9, err_msg=str(point))
        self.assertEqual(point_bearing(np.zeros(3)), (0.0, 0.0, 0.0))

    def test_pixel_and_point_bearings_agree(self) -> None:
        point = np.array([1.5, -0.7, 6.0])
        np.testing.assert_allclose(pixel_bearing(*project(point), K), point_bearing(point)[:2], atol=1e-9)

    def test_direction_labels(self) -> None:
        expected = {
            0: "ahead", -4: "ahead", 10: "slightly right", -20: "slightly left", 45: "ahead-right",
            -60: "ahead-left", 90: "right", -100: "left", 135: "behind-right", -150: "behind-left",
            180: "behind", -175: "behind", 370: "slightly right",
        }
        for azimuth, label in expected.items():
            self.assertEqual(direction_label(azimuth), label, azimuth)
        self.assertEqual([elevation_label(e) for e in (30, 0, -30)], ["above", "level", "below"])

    def test_locate_in_box_prefers_the_object_over_background(self) -> None:
        rng = np.random.default_rng(3)
        target = np.array([1.0, 0.2, 5.0])
        near = target + rng.normal(0, 0.03, (12, 3))
        background = np.column_stack([rng.uniform(-4, 6, 200), rng.uniform(-3, 3, 200), np.full(200, 20.0)])
        box = box_around(*project(target), half=40)
        position, support = locate_in_box(box, np.vstack([near, background]), K)
        self.assertGreaterEqual(support, 12)
        np.testing.assert_allclose(position, target, atol=0.15)

        self.assertEqual(locate_in_box(box, np.zeros((0, 3)), K), (None, 0))
        behind = near * np.array([1, 1, -1])
        self.assertEqual(locate_in_box(box, behind, K), (None, 0))
        position, support = locate_in_box(box, near[:2], K)
        self.assertIsNone(position, "two points are too few to trust")
        self.assertEqual(support, 2)

    def test_box_iou(self) -> None:
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


class TestObjectTracker(unittest.TestCase):
    def det(self, label: str, u: float, v: float, half: float = 30.0) -> Detection:
        return Detection(label, 0, 0.9, box_around(u, v, half))

    def test_ids_persist_per_object_and_class(self) -> None:
        tracker = ObjectTracker()
        first = tracker.update(0, [self.det("person", 100, 200), self.det("car", 400, 200)])
        second = tracker.update(1, [self.det("car", 405, 200), self.det("person", 104, 200)])
        self.assertEqual([t.track_id for t in first], [1, 2])
        self.assertEqual([t.track_id for t in second], [2, 1], "matched by overlap, not by order")
        # A dog exactly where the person was is a different object.
        (dog,) = tracker.update(2, [self.det("dog", 104, 200)])
        self.assertEqual(dog.track_id, 3)

    def test_track_is_retired_after_a_long_gap(self) -> None:
        tracker = ObjectTracker(max_missed_frames=5)
        (a,) = tracker.update(0, [self.det("person", 100, 200)])
        (b,) = tracker.update(4, [self.det("person", 100, 200)])
        (c,) = tracker.update(20, [self.det("person", 100, 200)])
        self.assertEqual((a.track_id, b.track_id), (1, 1))
        self.assertEqual(c.track_id, 2)
        self.assertEqual(sorted(tracker.tracks), [1, 2], "retired tracks stay available for summaries")

    def test_apparent_motion_and_range_trend(self) -> None:
        tracker = ObjectTracker()
        for frame in range(6):
            (track,) = tracker.update(frame, [self.det("person", 100 + 8 * frame, 200, half=30 + 2 * frame)])
        motion = tracker.motion(track)
        self.assertEqual(motion.apparent, "right")
        np.testing.assert_allclose(motion.apparent_px_per_frame, (8.0, 0.0), atol=1e-6)
        self.assertEqual(motion.range_trend, "approaching")
        self.assertIsNone(motion.world, "no depth, so no world motion")

        still = ObjectTracker()
        for frame in range(6):
            (track,) = still.update(frame, [self.det("car", 300, 200)])
        motion = still.motion(track)
        self.assertEqual((motion.apparent, motion.range_trend), ("still", "steady"))

        up = ObjectTracker()
        for frame in range(6):
            (track,) = up.update(frame, [self.det("bird", 300, 300 - 9 * frame, half=30 - 2 * frame)])
        motion = up.motion(track)
        self.assertEqual((motion.apparent, motion.range_trend), ("up", "receding"))

        self.assertEqual(ObjectTracker().motion(track).apparent, "up")
        (single,) = ObjectTracker().update(0, [self.det("car", 1, 1)])
        self.assertEqual(tracker.motion(single).apparent, "unknown")

    def test_world_motion_is_relative_to_the_camera_view(self) -> None:
        T_cw = pose(np.zeros(3))
        cases = {"toward": (0, 0, -0.5), "away": (0, 0, 0.5), "right": (0.5, 0, 0), "left": (-0.5, 0, 0), "stationary": (0.001, 0, 0)}
        for expected, velocity in cases.items():
            tracker = ObjectTracker()
            for frame in range(6):
                world = np.array([0.0, 0.0, 10.0]) + frame * np.array(velocity)
                (track,) = tracker.update(frame, [self.det("person", *project(world))], [world])
            motion = tracker.motion(track, T_cw)
            self.assertEqual(motion.world, expected)
            np.testing.assert_allclose(motion.world_velocity, velocity, atol=1e-9)

        # Same world motion (+x) seen by a camera turned 180 degrees about y is motion to its left.
        turned = np.diag([-1.0, 1.0, -1.0, 1.0])
        tracker = ObjectTracker()
        for frame in range(6):
            world = np.array([0.0, 0.0, -10.0]) + frame * np.array([0.5, 0.0, 0.0])
            (track,) = tracker.update(frame, [self.det("person", 320, 240)], [world])
        self.assertEqual(tracker.motion(track, turned).world, "left")


class TestMapClusters(unittest.TestCase):
    def test_separates_blobs_and_drops_noise(self) -> None:
        rng = np.random.default_rng(1)
        big = rng.normal([0, 0, 10], [0.5, 1.0, 0.5], (400, 3))
        small = rng.normal([12, 0, 14], [0.4, 1.0, 0.4], (150, 3))
        noise = rng.uniform([-10, -2, 0], [25, 2, 30], (40, 3))
        clusters = cluster_map_points(np.vstack([big, small, noise, [[np.nan, 0, 0]]]), cell=0.5)
        self.assertEqual([c.cluster_id for c in clusters], [1, 2])
        self.assertGreater(clusters[0].count, clusters[1].count)
        np.testing.assert_allclose(clusters[0].centroid, [0, 0, 10], atol=0.3)
        np.testing.assert_allclose(clusters[1].centroid, [12, 0, 14], atol=0.3)
        self.assertTrue((clusters[0].minimum <= clusters[0].centroid).all() and (clusters[0].centroid <= clusters[0].maximum).all())

    def test_default_cell_and_limits(self) -> None:
        rng = np.random.default_rng(2)
        blobs = [rng.normal([8 * i, 0, 10], 0.3, (100 + 10 * i, 3)) for i in range(5)]
        clusters = cluster_map_points(np.vstack(blobs), max_clusters=3)
        self.assertEqual(len(clusters), 3)
        self.assertEqual([c.count for c in clusters], sorted((c.count for c in clusters), reverse=True))
        self.assertEqual(cluster_map_points(np.zeros((5, 3))), [])
        self.assertEqual(cluster_map_points(np.zeros((0, 3))), [])
        with self.assertRaises(ValueError):
            cluster_map_points(np.vstack(blobs), cell=0.0)


class TestLayout(unittest.TestCase):
    def test_draws_camera_objects_and_handles_empty_input(self) -> None:
        empty = draw_layout([], np.zeros((0, 3)))
        self.assertEqual(empty.shape, (800, 800, 3))

        rng = np.random.default_rng(4)
        points = rng.normal([0, 0, 10], [3, 1, 1], (500, 3))
        trajectory = [np.array([x, 0.0, 0.0]) for x in np.linspace(-4, 4, 20)]
        objects = [
            {"track_id": 1, "label": "person", "position": np.array([2.0, 0.0, 8.0]), "velocity": (0.3, 0.0, 0.0)},
            {"track_id": 2, "label": "car", "position": None},
        ]
        image = draw_layout(trajectory, points, pose(trajectory[-1]), cluster_map_points(points), objects)
        self.assertEqual(image.shape, (800, 800, 3))
        self.assertGreater(int(np.count_nonzero((image != BACKGROUND).any(axis=2))), 800)
        # The camera sits at the right end of the path, below the map (z up means smaller row index).
        camera_pixels = np.argwhere((image == CAMERA_COLOR).all(axis=2))
        self.assertGreater(len(camera_pixels), 20)
        self.assertGreater(camera_pixels[:, 1].mean(), 400)
        self.assertGreater(camera_pixels[:, 0].mean(), 400)

        huge = draw_layout([np.zeros(3), np.array([1e30, 0, np.nan])], points)
        self.assertEqual(huge.shape, (800, 800, 3))


class TestSceneAnalyzer(unittest.TestCase):
    """A person stands still at a known place while the camera moves; a car has no depth."""

    def setUp(self) -> None:
        rng = np.random.default_rng(6)
        self.person = np.array([2.0, 0.0, 10.0])
        self.map_chunks = [
            self.person + rng.normal(0, 0.05, (60, 3)),
            np.column_stack([rng.uniform(-8, 8, 300), rng.uniform(-2, 2, 300), rng.normal(25, 0.3, 300)]),
        ]
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)

    def run_frames(self, analyzer: SceneAnalyzer, centers: list[np.ndarray]) -> np.ndarray:
        for index, center in enumerate(centers):
            analyzer.process(index + 1, index + 1, self.image, K, pose(center), self.map_chunks)
        return pose(centers[-1])

    def scripted(self, centers: list[np.ndarray]) -> ScriptedDetector:
        calls = []
        for center in centers:
            u, v = project(self.person - center)
            calls.append([Detection("person", 0, 0.9, box_around(u, v, 35)), Detection("car", 2, 0.8, (500, 40, 560, 80))])
        return ScriptedDetector(calls)

    def test_reports_direction_distance_and_motion(self) -> None:
        centers = [np.array([0.2 * i, 0.0, 0.0]) for i in range(8)]
        analyzer = SceneAnalyzer(self.scripted(centers))
        T_final = self.run_frames(analyzer, centers)

        person, car = analyzer.latest
        relative = self.person - centers[-1]
        azimuth, _, distance = point_bearing(relative)
        self.assertAlmostEqual(person.azimuth_deg, azimuth, delta=0.01)
        self.assertEqual(person.direction, "ahead")  # 0.6 right over 10 forward = 3.4 degrees
        self.assertAlmostEqual(person.distance, distance, delta=0.2)
        np.testing.assert_allclose(person.position_world, self.person, atol=0.2)
        # The camera slides right, so the person drifts left across the image yet is not moving.
        self.assertEqual(person.motion.apparent, "left")
        self.assertEqual(person.motion.world, "stationary")
        self.assertIsNone(car.distance)
        self.assertIsNone(car.motion.world)
        self.assertEqual(car.elevation, "above")

        summaries = {s["label"]: s for s in analyzer.object_summaries(T_final, scale_m=2.0)}
        self.assertEqual(summaries["person"]["detections"], 8)
        self.assertAlmostEqual(summaries["person"]["from_final_camera"]["distance_m"], 2.0 * distance, delta=0.5)
        self.assertIsNone(summaries["car"]["from_final_camera"])
        self.assertEqual([o["label"] for o in analyzer.layout_objects(T_final)], ["person"])
        self.assertIsNone(analyzer.layout_objects(T_final)[0]["velocity"], "no arrow for a stationary object")

    def test_first_frame_direction_is_slightly_right(self) -> None:
        analyzer = SceneAnalyzer(self.scripted([np.zeros(3)]))
        self.run_frames(analyzer, [np.zeros(3)])
        self.assertEqual(analyzer.latest[0].direction, "slightly right")  # atan(2 / 10) = 11.3 degrees
        self.assertEqual(analyzer.object_summaries(pose(np.zeros(3))), [], "one sighting is below min_hits")

    def test_detect_interval_skips_frames(self) -> None:
        centers = [np.array([0.2 * i, 0.0, 0.0]) for i in range(8)]
        detector = self.scripted(centers)
        analyzer = SceneAnalyzer(detector, detect_interval=4)
        self.run_frames(analyzer, centers)
        self.assertEqual(detector.calls, 2)  # indices 4 and 8
        self.assertEqual(len(analyzer.observations), 4)

    def test_without_detector_only_clusters_and_layout_are_exported(self) -> None:
        analyzer = SceneAnalyzer()
        self.assertEqual(analyzer.process(1, 1, self.image, K, np.eye(4), self.map_chunks), [])
        with tempfile.TemporaryDirectory() as tmp:
            paths = analyzer.export([np.zeros(3), np.array([1.0, 0, 0])], np.vstack(self.map_chunks), np.eye(4), Path(tmp) / "objects", Path(tmp) / "plots" / "layout.png")
            self.assertEqual(sorted(paths), ["clusters", "layout", "scene"])
            scene = json.loads(paths["scene"].read_text())
            self.assertFalse(scene["detector_enabled"])
            self.assertEqual(scene["objects"], [])
            nearest = min(scene["clusters"], key=lambda c: c["distance"])
            self.assertAlmostEqual(nearest["distance"], float(np.linalg.norm(self.person)), delta=0.3)
            self.assertEqual(nearest["direction"], "slightly right")
            self.assertIsNotNone(cv2.imread(str(paths["layout"])))

    def test_export_writes_objects_csv_scene_json_and_annotated_frame(self) -> None:
        centers = [np.array([0.2 * i, 0.0, 0.0]) for i in range(8)]
        analyzer = SceneAnalyzer(self.scripted(centers))
        T_final = self.run_frames(analyzer, centers)
        with tempfile.TemporaryDirectory() as tmp:
            paths = analyzer.export(centers, np.vstack(self.map_chunks), T_final, tmp, Path(tmp) / "layout.png", scale_m=2.0)
            with paths["objects"].open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 16)
            last_person = [r for r in rows if r["label"] == "person"][-1]
            self.assertEqual((last_person["direction"], last_person["apparent_motion"], last_person["world_motion"]), ("ahead", "left", "stationary"))
            self.assertEqual([r for r in rows if r["label"] == "car"][-1]["distance"], "")
            scene = json.loads(paths["scene"].read_text())
            self.assertEqual(scene["scale_m_per_unit"], 2.0)
            self.assertEqual({o["label"] for o in scene["objects"]}, {"person", "car"})
            with paths["clusters"].open() as f:
                self.assertGreaterEqual(len(list(csv.DictReader(f))), 2)

        annotated = SceneAnalyzer.annotate(self.image, analyzer.latest)
        self.assertEqual(annotated.shape, self.image.shape)
        self.assertGreater(int(np.count_nonzero(annotated)), 0)
        self.assertEqual(int(np.count_nonzero(self.image)), 0, "the input frame is not modified")


if __name__ == "__main__":
    unittest.main()
