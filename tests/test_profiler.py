"""Functional tests for stage profiling, ORB feature auto-tuning, and their use in the tracking loop."""

from __future__ import annotations

import unittest

import cv2

from src.pipeline import track_sequence
from src.pose_estimator import make_intrinsics
from src.profiler import FeatureAutoTuner, StageProfiler
from src.tracker import MonocularTracker
from tests.synthetic import HEIGHT, STEP, WIDTH, moving_scene_frames


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestStageProfiler(unittest.TestCase):
    def test_accumulates_stages_and_frame_rate(self) -> None:
        clock = FakeClock()
        profiler = StageProfiler(clock=clock)
        for detect_s, track_s in ((0.010, 0.030), (0.020, 0.040)):
            with profiler.stage("read", in_frame=False):
                clock.now += 1.0  # waiting on the camera must not count as processing time
            profiler.start_frame()
            with profiler.stage("detect"):
                clock.now += detect_s
            with profiler.stage("track"):
                clock.now += track_s
            self.assertAlmostEqual(profiler.end_frame(), detect_s + track_s)

        summary = profiler.summary()
        self.assertEqual(summary["frames"], 2)
        self.assertAlmostEqual(summary["fps"], 2 / 0.100)
        self.assertAlmostEqual(summary["mean_frame_ms"], 50.0)
        self.assertAlmostEqual(summary["stages"]["detect"]["mean_ms"], 15.0)
        self.assertAlmostEqual(summary["stages"]["detect"]["max_ms"], 20.0)
        self.assertAlmostEqual(summary["stages"]["track"]["share"], 0.7)
        self.assertIsNone(summary["stages"]["read"]["share"])

        lines = profiler.report().splitlines()
        self.assertIn("fps=20.0", lines[0])
        self.assertEqual([line.split()[0] for line in lines[2:]], ["track", "detect", "read"])

    def test_stage_time_recorded_when_stage_raises(self) -> None:
        clock = FakeClock()
        profiler = StageProfiler(clock=clock)
        with self.assertRaises(RuntimeError), profiler.stage("boom"):
            clock.now += 0.5
            raise RuntimeError
        self.assertAlmostEqual(profiler.stages["boom"].total_s, 0.5)

    def test_empty_profiler_reports_zeros(self) -> None:
        profiler = StageProfiler()
        self.assertEqual(profiler.fps, 0.0)
        self.assertEqual(profiler.end_frame(), 0.0)
        self.assertIn("frames=0", profiler.report())


class TestFeatureAutoTuner(unittest.TestCase):
    def run_tuner(self, tuner: FeatureAutoTuner, frame_s: float, matches: int, frames: int = 200) -> int:
        for _ in range(frames):
            tuner.update(frame_s, matches)
        return tuner.nfeatures

    def test_slow_frames_with_plenty_of_matches_shrink_budget_to_floor(self) -> None:
        tuner = FeatureAutoTuner(target_fps=20, initial=2000, min_features=500, min_matches=60)
        self.assertEqual(self.run_tuner(tuner, frame_s=0.2, matches=400), 500)

    def test_fast_frames_grow_budget_to_ceiling(self) -> None:
        tuner = FeatureAutoTuner(target_fps=20, initial=1000, max_features=3000)
        self.assertEqual(self.run_tuner(tuner, frame_s=0.01, matches=400), 3000)

    def test_low_matches_grow_budget_even_when_slow(self) -> None:
        tuner = FeatureAutoTuner(target_fps=20, initial=1000, max_features=3000, min_matches=60)
        self.assertEqual(self.run_tuner(tuner, frame_s=0.2, matches=65), 3000)

    def test_slow_frames_do_not_shrink_when_matches_are_marginal(self) -> None:
        tuner = FeatureAutoTuner(target_fps=20, initial=1000, min_matches=60)
        self.assertEqual(self.run_tuner(tuner, frame_s=0.2, matches=100), 1000)

    def test_on_target_frames_hold_steady(self) -> None:
        tuner = FeatureAutoTuner(target_fps=20, initial=1500)
        self.assertEqual(self.run_tuner(tuner, frame_s=0.05, matches=400), 1500)
        self.assertEqual(tuner.adjustments, 0)

    def test_converges_without_oscillating_when_cost_tracks_budget(self) -> None:
        # Frame time proportional to the budget: 2000 features -> 100 ms; target is 50 ms.
        tuner = FeatureAutoTuner(target_fps=20, initial=2000, cooldown_frames=5)
        history = []
        for _ in range(400):
            history.append(tuner.update(tuner.nfeatures * 0.05 / 1000.0, 400))
        self.assertEqual(len(set(history[-100:])), 1, "settles on one value")
        self.assertLessEqual(tuner.nfeatures * 0.05 / 1000.0, 1.1 * 0.05)
        self.assertGreaterEqual(tuner.nfeatures * 0.05 / 1000.0, 0.8 * 0.05)

    def test_cooldown_spaces_out_adjustments(self) -> None:
        tuner = FeatureAutoTuner(target_fps=20, initial=2000, cooldown_frames=10)
        self.run_tuner(tuner, frame_s=0.2, matches=400, frames=30)
        self.assertEqual(tuner.adjustments, 2)  # frames 11 and 22

    def test_initial_is_clamped_and_bad_config_rejected(self) -> None:
        self.assertEqual(FeatureAutoTuner(initial=10, min_features=500).nfeatures, 500)
        self.assertEqual(FeatureAutoTuner(initial=99999, max_features=4000).nfeatures, 4000)
        for kwargs in ({"target_fps": 0}, {"min_features": 0}, {"min_features": 900, "max_features": 800}, {"smoothing": 0}):
            with self.assertRaises(ValueError):
                FeatureAutoTuner(**kwargs)


class TestTrackingLoopPerformanceHooks(unittest.TestCase):
    """Run the real loop on a synthetic moving scene."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.images = moving_scene_frames(60)
        cls.K = make_intrinsics(WIDTH, HEIGHT)

    def frames(self):
        return ((index, image, self.K) for index, image in enumerate(self.images))

    def test_profiles_every_stage_and_tracks_the_motion(self) -> None:
        cv2.setRNGSeed(1)
        run = track_sequence(self.frames(), MonocularTracker(), display_enabled=False)
        summary = run.profiler.summary()
        self.assertEqual(summary["frames"], len(self.images) - 1)
        self.assertGreater(summary["fps"], 0.0)
        self.assertEqual(summary["stages"]["read"]["calls"], len(self.images))  # includes end-of-input
        for name in ("detect", "track", "render", "display"):
            self.assertEqual(summary["stages"][name]["calls"], len(self.images) - 1, name)
        self.assertEqual(run.nfeatures, 2000)

        # The camera slides along +x in equal steps: the path must head that way, evenly spaced.
        # Tolerances are loose because near-planar scenes occasionally yield a skewed step.
        frame_ids = [row[0] for row in run.trajectory_rows]
        xs = [row[1] for row in run.trajectory_rows]
        self.assertGreaterEqual(len(xs), 10)
        self.assertGreater(xs[-1], 0.0)
        units_per_step = xs[-1] / (frame_ids[-1] * STEP)
        for frame_id, row in zip(frame_ids, run.trajectory_rows):
            self.assertAlmostEqual(row[1], units_per_step * frame_id * STEP, delta=0.2 * xs[-1])
            self.assertLess(abs(row[2]) + abs(row[3]), 0.2 * xs[-1])

    def test_max_frames_limit_still_applies(self) -> None:
        run = track_sequence(self.frames(), MonocularTracker(), max_frames=5, display_enabled=False)
        self.assertEqual(run.frames_processed, 5)
        self.assertEqual(run.profiler.frames, 4)
        one = track_sequence(self.frames(), MonocularTracker(), max_frames=1, display_enabled=False)
        self.assertEqual((one.frames_processed, one.profiler.frames), (1, 0))

    def test_tuner_drives_the_orb_budget(self) -> None:
        cv2.setRNGSeed(1)
        # An unreachable frame rate with plentiful matches must walk the budget down to its floor.
        tuner = FeatureAutoTuner(target_fps=1e6, initial=2000, min_features=800, cooldown_frames=1)
        run = track_sequence(self.frames(), MonocularTracker(), max_frames=30, display_enabled=False, tuner=tuner)
        self.assertEqual(run.nfeatures, 800)
        self.assertGreaterEqual(len(run.trajectory_rows), 8, "tracking survives the smaller budget")

        cv2.setRNGSeed(1)
        relaxed = FeatureAutoTuner(target_fps=1e-3, initial=1000, max_features=1500, cooldown_frames=1)
        run = track_sequence(self.frames(), MonocularTracker(), max_frames=30, display_enabled=False, tuner=relaxed)
        self.assertEqual(run.nfeatures, 1500)


if __name__ == "__main__":
    unittest.main()
