#!/usr/bin/env python3
"""
Per-stage timing and ORB feature-count auto-tuning for the tracking loop.
`StageProfiler` shows where each frame's time goes; `FeatureAutoTuner` trades ORB feature count against a target frame rate without starving the tracker of matches.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class StageStats:
    """Accumulated wall time for one named stage."""

    calls: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    in_frame: bool = True

    @property
    def mean_ms(self) -> float:
        """Return the mean time per call in milliseconds."""
        return 1000.0 * self.total_s / self.calls if self.calls else 0.0


class StageProfiler:
    """Accumulate wall-clock time per pipeline stage and per frame."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        """Create an empty profiler; `clock` is injectable for tests."""
        self._clock = clock
        self.stages: dict[str, StageStats] = {}
        self.frames = 0
        self.frame_total_s = 0.0
        self.last_frame_s = 0.0
        self._frame_start: float | None = None

    @contextmanager
    def stage(self, name: str, in_frame: bool = True) -> Iterator[None]:
        """Time one stage; time is recorded even if the stage raises.

        Pass in_frame=False for work outside start_frame/end_frame (e.g. waiting on the
        camera), which is reported but not counted toward frame time or fps.
        """
        start = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - start
            stats = self.stages.setdefault(name, StageStats(in_frame=in_frame))
            stats.calls += 1
            stats.total_s += elapsed
            stats.max_s = max(stats.max_s, elapsed)

    def start_frame(self) -> None:
        """Mark the start of one frame's processing."""
        self._frame_start = self._clock()

    def end_frame(self) -> float:
        """Mark the end of the frame started by `start_frame` and return its duration in seconds."""
        if self._frame_start is None:
            return 0.0
        self.last_frame_s = self._clock() - self._frame_start
        self._frame_start = None
        self.frames += 1
        self.frame_total_s += self.last_frame_s
        return self.last_frame_s

    @property
    def fps(self) -> float:
        """Return the mean processing rate over all completed frames."""
        return self.frames / self.frame_total_s if self.frame_total_s > 0.0 else 0.0

    def summary(self) -> dict:
        """Return frame rate plus per-stage mean/max milliseconds and share of frame time.

        `share` is None for stages timed outside the frame.
        """
        stages = {}
        for name, stats in self.stages.items():
            share = None
            if stats.in_frame:
                share = stats.total_s / self.frame_total_s if self.frame_total_s > 0.0 else 0.0
            stages[name] = {
                "calls": stats.calls,
                "mean_ms": stats.mean_ms,
                "max_ms": 1000.0 * stats.max_s,
                "share": share,
            }
        mean_frame_ms = 1000.0 * self.frame_total_s / self.frames if self.frames else 0.0
        return {"frames": self.frames, "fps": self.fps, "mean_frame_ms": mean_frame_ms, "stages": stages}

    def report(self) -> str:
        """Format the summary as an aligned text table, slowest stage first."""
        summary = self.summary()
        lines = [
            f"frames={summary['frames']} fps={summary['fps']:.1f} mean_frame_ms={summary['mean_frame_ms']:.2f}",
            f"{'stage':<12}{'mean_ms':>10}{'max_ms':>10}{'share':>8}",
        ]
        ordered = sorted(
            summary["stages"].items(),
            key=lambda item: -1.0 if item[1]["share"] is None else item[1]["share"],
            reverse=True,
        )
        for name, stats in ordered:
            share = "       -" if stats["share"] is None else f"{100.0 * stats['share']:>7.1f}%"
            lines.append(f"{name:<12}{stats['mean_ms']:>10.2f}{stats['max_ms']:>10.2f}{share}")
        return "\n".join(lines)


class FeatureAutoTuner:
    """Adjust the ORB feature budget to hold a target frame rate.

    Tracking quality has priority over speed: when matches run low the budget grows even if
    frames are already slow, because a lost track costs far more than a late frame. The
    budget shrinks only while matches are comfortably above the tracker's minimum, and grows
    back when there is frame-time headroom.
    """

    def __init__(
        self,
        target_fps: float = 15.0,
        initial: int = 2000,
        min_features: int = 500,
        max_features: int = 4000,
        min_matches: int = 60,
        cooldown_frames: int = 10,
        smoothing: float = 0.2,
    ) -> None:
        """Configure the controller.

        min_matches: the tracker's `min_matches`; the tuner keeps a safety margin above it.
        cooldown_frames: frames to wait after a change so its effect shows in the average.
        smoothing: weight of the newest frame in the frame-time/match moving averages.
        """
        if target_fps <= 0.0:
            raise ValueError("target_fps must be positive")
        if not 0 < min_features <= max_features:
            raise ValueError("need 0 < min_features <= max_features")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self.target_frame_s = 1.0 / float(target_fps)
        self.min_features = int(min_features)
        self.max_features = int(max_features)
        self.min_matches = int(min_matches)
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.smoothing = float(smoothing)
        self.nfeatures = min(self.max_features, max(self.min_features, int(initial)))
        self.adjustments = 0
        self._frame_s: float | None = None
        self._matches: float | None = None
        self._cooldown = self.cooldown_frames

    def _smooth(self, average: float | None, value: float) -> float:
        return value if average is None else (1.0 - self.smoothing) * average + self.smoothing * value

    def update(self, frame_time_s: float, n_matches: int) -> int:
        """Record one frame's duration and match count; return the feature budget to use next."""
        self._frame_s = self._smooth(self._frame_s, float(frame_time_s))
        self._matches = self._smooth(self._matches, float(n_matches))
        if self._cooldown > 0:
            self._cooldown -= 1
            return self.nfeatures

        # Matches are capped by the tracker's max_matches, so thresholds stay close to the minimum.
        starving = n_matches < 1.25 * self.min_matches or self._matches < 1.5 * self.min_matches
        comfortable = self._matches > 2.0 * self.min_matches
        if starving:
            target = self.nfeatures * 1.25
        elif self._frame_s > 1.1 * self.target_frame_s and comfortable:
            target = self.nfeatures * 0.85
        elif self._frame_s < 0.8 * self.target_frame_s:
            target = self.nfeatures * 1.1
        else:
            return self.nfeatures

        new_value = min(self.max_features, max(self.min_features, round(target)))
        if new_value != self.nfeatures:
            self.nfeatures = new_value
            self.adjustments += 1
            self._cooldown = self.cooldown_frames
        return self.nfeatures
