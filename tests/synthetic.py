"""Synthetic two-depth scene for end-to-end tests: a camera sliding sideways past textured planes."""

from __future__ import annotations

import cv2
import numpy as np

from src.pose_estimator import make_intrinsics

WIDTH, HEIGHT = 640, 480
BACKGROUND_SHIFT_PX = 3  # per frame; the foreground, at half the depth, shifts twice as far
BACKGROUND_DEPTH = 10.0
# Camera translation per frame (EyeMap units) that produces exactly BACKGROUND_SHIFT_PX.
STEP = BACKGROUND_SHIFT_PX * BACKGROUND_DEPTH / make_intrinsics(WIDTH, HEIGHT)[0, 0]


def _texture(rng: np.random.Generator, height: int, width: int, shapes: int) -> np.ndarray:
    """Draw random filled shapes: corner-rich texture that survives video compression."""
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    for _ in range(shapes):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        w, h = int(rng.integers(8, 40)), int(rng.integers(8, 40))
        if rng.random() < 0.5:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.circle(image, (x, y), w // 2, color, -1)
    return image


def moving_scene_frames(n_frames: int = 30, seed: int = 5) -> list[np.ndarray]:
    """Return BGR frames seen by a camera translating along +x by STEP per frame."""
    rng = np.random.default_rng(seed)
    travel = 2 * BACKGROUND_SHIFT_PX * n_frames
    background = _texture(rng, HEIGHT, WIDTH + travel, 900)
    foreground = _texture(rng, HEIGHT, WIDTH + travel, 900)
    # Foreground covers a band of columns in every frame, so both depths are always visible.
    mask = np.zeros((HEIGHT, WIDTH + travel), dtype=bool)
    for start in range(0, WIDTH + travel, 320):
        mask[120:360, start : start + 160] = True

    frames = []
    for index in range(n_frames):
        near = 2 * BACKGROUND_SHIFT_PX * index
        far = BACKGROUND_SHIFT_PX * index
        frame = background[:, far : far + WIDTH].copy()
        visible = mask[:, near : near + WIDTH]
        frame[visible] = foreground[:, near : near + WIDTH][visible]
        frames.append(frame)
    return frames
