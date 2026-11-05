## Requirements

Python 3.10 or higher.
Dependencies are declared in `pyproject.toml`, including OpenCV window support.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# optional: Open3D point-cloud viewer
pip install -e ".[viz]"
```

## Verify Installation

```bash
python -m unittest discover tests
```

Expected: all tests pass.

## Run With Webcam

```bash
python main.py --source webcam
```

## Use Camera Calibration

Without calibration, intrinsics are guessed from the frame size. To calibrate with a
9x6 inner-corner checkerboard, move it around in front of the camera while running:

```bash
python -m src.core.calibration --source 0 --out configs/calib.npz
python main.py --source webcam --calib configs/calib.npz
```

Or set `calibration_file: configs/calib.npz` in `config/config.yaml`.

## Run With Video File

```bash
python main.py --source video --path /path/to/video.mp4 --mode trajectory
```

## Run With Virtual KITTI 2

Place the dataset under data/ as:

```text
data/vkitti_2.0.3_rgb/
data/vkitti_2.0.3_depth/
data/vkitti_2.0.3_textgt/
```

Then run:

```bash
python -m scripts.run_vkitti \
  --rgb-dir data/vkitti_2.0.3_rgb \
  --depth-dir data/vkitti_2.0.3_depth \
  --text-dir data/vkitti_2.0.3_textgt \
  --scene Scene01 --variant clone --max-frames 200 --no-display
```

## Evaluate a Run

```bash
python -m src.mvp.evaluate \
  --est-csv outputs/<run_id>/trajectory.csv \
  --gt-csv  outputs/<run_id>/gt_trajectory.csv \
  --ply     outputs/<run_id>/sparse_map.ply
```

Evaluation aligns origin, rotation, and one global scale to ground truth and
reports the fitted scale. Add `--no-align` for trajectories already in the same
metric frame. Installed equivalents are `eyemap-run`, `eyemap-vkitti`, and
`eyemap-evaluate`; these work outside the checkout.

Use `--no-display` with `main.py`, `scripts.run_vkitti`, `src.mvp.live_mvp`, or
`src.core.calibration` for runs without OpenCV windows. Point-cloud viewers are
opened only when explicitly requested.
