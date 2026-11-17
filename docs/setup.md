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

## Check and Tune Frame Rate

Every run logs a per-stage timing table (also saved in `logs/run.log`). `read` is time spent
waiting for the next frame and is not counted in `fps`.

To let EyeMap adapt the ORB feature count to a frame-rate goal:

```bash
python main.py --source video --path /path/to/video.mp4 --auto-tune --target-fps 20
```

The budget moves between `orb_features_min` and `orb_features_max` (see `config/config.yaml`,
where `orb_auto_tune: true` enables it permanently). It is only reduced while matches stay well
above `minimum_matches`.

## Recover Metric Scale From GPS

```bash
python main.py --source video --path flight.mp4 --gps-csv flight_gps.csv
```

The CSV needs `frame_id,lat,lon` columns and may include `alt`. See
[drone_deployment.md](drone_deployment.md) for outputs and limits.

## Object Directions and Layout

Every run writes a top-down layout (`plots/layout.png`) and map clusters (`objects/`). For
labelled objects, their directions and motion, pass a YOLO-style ONNX model:

```bash
python main.py --source video --path /path/to/video.mp4 --detect-model models/yolo11n.onnx
```

See [objects.md](objects.md) for the outputs and their limits.

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
