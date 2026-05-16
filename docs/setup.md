## Requirements

Python 3.10 or higher.
Dependencies listed in requirements.txt.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
python scripts/run_vkitti.py \
  --rgb-dir data/vkitti_2.0.3_rgb \
  --depth-dir data/vkitti_2.0.3_depth \
  --text-dir data/vkitti_2.0.3_textgt \
  --scene Scene01 --variant clone --max-frames 200
```

## Evaluate a Run

```bash
python src/mvp/evaluate.py \
  --est-csv outputs/<run_id>/trajectory.csv \
  --gt-csv  outputs/<run_id>/gt_trajectory.csv \
  --ply     outputs/<run_id>/sparse_map.ply
```
