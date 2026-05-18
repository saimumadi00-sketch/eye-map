# EyeMap

Open-source monocular terrain mapping for drone deployment.

## Status badges

[![CI](https://github.com/saimumadi00-sketch/eye-map/actions/workflows/ci.yml/badge.svg)](https://github.com/saimumadi00-sketch/eye-map/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## What is EyeMap

EyeMap is a Python monocular mapping foundation that turns live or recorded video into relative camera trajectories and sparse terrain point clouds. It gives drone developers a small, readable base they can fork, evaluate, and extend toward metric-scale mapping, onboard sensing, and flight-controller integration.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system map.

```text
Camera Input -> Frame Preprocessor -> Feature Detector -> Feature Matcher
-> Pose Estimator -> Triangulator -> Point Cloud -> Exporter -> Evaluator
```

## Quick Start

```bash
git clone https://github.com/saimumadi00-sketch/eye-map.git
cd eye-map
pip install -e .
python main.py --source webcam
```

## Running on Virtual KITTI 2

Expected dataset layout:

```text
data/
├── vkitti_2.0.3_rgb/
├── vkitti_2.0.3_depth/
└── vkitti_2.0.3_textgt/
```

Run one sequence:

```bash
python scripts/run_vkitti.py \
  --rgb-dir data/vkitti_2.0.3_rgb \
  --depth-dir data/vkitti_2.0.3_depth \
  --text-dir data/vkitti_2.0.3_textgt \
  --scene Scene01 \
  --variant clone \
  --max-frames 200
```

The run writes an output directory with estimated trajectory files, ground-truth trajectory files, saved keyframes, and a sparse map when enough valid points are triangulated.

## Evaluation

Trajectory error metrics compare estimated camera centers against ground truth, `drift_ratio` measures accumulated relative drift, and point-cloud statistics summarize the exported sparse geometry.

```bash
python src/mvp/evaluate.py \
  --est-csv outputs/<run_id>/trajectory.csv \
  --gt-csv outputs/<run_id>/gt_trajectory.csv \
  --ply outputs/<run_id>/sparse_map.ply
```

## Drone Deployment

See [docs/drone_deployment.md](docs/drone_deployment.md). Offline reconstruction from drone footage works today; live drone deployment still needs contributors to implement the RealSense, GPS-fusion, and MAVLink stubs.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The highest-leverage contributions are RealSense capture, GPS scale recovery, MAVLink publishing, and hardware-backed deployment testing.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md).

### v0.2 — Drone Sensor Integration

- [ ] RealSenseCapture implementation
- [ ] GPS scale fusion
- [ ] MAVLink publisher for ArduPilot
- [ ] MAVLink publisher for PX4
- [ ] Jetson Orin deployment guide

## License

MIT
