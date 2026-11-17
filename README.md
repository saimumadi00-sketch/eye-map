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

The default installation includes OpenCV window support. The optional Open3D
point-cloud viewer (`--show-pointcloud`, `src.mvp.view_cloud`) needs
`pip install -e ".[viz]"`. Use `--no-display` for
batch runs or machines without a display:

```bash
eyemap-run --source video --path data/sample_videos/terrain.mp4 --no-display
```

Installed commands also work outside the checkout; `eyemap-run` loads its packaged
default configuration unless you supply `--config`.

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
python -m scripts.run_vkitti \
  --rgb-dir data/vkitti_2.0.3_rgb \
  --depth-dir data/vkitti_2.0.3_depth \
  --text-dir data/vkitti_2.0.3_textgt \
  --scene Scene01 \
  --variant clone \
  --max-frames 200 \
  --no-display
```

The run writes an output directory with estimated trajectory files, ground-truth trajectory files, saved keyframes, and a sparse map when enough valid points are triangulated.

## Evaluation

Trajectory error metrics match frame IDs and fit one global rotation, translation,
and scale before comparing estimated camera centers against ground truth. The
report includes the fitted scale and errors in ground-truth units. Use `--no-align`
when both trajectories already share a metric coordinate frame. Alignment does
not correct local scale drift. `drift_ratio` is endpoint distance divided by path
length; it describes path closure, while point-cloud statistics summarize geometry.

```bash
python -m src.mvp.evaluate \
  --est-csv outputs/<run_id>/trajectory.csv \
  --gt-csv outputs/<run_id>/gt_trajectory.csv \
  --ply outputs/<run_id>/sparse_map.ply
```

## Object Directions and Layout

Each run writes a top-down layout (`plots/layout.png`) and groups the sparse map into clusters with their direction and distance from the camera. With a YOLO-style ONNX model (`--detect-model`), EyeMap also labels objects, reports which way each lies and how it is moving, and saves annotated frames. See [docs/objects.md](docs/objects.md).

## Drone Deployment

See [docs/drone_deployment.md](docs/drone_deployment.md). Offline reconstruction from drone footage works today, including metric scale from a GPS log (`--gps-csv`) and MAVLink pose publishing to ArduPilot/PX4. The RealSense capture stub and a live GPS feed still need contributors, and the publisher has not been flown on real hardware.

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
