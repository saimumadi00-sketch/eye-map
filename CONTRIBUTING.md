# Contributing to EyeMap

## Getting Started

```bash
git clone https://github.com/saimumadi00-sketch/eye-map.git
cd eye-map
pip install -e ".[dev]"
python -m pytest tests/
```

## Project Structure

| Directory | What it contains |
| --- | --- |
| `src/` | Core monocular visual-odometry pipeline modules. |
| `src/core/` | Shared utilities and camera-calibration helpers. |
| `src/data/` | Virtual KITTI 2 loading and dataset adapters. |
| `src/mvp/` | Runnable MVP scripts, evaluation, terrain-grid generation, and point-cloud viewing. |
| `src/offline/` | Optional COLMAP-based sparse and dense reconstruction helpers. |
| `src/sensors/` | Camera interfaces and hardware input adapters such as RealSense. |
| `src/integration/` | Drone-facing fusion and flight-controller integration stubs. |
| `scripts/` | Convenience launch scripts and dataset runners. |
| `tests/` | Automated tests for public behavior. |
| `docs/` | Architecture, setup, deployment, methodology, and roadmap notes. |

## How to Implement a Stub

1. Find files with the `CONTRIBUTOR STUB` comment.
2. Remove the `NotImplementedError`.
3. Install any extra dependency listed.
4. Follow the docstring specification.
5. Add a test in `tests/`.
6. Update `docs/drone_deployment.md` if relevant.

## Coding Standards

- Python 3.10+
- Type hints on all function signatures
- `logging`, not `print()`
- `pathlib.Path`, not `os.path` strings
- All paths relative to the repo root
- One test per new public function

## Pull Request Process

- Branch from `main`.
- All tests must pass: `python -m pytest tests/`.
- Update the relevant file in `docs/`.
- The PR description must state what stub was implemented, how it was tested, and what hardware was used if applicable.

## Priority Contributions Needed

1. RealSenseCapture implementation (`src/sensors/realsense_capture.py`)
2. `GPSFusion.estimate_scale` (`src/integration/gps_fusion.py`)
3. `MAVLinkPublisher.connect` and `send_vision_position_estimate`
4. Jetson Orin deployment testing
5. ArduPilot `VISION_POSITION_ESTIMATE` integration testing
