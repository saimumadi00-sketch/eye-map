# Roadmap

## Current Status

- Phase 1: Visual Odometry — COMPLETE
- Phase 2: Calibration and Evaluation — COMPLETE
- Phase 3: VKITTI 2 Evaluation Dataset — COMPLETE
- Phase 4: Drone Integration — IN PROGRESS

## Roadmap

### v0.2 — Drone Sensor Integration

- [ ] RealSenseCapture implementation
- [x] GPS scale fusion (offline via `--gps-csv`; live GPS feed into the loop still open)
- [x] MAVLink publisher for ArduPilot (tested against a pymavlink peer; not yet on hardware)
- [x] MAVLink publisher for PX4 (same message; note PX4's 30-50 Hz rate requirement)
- [ ] Jetson Orin deployment guide

### v0.3 — Real-Time Performance

- [x] Frame rate profiling (per-stage report in every run log)
- [x] ORB feature count auto-tuning (`--auto-tune`, `--target-fps`)
- [ ] GPU-accelerated feature detection
- [ ] C++ backend option via pybind11

### v0.4 — Loop Closure

- [ ] Place recognition module
- [ ] Pose graph optimization
- [ ] Map reuse across flights

### v0.5 — Autonomous Navigation

- [ ] Dense depth map from RealSense
- [ ] Obstacle detection layer
- [ ] Waypoint-based mapping mission
