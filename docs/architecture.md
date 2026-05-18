# Architecture

## Pipeline Overview

```text
Camera Input [IMPLEMENTED: video_input.py, CameraInterface/OpenCVCamera]
      |
      v
Frame Preprocessor [IMPLEMENTED: video_input.py]
      |
      v
Feature Detector [IMPLEMENTED: feature_detector.py]
      |
      v
Feature Matcher [IMPLEMENTED: feature_matcher.py]
      |
      v
Pose Estimator [IMPLEMENTED: pose_estimator.py]
      |
      v
Triangulator [IMPLEMENTED: triangulation.py]
      |
      v
Point Cloud [IMPLEMENTED: pointcloud.py]
      |
      v
Exporter [IMPLEMENTED: exporter.py]
      |
      v
Evaluator [IMPLEMENTED: mvp/evaluate.py]

Drone hardware sidecars:
  RealSenseCapture [STUB] -> Camera Input
  GPSFusion [STUB] -------> metric scale recovery
  MAVLinkPublisher [STUB] -> flight controller output
```

## Module Reference

`src/` contains the main reusable pipeline modules: `video_input.py`, `feature_detector.py`, `feature_matcher.py`, `pose_estimator.py`, `triangulation.py`, `pointcloud.py`, `keyframe_manager.py`, `exporter.py`, `visualizer.py`, and `utils.py`. Together they form the live monocular mapping path used by `main.py`.

`src/core/` contains shared low-level helpers in `utils.py` plus camera calibration support in `calibration.py`. These utilities keep common file, PLY, timestamp, intrinsic, and calibration behavior out of higher-level modules.

`src/data/` contains `vkitti_loader.py` and `vkitti_adapter.py`. It turns Virtual KITTI 2 assets into EyeMap-compatible frames, calibration records, trajectories, and evaluation artifacts.

`src/mvp/` contains `live_mvp.py`, `terrain_grid.py`, `view_cloud.py`, and `evaluate.py`. These scripts preserve the compact prototype path, derive simple terrain summaries, visualize clouds, and report evaluation metrics.

`src/offline/` contains `colmap_runner.py` and `dense_builder.py`. It wraps optional COLMAP reconstruction steps for sparse and dense offline refinement after keyframes have been collected.

`src/sensors/` contains `camera_interface.py` and `realsense_capture.py`. It defines a reusable camera contract, ships an implemented OpenCV adapter, and reserves a contributor stub for Intel RealSense D435i support.

`src/integration/` contains `gps_fusion.py` and `mavlink_publisher.py`. It is the drone-facing extension layer for metric-scale recovery and flight-controller pose publishing.

## Extending the Pipeline

To add a new camera, implement `CameraInterface`, then register the new source in `main.py` so the pipeline can construct it from command-line or configuration input. Keep the returned frames in OpenCV BGR format and provide calibrated or estimated intrinsics through `get_intrinsics()`.

To add new sensor fusion, follow the `GPSFusion` pattern: store synchronized readings, expose an explicit calibration step, and keep scaled outputs separate from raw visual-odometry state until calibration succeeds.

To add a new flight stack, follow the `MAVLinkPublisher` pattern: isolate connection setup, message formatting, and shutdown behind a small adapter so ArduPilot, PX4, or future transports do not leak into the vision modules.

## Data Flow

`read_frame()` returns `(success, frame)` where `frame` is a BGR image. The next stage expects a valid image array and passes it into `detect_features()`, which returns grayscale image data, keypoints, and binary descriptors.

`match_features()` consumes descriptor arrays and returns OpenCV match objects. `matched_points()` converts those matches into paired 2D point arrays that `estimate_relative_pose_from_points()` expects.

`estimate_relative_pose_from_points()` returns a result dictionary containing `success`, `R`, `t`, and an `inlier_mask` when pose recovery succeeds. `triangulate_relative()` consumes the inlier correspondences plus `K`, `R`, and `t`, then returns relative 3D points.

`filter_triangulated_points()` removes invalid geometry, and `transform_points()` converts surviving points into world coordinates. `save_pointcloud()` accepts the accumulated `(N, 3)` array and writes a PLY file path that downstream viewers and evaluators can consume.
