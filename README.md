# Real-Time Terrain Mapping Using Live Video Stream

Short description: a lightweight computer vision MVP that estimates relative camera motion from live or recorded video, saves keyframes, exports a trajectory, and builds a sparse terrain/scene point cloud.

## Final Architecture

```text
video input
-> ORB feature detection
-> descriptor matching
-> essential matrix + relative pose
-> accumulated camera trajectory
-> sparse triangulation
-> keyframe, trajectory, match, and point cloud export
-> optional terrain/elevation approximation
```

This is a student-level monocular visual odometry and sparse mapping system. It is intended for a working demo and academic explanation, not professional GIS or survey-grade mapping.

## Folder Structure

```text
EyeMap/
+-- README.md
+-- main.py
+-- config/
|   `-- config.yaml
+-- configs/
|   `-- mvp.yaml
+-- src/
|   +-- video_input.py
|   +-- feature_detector.py
|   +-- feature_matcher.py
|   +-- pose_estimator.py
|   +-- triangulation.py
|   +-- keyframe_manager.py
|   +-- pointcloud.py
|   +-- visualizer.py
|   +-- exporter.py
|   +-- utils.py
|   +-- core/
|   |   `-- utils.py
|   +-- data/
|   |   `-- vkitti_loader.py
|   `-- mvp/
|       +-- live_mvp.py
|       +-- terrain_grid.py
|       `-- view_cloud.py
+-- tests/
+-- data/
|   +-- input/
|   +-- sample_videos/
|   `-- sample_frames/
+-- docs/
|   +-- methodology.md
|   +-- testing.md
|   +-- limitations.md
|   `-- report_notes.md
`-- outputs/
```

`outputs/` is created at runtime and ignored by Git.

## Main Commands

Run with webcam:

```bash
python main.py --source webcam
```

Run with recorded video:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4
```

Show ORB features:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --mode features
```

Show feature matches:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --mode matches
```

Run trajectory mode and save outputs:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --mode trajectory --save-keyframes --max-frames 200
```

Export sparse point cloud:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --mode pointcloud --export-pointcloud --max-frames 200
```

Headless run for testing:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --max-frames 200 --no-display
```

## Expected Outputs

Each run creates a timestamped folder under `outputs/`:

```text
outputs/<run_id>/
+-- keyframes/
+-- matches/
+-- trajectory/
|   `-- trajectory.csv
+-- pointclouds/
|   `-- sparse_map.ply
+-- plots/
|   `-- trajectory.png
+-- logs/
|   `-- run.log
`-- keyframes.json
```

Point cloud export only appears when enough valid triangulated points are available.

## Testing

Run the functional tests:

```bash
python -m unittest discover tests
```

See [docs/testing.md](docs/testing.md) for the manual demo checklist.

## Existing MVP Scripts

The original compact prototype is still available:

```bash
python src/mvp/live_mvp.py --source 0
```

Terrain grid generation from a sparse PLY:

```bash
python src/mvp/terrain_grid.py --ply outputs/<run_id>/pointclouds/sparse_map.ply --out outputs/<run_id>/terrain_grid.csv
```

Point cloud viewer:

```bash
python src/mvp/view_cloud.py --ply outputs/<run_id>/pointclouds/sparse_map.ply
```

## Report-Friendly Explanation

The system demonstrates a complete visual odometry pipeline: it reads video frames, detects ORB features, matches features between consecutive frames, estimates relative camera pose using the essential matrix, accumulates a camera trajectory, triangulates sparse 3D points, and saves keyframes for offline refinement.

The output trajectory and point cloud are in relative units. A monocular camera cannot directly recover metric scale unless additional scale information is added.

## Known Limitations

- Monocular scale is relative, not metric.
- Textureless surfaces produce few reliable features.
- Fast camera motion creates blur and matching failures.
- Moving objects and vegetation can corrupt feature matches.
- Sparse live mapping is less accurate than offline photogrammetry.

See [docs/limitations.md](docs/limitations.md) for details.

## Future Improvements

- Camera calibration
- Better keyframe selection based on motion and match quality
- COLMAP offline reconstruction from saved keyframes
- ORB-SLAM3 comparison
- Stereo, GPS, or IMU scale recovery
- Virtual KITTI 2 evaluation using `src/data/vkitti_loader.py`
