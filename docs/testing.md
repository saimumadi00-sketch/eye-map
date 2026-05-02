# Testing Checklist

Use this checklist after dependencies are available and before recording report results.

## Test 1: Open Recorded Video

Command:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --mode display --max-frames 100
```

Expected result: the video opens, frames update, and pressing `q` exits safely.

## Test 2: Detect Features

Command:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --mode features --max-frames 100
```

Expected result: green ORB keypoints appear on textured objects and terrain.

## Test 3: Match Features

Command:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --mode matches --max-frames 100
```

Expected result: line visualizations connect matched features between consecutive frames.

## Test 4: Estimate Camera Pose

Command:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --mode trajectory --max-frames 200
```

Expected result: the status overlay reports `pose=ok` for many frames and a trajectory window updates.

## Test 5: Save Keyframes

Command:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --save-keyframes --max-frames 200
```

Expected result: `outputs/<run>/keyframes/` contains frame images and `outputs/<run>/keyframes.json` exists.

## Test 6: Plot Trajectory

Command:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --mode trajectory --max-frames 200 --no-display
```

Expected result: `outputs/<run>/trajectory/trajectory.csv` and `outputs/<run>/plots/trajectory.png` are created.

## Test 7: Export Point Cloud

Command:

```bash
python main.py --source video --path data/sample_videos/terrain.mp4 --export-pointcloud --max-frames 200 --no-display
```

Expected result: `outputs/<run>/pointclouds/sparse_map.ply` is created if enough valid triangulated points are found.

## Test 8: Poor Video Handling

Command:

```bash
python main.py --source video --path data/sample_videos/poor_texture.mp4 --mode trajectory --max-frames 100
```

Expected result: the system does not crash; it reports low matches or skipped pose updates.

## Unit and Functional Tests

Command:

```bash
python -m unittest discover tests
```

Expected result: tests for video loading, feature detection, matching, pose estimation, and exporters pass. Webcam hardware is not required.
