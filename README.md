# Real-Time Terrain Mapping Using Live Video Stream (MVP)

Short description: A lightweight computer vision MVP that estimates camera motion from live video, builds a sparse terrain/scene point cloud, and exports keyframes for later offline reconstruction.

This repository contains a student-friendly baseline for:

1. Live camera/video input  
2. Visual feature tracking  
3. Relative camera pose estimation  
4. Sparse 3D point triangulation  
5. Live trajectory display + exported keyframes/point cloud  

The current baseline is intentionally simple and optimized for a working demo first.

## Repo Structure

```text
EyeMap/
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- configs/
|   +-- mvp.yaml
+-- docs/
+-- scripts/
`-- src/
    +-- core/
    +-- offline/
    `-- mvp/
        +-- live_mvp.py
        +-- terrain_grid.py
        `-- view_cloud.py
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/mvp/live_mvp.py --source 0
```

- Press `q` to quit the live run.
- Outputs are saved in `outputs/<timestamp>/`.

## Current Outputs

- `trajectory.csv`: Camera center positions over time
- `sparse_map.ply`: Sparse 3D point cloud (if enough points were triangulated)
- `keyframes/`: Saved RGB keyframes
- `keyframes.json`: Keyframe metadata and poses

## Scripts

- `src/mvp/live_mvp.py`: Live MVP pipeline
- `src/mvp/terrain_grid.py`: Simple elevation-grid approximation from exported sparse cloud
- `src/mvp/view_cloud.py`: Open3D point-cloud viewer for `.ply`

## Installation (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `open3d` fails to install on your machine, keep using the pipeline and view `.ply` in MeshLab as fallback.

## Run Commands

Webcam:

```bash
python src/mvp/live_mvp.py --source 0
```

Video file:

```bash
python src/mvp/live_mvp.py --source /path/to/video.mp4
```

Run + auto-open exported sparse point cloud:

```bash
python src/mvp/live_mvp.py --source 0 --show-pointcloud
```

View point cloud later:

```bash
python src/mvp/view_cloud.py --ply outputs/<run_id>/sparse_map.ply
```

Generate terrain/elevation grid from sparse cloud:

```bash
python src/mvp/terrain_grid.py --ply outputs/<run_id>/sparse_map.ply --cell 0.25 --out outputs/<run_id>/terrain_grid.csv
```

## Phase-1 Success Checklist

- Live frame window opens and updates in real time.
- Trajectory window draws a moving path as camera moves.
- `outputs/<run_id>/trajectory.csv` is created.
- `outputs/<run_id>/keyframes/` contains saved images.
- `outputs/<run_id>/sparse_map.ply` is generated with non-zero points.
- Sparse cloud opens in Open3D or MeshLab.

## Notes

- This is monocular visual odometry + sparse mapping, so scale is relative (not metric-true).
- Expect drift over long sequences; this is normal for a Phase-1 student MVP.
