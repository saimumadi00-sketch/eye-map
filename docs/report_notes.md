# Report Notes

## Short Project Summary

This project implements a real-time terrain mapping MVP using monocular video. It detects visual features, matches them between frames, estimates relative camera motion, saves keyframes, and exports a sparse 3D point cloud.

## Defendable MVP Claim

The system is not designed to be a professional surveying tool. It is a computer vision demonstration that shows the main steps of visual odometry and sparse scene reconstruction using accessible hardware.

## Demo Evidence To Capture

- Input video frame with detected ORB features
- Match visualization between consecutive frames
- Camera trajectory plot
- Saved keyframe folder
- Sparse point cloud viewer screenshot
- Terrain grid CSV or elevation visualization from `src/mvp/terrain_grid.py`

## Viva Talking Point

The most important limitation is monocular scale ambiguity. The shape and motion are recovered in relative units, and metric terrain mapping would require extra scale information from stereo, GPS, IMU, or known scene measurements.
