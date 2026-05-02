# Methodology

## Problem

The project demonstrates a practical camera-based terrain mapping pipeline. A live or recorded video stream is used to estimate camera motion, save useful keyframes, and reconstruct a rough sparse 3D structure of the observed scene.

## Why Camera-Based Terrain Mapping Is Useful

Cameras are inexpensive, lightweight, and available on phones, webcams, drones, and robots. A monocular camera cannot replace professional LiDAR or survey-grade photogrammetry, but it is useful for demonstrating visual odometry, structure-from-motion, and terrain reconstruction concepts in a student project.

## Workflow

The MVP pipeline is:

```text
video frame -> grayscale -> ORB keypoints -> descriptor matching -> essential matrix
-> relative pose -> accumulated trajectory -> sparse triangulated points -> exported artifacts
```

## Feature Detection

The system uses ORB features because ORB is fast, CPU-friendly, and available in OpenCV without patent restrictions. Each frame is converted to grayscale, then ORB detects keypoints and computes binary descriptors.

## Feature Matching

Descriptors from consecutive frames are matched using Hamming-distance brute force matching. Lowe's ratio test removes ambiguous matches so the pose estimator receives cleaner feature correspondences.

## Camera Pose Estimation

Matched feature coordinates are used to compute the essential matrix with RANSAC. OpenCV then recovers a relative rotation and translation direction between frames.

Because this is monocular video, translation has unknown metric scale. The MVP normalizes translation so the trajectory is stable for visualization, but the output is in relative units rather than meters.

## Sparse Reconstruction

When pose estimation succeeds, matched inlier points are triangulated into sparse 3D points. Invalid points, points behind either camera, and extreme outliers are filtered before export.

## Keyframes

Keyframes are saved at a fixed interval. These images and their metadata can later be used as input to offline reconstruction tools such as COLMAP.

## Terrain Approximation

The sparse 3D point cloud can be treated as a rough terrain or scene map. The existing terrain grid script bins points into x-z grid cells and estimates a representative elevation value from the y coordinate.

## Future Improvements

Future versions can improve scale and stability by adding camera calibration, stereo cameras, GPS, IMU, ORB-SLAM3, or offline COLMAP reconstruction.
