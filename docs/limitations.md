# Limitations

## Monocular Scale Ambiguity

A single moving camera cannot directly recover real-world metric scale. The current MVP estimates a relative trajectory and relative sparse map unless a known scale reference, GPS, IMU, stereo camera, or calibrated external measurement is added.

With GPS fixes for the flight (`--gps-csv`), `GPSFusion` recovers one global metric scale and a local East-North-Up frame. Scale drift within a long sequence is not corrected.

## Near-Planar Scenes

When most matches lie on one plane (flat ground seen from altitude, a wall), the essential matrix is poorly constrained and an occasional frame pair yields a translation direction tens of degrees off. On a synthetic two-plane scene with 80% of matches on the far plane, a few percent of pairs were affected. The tracker does not yet detect this (for example by comparing a homography model against the essential matrix), so expect occasional kinks in the trajectory over flat terrain.

## Textureless Terrain

Open ground, plain walls, sand, water, and smooth roads may not contain enough stable visual features. In those cases, ORB matching becomes weak and pose estimation may fail.

## Motion Blur

Fast camera movement causes blur and reduces feature detection quality. The best demo videos use slow, smooth motion and visible texture.

## Dynamic Objects

Moving cars, people, vegetation, and shadows can create incorrect matches. The MVP does not yet segment or remove dynamic objects.

## Sparse Reconstruction Quality

The live sparse map is useful for demonstrating structure-from-motion, but it is less accurate and less dense than offline photogrammetry. Dense terrain reconstruction should be treated as a later extension.

## Hardware and Lighting

Low light, rolling shutter distortion, low frame rate, and poor camera focus all reduce tracking stability.

## Processing Speed

Brute-force ORB matching grows with the square of the feature count and dominates frame time (about two thirds at 2000 features in a 640x480 synthetic benchmark on a cloud CPU, with detection most of the rest). `--auto-tune` lowers the feature budget to hold a target frame rate, but only while matches stay comfortably above `minimum_matches`. There is no GPU path yet.
