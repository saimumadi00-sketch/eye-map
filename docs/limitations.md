# Limitations

## Monocular Scale Ambiguity

A single moving camera cannot directly recover real-world metric scale. The current MVP estimates a relative trajectory and relative sparse map unless a known scale reference, GPS, IMU, stereo camera, or calibrated external measurement is added.

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
