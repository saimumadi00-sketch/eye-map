# Drone Deployment

## Recommended Hardware

| Component | Recommended part | Why |
| --- | --- | --- |
| Companion computer | Jetson Orin Nano | GPU headroom for future real-time vision acceleration. |
| Flight controller | Pixhawk 6C | Mature ArduPilot and PX4 support with TELEM ports for companion links. |
| Camera | Intel RealSense D435i | RGB-D sensing plus IMU support for depth-aided mapping experiments. |
| Frame | F450 450mm quad | Common, repairable test frame with room for companion hardware. |
| Communication | MAVLink over UART | Standard low-latency bridge between Jetson and flight controller. |

## Wiring

- Connect Jetson UART TX to Pixhawk TELEM2 RX.
- Connect Jetson UART RX to Pixhawk TELEM2 TX.
- Connect ground to ground between Jetson and Pixhawk.
- Connect the Intel RealSense D435i to the Jetson over USB3.
- Placeholder diagram note: see issue #X for wiring diagram contribution.

## Software Setup on Jetson

```bash
pip install -e ".[drone]"
python -c "import pyrealsense2"
python -c "import pymavlink"
```

## Implementing the Stubs

- [`src/sensors/realsense_capture.py`](../src/sensors/realsense_capture.py): replace the contributor stub with live RGB, depth, and calibrated-intrinsics acquisition from the D435i.
- [`src/integration/gps_fusion.py`](../src/integration/gps_fusion.py): align GPS arc length with visual-trajectory arc length to recover metric scale.
- [`src/integration/mavlink_publisher.py`](../src/integration/mavlink_publisher.py): establish the MAVLink connection and publish EyeMap pose estimates to the flight controller.

## Running EyeMap on a Drone

What works today: offline reconstruction from recorded drone footage. What still needs implementation: `RealSenseCapture`, `GPSFusion`, and `MAVLinkPublisher`.

Suggested integration test: hover in place and verify that the emitted pose estimate remains stable before attempting translational flight.

## Known Limitations

- Monocular scale ambiguity without GPS fusion.
- Python pipeline not real-time on CPU — requires Jetson GPU.
- No loop closure — drift accumulates on long flights.
- Dense reconstruction requires CUDA.
