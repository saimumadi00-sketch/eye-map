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
pip install -e ".[drone]"        # pymavlink + pyserial
python -c "import pymavlink"
python -c "import pyrealsense2"  # only for RealSenseCapture; on Jetson, build librealsense with Python bindings
```

## Metric Scale from GPS

Monocular tracking gives the path only up to one unknown scale. `GPSFusion`
([`src/integration/gps_fusion.py`](../src/integration/gps_fusion.py)) pairs GPS fixes with camera
centers by frame id and recovers metres per EyeMap unit, plus the rotation and translation into a
local East-North-Up (ENU) frame anchored at the first paired fix.

Offline, pass a CSV with `frame_id,lat,lon` and an optional `alt` column:

```bash
python main.py --source video --path flight.mp4 --gps-csv flight_gps.csv --no-display
```

The run writes `trajectory/trajectory_metric_enu.csv` (metres east/north/up) and logs
`gps_scale_m_per_unit` and the ENU origin in `logs/run.log`.

In code:

```python
fusion = GPSFusion()                       # or GPSFusion(method="arc_length")
fusion.add_gps(frame_id, lat, lon, alt_m)  # whenever a fix arrives
fusion.add_pose(frame_id, x, y, z)         # camera center for the same frame
scale = fusion.estimate_scale()            # None until calibration is possible
metres = fusion.apply_scale(points)        # EyeMap axes, metric
enu = fusion.to_enu(points)                # georeferenced, metric
```

- `estimate_scale()` returns `None` until at least 3 frames have both readings and the GPS track
  spans 5 m (`min_baseline_m`); below that, receiver noise dominates. Fly a few metres before relying on it.
- The default `similarity` method is a least-squares fit, so GPS noise averages out. `arc_length`
  is the path-length ratio described in the original stub, measured over chords of at least
  `min_segment_m` so jitter does not inflate the GPS path.
- One global scale is fitted. Scale drift inside a long flight is not corrected (no loop closure yet).
- A straight-line flight fixes scale and position but not the roll about the flight direction;
  include a turn if you need a full ENU orientation.

## Publishing Poses to the Flight Controller

`MAVLinkPublisher` ([`src/integration/mavlink_publisher.py`](../src/integration/mavlink_publisher.py))
sends `VISION_POSITION_ESTIMATE`, which both ArduPilot and PX4 accept, and a 1 Hz companion heartbeat.

```python
from src.integration.mavlink_publisher import MAVLinkPublisher, MOUNT_DOWNWARD

with MAVLinkPublisher("/dev/ttyTHS1", baud=921600, camera_to_body=MOUNT_DOWNWARD) as fc:
    print(fc.autopilot)                      # "ArduPilot" or "PX4"
    fc.send_pose(T_cw, scale=fusion.scale_factor, timestamp_us=capture_time_us)
```

- `connect()` raises `TimeoutError` if no heartbeat arrives within `heartbeat_timeout_s`.
- `send_pose()` converts EyeMap's camera pose (OpenCV axes, relative to the first image) into the
  Forward-Right-Down frame MAVLink expects. Choose the mount: `MOUNT_FORWARD` (default) or
  `MOUNT_DOWNWARD` (lens down, top of image toward the nose), or pass your own 3x3 rotation.
- The pose is relative to the vehicle's attitude at the first image, so yaw is relative to the
  starting heading, not north. Start level.
- `send_pose()` requires a metric scale and refuses a non-positive one. Never publish a
  relative-scale pose to a flight controller. Non-finite poses are dropped, not sent.
- SITL: use `"tcp:127.0.0.1:5760"` or `"udpout:127.0.0.1:14550"` as the connection string.

Autopilot parameters (confirm against the documentation for your firmware version; reboot after changing):

| Autopilot | Parameters |
| --- | --- |
| ArduPilot (EKF3) | `SERIALx_PROTOCOL = 2` on the companion port; `VISO_TYPE = 1`; `EK3_SRC1_POSXY = 6`, `EK3_SRC1_POSZ = 6` (or `1` for baro), `EK3_SRC1_YAW = 6` (ExternalNav). EyeMap sends no velocity, so leave `EK3_SRC1_VELXY`/`VELZ` at `0`. |
| PX4 (EKF2) | `MAV_x_CONFIG` on the companion port; enable position and yaw fusion in `EKF2_EV_CTRL`; set `EKF2_HGT_REF` to Vision if vision should be the height reference; tune `EKF2_EV_DELAY`. |

PX4's [external position estimation guide](https://docs.px4.io/main/en/ros/external_position_estimation)
asks for these messages at 30-50 Hz and will not fuse a slower stream. EyeMap's Python pipeline
only emits a pose when enough parallax has accumulated, typically well below that rate, so treat
the published pose as an experiment for bench and SITL work, not as a flight-ready position source.

## What Is Still a Stub

- [`src/sensors/realsense_capture.py`](../src/sensors/realsense_capture.py): live RGB, depth, and
  calibrated-intrinsics acquisition from the D435i. It needs the hardware to implement and test.
- The live loop does not yet call `GPSFusion`/`MAVLinkPublisher` itself: there is no live GPS
  reader. `--gps-csv` covers recorded flights; wiring a live MAVLink GPS feed into the loop is the next step.

## Running EyeMap on a Drone

What works today: offline reconstruction from recorded drone footage, metric scale and ENU
georeferencing from a GPS log, and MAVLink pose publishing (verified against a pymavlink peer
over UDP, not yet on a real flight controller).

Suggested integration test: hover in place and verify that the emitted pose estimate remains stable before attempting translational flight.

## Known Limitations

- Monocular scale ambiguity unless GPS fixes are supplied.
- Python pipeline is CPU-bound; brute-force ORB matching dominates frame time. Use `--auto-tune` to trade features for frame rate.
- No loop closure — drift accumulates on long flights.
- Dense reconstruction requires CUDA.
