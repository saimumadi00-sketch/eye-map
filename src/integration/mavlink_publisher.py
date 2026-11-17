"""MAVLink pose publisher for ArduPilot and PX4 integration.

Sends EyeMap pose estimates to a flight controller as VISION_POSITION_ESTIMATE messages,
which both ArduPilot (EKF3 ExternalNav source) and PX4 (EKF2 external vision) accept.
See docs/drone_deployment.md for wiring and the autopilot parameters to set.

Frames: MAVLink expects position in a local Forward-Right-Down style frame and attitude as
roll/pitch/yaw of the vehicle body in that frame. EyeMap tracks the camera in OpenCV axes
(x right, y down, z forward) relative to the first image, with unknown scale. `pose_to_frd`
converts between the two given the camera mount and the metric scale from `GPSFusion`.
The reference frame is the vehicle body frame at the first image, so yaw is relative to the
initial heading, not to north.

pymavlink is imported lazily: install it with `pip install ".[drone]"`.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Rotations taking OpenCV camera axes (x right, y down, z forward) to body FRD axes.
MOUNT_FORWARD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# Camera looking straight down with the top of the image toward the vehicle nose.
MOUNT_DOWNWARD = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

HEARTBEAT_INTERVAL_S = 1.0
AUTOPILOT_NAMES = {3: "ArduPilot", 12: "PX4"}  # MAV_AUTOPILOT_ARDUPILOTMEGA, MAV_AUTOPILOT_PX4

INSTALL_MESSAGE = (
    'pymavlink is not installed. Install the drone extras with `pip install ".[drone]"` '
    "to publish poses to a flight controller."
)


def pose_to_frd(
    T_cw: np.ndarray,
    scale: float = 1.0,
    camera_to_body: np.ndarray = MOUNT_FORWARD,
) -> tuple[float, float, float, float, float, float]:
    """Convert an EyeMap world-to-camera pose into (x, y, z, roll, pitch, yaw) for MAVLink.

    T_cw: 4x4 world-to-camera pose, world being the first camera frame.
    scale: metres per EyeMap unit (`GPSFusion.scale_factor`); 1.0 leaves relative units.
    camera_to_body: 3x3 rotation from OpenCV camera axes to body FRD axes (the mount).

    Position is in metres in the body frame of the first image (x forward, y right, z down);
    angles are radians, ZYX (yaw-pitch-roll) convention.
    """
    T_cw = np.asarray(T_cw, dtype=np.float64)
    mount = np.asarray(camera_to_body, dtype=np.float64)
    if T_cw.shape != (4, 4):
        raise ValueError("T_cw must have shape (4, 4)")
    if mount.shape != (3, 3):
        raise ValueError("camera_to_body must have shape (3, 3)")

    R_wc = T_cw[:3, :3].T
    center = -R_wc @ T_cw[:3, 3]
    position = float(scale) * (mount @ center)
    R = mount @ R_wc @ mount.T

    pitch = -math.asin(max(-1.0, min(1.0, float(R[2, 0]))))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return float(position[0]), float(position[1]), float(position[2]), roll, pitch, yaw


class MAVLinkPublisher:
    """Publish vision pose estimates to an ArduPilot or PX4 flight controller."""

    def __init__(
        self,
        connection_string: str = "/dev/ttyUSB0",
        baud: int = 115200,
        source_system: int = 1,
        source_component: int = 197,  # MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
        heartbeat_timeout_s: float = 10.0,
        camera_to_body: np.ndarray = MOUNT_FORWARD,
    ) -> None:
        """Store connection settings without opening the link.

        connection_string: serial device (e.g. /dev/ttyTHS1) or a pymavlink URL such as
            "udpout:127.0.0.1:14550" or "tcp:127.0.0.1:5760" (SITL).
        source_system: MAVLink system id to publish as; use the vehicle's id (usually 1).
        """
        self.connection_string = connection_string
        self.baud = int(baud)
        self.source_system = int(source_system)
        self.source_component = int(source_component)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.camera_to_body = np.asarray(camera_to_body, dtype=np.float64)
        self.connection: Any = None
        self.autopilot: str | None = None  # "ArduPilot", "PX4", or "autopilot <id>" once connected
        self.messages_sent = 0
        self._last_heartbeat = -math.inf

    def connect(self, connection_factory: Callable[..., Any] | None = None) -> None:
        """Open the link and wait for the flight controller's heartbeat.

        Raises ImportError if pymavlink is missing and TimeoutError if no heartbeat arrives
        within `heartbeat_timeout_s`. `connection_factory` replaces
        `mavutil.mavlink_connection` (used by tests and custom transports).
        """
        if self.connection is not None:
            return
        if connection_factory is None:
            try:
                from pymavlink import mavutil
            except ImportError as exc:
                raise ImportError(INSTALL_MESSAGE) from exc
            connection_factory = mavutil.mavlink_connection

        connection = connection_factory(
            self.connection_string,
            baud=self.baud,
            source_system=self.source_system,
            source_component=self.source_component,
        )
        self.connection = connection
        try:
            # Announce ourselves first: on UDP the peer only learns our address from a packet.
            self._send_heartbeat(force=True)
            heartbeat = connection.wait_heartbeat(timeout=self.heartbeat_timeout_s)
        except BaseException:
            self.close()
            raise
        if heartbeat is None:
            self.close()
            raise TimeoutError(
                f"No MAVLink heartbeat from {self.connection_string} within "
                f"{self.heartbeat_timeout_s:.0f}s. Check wiring, baud rate, and SERIALx_PROTOCOL."
            )
        autopilot_id = int(getattr(heartbeat, "autopilot", -1))
        self.autopilot = AUTOPILOT_NAMES.get(autopilot_id, f"autopilot {autopilot_id}")
        logger.info("MAVLink connected to %s via %s", self.autopilot, self.connection_string)

    def is_connected(self) -> bool:
        """Return whether the link is open."""
        return self.connection is not None

    def _send_heartbeat(self, force: bool = False) -> None:
        """Send a companion-computer heartbeat at most once per second."""
        now = time.monotonic()
        if not force and now - self._last_heartbeat < HEARTBEAT_INTERVAL_S:
            return
        # MAV_TYPE_ONBOARD_CONTROLLER=18, MAV_AUTOPILOT_INVALID=8, MAV_STATE_ACTIVE=4
        self.connection.mav.heartbeat_send(18, 8, 0, 0, 4)
        self._last_heartbeat = now

    def send_vision_position_estimate(
        self,
        timestamp_us: int | None,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> bool:
        """Send one VISION_POSITION_ESTIMATE (metres, radians, local FRD/NED-style frame).

        timestamp_us: capture time in microseconds (UNIX or since boot); None uses now.
        Returns False without sending when any value is non-finite, so a bad vision frame
        can never reach the flight controller's estimator.
        """
        if self.connection is None:
            raise RuntimeError("MAVLinkPublisher is not connected; call connect() first.")
        values = (float(x), float(y), float(z), float(roll), float(pitch), float(yaw))
        if not all(math.isfinite(v) for v in values):
            logger.warning("Dropping non-finite vision pose: %s", values)
            return False
        if timestamp_us is None:
            timestamp_us = time.time_ns() // 1000
        self._send_heartbeat()
        self.connection.mav.vision_position_estimate_send(int(timestamp_us), *values)
        self.messages_sent += 1
        return True

    def send_pose(self, T_cw: np.ndarray, scale: float, timestamp_us: int | None = None) -> bool:
        """Convert an EyeMap pose with `pose_to_frd` and publish it.

        `scale` must be the metric scale (metres per EyeMap unit); never publish a
        relative-scale pose to a flight controller.
        """
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be a positive, finite metres-per-unit factor")
        return self.send_vision_position_estimate(
            timestamp_us, *pose_to_frd(T_cw, scale=scale, camera_to_body=self.camera_to_body)
        )

    def close(self) -> None:
        """Close the MAVLink connection if it is open."""
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:  # closing a dead link must never raise
                logger.debug("Ignoring error while closing MAVLink connection", exc_info=True)

    def __enter__(self) -> MAVLinkPublisher:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
