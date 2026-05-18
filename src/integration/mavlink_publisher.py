# ============================================================
# CONTRIBUTOR STUB
# This file defines the interface for MAVLink pose publishing.
# To implement:
#   1. pip install pymavlink
#   2. Replace each NotImplementedError with real implementation
#   3. See docs/drone_deployment.md for Jetson-to-Pixhawk wiring
#   4. See docs/architecture.md for how this fits the pipeline
# ============================================================
"""MAVLink pose publisher for ArduPilot and PX4 integration.
CONTRIBUTOR STUB — implement to send EyeMap pose estimates to a flight controller."""

from __future__ import annotations

GUIDANCE_MESSAGE = (
    "MAVLinkPublisher.connect() is not implemented. "
    "Install pymavlink, establish a mavutil.mavlink_connection, "
    "and implement heartbeat handling. "
    "See docs/drone_deployment.md for wiring Jetson to Pixhawk."
)


class MAVLinkPublisher:
    """Define the MAVLink publishing contract for future contributors."""

    def __init__(self, connection_string: str = "/dev/ttyUSB0", baud: int = 115200) -> None:
        """Store connection settings without opening the serial link."""
        self.connection_string = connection_string
        self.baud = baud
        self.connection = None

    def connect(self) -> None:
        """Connect to a MAVLink endpoint once implemented by a contributor."""
        raise NotImplementedError(GUIDANCE_MESSAGE)

    def send_vision_position_estimate(
        self,
        timestamp_us: int,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> None:
        """Send a VISION_POSITION_ESTIMATE MAVLink message."""
        raise NotImplementedError(GUIDANCE_MESSAGE)

    def close(self) -> None:
        """Close the MAVLink connection when an implementation adds one."""
        pass
