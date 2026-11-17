"""Functional tests for MAVLink pose publishing: frame conversion, a fake link, and a real UDP loopback."""

from __future__ import annotations

import math
import socket
import threading
import unittest

import numpy as np

from src.integration.mavlink_publisher import (
    MOUNT_DOWNWARD,
    MOUNT_FORWARD,
    MAVLinkPublisher,
    pose_to_frd,
)


def make_T_cw(R_wc: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Build a world-to-camera pose from a camera-to-world rotation and camera center."""
    T = np.eye(4)
    T[:3, :3] = R_wc.T
    T[:3, 3] = -R_wc.T @ center
    return T


def rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class FakeMav:
    def __init__(self) -> None:
        self.heartbeats: list[tuple] = []
        self.poses: list[tuple] = []

    def heartbeat_send(self, *args) -> None:
        self.heartbeats.append(args)

    def vision_position_estimate_send(self, *args) -> None:
        self.poses.append(args)


class FakeConnection:
    def __init__(self, heartbeat) -> None:
        self.mav = FakeMav()
        self.heartbeat = heartbeat
        self.closed = False

    def wait_heartbeat(self, timeout=None):
        self.timeout = timeout
        return self.heartbeat

    def close(self) -> None:
        self.closed = True


class Heartbeat:
    def __init__(self, autopilot: int) -> None:
        self.autopilot = autopilot


def connected_publisher(autopilot: int = 3, **kwargs) -> tuple[MAVLinkPublisher, FakeConnection]:
    connection = FakeConnection(Heartbeat(autopilot))
    publisher = MAVLinkPublisher("fake", **kwargs)
    publisher.connect(connection_factory=lambda *a, **k: connection)
    return publisher, connection


class TestPoseToFrd(unittest.TestCase):
    def test_identity_pose_is_origin(self) -> None:
        self.assertEqual(pose_to_frd(np.eye(4)), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_forward_camera_axes_map_to_forward_right_down(self) -> None:
        # Camera moved 2 forward (cv z), 3 right (cv x), 1 down (cv y); scale 0.5 m/unit.
        x, y, z, roll, pitch, yaw = pose_to_frd(make_T_cw(np.eye(3), np.array([3.0, 1.0, 2.0])), scale=0.5)
        np.testing.assert_allclose([x, y, z], [1.0, 1.5, 0.5])
        np.testing.assert_allclose([roll, pitch, yaw], [0.0, 0.0, 0.0], atol=1e-12)

    def test_forward_camera_rotations(self) -> None:
        # cv +y points down, so rotating about it by +30 deg turns the nose right: yaw +30.
        _, _, _, roll, pitch, yaw = pose_to_frd(make_T_cw(rot_y(30.0), np.zeros(3)))
        np.testing.assert_allclose(np.degrees([roll, pitch, yaw]), [0.0, 0.0, 30.0], atol=1e-9)
        # cv +x points right, so rotating about it by +10 deg raises the nose: pitch +10.
        _, _, _, roll, pitch, yaw = pose_to_frd(make_T_cw(rot_x(10.0), np.zeros(3)))
        np.testing.assert_allclose(np.degrees([roll, pitch, yaw]), [0.0, 10.0, 0.0], atol=1e-9)

    def test_downward_camera(self) -> None:
        # Image-up is vehicle-forward: moving "up the image" (cv -y) is forward; cv z is down.
        T = make_T_cw(np.eye(3), np.array([0.0, -4.0, 1.0]))
        x, y, z, *_ = pose_to_frd(T, camera_to_body=MOUNT_DOWNWARD)
        np.testing.assert_allclose([x, y, z], [4.0, 0.0, 1.0], atol=1e-12)

    def test_mounts_are_proper_rotations(self) -> None:
        for mount in (MOUNT_FORWARD, MOUNT_DOWNWARD):
            np.testing.assert_allclose(mount @ mount.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(mount)), 1.0)

    def test_rejects_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            pose_to_frd(np.eye(3))
        with self.assertRaises(ValueError):
            pose_to_frd(np.eye(4), camera_to_body=np.eye(4))


class TestMAVLinkPublisherWithFakeLink(unittest.TestCase):
    def test_connect_announces_then_identifies_autopilot(self) -> None:
        publisher, connection = connected_publisher(autopilot=12, heartbeat_timeout_s=3.0)
        self.assertTrue(publisher.is_connected())
        self.assertEqual(publisher.autopilot, "PX4")
        self.assertEqual(len(connection.mav.heartbeats), 1, "heartbeat sent before waiting")
        self.assertEqual(connection.timeout, 3.0)
        self.assertEqual(connected_publisher(autopilot=3)[0].autopilot, "ArduPilot")

    def test_connect_passes_identity_to_factory(self) -> None:
        seen = {}

        def factory(device, **kwargs):
            seen.update(device=device, **kwargs)
            return FakeConnection(Heartbeat(3))

        MAVLinkPublisher("/dev/ttyTHS1", baud=921600).connect(connection_factory=factory)
        self.assertEqual(
            seen, {"device": "/dev/ttyTHS1", "baud": 921600, "source_system": 1, "source_component": 197}
        )

    def test_heartbeat_timeout_closes_link(self) -> None:
        connection = FakeConnection(None)
        publisher = MAVLinkPublisher("fake")
        with self.assertRaises(TimeoutError):
            publisher.connect(connection_factory=lambda *a, **k: connection)
        self.assertTrue(connection.closed)
        self.assertFalse(publisher.is_connected())

    def test_send_requires_connection(self) -> None:
        with self.assertRaises(RuntimeError):
            MAVLinkPublisher("fake").send_vision_position_estimate(0, 0, 0, 0, 0, 0, 0)

    def test_send_formats_message_and_counts(self) -> None:
        publisher, connection = connected_publisher()
        self.assertTrue(publisher.send_vision_position_estimate(123456, 1, 2, -3, 0.1, 0.2, 0.3))
        self.assertEqual(connection.mav.poses, [(123456, 1.0, 2.0, -3.0, 0.1, 0.2, 0.3)])
        self.assertTrue(publisher.send_vision_position_estimate(None, 0, 0, 0, 0, 0, 0))
        self.assertGreater(connection.mav.poses[1][0], 1_600_000_000_000_000, "defaults to UNIX microseconds")
        self.assertEqual(publisher.messages_sent, 2)

    def test_non_finite_pose_is_dropped(self) -> None:
        publisher, connection = connected_publisher()
        with self.assertLogs("src.integration.mavlink_publisher", level="WARNING"):
            self.assertFalse(publisher.send_vision_position_estimate(0, float("nan"), 0, 0, 0, 0, 0))
        self.assertEqual(connection.mav.poses, [])
        self.assertEqual(publisher.messages_sent, 0)

    def test_heartbeat_is_rate_limited(self) -> None:
        publisher, connection = connected_publisher()
        for _ in range(50):
            publisher.send_vision_position_estimate(0, 0, 0, 0, 0, 0, 0)
        self.assertEqual(len(connection.mav.heartbeats), 1)
        publisher._last_heartbeat -= 1.5  # pretend 1.5 s passed
        publisher.send_vision_position_estimate(0, 0, 0, 0, 0, 0, 0)
        self.assertEqual(len(connection.mav.heartbeats), 2)

    def test_send_pose_uses_mount_and_scale(self) -> None:
        publisher, connection = connected_publisher(camera_to_body=MOUNT_DOWNWARD)
        publisher.send_pose(make_T_cw(np.eye(3), np.array([0.0, -4.0, 1.0])), scale=2.0, timestamp_us=7)
        np.testing.assert_allclose(connection.mav.poses[0], (7, 8.0, 0.0, 2.0, 0.0, 0.0, 0.0), atol=1e-12)
        for bad_scale in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                publisher.send_pose(np.eye(4), scale=bad_scale)

    def test_close_is_idempotent_and_swallows_errors(self) -> None:
        publisher, connection = connected_publisher()
        connection.close = lambda: (_ for _ in ()).throw(OSError("port gone"))
        publisher.close()
        publisher.close()
        self.assertFalse(publisher.is_connected())


class TestMAVLinkLoopback(unittest.TestCase):
    """Send real MAVLink frames over UDP to a pymavlink peer acting as the flight controller."""

    def test_pose_round_trips_over_udp(self) -> None:
        try:
            from pymavlink import mavutil
        except ImportError:
            self.skipTest("pymavlink not installed")

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        vehicle = mavutil.mavlink_connection(f"udpin:127.0.0.1:{port}", source_system=1, source_component=1)
        self.addCleanup(vehicle.close)
        received = {}

        def flight_controller() -> None:
            # Learn the companion's address from its heartbeat, then answer as ArduPilot.
            received["companion_heartbeat"] = vehicle.recv_match(type="HEARTBEAT", blocking=True, timeout=5)
            vehicle.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_QUADROTOR, mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA, 0, 0, 0
            )
            received["pose"] = vehicle.recv_match(type="VISION_POSITION_ESTIMATE", blocking=True, timeout=5)

        thread = threading.Thread(target=flight_controller, daemon=True)
        thread.start()

        with MAVLinkPublisher(f"udpout:127.0.0.1:{port}", heartbeat_timeout_s=5.0) as publisher:
            self.assertEqual(publisher.autopilot, "ArduPilot")
            T = make_T_cw(rot_y(30.0), np.array([3.0, 1.0, 2.0]))
            self.assertTrue(publisher.send_pose(T, scale=0.5, timestamp_us=42))
            thread.join(timeout=10)

        heartbeat, pose = received.get("companion_heartbeat"), received.get("pose")
        self.assertIsNotNone(heartbeat)
        self.assertEqual(heartbeat.type, mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER)
        self.assertEqual(heartbeat.get_srcComponent(), mavutil.mavlink.MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY)
        self.assertIsNotNone(pose)
        self.assertEqual(pose.usec, 42)
        np.testing.assert_allclose([pose.x, pose.y, pose.z], [1.0, 1.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(
            np.degrees([pose.roll, pose.pitch, pose.yaw]), [0.0, 0.0, 30.0], atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
