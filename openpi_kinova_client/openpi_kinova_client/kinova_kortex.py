from __future__ import annotations

import logging
import math
import time
from types import ModuleType
from typing import Any

from .config import KinovaConfig
from .types import Action7, RobotState


logger = logging.getLogger(__name__)


class KortexNotInstalled(RuntimeError):
    pass


def _import_kortex() -> dict[str, Any]:
    try:
        from kortex_api.MqttTransport import MqttTransport
        from kortex_api.RouterClient import RouterClient
        from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
        from kortex_api.autogen.client_stubs.SessionClientRpc import SessionClient
        from kortex_api.autogen.messages import Base_pb2, Session_pb2
    except ImportError as exc:  # pragma: no cover - depends on robot laptop
        raise KortexNotInstalled(
            "kortex_api is not installed. Install Kinova's Kortex Python wheel in this environment."
        ) from exc

    return {
        "MqttTransport": MqttTransport,
        "RouterClient": RouterClient,
        "BaseClient": BaseClient,
        "SessionClient": SessionClient,
        "Base_pb2": Base_pb2,
        "Session_pb2": Session_pb2,
    }


class KinovaKortexController:
    """Minimal Kinova Gen3 controller using Kortex Python API.

    This intentionally exposes only:
    - state read
    - small Cartesian delta command
    - gripper position command
    - stop / quick stop
    """

    def __init__(self, config: KinovaConfig) -> None:
        self.config = config
        self.kortex: dict[str, Any] = {}
        self.transport = None
        self.router = None
        self.session_client = None
        self.base = None
        self._last_gripper_position = config.gripper_open

    def connect(self) -> None:
        self.kortex = _import_kortex()
        error_callback = lambda exc: logger.error("Kortex router error: %s", exc)
        self.transport = self.kortex["MqttTransport"]()
        self.router = self.kortex["RouterClient"](self.transport, error_callback)
        self.transport.connect(self.config.robot_ip, self.config.mqtt_port)

        session_info = self.kortex["Session_pb2"].CreateSessionInfo()
        session_info.username = self.config.username
        session_info.password = self.config.password
        session_info.session_inactivity_timeout = self.config.session_inactivity_timeout_ms
        session_info.connection_inactivity_timeout = self.config.connection_inactivity_timeout_ms

        self.session_client = self.kortex["SessionClient"](self.router)
        logger.info("Creating Kortex session with %s", self.config.robot_ip)
        self.session_client.CreateSession(session_info)
        self.base = self.kortex["BaseClient"](self.router)
        logger.info("Kortex session ready")

    def close(self) -> None:
        if self.session_client is not None:
            try:
                self.session_client.CloseSession()
            except Exception:
                logger.exception("Failed to close Kortex session cleanly")
        if self.transport is not None:
            self.transport.disconnect()

    def __enter__(self) -> "KinovaKortexController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def get_state(self) -> RobotState:
        if self.base is None:
            raise RuntimeError("Kinova controller is not connected")
        pose = self.base.GetMeasuredCartesianPose()
        joint_angles = self.base.GetMeasuredJointAngles()
        joints_rad = tuple(math.radians(j.value) for j in joint_angles.joint_angles)
        return RobotState(
            ee_xyz_m=(float(pose.x), float(pose.y), float(pose.z)),
            rpy_rad=(math.radians(pose.theta_x), math.radians(pose.theta_y), math.radians(pose.theta_z)),
            joint_position=joints_rad,
            gripper_position=self._last_gripper_position,
        )

    def apply_action(self, action: Action7) -> None:
        if self.config.use_twist:
            self.send_twist_delta(action)
        else:
            self.send_reach_pose_delta(action)
        self.set_gripper(action.gripper)

    def send_twist_delta(self, action: Action7) -> None:
        if self.base is None:
            raise RuntimeError("Kinova controller is not connected")
        Base_pb2: ModuleType = self.kortex["Base_pb2"]
        dt = max(0.05, self.config.command_dt_s)

        command = Base_pb2.TwistCommand()
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.duration = dt
        command.twist.linear_x = action.dx / dt
        command.twist.linear_y = action.dy / dt
        command.twist.linear_z = action.dz / dt
        # Kortex Cartesian orientation fields are expressed in degrees; twist angular
        # values are treated here as degrees/s to match that API convention.
        command.twist.angular_x = math.degrees(action.droll) / dt
        command.twist.angular_y = math.degrees(action.dpitch) / dt
        command.twist.angular_z = math.degrees(action.dyaw) / dt

        self.base.SendTwistCommand(command)
        time.sleep(dt)
        self.stop()

    def send_reach_pose_delta(self, action: Action7) -> None:
        if self.base is None:
            raise RuntimeError("Kinova controller is not connected")
        Base_pb2: ModuleType = self.kortex["Base_pb2"]
        current = self.base.GetMeasuredCartesianPose()

        command = Base_pb2.Action()
        command.name = "openpi small cartesian delta"
        target = command.reach_pose.target_pose
        target.x = current.x + action.dx
        target.y = current.y + action.dy
        target.z = current.z + action.dz
        target.theta_x = current.theta_x + math.degrees(action.droll)
        target.theta_y = current.theta_y + math.degrees(action.dpitch)
        target.theta_z = current.theta_z + math.degrees(action.dyaw)
        self.base.ExecuteAction(command)
        time.sleep(self.config.command_dt_s)

    def set_gripper(self, value: float) -> None:
        if self.base is None:
            raise RuntimeError("Kinova controller is not connected")
        Base_pb2: ModuleType = self.kortex["Base_pb2"]
        target = self.config.gripper_closed if value >= 0.5 else self.config.gripper_open

        command = Base_pb2.GripperCommand()
        command.mode = Base_pb2.GRIPPER_POSITION
        finger = command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = float(target)
        self.base.SendGripperCommand(command)
        self._last_gripper_position = float(target)

    def stop(self) -> None:
        if self.base is not None:
            self.base.Stop()

    def quick_stop(self) -> None:
        if self.base is not None:
            self.base.ApplyQuickStop()
