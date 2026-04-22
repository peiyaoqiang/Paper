from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .types import Action7, RobotState as OpenPIRobotState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.gripper_driver import GripperConfig, GripperDriver  # noqa: E402
from drivers.kinova_driver import KinovaConfig, KinovaDriver  # noqa: E402

from .ros2_rgb_camera import ROS2RGBCamera, ROS2RGBCameraConfig


def load_paper_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_camera(config: dict[str, Any], *, node_suffix: str = "openpi") -> ROS2RGBCamera:
    return ROS2RGBCamera(
        ROS2RGBCameraConfig(
            width=config["camera"]["width"],
            height=config["camera"]["height"],
            mode=config["camera"]["mode"],
            color_topic=config["camera"]["color_topic"],
            camera_info_topic=config["camera"].get("camera_info_topic", "/camera/camera/color/camera_info"),
            capture_timeout_s=config["camera"]["capture_timeout_s"],
            output_dir=config["camera"]["output_dir"],
            ros_node_name=f"{config['camera']['ros_node_name']}_{node_suffix}",
        )
    )


def build_robot(
    config: dict[str, Any],
    *,
    max_translation_step_m: float | None = None,
    max_rotation_step_deg: float | None = None,
    twist_command_duration_s: float | None = None,
    twist_stop_duration_s: float | None = None,
    combined_axis_commands: bool = False,
    twist_command_frame: str | None = None,
    node_suffix: str = "openpi",
) -> KinovaDriver:
    sequential_axis_commands = not combined_axis_commands
    if "sequential_axis_commands" in config["robot"] and not combined_axis_commands:
        sequential_axis_commands = bool(config["robot"]["sequential_axis_commands"])

    return KinovaDriver(
        KinovaConfig(
            max_translation_step_m=max_translation_step_m
            if max_translation_step_m is not None
            else config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=max_rotation_step_deg
            if max_rotation_step_deg is not None
            else config["robot"]["max_rotation_step_deg"],
            mode=config["robot"]["mode"],
            joint_state_topic=config["robot"]["joint_state_topic"],
            twist_command_topic=config["robot"]["twist_command_topic"],
            base_frame=config["robot"]["base_frame"],
            ee_frame=config["robot"]["ee_frame"],
            twist_command_frame=twist_command_frame
            if twist_command_frame is not None
            else config["robot"].get("twist_command_frame", "tool_frame"),
            sequential_axis_commands=sequential_axis_commands,
            ros_node_name=f"{config['robot']['ros_node_name']}_{node_suffix}",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=twist_command_duration_s
            if twist_command_duration_s is not None
            else config["robot"]["twist_command_duration_s"],
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=twist_stop_duration_s
            if twist_stop_duration_s is not None
            else config["robot"]["twist_stop_duration_s"],
        )
    )


def build_gripper(config: dict[str, Any], robot: KinovaDriver) -> GripperDriver:
    return GripperDriver(
        robot,
        GripperConfig(
            open_width_m=config["gripper"]["open_width_m"],
            close_width_m=config["gripper"]["close_width_m"],
            mode=config["gripper"].get("mode", "state_only"),
            ctag_serial_port=config["gripper"].get("ctag_serial_port", "/dev/ttyUSB0"),
            ctag_baudrate=config["gripper"].get("ctag_baudrate", 115200),
            ctag_device_id=config["gripper"].get("ctag_device_id", 1),
            ctag_timeout_s=config["gripper"].get("ctag_timeout_s", 0.2),
            ctag_open_pos_mm=config["gripper"].get("ctag_open_pos_mm", 850.0),
            ctag_close_pos_mm=config["gripper"].get("ctag_close_pos_mm", 0.0),
            ctag_max_stroke_mm=config["gripper"].get("ctag_max_stroke_mm", 850.0),
            ctag_speed=config["gripper"].get("ctag_speed", 80),
            ctag_close_torque=config["gripper"].get("ctag_close_torque", 80),
            ctag_open_torque=config["gripper"].get("ctag_open_torque", 40),
            ctag_acc_dec=config["gripper"].get("ctag_acc_dec", 80),
            ctag_parity=config["gripper"].get("ctag_parity", "N"),
            ctag_stopbits=config["gripper"].get("ctag_stopbits", 1),
            ctag_enable_rs485_mode=config["gripper"].get("ctag_enable_rs485_mode", False),
            ctag_accept_pos_reached_as_success=config["gripper"].get(
                "ctag_accept_pos_reached_as_success", False
            ),
            ctag_rs485_rts_level_for_tx=config["gripper"].get("ctag_rs485_rts_level_for_tx", True),
            ctag_rs485_rts_level_for_rx=config["gripper"].get("ctag_rs485_rts_level_for_rx", False),
            ctag_rs485_delay_before_tx=config["gripper"].get("ctag_rs485_delay_before_tx", 0.0),
            ctag_rs485_delay_before_rx=config["gripper"].get("ctag_rs485_delay_before_rx", 0.0),
            open_timeout_s=config["gripper"].get("open_timeout_s", 3.0),
            close_timeout_s=config["gripper"].get("close_timeout_s", 5.0),
        ),
    )


def rgb_from_camera_frame(rgb_path_hint: str) -> np.ndarray:
    return np.asarray(Image.open(rgb_path_hint).convert("RGB"), dtype=np.uint8)


def convert_robot_state(state, *, gripper_open_width_m: float) -> OpenPIRobotState:  # type: ignore[no-untyped-def]
    if hasattr(state, "joint_positions"):
        joint_values = state.joint_positions
        ee_xyz_m = state.ee_position_m
        ee_yaw_deg = state.ee_yaw_deg
        gripper_opening_m = state.gripper_opening_m
    else:
        joint_values = state.joint_position
        ee_xyz_m = state.ee_xyz_m
        ee_yaw_deg = math.degrees(state.rpy_rad[2])
        gripper_opening_m = state.gripper_position * gripper_open_width_m

    joints = tuple(float(value) for value in joint_values[:7])
    if len(joints) < 7:
        joints = joints + (0.0,) * (7 - len(joints))
    gripper_position = 0.0
    if gripper_open_width_m > 1e-9:
        gripper_position = float(np.clip(gripper_opening_m / gripper_open_width_m, 0.0, 1.0))
    return OpenPIRobotState(
        ee_xyz_m=tuple(float(value) for value in ee_xyz_m),
        rpy_rad=(0.0, 0.0, math.radians(float(ee_yaw_deg))),
        joint_position=joints,
        gripper_position=gripper_position,
    )


def action7_to_kinova_delta(action: Action7) -> tuple[tuple[float, float, float], float]:
    return ((action.dx, action.dy, action.dz), math.degrees(action.dyaw))
