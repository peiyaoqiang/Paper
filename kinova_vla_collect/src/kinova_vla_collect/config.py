from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TaskConfig:
    name: str
    prompt: str


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    robot: str
    camera: str


@dataclass(frozen=True)
class WorkspaceConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class ControlConfig:
    hz: float
    max_delta_m: float
    deadzone: float
    allow_motion_when_not_recording: bool
    max_steps: int
    workspace: WorkspaceConfig


@dataclass(frozen=True)
class HardwareConfig:
    dry_run: bool


@dataclass(frozen=True)
class XboxConfig:
    device_index: int
    mapping: dict[str, Any]
    debug: bool
    dry_run_mode: str


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fps: int
    serial: str | None


@dataclass(frozen=True)
class KinovaConfig:
    ip: str
    username: str
    password: str
    max_linear_speed: float
    mode: str
    joint_state_topic: str
    twist_command_topic: str
    base_frame: str
    ee_frame: str
    twist_command_frame: str
    sequential_axis_commands: bool
    state_timeout_s: float
    twist_command_duration_s: float
    twist_publish_rate_hz: float
    twist_stop_duration_s: float


@dataclass(frozen=True)
class GripperConfig:
    mode: str
    host: str
    port: int
    unit_id: int
    serial_port: str
    baudrate: int
    timeout_s: float
    open_pos_mm: float
    close_pos_mm: float
    max_stroke_mm: float
    speed: int
    close_torque: int
    open_torque: int
    acc_dec: int
    parity: str
    stopbits: int
    enable_rs485_mode: bool
    accept_pos_reached_as_success: bool
    rs485_rts_level_for_tx: bool
    rs485_rts_level_for_rx: bool
    rs485_delay_before_tx: float
    rs485_delay_before_rx: float
    open_timeout_s: float
    close_timeout_s: float
    tip_offset_ee_m: tuple[float, float, float]


@dataclass(frozen=True)
class AppConfig:
    task: TaskConfig
    dataset: DatasetConfig
    control: ControlConfig
    hardware: HardwareConfig
    xbox: XboxConfig
    camera: CameraConfig
    kinova: KinovaConfig
    gripper: GripperConfig


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid config section: {key}")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    task = _require_mapping(raw, "task")
    dataset = _require_mapping(raw, "dataset")
    control = _require_mapping(raw, "control")
    workspace = _require_mapping(control, "workspace")
    hardware = _require_mapping(raw, "hardware")
    xbox = _require_mapping(raw, "xbox")
    camera = _require_mapping(raw, "camera")
    kinova = _require_mapping(raw, "kinova")
    gripper = _require_mapping(raw, "gripper")

    return AppConfig(
        task=TaskConfig(name=str(task["name"]), prompt=str(task["prompt"])),
        dataset=DatasetConfig(
            root=Path(str(dataset["root"])),
            robot=str(dataset["robot"]),
            camera=str(dataset["camera"]),
        ),
        control=ControlConfig(
            hz=float(control["hz"]),
            max_delta_m=float(control["max_delta_m"]),
            deadzone=float(control["deadzone"]),
            allow_motion_when_not_recording=bool(control.get("allow_motion_when_not_recording", True)),
            max_steps=int(control.get("max_steps", 500)),
            workspace=WorkspaceConfig(
                x_min=float(workspace["x_min"]),
                x_max=float(workspace["x_max"]),
                y_min=float(workspace["y_min"]),
                y_max=float(workspace["y_max"]),
                z_min=float(workspace["z_min"]),
                z_max=float(workspace["z_max"]),
            ),
        ),
        hardware=HardwareConfig(dry_run=bool(hardware["dry_run"])),
        xbox=XboxConfig(
            device_index=int(xbox["device_index"]),
            mapping=dict(xbox.get("mapping", {})),
            debug=bool(xbox.get("debug", False)),
            dry_run_mode=str(xbox.get("dry_run_mode", "keyboard")),
        ),
        camera=CameraConfig(
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=int(camera["fps"]),
            serial=None if camera.get("serial") is None else str(camera["serial"]),
        ),
        kinova=KinovaConfig(
            ip=str(kinova["ip"]),
            username=str(kinova["username"]),
            password=str(kinova["password"]),
            max_linear_speed=float(kinova.get("max_linear_speed", 0.05)),
            mode=str(kinova.get("mode", "kortex_twist")),
            joint_state_topic=str(kinova.get("joint_state_topic", "/joint_states")),
            twist_command_topic=str(kinova.get("twist_command_topic", "/twist_controller/commands")),
            base_frame=str(kinova.get("base_frame", "base_link")),
            ee_frame=str(kinova.get("ee_frame", "end_effector_link")),
            twist_command_frame=str(kinova.get("twist_command_frame", "tool_frame")),
            sequential_axis_commands=bool(kinova.get("sequential_axis_commands", True)),
            state_timeout_s=float(kinova.get("state_timeout_s", 5.0)),
            twist_command_duration_s=float(kinova.get("twist_command_duration_s", 0.2)),
            twist_publish_rate_hz=float(kinova.get("twist_publish_rate_hz", 20.0)),
            twist_stop_duration_s=float(kinova.get("twist_stop_duration_s", 0.3)),
        ),
        gripper=GripperConfig(
            mode=str(gripper.get("mode", "tcp")),
            host=str(gripper["host"]),
            port=int(gripper["port"]),
            unit_id=int(gripper["unit_id"]),
            serial_port=str(gripper.get("serial_port", gripper.get("ctag_serial_port", "/dev/ttyUSB0"))),
            baudrate=int(gripper.get("baudrate", gripper.get("ctag_baudrate", 115200))),
            timeout_s=float(gripper.get("timeout_s", gripper.get("ctag_timeout_s", 4.0))),
            open_pos_mm=float(gripper.get("open_pos_mm", gripper.get("ctag_open_pos_mm", 0.0))),
            close_pos_mm=float(gripper.get("close_pos_mm", gripper.get("ctag_close_pos_mm", 120.0))),
            max_stroke_mm=float(gripper.get("max_stroke_mm", gripper.get("ctag_max_stroke_mm", 120.0))),
            speed=int(gripper.get("speed", gripper.get("ctag_speed", 30))),
            close_torque=int(gripper.get("close_torque", gripper.get("ctag_close_torque", 10))),
            open_torque=int(gripper.get("open_torque", gripper.get("ctag_open_torque", 100))),
            acc_dec=int(gripper.get("acc_dec", gripper.get("ctag_acc_dec", 2000))),
            parity=str(gripper.get("parity", gripper.get("ctag_parity", "N"))),
            stopbits=int(gripper.get("stopbits", gripper.get("ctag_stopbits", 1))),
            enable_rs485_mode=bool(gripper.get("enable_rs485_mode", gripper.get("ctag_enable_rs485_mode", False))),
            accept_pos_reached_as_success=bool(
                gripper.get(
                    "accept_pos_reached_as_success",
                    gripper.get("ctag_accept_pos_reached_as_success", True),
                )
            ),
            rs485_rts_level_for_tx=bool(
                gripper.get("rs485_rts_level_for_tx", gripper.get("ctag_rs485_rts_level_for_tx", True))
            ),
            rs485_rts_level_for_rx=bool(
                gripper.get("rs485_rts_level_for_rx", gripper.get("ctag_rs485_rts_level_for_rx", False))
            ),
            rs485_delay_before_tx=float(
                gripper.get("rs485_delay_before_tx", gripper.get("ctag_rs485_delay_before_tx", 0.0))
            ),
            rs485_delay_before_rx=float(
                gripper.get("rs485_delay_before_rx", gripper.get("ctag_rs485_delay_before_rx", 0.0))
            ),
            open_timeout_s=float(gripper.get("open_timeout_s", 4.0)),
            close_timeout_s=float(gripper.get("close_timeout_s", 5.0)),
            tip_offset_ee_m=tuple(float(value) for value in gripper.get("tip_offset_ee_m", [0.0, 0.0, 0.28])),
        ),
    )
