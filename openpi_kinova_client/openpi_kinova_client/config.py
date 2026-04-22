from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    connect_retry_s: float = 2.0
    ping_interval_s: float | None = None
    ping_timeout_s: float | None = None
    close_timeout_s: float = 10.0


@dataclass(frozen=True)
class ObservationConfig:
    prompt: str = "pick up the object"
    image_size: int = 224
    camera_index: int = 0
    dummy_images: bool = False


@dataclass(frozen=True)
class AdapterConfig:
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    gripper_threshold: float = 0.0
    invert_gripper: bool = False


@dataclass(frozen=True)
class SafetyConfig:
    max_abs_translation_m: float = 0.015
    max_abs_rotation_rad: float = 0.10
    workspace_min_xyz_m: tuple[float, float, float] = (0.20, -0.55, 0.02)
    workspace_max_xyz_m: tuple[float, float, float] = (0.80, 0.45, 0.60)
    workspace_enforced: bool = True
    max_action_age_s: float = 2.0
    estop_file: Path = Path("/tmp/openpi_kinova_estop")


@dataclass(frozen=True)
class KinovaConfig:
    robot_ip: str = "192.168.1.10"
    username: str = "admin"
    password: str = "admin"
    mqtt_port: int = 1883
    session_inactivity_timeout_ms: int = 60000
    connection_inactivity_timeout_ms: int = 2000
    command_dt_s: float = 0.25
    gripper_open: float = 0.0
    gripper_closed: float = 1.0
    use_twist: bool = True
