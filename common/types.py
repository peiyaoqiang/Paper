from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]


@dataclass
class CameraFrame:
    rgb_path_hint: str
    depth_path_hint: str
    width: int
    height: int


@dataclass
class RobotState:
    joint_positions: List[float]
    ee_position_m: Vector3
    ee_yaw_deg: float
    gripper_opening_m: float


@dataclass
class Observation:
    instruction: str
    frame: CameraFrame
    robot_state: RobotState


@dataclass
class PolicyAction:
    delta_xyz_m: Vector3
    delta_yaw_deg: float
    gripper_command: str
    confidence: float
    target_pixel: Optional[Tuple[int, int]] = None
    notes: str = ""


@dataclass
class SafeAction:
    delta_xyz_m: Vector3
    delta_yaw_deg: float
    gripper_command: str
    clipped: bool = False
    rejection_reason: str = ""


@dataclass
class RefinedGrasp:
    target_xyz_m: Vector3
    target_yaw_deg: float
    grasp_width_m: float
    quality: float
    source: str = "geometry"


@dataclass
class ExecutionResult:
    success: bool
    state_trace: List[str] = field(default_factory=list)
    message: str = ""
    failure_reason: str = ""
    grasp: Optional[RefinedGrasp] = None
