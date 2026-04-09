from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


@dataclass
class CameraFrame:
    rgb_path_hint: str
    depth_path_hint: str
    width: int
    height: int
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None


@dataclass
class RobotState:
    joint_positions: List[float]
    ee_position_m: Vector3
    ee_yaw_deg: float
    gripper_opening_m: float
    ee_quaternion_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)


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
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    contact_xyz_m: Optional[Vector3] = None


@dataclass
class ExecutionResult:
    success: bool
    state_trace: List[str] = field(default_factory=list)
    message: str = ""
    failure_reason: str = ""
    grasp: Optional[RefinedGrasp] = None
