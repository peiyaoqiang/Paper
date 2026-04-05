from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from common.types import RobotState


Vector3 = Tuple[float, float, float]


@dataclass
class KinovaConfig:
    max_translation_step_m: float
    max_rotation_step_deg: float


@dataclass
class KinovaDriver:
    """
    Minimal Kinova arm interface with an in-memory mock state.

    Replace movement calls with Kinova API or ROS services.
    """

    config: KinovaConfig
    ee_position_m: Vector3 = (0.45, 0.00, 0.25)
    ee_yaw_deg: float = 0.0
    joint_positions: List[float] = field(default_factory=lambda: [0.0] * 7)
    gripper_opening_m: float = 0.08

    def get_state(self) -> RobotState:
        return RobotState(
            joint_positions=list(self.joint_positions),
            ee_position_m=self.ee_position_m,
            ee_yaw_deg=self.ee_yaw_deg,
            gripper_opening_m=self.gripper_opening_m,
        )

    def move_cartesian_delta(self, delta_xyz_m: Vector3, delta_yaw_deg: float) -> None:
        self.ee_position_m = tuple(
            current + delta for current, delta in zip(self.ee_position_m, delta_xyz_m)
        )
        self.ee_yaw_deg += delta_yaw_deg

    def set_gripper_opening(self, width_m: float) -> None:
        self.gripper_opening_m = width_m
