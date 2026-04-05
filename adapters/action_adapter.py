from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from common.types import PolicyAction, SafeAction


Vector3 = Tuple[float, float, float]


@dataclass
class ActionAdapterConfig:
    max_translation_step_m: float
    max_rotation_step_deg: float
    workspace_xyz_min: Vector3
    workspace_xyz_max: Vector3


class ActionAdapter:
    def __init__(self, config: ActionAdapterConfig) -> None:
        self.config = config

    def _clip_delta(self, delta_xyz_m: Vector3) -> tuple[Vector3, bool]:
        clipped = False
        clipped_values = []
        for value in delta_xyz_m:
            if value > self.config.max_translation_step_m:
                clipped_values.append(self.config.max_translation_step_m)
                clipped = True
            elif value < -self.config.max_translation_step_m:
                clipped_values.append(-self.config.max_translation_step_m)
                clipped = True
            else:
                clipped_values.append(value)
        return (tuple(clipped_values), clipped)

    def adapt(self, action: PolicyAction) -> SafeAction:
        delta_xyz_m, clipped_xyz = self._clip_delta(action.delta_xyz_m)
        delta_yaw_deg = action.delta_yaw_deg
        clipped_yaw = False
        if delta_yaw_deg > self.config.max_rotation_step_deg:
            delta_yaw_deg = self.config.max_rotation_step_deg
            clipped_yaw = True
        elif delta_yaw_deg < -self.config.max_rotation_step_deg:
            delta_yaw_deg = -self.config.max_rotation_step_deg
            clipped_yaw = True

        return SafeAction(
            delta_xyz_m=delta_xyz_m,
            delta_yaw_deg=delta_yaw_deg,
            gripper_command=action.gripper_command,
            clipped=clipped_xyz or clipped_yaw,
            rejection_reason="" if not (clipped_xyz or clipped_yaw) else "Action clipped to safety limits",
        )
