from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from common.types import PolicyAction, RobotState, SafeAction


Vector3 = Tuple[float, float, float]


@dataclass
class ActionAdapterConfig:
    max_translation_step_m: float
    max_rotation_step_deg: float
    workspace_xyz_min: Vector3
    workspace_xyz_max: Vector3
    safety_clipping_enabled: bool = True
    workspace_enforced: bool = True


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

    def _clip_to_workspace(self, delta_xyz_m: Vector3, robot_state: RobotState | None) -> tuple[Vector3, bool]:
        if not self.config.workspace_enforced:
            return (delta_xyz_m, False)
        if robot_state is None:
            return (delta_xyz_m, False)

        clipped = False
        clipped_values = []
        for axis_idx, delta in enumerate(delta_xyz_m):
            current = robot_state.ee_position_m[axis_idx]
            min_limit = self.config.workspace_xyz_min[axis_idx]
            max_limit = self.config.workspace_xyz_max[axis_idx]
            next_value = current + delta

            # If the robot is already outside the workspace on this axis, do not command a large
            # jump back to the boundary in a single step. Only allow motion that heads back inward.
            if current < min_limit:
                if delta < 0.0:
                    clipped_values.append(0.0)
                    clipped = True
                else:
                    clipped_values.append(delta)
                continue
            if current > max_limit:
                if delta > 0.0:
                    clipped_values.append(0.0)
                    clipped = True
                else:
                    clipped_values.append(delta)
                continue

            if next_value < min_limit:
                clipped_values.append(min_limit - current)
                clipped = True
            elif next_value > max_limit:
                clipped_values.append(max_limit - current)
                clipped = True
            else:
                clipped_values.append(delta)
        return (tuple(clipped_values), clipped)

    def adapt(self, action: PolicyAction, robot_state: RobotState | None = None) -> SafeAction:
        if self.config.safety_clipping_enabled:
            delta_xyz_m, clipped_xyz_step = self._clip_delta(action.delta_xyz_m)
        else:
            delta_xyz_m = action.delta_xyz_m
            clipped_xyz_step = False
        delta_xyz_m, clipped_xyz_workspace = self._clip_to_workspace(delta_xyz_m, robot_state)
        delta_yaw_deg = action.delta_yaw_deg
        clipped_yaw = False
        if self.config.safety_clipping_enabled:
            if delta_yaw_deg > self.config.max_rotation_step_deg:
                delta_yaw_deg = self.config.max_rotation_step_deg
                clipped_yaw = True
            elif delta_yaw_deg < -self.config.max_rotation_step_deg:
                delta_yaw_deg = -self.config.max_rotation_step_deg
                clipped_yaw = True

        clipped = clipped_xyz_step or clipped_xyz_workspace or clipped_yaw
        if clipped_xyz_workspace:
            rejection_reason = "Action clipped to workspace limits"
        elif clipped:
            rejection_reason = "Action clipped to safety limits"
        else:
            rejection_reason = ""

        return SafeAction(
            delta_xyz_m=delta_xyz_m,
            delta_yaw_deg=delta_yaw_deg,
            gripper_command=action.gripper_command,
            clipped=clipped,
            rejection_reason=rejection_reason,
        )
