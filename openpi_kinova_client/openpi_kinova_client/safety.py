from __future__ import annotations

import math
import time

from .config import SafetyConfig
from .types import Action7, RobotState, SafeAction


ZERO_ACTION = Action7(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class EmergencyStop(RuntimeError):
    pass


class SafetyLimiter:
    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    def estop_active(self) -> bool:
        return self.config.estop_file.exists()

    def filter(
        self,
        action: Action7,
        robot_state: RobotState | None,
        *,
        action_timestamp: float | None = None,
    ) -> SafeAction:
        if self.estop_active():
            return SafeAction(ZERO_ACTION, clipped=True, stop=True, reason=f"estop file exists: {self.config.estop_file}")

        if action_timestamp is not None and time.monotonic() - action_timestamp > self.config.max_action_age_s:
            return SafeAction(ZERO_ACTION, clipped=True, stop=True, reason="stale action")

        values = action.as_tuple()
        if not all(math.isfinite(value) for value in values):
            return SafeAction(ZERO_ACTION, clipped=True, stop=True, reason="non-finite action")

        clipped = False
        dx, c0 = self._clip_abs(action.dx, self.config.max_abs_translation_m)
        dy, c1 = self._clip_abs(action.dy, self.config.max_abs_translation_m)
        dz, c2 = self._clip_abs(action.dz, self.config.max_abs_translation_m)
        droll, c3 = self._clip_abs(action.droll, self.config.max_abs_rotation_rad)
        dpitch, c4 = self._clip_abs(action.dpitch, self.config.max_abs_rotation_rad)
        dyaw, c5 = self._clip_abs(action.dyaw, self.config.max_abs_rotation_rad)
        clipped = any((c0, c1, c2, c3, c4, c5))

        if robot_state is not None and self.config.workspace_enforced:
            (dx, dy, dz), workspace_clipped = self._clip_delta_to_workspace((dx, dy, dz), robot_state.ee_xyz_m)
            clipped = clipped or workspace_clipped

        safe = Action7(dx, dy, dz, droll, dpitch, dyaw, min(1.0, max(0.0, action.gripper)))
        return SafeAction(safe, clipped=clipped, stop=False, reason="clipped to safety limits" if clipped else "")

    @staticmethod
    def _clip_abs(value: float, limit: float) -> tuple[float, bool]:
        if value > limit:
            return limit, True
        if value < -limit:
            return -limit, True
        return value, False

    def _clip_delta_to_workspace(
        self,
        delta_xyz: tuple[float, float, float],
        current_xyz: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], bool]:
        clipped_values: list[float] = []
        clipped = False
        for current, delta, lo, hi in zip(
            current_xyz,
            delta_xyz,
            self.config.workspace_min_xyz_m,
            self.config.workspace_max_xyz_m,
        ):
            # If the current pose is already outside the workspace, do not command
            # a large jump back to the boundary. Only allow the already-clipped
            # policy delta if it moves inward; block motion that goes farther out.
            if current < lo:
                if delta < 0.0:
                    clipped_values.append(0.0)
                    clipped = True
                else:
                    clipped_values.append(delta)
                continue
            if current > hi:
                if delta > 0.0:
                    clipped_values.append(0.0)
                    clipped = True
                else:
                    clipped_values.append(delta)
                continue

            target = current + delta
            if target < lo:
                clipped_values.append(lo - current)
                clipped = True
            elif target > hi:
                clipped_values.append(hi - current)
                clipped = True
            else:
                clipped_values.append(delta)
        return (clipped_values[0], clipped_values[1], clipped_values[2]), clipped
