from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class WorkspaceLimits:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains_position(self, position: FloatArray) -> bool:
        x, y, z = float(position[0]), float(position[1]), float(position[2])
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


@dataclass(frozen=True)
class SafetyLimiter:
    max_delta_m: float
    workspace: WorkspaceLimits
    max_delta_rad: float = 0.03490658503988659

    def limit_action(self, action: FloatArray, current_position: FloatArray) -> FloatArray:
        limited = np.array(action, dtype=np.float32, copy=True)
        if limited.shape not in {(4,), (7,)}:
            raise ValueError(f"Expected action shape (4,) or (7,), got {limited.shape}")
        limited[:3] = np.clip(limited[:3], -self.max_delta_m, self.max_delta_m)
        if limited.shape == (7,):
            limited[3:6] = np.clip(limited[3:6], -self.max_delta_rad, self.max_delta_rad)
        target_position = current_position.astype(np.float32) + limited[:3]
        if not self.workspace.contains_position(target_position):
            limited[:3] = 0.0
        gripper_index = 6 if limited.shape == (7,) else 3
        limited[gripper_index] = float(np.clip(limited[gripper_index], -1.0, 1.0))
        if limited[gripper_index] > 0.5:
            limited[gripper_index] = 1.0
        elif limited[gripper_index] < -0.5:
            limited[gripper_index] = -1.0
        else:
            limited[gripper_index] = 0.0
        return limited
