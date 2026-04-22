from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotState:
    ee_xyz_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]
    joint_position: tuple[float, ...]
    gripper_position: float


@dataclass(frozen=True)
class Action7:
    dx: float
    dy: float
    dz: float
    droll: float
    dpitch: float
    dyaw: float
    gripper: float

    def as_tuple(self) -> tuple[float, float, float, float, float, float, float]:
        return (self.dx, self.dy, self.dz, self.droll, self.dpitch, self.dyaw, self.gripper)


@dataclass(frozen=True)
class SafeAction:
    action: Action7
    clipped: bool
    stop: bool
    reason: str = ""
