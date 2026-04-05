from __future__ import annotations

from dataclasses import dataclass

from drivers.kinova_driver import KinovaDriver


@dataclass
class GripperConfig:
    open_width_m: float
    close_width_m: float


class GripperDriver:
    """Thin wrapper around the robot gripper state."""

    def __init__(self, robot: KinovaDriver, config: GripperConfig) -> None:
        self.robot = robot
        self.config = config

    def open(self) -> None:
        self.robot.set_gripper_opening(self.config.open_width_m)

    def close(self) -> None:
        self.robot.set_gripper_opening(self.config.close_width_m)
