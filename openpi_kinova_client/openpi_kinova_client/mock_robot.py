from __future__ import annotations

from dataclasses import dataclass, field

from .types import Action7, RobotState


@dataclass
class MockRobot:
    state: RobotState = field(
        default_factory=lambda: RobotState(
            ee_xyz_m=(0.45, 0.0, 0.25),
            rpy_rad=(0.0, 0.0, 0.0),
            joint_position=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            gripper_position=0.0,
        )
    )

    def get_state(self) -> RobotState:
        return self.state

    def apply_action(self, action: Action7) -> None:
        x, y, z = self.state.ee_xyz_m
        r, p, yaw = self.state.rpy_rad
        self.state = RobotState(
            ee_xyz_m=(x + action.dx, y + action.dy, z + action.dz),
            rpy_rad=(r + action.droll, p + action.dpitch, yaw + action.dyaw),
            joint_position=self.state.joint_position,
            gripper_position=action.gripper,
        )

    def stop(self) -> None:
        pass
