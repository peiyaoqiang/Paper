from __future__ import annotations

from dataclasses import dataclass

from common.types import Observation, PolicyAction


@dataclass
class OpenVLAConfig:
    model_name: str = "openvla-mock"


class OpenVLAWrapper:
    """
    Mock wrapper that exposes the interface we want for the real system.

    Replace `predict_action` with actual model loading and inference.
    """

    def __init__(self, config: OpenVLAConfig) -> None:
        self.config = config

    def predict_action(self, observation: Observation) -> PolicyAction:
        instruction = observation.instruction.lower()
        is_pick = "pick" in instruction or "grasp" in instruction
        target_pixel = (observation.frame.width // 2, observation.frame.height // 2)
        delta_z = -0.03 if is_pick else 0.0
        notes = f"Mock {self.config.model_name} action for instruction: {observation.instruction}"
        return PolicyAction(
            delta_xyz_m=(0.00, 0.00, delta_z),
            delta_yaw_deg=0.0,
            gripper_command="open" if is_pick else "hold",
            confidence=0.68,
            target_pixel=target_pixel,
            notes=notes,
        )
