from __future__ import annotations

from typing import Any

import numpy as np

from .config import AdapterConfig
from .types import Action7


def extract_action_chunk(policy_response: dict[str, Any] | np.ndarray) -> np.ndarray:
    actions = policy_response["actions"] if isinstance(policy_response, dict) else policy_response
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    if chunk.ndim != 2:
        raise ValueError(f"Expected action chunk with shape [horizon, dim], got {chunk.shape}")
    if chunk.shape[1] < 7:
        raise ValueError(f"Expected at least 7 action dims, got {chunk.shape[1]}")
    return chunk


class ActionAdapter:
    """Map one openpi action row to [dx, dy, dz, droll, dpitch, dyaw, gripper]."""

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    def row_to_action7(self, row: np.ndarray) -> Action7:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if values.size < 7:
            raise ValueError(f"Expected at least 7 action dims, got {values.size}")

        raw_gripper = float(values[6])
        close = raw_gripper > self.config.gripper_threshold
        if self.config.invert_gripper:
            close = not close

        return Action7(
            dx=float(values[0]) * self.config.position_scale,
            dy=float(values[1]) * self.config.position_scale,
            dz=float(values[2]) * self.config.position_scale,
            droll=float(values[3]) * self.config.rotation_scale,
            dpitch=float(values[4]) * self.config.rotation_scale,
            dyaw=float(values[5]) * self.config.rotation_scale,
            gripper=1.0 if close else 0.0,
        )

    def chunk_to_actions(self, policy_response: dict[str, Any] | np.ndarray) -> list[Action7]:
        chunk = extract_action_chunk(policy_response)
        return [self.row_to_action7(row) for row in chunk]
