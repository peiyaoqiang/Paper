from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from common.types import Vector3


@dataclass
class TFConfig:
    camera_to_ee_translation_m: Vector3
    fx: float
    fy: float
    cx: float
    cy: float


class TFManager:
    """
    Minimal transform helper.

    Assumes aligned camera axes for the first scaffold version.
    """

    def __init__(self, config: TFConfig) -> None:
        self.config = config

    def project_pixel_to_camera_xyz(
        self, pixel_xy: Tuple[int, int], depth_m: float
    ) -> Vector3:
        u, v = pixel_xy
        x = (u - self.config.cx) * depth_m / self.config.fx
        y = (v - self.config.cy) * depth_m / self.config.fy
        z = depth_m
        return (x, y, z)

    def camera_xyz_to_base_xyz(self, camera_xyz_m: Vector3, ee_xyz_m: Vector3) -> Vector3:
        return tuple(
            ee + camera_offset + point
            for ee, camera_offset, point in zip(
                ee_xyz_m,
                self.config.camera_to_ee_translation_m,
                camera_xyz_m,
            )
        )
