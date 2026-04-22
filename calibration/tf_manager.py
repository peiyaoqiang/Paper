from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from common.types import Quaternion, Vector3


@dataclass
class TFConfig:
    camera_to_ee_translation_m: Vector3
    camera_to_ee_quaternion_xyzw: Tuple[float, float, float, float]
    fx: float
    fy: float
    cx: float
    cy: float


class TFManager:
    """
    Minimal transform helper.

    Applies the calibrated hand-eye transform.

    The stored calibration in this project comes from easy_handeye2 and is
    published as:
    `parent=end_effector_link`, `child=camera_color_optical_frame`.
    That means the calibrated translation/quaternion represent the EE->camera
    transform, not camera->EE. To map a 3D point from camera coordinates into
    EE coordinates, we must invert that rigid transform.
    """

    def __init__(self, config: TFConfig) -> None:
        self.config = config

    def project_pixel_to_camera_xyz(
        self,
        pixel_xy: Tuple[int, int],
        depth_m: float,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
    ) -> Vector3:
        u, v = pixel_xy
        fx = self.config.fx if fx is None else fx
        fy = self.config.fy if fy is None else fy
        cx = self.config.cx if cx is None else cx
        cy = self.config.cy if cy is None else cy
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        return (x, y, z)

    def _quat_to_rotation_matrix(self) -> tuple[tuple[float, float, float], ...]:
        qx, qy, qz, qw = self.config.camera_to_ee_quaternion_xyzw
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            raise ValueError("camera_to_ee_quaternion_xyzw must not be zero.")
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        return (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        )

    def _quat_conjugate(self) -> Quaternion:
        qx, qy, qz, qw = self.config.camera_to_ee_quaternion_xyzw
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            raise ValueError("camera_to_ee_quaternion_xyzw must not be zero.")
        return (-qx / norm, -qy / norm, -qz / norm, qw / norm)

    def _quat_to_rotation_matrix_from_xyzw(
        self,
        quaternion_xyzw: Quaternion,
    ) -> tuple[tuple[float, float, float], ...]:
        qx, qy, qz, qw = quaternion_xyzw
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            raise ValueError("quaternion_xyzw must not be zero.")
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        return (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        )

    def _rotate_camera_to_ee(self, camera_xyz_m: Vector3) -> Vector3:
        rotation = self._quat_to_rotation_matrix_from_xyzw(self._quat_conjugate())
        return tuple(
            sum(rotation[row][col] * camera_xyz_m[col] for col in range(3))
            for row in range(3)
        )

    def camera_vector_to_ee_vector(self, camera_vector_m: Vector3) -> Vector3:
        return self._rotate_camera_to_ee(camera_vector_m)

    def _rotate_ee_to_base_yaw_only(self, ee_relative_xyz_m: Vector3, ee_yaw_deg: float) -> Vector3:
        yaw_rad = math.radians(ee_yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        x, y, z = ee_relative_xyz_m
        return (
            cos_yaw * x - sin_yaw * y,
            sin_yaw * x + cos_yaw * y,
            z,
        )

    def _rotate_ee_to_base_full(self, ee_relative_xyz_m: Vector3, ee_quaternion_xyzw: Quaternion) -> Vector3:
        qx, qy, qz, qw = ee_quaternion_xyzw
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            raise ValueError("ee_quaternion_xyzw must not be zero.")
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        rotation = (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        )
        return tuple(
            sum(rotation[row][col] * ee_relative_xyz_m[col] for col in range(3))
            for row in range(3)
        )

    def camera_xyz_to_base_xyz(
        self,
        camera_xyz_m: Vector3,
        ee_xyz_m: Vector3,
        ee_yaw_deg: float,
        ee_quaternion_xyzw: Quaternion | None = None,
    ) -> Vector3:
        ee_to_camera_translation = self.config.camera_to_ee_translation_m
        camera_minus_translation = tuple(
            point - offset
            for point, offset in zip(camera_xyz_m, ee_to_camera_translation)
        )
        ee_relative_xyz = self._rotate_camera_to_ee(camera_minus_translation)
        if ee_quaternion_xyzw is not None:
            ee_relative_in_base = self._rotate_ee_to_base_full(ee_relative_xyz, ee_quaternion_xyzw)
        else:
            ee_relative_in_base = self._rotate_ee_to_base_yaw_only(ee_relative_xyz, ee_yaw_deg)
        return tuple(
            ee + offset
            for ee, offset in zip(
                ee_xyz_m,
                ee_relative_in_base,
            )
        )

    def camera_vector_to_base_offset(
        self,
        camera_vector_m: Vector3,
        ee_yaw_deg: float,
        ee_quaternion_xyzw: Quaternion | None = None,
    ) -> Vector3:
        ee_relative_vector = self.camera_vector_to_ee_vector(camera_vector_m)
        return self.ee_relative_xyz_to_base_offset(
            ee_relative_vector,
            ee_yaw_deg,
            ee_quaternion_xyzw,
        )

    def ee_relative_xyz_to_base_offset(
        self,
        ee_relative_xyz_m: Vector3,
        ee_yaw_deg: float,
        ee_quaternion_xyzw: Quaternion | None = None,
    ) -> Vector3:
        if ee_quaternion_xyzw is not None:
            return self._rotate_ee_to_base_full(ee_relative_xyz_m, ee_quaternion_xyzw)
        return self._rotate_ee_to_base_yaw_only(ee_relative_xyz_m, ee_yaw_deg)

    def ee_relative_xyz_to_base_xyz(
        self,
        ee_relative_xyz_m: Vector3,
        ee_xyz_m: Vector3,
        ee_yaw_deg: float,
        ee_quaternion_xyzw: Quaternion | None = None,
    ) -> Vector3:
        base_offset = self.ee_relative_xyz_to_base_offset(
            ee_relative_xyz_m,
            ee_yaw_deg,
            ee_quaternion_xyzw,
        )
        return tuple(
            ee_axis + offset_axis
            for ee_axis, offset_axis in zip(ee_xyz_m, base_offset)
        )
