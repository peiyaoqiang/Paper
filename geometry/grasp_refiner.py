from __future__ import annotations

from dataclasses import dataclass

from calibration.tf_manager import TFManager
from common.types import Observation, PolicyAction, RefinedGrasp, Vector3
from geometry.depth_filter import DepthFilter


@dataclass
class GraspRefinerConfig:
    approach_height_m: float
    refine_height_m: float
    gripper_tip_offset_ee_m: Vector3 = (0.0, 0.0, 0.0)
    default_grasp_width_m: float = 0.05


class GraspRefiner:
    def __init__(
        self,
        depth_filter: DepthFilter,
        tf_manager: TFManager,
        config: GraspRefinerConfig,
    ) -> None:
        self.depth_filter = depth_filter
        self.tf_manager = tf_manager
        self.config = config

    def refine(self, policy_action: PolicyAction, observation: Observation) -> RefinedGrasp:
        robot_state = observation.robot_state
        if policy_action.target_pixel is None:
            return RefinedGrasp(
                target_xyz_m=robot_state.ee_position_m,
                target_yaw_deg=robot_state.ee_yaw_deg,
                grasp_width_m=self.config.default_grasp_width_m,
                quality=0.30,
                source="fallback",
                contact_xyz_m=None,
            )

        depth = self.depth_filter.sample_target_depth(
            policy_action.target_pixel,
            observation.frame.depth_path_hint,
        )
        camera_xyz = self.tf_manager.project_pixel_to_camera_xyz(
            depth.pixel_xy,
            depth.depth_m,
            fx=observation.frame.fx,
            fy=observation.frame.fy,
            cx=observation.frame.cx,
            cy=observation.frame.cy,
        )
        base_xyz = self.tf_manager.camera_xyz_to_base_xyz(
            camera_xyz,
            robot_state.ee_position_m,
            robot_state.ee_yaw_deg,
            robot_state.ee_quaternion_xyzw,
        )
        tip_offset_in_base = self.tf_manager.ee_relative_xyz_to_base_offset(
            self.config.gripper_tip_offset_ee_m,
            robot_state.ee_yaw_deg,
            robot_state.ee_quaternion_xyzw,
        )
        wrist_target_xyz = tuple(
            target_axis - offset_axis
            for target_axis, offset_axis in zip(base_xyz, tip_offset_in_base)
        )
        refined_xyz = (
            wrist_target_xyz[0],
            wrist_target_xyz[1],
            max(wrist_target_xyz[2], self.config.refine_height_m),
        )

        return RefinedGrasp(
            target_xyz_m=refined_xyz,
            target_yaw_deg=robot_state.ee_yaw_deg + policy_action.delta_yaw_deg,
            grasp_width_m=self.config.default_grasp_width_m,
            quality=0.82 if depth.valid else 0.40,
            source="depth_refinement",
            contact_xyz_m=base_xyz,
        )
