from __future__ import annotations

from dataclasses import dataclass

from calibration.tf_manager import TFManager
from common.types import PolicyAction, RefinedGrasp, RobotState
from geometry.depth_filter import DepthFilter


@dataclass
class GraspRefinerConfig:
    approach_height_m: float
    refine_height_m: float


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

    def refine(self, policy_action: PolicyAction, robot_state: RobotState) -> RefinedGrasp:
        if policy_action.target_pixel is None:
            return RefinedGrasp(
                target_xyz_m=robot_state.ee_position_m,
                target_yaw_deg=robot_state.ee_yaw_deg,
                grasp_width_m=0.05,
                quality=0.30,
                source="fallback",
            )

        depth = self.depth_filter.sample_target_depth(policy_action.target_pixel)
        camera_xyz = self.tf_manager.project_pixel_to_camera_xyz(depth.pixel_xy, depth.depth_m)
        base_xyz = self.tf_manager.camera_xyz_to_base_xyz(camera_xyz, robot_state.ee_position_m)
        refined_xyz = (base_xyz[0], base_xyz[1], max(base_xyz[2], self.config.refine_height_m))

        return RefinedGrasp(
            target_xyz_m=refined_xyz,
            target_yaw_deg=robot_state.ee_yaw_deg + policy_action.delta_yaw_deg,
            grasp_width_m=0.05,
            quality=0.82 if depth.valid else 0.40,
            source="depth_refinement",
        )
