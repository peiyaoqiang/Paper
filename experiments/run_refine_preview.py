from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from adapters.action_adapter import ActionAdapter, ActionAdapterConfig
from calibration.tf_manager import TFConfig, TFManager
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver
from geometry.depth_filter import DepthFilter
from geometry.grasp_refiner import GraspRefiner, GraspRefinerConfig
from common.types import Observation
from policy.openvla_wrapper import OpenVLAConfig, OpenVLAWrapper


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview refined 3D target using real hand-eye calibration.")
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Override the instruction from configs/default_config.json.",
    )
    parser.add_argument(
        "--target-color",
        type=str,
        choices=("red", "green"),
        default="",
        help="If set, replace OpenVLA target_pixel with the detected ball centroid of this color.",
    )
    return parser.parse_args()


def detect_ball_centroid(rgb_path: str, target_color: str) -> tuple[int, int] | None:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    if target_color == "red":
        mask = (r > 100) & (r > g + 35) & (r > b + 35)
    else:
        mask = (g > 80) & (g > r + 20) & (g > b + 20)

    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    return (int(round(xs.mean())), int(round(ys.mean())))


def main() -> None:
    args = parse_args()
    config = load_config()

    camera = RealSenseDriver(
        RealSenseConfig(
            width=config["camera"]["width"],
            height=config["camera"]["height"],
            mode=config["camera"]["mode"],
            color_topic=config["camera"]["color_topic"],
            aligned_depth_topic=config["camera"]["aligned_depth_topic"],
            camera_info_topic=config["camera"].get("camera_info_topic", "/camera/camera/color/camera_info"),
            capture_timeout_s=config["camera"]["capture_timeout_s"],
            output_dir=config["camera"]["output_dir"],
            ros_node_name=f"{config['camera']['ros_node_name']}_refine_preview",
        )
    )
    robot = KinovaDriver(
        KinovaConfig(
            max_translation_step_m=config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            mode=config["robot"]["mode"],
            joint_state_topic=config["robot"]["joint_state_topic"],
            twist_command_topic=config["robot"]["twist_command_topic"],
            base_frame=config["robot"]["base_frame"],
            ee_frame=config["robot"]["ee_frame"],
            twist_command_frame=config["robot"].get("twist_command_frame", "tool_frame"),
            ros_node_name=f"{config['robot']['ros_node_name']}_refine_preview",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=config["robot"]["twist_command_duration_s"],
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=config["robot"]["twist_stop_duration_s"],
        )
    )
    policy = OpenVLAWrapper(
        OpenVLAConfig(
            model_name=config["policy"]["model_name"],
            mode=config["policy"]["mode"],
            remote_url=config["policy"]["remote_url"],
            remote_timeout_s=config["policy"]["remote_timeout_s"],
            unnorm_key=config["policy"]["unnorm_key"],
            image_input_key=config["policy"]["image_input_key"],
            remote_action_gripper_semantics=config["policy"].get("remote_action_gripper_semantics", "open_high"),
        )
    )
    action_adapter = ActionAdapter(
        ActionAdapterConfig(
            max_translation_step_m=config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            workspace_xyz_min=tuple(config["robot"]["workspace_xyz_min"]),
            workspace_xyz_max=tuple(config["robot"]["workspace_xyz_max"]),
            workspace_enforced=config["robot"].get("workspace_enforced", True),
        )
    )
    tf_manager = TFManager(
        TFConfig(
            camera_to_ee_translation_m=tuple(config["calibration"]["camera_to_ee_translation_m"]),
            camera_to_ee_quaternion_xyzw=tuple(config["calibration"]["camera_to_ee_quaternion_xyzw"]),
            fx=config["camera"]["fx"],
            fy=config["camera"]["fy"],
            cx=config["camera"]["cx"],
            cy=config["camera"]["cy"],
        )
    )
    grasp_refiner = GraspRefiner(
        depth_filter=DepthFilter(),
        tf_manager=tf_manager,
        config=GraspRefinerConfig(
            approach_height_m=config["task"]["approach_height_m"],
            refine_height_m=config["task"]["refine_height_m"],
            gripper_tip_offset_ee_m=tuple(config["gripper"].get("tip_offset_ee_m", [0.0, 0.0, 0.0])),
            default_grasp_width_m=config["gripper"]["open_width_m"],
        ),
    )

    robot_state = robot.get_state()
    frame = camera.capture_frame()
    observation = Observation(
        instruction=args.instruction.strip() or config["task"]["instruction"],
        frame=frame,
        robot_state=robot_state,
    )
    policy_action = policy.predict_action(observation)
    detected_target_pixel = None
    if args.target_color:
        detected_target_pixel = detect_ball_centroid(frame.rgb_path_hint, args.target_color)
        if detected_target_pixel is None:
            raise RuntimeError(
                f"Could not detect a {args.target_color} ball centroid in {frame.rgb_path_hint}."
            )
        policy_action = replace(policy_action, target_pixel=detected_target_pixel)
    safe_action = action_adapter.adapt(policy_action, robot_state)
    depth_sample = grasp_refiner.depth_filter.sample_target_depth(
        policy_action.target_pixel,
        frame.depth_path_hint,
    )
    camera_xyz = tf_manager.project_pixel_to_camera_xyz(
        depth_sample.pixel_xy,
        depth_sample.depth_m,
        fx=frame.fx,
        fy=frame.fy,
        cx=frame.cx,
        cy=frame.cy,
    )
    camera_minus_translation = tuple(
        point - offset
        for point, offset in zip(
            camera_xyz,
            tf_manager.config.camera_to_ee_translation_m,
        )
    )
    camera_in_ee = tf_manager._rotate_camera_to_ee(camera_minus_translation)
    ee_relative_xyz = camera_in_ee
    base_xyz = tf_manager.camera_xyz_to_base_xyz(
        camera_xyz,
        robot_state.ee_position_m,
        robot_state.ee_yaw_deg,
        robot_state.ee_quaternion_xyzw,
    )
    refined_grasp = grasp_refiner.refine(policy_action, observation)
    tip_offset_ee_m = tuple(config["gripper"].get("tip_offset_ee_m", [0.0, 0.0, 0.0]))
    current_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
        tip_offset_ee_m,
        robot_state.ee_position_m,
        robot_state.ee_yaw_deg,
        robot_state.ee_quaternion_xyzw,
    )
    refined_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
        tip_offset_ee_m,
        refined_grasp.target_xyz_m,
        refined_grasp.target_yaw_deg,
        robot_state.ee_quaternion_xyzw,
    )
    tip_to_contact_delta = None
    tip_to_contact_distance_m = None
    if refined_grasp.contact_xyz_m is not None:
        tip_to_contact_delta = tuple(
            contact_axis - tip_axis
            for contact_axis, tip_axis in zip(refined_grasp.contact_xyz_m, current_tip_xyz)
        )
        tip_to_contact_distance_m = math.sqrt(
            sum(component * component for component in tip_to_contact_delta)
        )

    print("Instruction:", observation.instruction)
    print("RGB path:", frame.rgb_path_hint)
    print("Depth path:", frame.depth_path_hint)
    print("Robot ee_position_m:", robot_state.ee_position_m)
    print("Robot ee_yaw_deg:", robot_state.ee_yaw_deg)
    print("Robot ee_quaternion_xyzw:", robot_state.ee_quaternion_xyzw)
    print("Frame intrinsics fx/fy/cx/cy:", (frame.fx, frame.fy, frame.cx, frame.cy))
    print("Configured gripper tip offset ee_m:", tip_offset_ee_m)
    print("Detected target color:", args.target_color or "none")
    print("Detected target centroid:", detected_target_pixel)
    print("Policy target_pixel:", policy_action.target_pixel)
    print("Depth sample valid:", depth_sample.valid)
    print("Depth sample depth_m:", depth_sample.depth_m)
    print("Camera xyz:", camera_xyz)
    print("Camera xyz after subtracting EE->camera translation:", camera_minus_translation)
    print("Camera xyz rotated into EE:", camera_in_ee)
    print("EE-relative xyz after inverting handeye:", ee_relative_xyz)
    print("Base xyz before refine height clamp:", base_xyz)
    print("Current tip xyz:", current_tip_xyz)
    print("Policy delta_xyz_m:", policy_action.delta_xyz_m)
    print("Safe delta_xyz_m:", safe_action.delta_xyz_m)
    print("Safe clipped:", safe_action.clipped)
    if safe_action.rejection_reason:
        print("Safe action note:", safe_action.rejection_reason)
    print("Refined contact xyz:", refined_grasp.contact_xyz_m)
    print("Refined grasp target xyz:", refined_grasp.target_xyz_m)
    print("Predicted tip xyz at refined target:", refined_tip_xyz)
    print("Tip to contact delta xyz:", tip_to_contact_delta)
    print("Tip to contact distance m:", tip_to_contact_distance_m)
    print("Refined grasp target yaw:", refined_grasp.target_yaw_deg)
    print("Refined grasp quality:", refined_grasp.quality)
    print("Refined grasp source:", refined_grasp.source)


if __name__ == "__main__":
    main()
