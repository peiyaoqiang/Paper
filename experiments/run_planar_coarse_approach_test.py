from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from analysis.trial_logger import TrialLogger
from adapters.action_adapter import ActionAdapter, ActionAdapterConfig
from calibration.tf_manager import TFConfig, TFManager
from common.types import ExecutionResult, Observation, PolicyAction, RefinedGrasp
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver
from geometry.depth_filter import DepthFilter
from geometry.grasp_refiner import GraspRefiner, GraspRefinerConfig
from policy.openvla_wrapper import OpenVLAConfig, OpenVLAWrapper


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Planar coarse approach test using detected ball pixel and hand-eye refinement."
    )
    parser.add_argument("--instruction", type=str, default="", help="Override default instruction.")
    parser.add_argument(
        "--target-color",
        type=str,
        choices=("red", "green"),
        required=True,
        help="Ball color to track and refine toward.",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of planar approach steps.")
    parser.add_argument(
        "--planar-mode",
        type=str,
        choices=("image_servo", "refine_xy"),
        default="image_servo",
        help="Planar control source. `image_servo` is the current recommended mode for real red-ball approach.",
    )
    parser.add_argument(
        "--image-servo-x-gain-m-per-px",
        type=float,
        default=0.00047,
        help="Base-frame x delta per image x pixel error for image-servo planar control.",
    )
    parser.add_argument(
        "--image-servo-y-gain-m-per-px",
        type=float,
        default=0.00059,
        help="Base-frame y delta per image y pixel error for image-servo planar control.",
    )
    parser.add_argument(
        "--image-servo-deadband-px",
        type=float,
        default=12.0,
        help="Ignore small centroid errors within this many pixels when image-servo planar control is enabled.",
    )
    parser.add_argument(
        "--success-radius-px",
        type=float,
        default=80.0,
        help="Consider success if target enters this image-center radius.",
    )
    parser.add_argument(
        "--min-improvement-px",
        type=float,
        default=15.0,
        help="Require at least this much net improvement when target starts near the center.",
    )
    parser.add_argument(
        "--enable-z",
        action="store_true",
        help="Enable guarded z motion instead of pure planar motion.",
    )
    parser.add_argument(
        "--max-z-down-step-m",
        type=float,
        default=0.005,
        help="Maximum downward z step when guarded z is enabled.",
    )
    parser.add_argument(
        "--max-z-up-step-m",
        type=float,
        default=0.01,
        help="Maximum upward z step when guarded z is enabled.",
    )
    parser.add_argument(
        "--allow-z-when-center-distance-below-px",
        type=float,
        default=100.0,
        help="Only allow negative z motion when the target is already this close to image center.",
    )
    return parser.parse_args()


def _build_camera(config: dict) -> RealSenseDriver:
    return RealSenseDriver(
        RealSenseConfig(
            width=config["camera"]["width"],
            height=config["camera"]["height"],
            mode=config["camera"]["mode"],
            color_topic=config["camera"]["color_topic"],
            aligned_depth_topic=config["camera"]["aligned_depth_topic"],
            camera_info_topic=config["camera"].get("camera_info_topic", "/camera/camera/color/camera_info"),
            capture_timeout_s=config["camera"]["capture_timeout_s"],
            output_dir=config["camera"]["output_dir"],
            ros_node_name=f"{config['camera']['ros_node_name']}_planar_coarse_test",
        )
    )


def _build_robot(config: dict) -> KinovaDriver:
    return KinovaDriver(
        KinovaConfig(
            max_translation_step_m=config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            mode=config["robot"]["mode"],
            joint_state_topic=config["robot"]["joint_state_topic"],
            twist_command_topic=config["robot"]["twist_command_topic"],
            base_frame=config["robot"]["base_frame"],
            ee_frame=config["robot"]["ee_frame"],
            twist_command_frame=config["robot"].get("twist_command_frame", "tool_frame"),
            ros_node_name=f"{config['robot']['ros_node_name']}_planar_coarse_test",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=config["robot"]["twist_command_duration_s"],
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=config["robot"]["twist_stop_duration_s"],
        )
    )


def _build_policy(config: dict) -> OpenVLAWrapper:
    return OpenVLAWrapper(
        OpenVLAConfig(
            model_name=config["policy"]["model_name"],
            mode=config["policy"]["mode"],
            remote_url=config["policy"]["remote_url"],
            remote_timeout_s=config["policy"]["remote_timeout_s"],
            unnorm_key=config["policy"]["unnorm_key"],
            image_input_key=config["policy"]["image_input_key"],
        )
    )


def detect_ball_centroid(rgb_path: str, target_color: str) -> tuple[float, float] | None:
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
    return (float(xs.mean()), float(ys.mean()))


def center_distance_px(centroid: tuple[float, float] | None, width: int, height: int) -> float | None:
    if centroid is None:
        return None
    cx = width / 2.0
    cy = height / 2.0
    return math.hypot(centroid[0] - cx, centroid[1] - cy)


def image_servo_planar_delta(
    centroid: tuple[float, float],
    width: int,
    height: int,
    x_gain_m_per_px: float,
    y_gain_m_per_px: float,
    deadband_px: float,
) -> tuple[float, float]:
    center_x = width / 2.0
    center_y = height / 2.0
    error_x_px = centroid[0] - center_x
    error_y_px = centroid[1] - center_y

    if abs(error_x_px) <= deadband_px:
        error_x_px = 0.0
    if abs(error_y_px) <= deadband_px:
        error_y_px = 0.0

    # Empirical mapping from recent red-ball calibration:
    # - increasing base x moves the target right in the image
    # - increasing base y moves the target up in the image
    delta_x_m = -error_x_px * x_gain_m_per_px
    delta_y_m = error_y_px * y_gain_m_per_px
    return (delta_x_m, delta_y_m)


def main() -> None:
    args = parse_args()
    config = load_config()
    trial_logger = TrialLogger(config["logging"]["log_dir"]) if config["logging"]["enabled"] else None
    instruction = args.instruction.strip() or config["task"]["instruction"]

    camera = _build_camera(config)
    robot = _build_robot(config)
    policy = _build_policy(config)
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

    print("Instruction:", instruction)
    print("Tracked target color:", args.target_color)
    print("Requested planar steps:", args.steps)
    print("Planar mode:", args.planar_mode)
    print("Guarded z enabled:", args.enable_z)
    tip_offset_ee_m = tuple(config["gripper"].get("tip_offset_ee_m", [0.0, 0.0, 0.0]))

    step_records = []
    first_observation: Observation | None = None
    last_policy_action = None
    last_safe_action = None
    last_refined_grasp = None
    initial_distance_px: float | None = None
    final_distance_px: float | None = None

    for step_idx in range(args.steps):
        before_state = robot.get_state()
        before_frame = camera.capture_frame()
        before_centroid = detect_ball_centroid(before_frame.rgb_path_hint, args.target_color)
        if before_centroid is None:
            raise RuntimeError(f"Could not detect a {args.target_color} ball in {before_frame.rgb_path_hint}.")
        before_distance_px = center_distance_px(before_centroid, before_frame.width, before_frame.height)
        if initial_distance_px is None:
            initial_distance_px = before_distance_px

        observation = Observation(instruction=instruction, frame=before_frame, robot_state=before_state)
        if first_observation is None:
            first_observation = observation

        policy_action = policy.predict_action(observation)
        policy_action = replace(
            policy_action,
            target_pixel=(int(round(before_centroid[0])), int(round(before_centroid[1]))),
        )
        refined_grasp = grasp_refiner.refine(policy_action, observation)
        before_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
            tip_offset_ee_m,
            before_state.ee_position_m,
            before_state.ee_yaw_deg,
            before_state.ee_quaternion_xyzw,
        )
        before_tip_to_contact_delta = None
        before_tip_to_contact_distance_m = None
        if refined_grasp.contact_xyz_m is not None:
            before_tip_to_contact_delta = tuple(
                contact_axis - tip_axis
                for contact_axis, tip_axis in zip(refined_grasp.contact_xyz_m, before_tip_xyz)
            )
            before_tip_to_contact_distance_m = math.sqrt(
                sum(component * component for component in before_tip_to_contact_delta)
            )
        image_servo_delta_xy = image_servo_planar_delta(
            before_centroid,
            before_frame.width,
            before_frame.height,
            x_gain_m_per_px=args.image_servo_x_gain_m_per_px,
            y_gain_m_per_px=args.image_servo_y_gain_m_per_px,
            deadband_px=args.image_servo_deadband_px,
        )
        raw_dz = refined_grasp.target_xyz_m[2] - before_state.ee_position_m[2]
        guarded_dz = 0.0
        if args.enable_z:
            min_allowed_z = max(config["task"]["refine_height_m"], config["robot"]["workspace_xyz_min"][2])
            desired_target_z = max(refined_grasp.target_xyz_m[2], min_allowed_z)
            guarded_dz = desired_target_z - before_state.ee_position_m[2]
            guarded_dz = max(-args.max_z_down_step_m, min(args.max_z_up_step_m, guarded_dz))
            if before_distance_px is None or before_distance_px > args.allow_z_when_center_distance_below_px:
                guarded_dz = max(0.0, guarded_dz)
        if args.planar_mode == "image_servo":
            planar_delta_xy = image_servo_delta_xy
            planar_notes = "Planar coarse approach using image-servo x/y and guarded z"
            planar_metadata = {
                "planar_mode": args.planar_mode,
                "image_servo_delta_xy_m": image_servo_delta_xy,
                "image_servo_x_gain_m_per_px": args.image_servo_x_gain_m_per_px,
                "image_servo_y_gain_m_per_px": args.image_servo_y_gain_m_per_px,
                "image_servo_deadband_px": args.image_servo_deadband_px,
            }
        else:
            planar_delta_xy = (
                refined_grasp.target_xyz_m[0] - before_state.ee_position_m[0],
                refined_grasp.target_xyz_m[1] - before_state.ee_position_m[1],
            )
            planar_notes = "Planar coarse approach toward refined target"
            planar_metadata = {"planar_mode": args.planar_mode}
        planar_policy_action = PolicyAction(
            delta_xyz_m=(
                planar_delta_xy[0],
                planar_delta_xy[1],
                guarded_dz,
            ),
            delta_yaw_deg=0.0,
            gripper_command=policy_action.gripper_command,
            confidence=policy_action.confidence,
            target_pixel=policy_action.target_pixel,
            notes=planar_notes if args.enable_z else planar_notes.replace(" and guarded z", ""),
            metadata={
                **policy_action.metadata,
                **planar_metadata,
                "refined_target_xyz_m": refined_grasp.target_xyz_m,
                "source_policy_delta_xyz_m": policy_action.delta_xyz_m,
                "guarded_z_enabled": args.enable_z,
                "raw_refined_dz_m": raw_dz,
                "guarded_dz_m": guarded_dz,
            },
        )
        safe_action = action_adapter.adapt(planar_policy_action, before_state)

        print(f"Step {step_idx + 1}")
        print("  RGB path:", before_frame.rgb_path_hint)
        print("  Before ee_position_m:", before_state.ee_position_m)
        print("  Before tip_position_m:", before_tip_xyz)
        print("  Target centroid before:", before_centroid)
        print("  Target center distance before px:", before_distance_px)
        print("  Contact xyz before:", refined_grasp.contact_xyz_m)
        print("  Tip to contact delta xyz before:", before_tip_to_contact_delta)
        print("  Tip to contact distance before m:", before_tip_to_contact_distance_m)
        print("  Refined grasp target xyz:", refined_grasp.target_xyz_m)
        print("  Image-servo delta_xy_m:", image_servo_delta_xy)
        if args.enable_z:
            print("  Raw refined dz_m:", raw_dz)
            print("  Guarded dz_m:", guarded_dz)
        print("  Planar policy delta_xyz_m:", planar_policy_action.delta_xyz_m)
        print("  Safe delta_xyz_m:", safe_action.delta_xyz_m)
        print("  Safe clipped:", safe_action.clipped)
        if safe_action.rejection_reason:
            print("  Safe action note:", safe_action.rejection_reason)

        robot.move_cartesian_delta(safe_action.delta_xyz_m, safe_action.delta_yaw_deg)

        after_state = robot.get_state()
        after_frame = camera.capture_frame()
        after_centroid = detect_ball_centroid(after_frame.rgb_path_hint, args.target_color)
        after_distance_px = center_distance_px(after_centroid, after_frame.width, after_frame.height)
        final_distance_px = after_distance_px
        after_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
            tip_offset_ee_m,
            after_state.ee_position_m,
            after_state.ee_yaw_deg,
            after_state.ee_quaternion_xyzw,
        )

        print("  After ee_position_m:", after_state.ee_position_m)
        print("  After tip_position_m:", after_tip_xyz)
        print("  Target centroid after:", after_centroid)
        print("  Target center distance after px:", after_distance_px)

        step_records.append(
            {
                "step_index": step_idx + 1,
                "before_rgb_path_hint": before_frame.rgb_path_hint,
                "after_rgb_path_hint": after_frame.rgb_path_hint,
                "before_ee_position_m": before_state.ee_position_m,
                "after_ee_position_m": after_state.ee_position_m,
                "before_tip_position_m": before_tip_xyz,
                "after_tip_position_m": after_tip_xyz,
                "before_target_centroid": before_centroid,
                "after_target_centroid": after_centroid,
                "before_target_center_distance_px": before_distance_px,
                "after_target_center_distance_px": after_distance_px,
                "contact_xyz_m": refined_grasp.contact_xyz_m,
                "before_tip_to_contact_delta_xyz_m": before_tip_to_contact_delta,
                "before_tip_to_contact_distance_m": before_tip_to_contact_distance_m,
                "policy_delta_xyz_m": policy_action.delta_xyz_m,
                "planar_policy_delta_xyz_m": planar_policy_action.delta_xyz_m,
                "raw_refined_dz_m": raw_dz,
                "guarded_dz_m": guarded_dz,
                "safe_delta_xyz_m": safe_action.delta_xyz_m,
                "safe_action_clipped": safe_action.clipped,
                "refined_grasp_target_xyz_m": refined_grasp.target_xyz_m,
                "policy_metadata": policy_action.metadata,
            }
        )
        last_policy_action = planar_policy_action
        last_safe_action = safe_action
        last_refined_grasp = refined_grasp

    final_state = robot.get_state()
    reached_radius = (
        initial_distance_px is not None
        and final_distance_px is not None
        and initial_distance_px > args.success_radius_px
        and final_distance_px <= args.success_radius_px
    )
    net_improvement_px = None
    if initial_distance_px is not None and final_distance_px is not None:
        net_improvement_px = initial_distance_px - final_distance_px
    improved_enough = net_improvement_px is not None and net_improvement_px >= args.min_improvement_px
    success = bool(reached_radius or improved_enough)
    final_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
        tip_offset_ee_m,
        final_state.ee_position_m,
        final_state.ee_yaw_deg,
        final_state.ee_quaternion_xyzw,
    )
    result = ExecutionResult(
        success=success,
        state_trace=[f"planar_coarse_step_{idx + 1}" for idx in range(args.steps)],
        message="Planar coarse approach test completed",
        failure_reason="" if success else "Target did not get closer in planar coarse approach",
        grasp=RefinedGrasp(
            target_xyz_m=final_tip_xyz,
            target_yaw_deg=final_state.ee_yaw_deg,
            grasp_width_m=final_state.gripper_opening_m,
            quality=1.0,
            source="planar_coarse_approach_test",
        ),
    )

    print("Initial target center distance px:", initial_distance_px)
    print("Final target center distance px:", final_distance_px)
    print("Net improvement px:", net_improvement_px)
    print("Success radius px:", args.success_radius_px)
    print("Minimum improvement px:", args.min_improvement_px)
    print("Reached radius:", reached_radius)
    print("Improved enough:", improved_enough)
    print("Success:", success)
    print("Result message:", result.message)
    if result.failure_reason:
        print("Failure reason:", result.failure_reason)

    if (
        trial_logger is not None
        and first_observation is not None
        and last_policy_action is not None
        and last_safe_action is not None
        and last_refined_grasp is not None
    ):
        trial_logger.log_trial(
            instruction=instruction,
            observation=first_observation,
            policy_action=last_policy_action,
            safe_action=last_safe_action,
            refined_grasp=last_refined_grasp,
            result=result,
            final_robot_state=final_state,
            metadata={
                "test_type": "planar_coarse_approach_test",
                "target_color": args.target_color,
                "requested_steps": args.steps,
                "planar_mode": args.planar_mode,
                "image_servo_x_gain_m_per_px": args.image_servo_x_gain_m_per_px,
                "image_servo_y_gain_m_per_px": args.image_servo_y_gain_m_per_px,
                "image_servo_deadband_px": args.image_servo_deadband_px,
                "success_radius_px": args.success_radius_px,
                "min_improvement_px": args.min_improvement_px,
                "guarded_z_enabled": args.enable_z,
                "max_z_down_step_m": args.max_z_down_step_m,
                "max_z_up_step_m": args.max_z_up_step_m,
                "allow_z_when_center_distance_below_px": args.allow_z_when_center_distance_below_px,
                "initial_target_center_distance_px": initial_distance_px,
                "final_target_center_distance_px": final_distance_px,
                "net_improvement_px": net_improvement_px,
                "reached_radius": reached_radius,
                "improved_enough": improved_enough,
                "step_records": step_records,
            },
        )
        print("Trial log:", trial_logger.log_path)


if __name__ == "__main__":
    main()
