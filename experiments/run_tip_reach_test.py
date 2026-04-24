from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        description="Closed-loop 3D tip-reach test: repeatedly refine the target and move the gripper tip toward it."
    )
    parser.add_argument("--instruction", type=str, default="", help="Override default instruction.")
    parser.add_argument(
        "--target-color",
        type=str,
        choices=("red", "green"),
        required=True,
        help="Ball color to track and use as the target pixel.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Maximum number of closed-loop reach steps.")
    parser.add_argument(
        "--success-distance-m",
        type=float,
        default=0.03,
        help="Consider success once the gripper tip is within this 3D distance of the refined contact point.",
    )
    parser.add_argument(
        "--use-policy-yaw",
        action="store_true",
        help="Apply the OpenVLA yaw delta during tip reach. Disabled by default to keep position control stable.",
    )
    parser.add_argument(
        "--image-servo-x-gain-m-per-px",
        type=float,
        default=0.00047,
        help="Base-frame x delta per image x pixel error while centering the target under the gripper.",
    )
    parser.add_argument(
        "--image-servo-y-gain-m-per-px",
        type=float,
        default=0.00059,
        help="Base-frame y delta per image y pixel error while centering the target under the gripper.",
    )
    parser.add_argument(
        "--image-servo-deadband-px",
        type=float,
        default=12.0,
        help="Ignore small image centroid errors within this deadband.",
    )
    parser.add_argument(
        "--enable-axis-approach-below-px",
        type=float,
        default=60.0,
        help="Only advance along the gripper forward axis once the target is within this image-center radius.",
    )
    parser.add_argument(
        "--enable-axis-approach",
        action="store_true",
        help="Enable guarded forward motion along the gripper axis after the target is visually centered.",
    )
    parser.add_argument(
        "--axis-approach-step-m",
        type=float,
        default=0.004,
        help="Maximum forward approach distance per step along the gripper axis once axis approach is enabled.",
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
            ros_node_name=f"{config['camera']['ros_node_name']}_tip_reach_test",
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
            ros_node_name=f"{config['robot']['ros_node_name']}_tip_reach_test",
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
            remote_action_gripper_semantics=config["policy"].get("remote_action_gripper_semantics", "open_high"),
        )
    )


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


def vector_norm(vector_xyz: tuple[float, float, float] | None) -> float | None:
    if vector_xyz is None:
        return None
    return math.sqrt(sum(component * component for component in vector_xyz))


def center_distance_px(pixel_xy: tuple[int, int], width: int, height: int) -> float:
    center_x = width / 2.0
    center_y = height / 2.0
    return math.hypot(pixel_xy[0] - center_x, pixel_xy[1] - center_y)


def image_servo_planar_delta(
    pixel_xy: tuple[int, int],
    width: int,
    height: int,
    x_gain_m_per_px: float,
    y_gain_m_per_px: float,
    deadband_px: float,
) -> tuple[float, float]:
    center_x = width / 2.0
    center_y = height / 2.0
    error_x_px = pixel_xy[0] - center_x
    error_y_px = pixel_xy[1] - center_y

    if abs(error_x_px) <= deadband_px:
        error_x_px = 0.0
    if abs(error_y_px) <= deadband_px:
        error_y_px = 0.0

    delta_x_m = -error_x_px * x_gain_m_per_px
    delta_y_m = error_y_px * y_gain_m_per_px
    return (delta_x_m, delta_y_m)


def main() -> None:
    args = parse_args()
    config = load_config()
    trial_logger = TrialLogger(config["logging"]["log_dir"]) if config["logging"]["enabled"] else None
    instruction = args.instruction.strip() or config["task"]["instruction"]
    tip_offset_ee_m = tuple(config["gripper"].get("tip_offset_ee_m", [0.0, 0.0, 0.0]))

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
            gripper_tip_offset_ee_m=tip_offset_ee_m,
            default_grasp_width_m=config["gripper"]["open_width_m"],
        ),
    )

    print("Instruction:", instruction)
    print("Tracked target color:", args.target_color)
    print("Requested tip-reach steps:", args.steps)
    print("Success distance m:", args.success_distance_m)
    print("Use policy yaw:", args.use_policy_yaw)
    print("Axis approach enabled:", args.enable_axis_approach)
    print("Axis approach starts below px:", args.enable_axis_approach_below_px)

    step_records = []
    first_observation: Observation | None = None
    last_policy_action = None
    last_safe_action = None
    last_refined_grasp = None
    initial_tip_distance_m: float | None = None
    final_tip_distance_m: float | None = None

    for step_idx in range(args.steps):
        before_state = robot.get_state()
        before_frame = camera.capture_frame()
        target_pixel = detect_ball_centroid(before_frame.rgb_path_hint, args.target_color)
        if target_pixel is None:
            raise RuntimeError(f"Could not detect a {args.target_color} ball in {before_frame.rgb_path_hint}.")

        observation = Observation(instruction=instruction, frame=before_frame, robot_state=before_state)
        if first_observation is None:
            first_observation = observation
        target_center_distance_px = center_distance_px(target_pixel, before_frame.width, before_frame.height)

        policy_action = policy.predict_action(observation)
        policy_action = PolicyAction(
            delta_xyz_m=policy_action.delta_xyz_m,
            delta_yaw_deg=policy_action.delta_yaw_deg,
            gripper_command=policy_action.gripper_command,
            confidence=policy_action.confidence,
            target_pixel=target_pixel,
            notes=policy_action.notes,
            metadata=policy_action.metadata,
        )
        refined_grasp = grasp_refiner.refine(policy_action, observation)

        current_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
            tip_offset_ee_m,
            before_state.ee_position_m,
            before_state.ee_yaw_deg,
            before_state.ee_quaternion_xyzw,
        )
        contact_xyz = refined_grasp.contact_xyz_m
        if contact_xyz is None:
            raise RuntimeError("Tip-reach requires a valid refined contact point, but refine returned none.")

        tip_to_contact_delta = tuple(
            contact_axis - tip_axis
            for contact_axis, tip_axis in zip(contact_xyz, current_tip_xyz)
        )
        tip_to_contact_distance_m = vector_norm(tip_to_contact_delta)
        final_tip_distance_m = tip_to_contact_distance_m
        if initial_tip_distance_m is None:
            initial_tip_distance_m = tip_to_contact_distance_m
        image_servo_delta_xy = image_servo_planar_delta(
            target_pixel,
            before_frame.width,
            before_frame.height,
            x_gain_m_per_px=args.image_servo_x_gain_m_per_px,
            y_gain_m_per_px=args.image_servo_y_gain_m_per_px,
            deadband_px=args.image_servo_deadband_px,
        )
        tip_axis_norm = vector_norm(tip_offset_ee_m)
        if tip_axis_norm is None or tip_axis_norm <= 1e-6:
            tip_axis_ee = (0.0, 0.0, 1.0)
        else:
            tip_axis_ee = tuple(component / tip_axis_norm for component in tip_offset_ee_m)
        tip_axis_base = tf_manager.ee_relative_xyz_to_base_offset(
            tip_axis_ee,
            before_state.ee_yaw_deg,
            before_state.ee_quaternion_xyzw,
        )
        axial_tip_error_m = sum(
            delta_component * axis_component
            for delta_component, axis_component in zip(tip_to_contact_delta, tip_axis_base)
        )
        axis_approach_delta_xyz = (0.0, 0.0, 0.0)
        if (
            args.enable_axis_approach
            and target_center_distance_px <= args.enable_axis_approach_below_px
            and axial_tip_error_m > 0.0
        ):
            forward_step_m = min(axial_tip_error_m, args.axis_approach_step_m)
            axis_approach_delta_xyz = tuple(
                forward_step_m * axis_component
                for axis_component in tip_axis_base
            )

        print(f"Step {step_idx + 1}")
        print("  RGB path:", before_frame.rgb_path_hint)
        print("  Before ee_position_m:", before_state.ee_position_m)
        print("  Before tip_position_m:", current_tip_xyz)
        print("  Target pixel:", target_pixel)
        print("  Target center distance px:", target_center_distance_px)
        print("  Refined contact xyz:", contact_xyz)
        print("  Tip to contact delta xyz:", tip_to_contact_delta)
        print("  Tip to contact distance m:", tip_to_contact_distance_m)
        print("  Image-servo delta_xy_m:", image_servo_delta_xy)
        print("  Tip axis base:", tip_axis_base)
        print("  Axial tip error m:", axial_tip_error_m)
        print("  Axis-approach delta xyz:", axis_approach_delta_xyz)
        print("  Refined wrist target xyz:", refined_grasp.target_xyz_m)

        if tip_to_contact_distance_m is not None and tip_to_contact_distance_m <= args.success_distance_m:
            print("  Reach note: Tip is already within the success distance; stopping early.")
            zero_action = PolicyAction(
                delta_xyz_m=(0.0, 0.0, 0.0),
                delta_yaw_deg=0.0,
                gripper_command=policy_action.gripper_command,
                confidence=policy_action.confidence,
                target_pixel=policy_action.target_pixel,
                notes="Already within tip reach success distance",
                metadata=policy_action.metadata,
            )
            last_policy_action = zero_action
            last_safe_action = action_adapter.adapt(zero_action, before_state)
            last_refined_grasp = refined_grasp
            break

        wrist_delta_xyz = (
            image_servo_delta_xy[0] + axis_approach_delta_xyz[0],
            image_servo_delta_xy[1] + axis_approach_delta_xyz[1],
            axis_approach_delta_xyz[2],
        )
        tip_reach_action = PolicyAction(
            delta_xyz_m=wrist_delta_xyz,
            delta_yaw_deg=policy_action.delta_yaw_deg if args.use_policy_yaw else 0.0,
            gripper_command=policy_action.gripper_command,
            confidence=policy_action.confidence,
            target_pixel=policy_action.target_pixel,
            notes="Closed-loop tip reach toward refined 3D contact point",
            metadata={
                **policy_action.metadata,
                "control_mode": "tip_reach",
                "target_color": args.target_color,
                "refined_contact_xyz_m": contact_xyz,
                "refined_wrist_target_xyz_m": refined_grasp.target_xyz_m,
                "current_tip_xyz_m": current_tip_xyz,
                "tip_to_contact_delta_xyz_m": tip_to_contact_delta,
                "tip_to_contact_distance_m": tip_to_contact_distance_m,
                "target_center_distance_px": target_center_distance_px,
                "image_servo_delta_xy_m": image_servo_delta_xy,
                "tip_axis_base": tip_axis_base,
                "axial_tip_error_m": axial_tip_error_m,
                "axis_approach_delta_xyz_m": axis_approach_delta_xyz,
                "tip_offset_ee_m": tip_offset_ee_m,
                "policy_yaw_used": args.use_policy_yaw,
            },
        )
        safe_action = action_adapter.adapt(tip_reach_action, before_state)

        print("  Tip-reach wrist delta xyz:", wrist_delta_xyz)
        print("  Safe delta_xyz_m:", safe_action.delta_xyz_m)
        print("  Safe delta_yaw_deg:", safe_action.delta_yaw_deg)
        print("  Safe clipped:", safe_action.clipped)
        if safe_action.rejection_reason:
            print("  Safe action note:", safe_action.rejection_reason)

        robot.move_cartesian_delta(safe_action.delta_xyz_m, safe_action.delta_yaw_deg)

        after_state = robot.get_state()
        after_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
            tip_offset_ee_m,
            after_state.ee_position_m,
            after_state.ee_yaw_deg,
            after_state.ee_quaternion_xyzw,
        )
        after_tip_to_contact_delta = tuple(
            contact_axis - tip_axis
            for contact_axis, tip_axis in zip(contact_xyz, after_tip_xyz)
        )
        after_tip_to_contact_distance_m = vector_norm(after_tip_to_contact_delta)
        final_tip_distance_m = after_tip_to_contact_distance_m

        print("  After ee_position_m:", after_state.ee_position_m)
        print("  After tip_position_m:", after_tip_xyz)
        print("  Tip to contact delta xyz after:", after_tip_to_contact_delta)
        print("  Tip to contact distance after m:", after_tip_to_contact_distance_m)

        step_records.append(
            {
                "step_index": step_idx + 1,
                "rgb_path_hint": before_frame.rgb_path_hint,
                "before_ee_position_m": before_state.ee_position_m,
                "after_ee_position_m": after_state.ee_position_m,
                "before_tip_position_m": current_tip_xyz,
                "after_tip_position_m": after_tip_xyz,
                "target_pixel": target_pixel,
                "target_center_distance_px": target_center_distance_px,
                "refined_contact_xyz_m": contact_xyz,
                "refined_wrist_target_xyz_m": refined_grasp.target_xyz_m,
                "tip_to_contact_delta_xyz_m_before": tip_to_contact_delta,
                "tip_to_contact_distance_m_before": tip_to_contact_distance_m,
                "image_servo_delta_xy_m": image_servo_delta_xy,
                "tip_axis_base": tip_axis_base,
                "axial_tip_error_m": axial_tip_error_m,
                "axis_approach_delta_xyz_m": axis_approach_delta_xyz,
                "tip_reach_wrist_delta_xyz_m": wrist_delta_xyz,
                "safe_delta_xyz_m": safe_action.delta_xyz_m,
                "safe_delta_yaw_deg": safe_action.delta_yaw_deg,
                "safe_action_clipped": safe_action.clipped,
                "tip_to_contact_delta_xyz_m_after": after_tip_to_contact_delta,
                "tip_to_contact_distance_m_after": after_tip_to_contact_distance_m,
                "policy_metadata": tip_reach_action.metadata,
            }
        )

        last_policy_action = tip_reach_action
        last_safe_action = safe_action
        last_refined_grasp = refined_grasp

        if after_tip_to_contact_distance_m is not None and after_tip_to_contact_distance_m <= args.success_distance_m:
            print("  Reach note: Tip entered the success distance; stopping early.")
            break

    final_state = robot.get_state()
    final_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
        tip_offset_ee_m,
        final_state.ee_position_m,
        final_state.ee_yaw_deg,
        final_state.ee_quaternion_xyzw,
    )
    success = final_tip_distance_m is not None and final_tip_distance_m <= args.success_distance_m
    result = ExecutionResult(
        success=bool(success),
        state_trace=[f"tip_reach_step_{idx + 1}" for idx in range(len(step_records))],
        message="Tip reach test completed",
        failure_reason="" if success else "Gripper tip did not reach the refined target distance threshold",
        grasp=RefinedGrasp(
            target_xyz_m=final_tip_xyz,
            target_yaw_deg=final_state.ee_yaw_deg,
            grasp_width_m=final_state.gripper_opening_m,
            quality=1.0,
            source="tip_reach_test",
            contact_xyz_m=last_refined_grasp.contact_xyz_m if last_refined_grasp is not None else None,
        ),
    )

    print("Initial tip-to-contact distance m:", initial_tip_distance_m)
    print("Final tip-to-contact distance m:", final_tip_distance_m)
    print("Success distance m:", args.success_distance_m)
    print("Success:", success)
    print("Final tip xyz:", final_tip_xyz)
    if last_refined_grasp is not None:
        print("Final refined contact xyz:", last_refined_grasp.contact_xyz_m)
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
                "test_type": "tip_reach_test",
                "target_color": args.target_color,
                "requested_steps": args.steps,
                "success_distance_m": args.success_distance_m,
                "image_servo_x_gain_m_per_px": args.image_servo_x_gain_m_per_px,
                "image_servo_y_gain_m_per_px": args.image_servo_y_gain_m_per_px,
                "image_servo_deadband_px": args.image_servo_deadband_px,
                "enable_axis_approach_below_px": args.enable_axis_approach_below_px,
                "tip_offset_ee_m": tip_offset_ee_m,
                "policy_yaw_used": args.use_policy_yaw,
                "initial_tip_to_contact_distance_m": initial_tip_distance_m,
                "final_tip_to_contact_distance_m": final_tip_distance_m,
                "step_records": step_records,
            },
        )
        print("Trial log:", trial_logger.log_path)


if __name__ == "__main__":
    main()
