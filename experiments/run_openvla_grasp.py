from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.trial_logger import TrialLogger
from adapters.action_adapter import ActionAdapter, ActionAdapterConfig
from calibration.tf_manager import TFConfig, TFManager
from common.types import ExecutionResult, Observation, PolicyAction, RefinedGrasp
from drivers.gripper_driver import GripperConfig, GripperDriver
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
        description="OpenVLA-guided real grasp: closed-loop tip reach, then real gripper close and lift."
    )
    parser.add_argument("--instruction", type=str, default="", help="Override default instruction.")
    parser.add_argument(
        "--target-color",
        type=str,
        choices=("red", "green"),
        required=True,
        help="Ball color to track during closed-loop approach.",
    )
    parser.add_argument("--steps", type=int, default=30, help="Maximum number of closed-loop approach steps.")
    parser.add_argument(
        "--success-distance-m",
        type=float,
        default=0.03,
        help="Start grasp closure once the gripper tip is within this 3D distance of the refined contact point.",
    )
    parser.add_argument(
        "--image-servo-x-gain-m-per-px",
        type=float,
        default=0.00047,
        help="Base-frame x delta per image x pixel error while centering the target.",
    )
    parser.add_argument(
        "--image-servo-y-gain-m-per-px",
        type=float,
        default=0.00059,
        help="Base-frame y delta per image y pixel error while centering the target.",
    )
    parser.add_argument(
        "--image-servo-deadband-px",
        type=float,
        default=12.0,
        help="Ignore small image centroid errors within this deadband.",
    )
    parser.add_argument(
        "--enable-axis-approach",
        action="store_true",
        help="Enable guarded forward motion along the gripper axis once the target is visually centered.",
    )
    parser.add_argument(
        "--enable-axis-approach-below-px",
        type=float,
        default=60.0,
        help="Only advance along the gripper axis after the target enters this image-center radius.",
    )
    parser.add_argument(
        "--axis-approach-step-m",
        type=float,
        default=0.004,
        help="Maximum forward approach distance per step along the gripper axis.",
    )
    parser.add_argument(
        "--use-policy-yaw",
        action="store_true",
        help="Apply OpenVLA yaw deltas during approach. Disabled by default to keep motion stable.",
    )
    parser.add_argument(
        "--skip-initial-open",
        action="store_true",
        help="Skip opening the gripper before approach.",
    )
    parser.add_argument(
        "--lift-height-m",
        type=float,
        default=None,
        help="Override lift height after grasp. Defaults to configs/default_config.json.",
    )
    parser.add_argument(
        "--max-translation-step-m",
        type=float,
        default=None,
        help="Override the per-step translation clip used by the ActionAdapter and Kinova driver.",
    )
    parser.add_argument(
        "--twist-command-duration-s",
        type=float,
        default=None,
        help="Override twist publish duration for each command segment.",
    )
    parser.add_argument(
        "--twist-stop-duration-s",
        type=float,
        default=None,
        help="Override zero-twist braking duration after each command segment.",
    )
    parser.add_argument(
        "--combined-axis-commands",
        action="store_true",
        help="Send x/y/z together in one twist command instead of sequential axis commands.",
    )
    parser.add_argument(
        "--grasp-center-distance-px",
        type=float,
        default=70.0,
        help="Allow grasp if the target is this close to the image center and the local pose error is also small.",
    )
    parser.add_argument(
        "--grasp-planar-error-m",
        type=float,
        default=0.02,
        help="Allow grasp if the tip planar error to the refined contact point is below this threshold.",
    )
    parser.add_argument(
        "--grasp-axial-error-m",
        type=float,
        default=0.025,
        help="Allow grasp if the tip axial error to the refined contact point is below this threshold.",
    )
    parser.add_argument(
        "--grasp-stable-steps",
        type=int,
        default=2,
        help="Require this many consecutive grasp-ready checks before closing the gripper.",
    )
    parser.add_argument(
        "--settle-before-close-s",
        type=float,
        default=0.15,
        help="Pause briefly before closing once the grasp trigger condition is met.",
    )
    parser.add_argument(
        "--grasp-on-center-only",
        action="store_true",
        help="Trigger grasp from visual centering only, without requiring the 3D axial/planar thresholds.",
    )
    return parser.parse_args()


def build_camera(config: dict) -> RealSenseDriver:
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
            ros_node_name=f"{config['camera']['ros_node_name']}_openvla_grasp",
        )
    )


def build_robot(config: dict, args: argparse.Namespace) -> KinovaDriver:
    max_translation_step_m = (
        args.max_translation_step_m
        if args.max_translation_step_m is not None
        else config["robot"]["max_translation_step_m"]
    )
    twist_command_duration_s = (
        args.twist_command_duration_s
        if args.twist_command_duration_s is not None
        else config["robot"]["twist_command_duration_s"]
    )
    twist_stop_duration_s = (
        args.twist_stop_duration_s
        if args.twist_stop_duration_s is not None
        else config["robot"]["twist_stop_duration_s"]
    )
    sequential_axis_commands = not args.combined_axis_commands
    if "sequential_axis_commands" in config["robot"] and not args.combined_axis_commands:
        sequential_axis_commands = bool(config["robot"]["sequential_axis_commands"])

    return KinovaDriver(
        KinovaConfig(
            max_translation_step_m=max_translation_step_m,
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            mode=config["robot"]["mode"],
            joint_state_topic=config["robot"]["joint_state_topic"],
            twist_command_topic=config["robot"]["twist_command_topic"],
            base_frame=config["robot"]["base_frame"],
            ee_frame=config["robot"]["ee_frame"],
            twist_command_frame=config["robot"].get("twist_command_frame", "tool_frame"),
            sequential_axis_commands=sequential_axis_commands,
            ros_node_name=f"{config['robot']['ros_node_name']}_openvla_grasp",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=twist_command_duration_s,
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=twist_stop_duration_s,
        )
    )


def build_policy(config: dict) -> OpenVLAWrapper:
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


def build_gripper(config: dict, robot: KinovaDriver) -> GripperDriver:
    return GripperDriver(
        robot,
        GripperConfig(
            open_width_m=config["gripper"]["open_width_m"],
            close_width_m=config["gripper"]["close_width_m"],
            mode=config["gripper"].get("mode", "state_only"),
            ctag_serial_port=config["gripper"].get("ctag_serial_port", "/dev/ttyUSB0"),
            ctag_baudrate=config["gripper"].get("ctag_baudrate", 115200),
            ctag_device_id=config["gripper"].get("ctag_device_id", 1),
            ctag_timeout_s=config["gripper"].get("ctag_timeout_s", 0.2),
            ctag_open_pos_mm=config["gripper"].get("ctag_open_pos_mm", 850.0),
            ctag_close_pos_mm=config["gripper"].get("ctag_close_pos_mm", 0.0),
            ctag_max_stroke_mm=config["gripper"].get("ctag_max_stroke_mm", 850.0),
            ctag_speed=config["gripper"].get("ctag_speed", 80),
            ctag_close_torque=config["gripper"].get("ctag_close_torque", 80),
            ctag_open_torque=config["gripper"].get("ctag_open_torque", 40),
            ctag_acc_dec=config["gripper"].get("ctag_acc_dec", 80),
            ctag_parity=config["gripper"].get("ctag_parity", "N"),
            ctag_stopbits=config["gripper"].get("ctag_stopbits", 1),
            ctag_enable_rs485_mode=config["gripper"].get("ctag_enable_rs485_mode", False),
            ctag_accept_pos_reached_as_success=config["gripper"].get(
                "ctag_accept_pos_reached_as_success", False
            ),
            ctag_rs485_rts_level_for_tx=config["gripper"].get("ctag_rs485_rts_level_for_tx", True),
            ctag_rs485_rts_level_for_rx=config["gripper"].get("ctag_rs485_rts_level_for_rx", False),
            ctag_rs485_delay_before_tx=config["gripper"].get("ctag_rs485_delay_before_tx", 0.0),
            ctag_rs485_delay_before_rx=config["gripper"].get("ctag_rs485_delay_before_rx", 0.0),
            open_timeout_s=config["gripper"].get("open_timeout_s", 3.0),
            close_timeout_s=config["gripper"].get("close_timeout_s", 5.0),
        ),
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


def decompose_tip_error(
    delta_xyz: tuple[float, float, float],
    axis_xyz: tuple[float, float, float],
) -> tuple[float, float]:
    axial_error_m = sum(
        delta_component * axis_component
        for delta_component, axis_component in zip(delta_xyz, axis_xyz)
    )
    planar_delta = tuple(
        delta_component - axial_error_m * axis_component
        for delta_component, axis_component in zip(delta_xyz, axis_xyz)
    )
    planar_error_m = math.sqrt(sum(component * component for component in planar_delta))
    return (axial_error_m, planar_error_m)


def is_grasp_ready(
    target_center_distance_px: float,
    tip_to_contact_distance_m: float | None,
    axial_tip_error_m: float,
    planar_tip_error_m: float,
    success_distance_m: float,
    grasp_center_distance_px: float,
    grasp_axial_error_m: float,
    grasp_planar_error_m: float,
    grasp_on_center_only: bool,
) -> tuple[bool, str]:
    if tip_to_contact_distance_m is not None and tip_to_contact_distance_m <= success_distance_m:
        return (True, "3d_distance")
    if target_center_distance_px <= grasp_center_distance_px and grasp_on_center_only:
        return (True, "visual_center_only")
    if (
        target_center_distance_px <= grasp_center_distance_px
        and abs(axial_tip_error_m) <= grasp_axial_error_m
        and planar_tip_error_m <= grasp_planar_error_m
    ):
        return (True, "visual_alignment")
    return (False, "")


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

    return (-error_x_px * x_gain_m_per_px, error_y_px * y_gain_m_per_px)


def main() -> None:
    args = parse_args()
    config = load_config()
    trial_logger = TrialLogger(config["logging"]["log_dir"]) if config["logging"]["enabled"] else None
    instruction = args.instruction.strip() or config["task"]["instruction"]
    lift_height_m = args.lift_height_m if args.lift_height_m is not None else config["task"]["lift_height_m"]
    tip_offset_ee_m = tuple(config["gripper"].get("tip_offset_ee_m", [0.0, 0.0, 0.0]))
    max_translation_step_m = (
        args.max_translation_step_m
        if args.max_translation_step_m is not None
        else config["robot"]["max_translation_step_m"]
    )

    camera = build_camera(config)
    robot = build_robot(config, args)
    gripper = build_gripper(config, robot)
    policy = build_policy(config)
    action_adapter = ActionAdapter(
        ActionAdapterConfig(
            max_translation_step_m=max_translation_step_m,
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
    print("Requested grasp steps:", args.steps)
    print("Success distance m:", args.success_distance_m)
    print("Axis approach enabled:", args.enable_axis_approach)
    print("Gripper mode:", config["gripper"].get("mode", "state_only"))
    print("Max translation step m:", max_translation_step_m)
    print("Twist command duration s:", robot.config.twist_command_duration_s)
    print("Twist stop duration s:", robot.config.twist_stop_duration_s)
    print("Sequential axis commands:", robot.config.sequential_axis_commands)

    step_records = []
    first_observation: Observation | None = None
    last_policy_action = None
    last_safe_action = None
    last_refined_grasp = None
    final_tip_distance_m: float | None = None
    close_success = False
    lift_executed = False
    consecutive_grasp_ready_steps = 0

    try:
        remote_ok, remote_message = policy.check_remote_health()
        print("OpenVLA remote health:", remote_message)
        if not remote_ok:
            raise RuntimeError(remote_message)

        if not args.skip_initial_open:
            open_success = gripper.open()
            print("Initial gripper open success:", open_success)
            if not open_success:
                raise RuntimeError("Failed to open the gripper before grasp.")

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
                raise RuntimeError("Grasp execution requires a valid refined contact point.")

            tip_to_contact_delta = tuple(
                contact_axis - tip_axis
                for contact_axis, tip_axis in zip(contact_xyz, current_tip_xyz)
            )
            tip_to_contact_distance_m = vector_norm(tip_to_contact_delta)
            final_tip_distance_m = tip_to_contact_distance_m

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
            axial_tip_error_m, planar_tip_error_m = decompose_tip_error(
                tip_to_contact_delta,
                tip_axis_base,
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
            print("  Target pixel:", target_pixel)
            print("  Target center distance px:", target_center_distance_px)
            print("  Refined contact xyz:", contact_xyz)
            print("  Tip to contact distance m:", tip_to_contact_distance_m)
            print("  Image-servo delta_xy_m:", image_servo_delta_xy)
            print("  Axial tip error m:", axial_tip_error_m)
            print("  Planar tip error m:", planar_tip_error_m)
            print("  Axis-approach delta xyz:", axis_approach_delta_xyz)
            grasp_ready_before_move, grasp_trigger_reason = is_grasp_ready(
                target_center_distance_px=target_center_distance_px,
                tip_to_contact_distance_m=tip_to_contact_distance_m,
                axial_tip_error_m=axial_tip_error_m,
                planar_tip_error_m=planar_tip_error_m,
                success_distance_m=args.success_distance_m,
                grasp_center_distance_px=args.grasp_center_distance_px,
                grasp_axial_error_m=args.grasp_axial_error_m,
                grasp_planar_error_m=args.grasp_planar_error_m,
                grasp_on_center_only=args.grasp_on_center_only,
            )
            consecutive_grasp_ready_steps = (
                consecutive_grasp_ready_steps + 1 if grasp_ready_before_move else 0
            )
            print("  Grasp ready before move:", grasp_ready_before_move)
            print("  Grasp trigger reason before move:", grasp_trigger_reason or "none")
            print("  Consecutive grasp-ready steps:", consecutive_grasp_ready_steps)

            if consecutive_grasp_ready_steps >= max(args.grasp_stable_steps, 1):
                print("  Reach note: grasp trigger satisfied before move, closing gripper.")
                last_policy_action = PolicyAction(
                    delta_xyz_m=(0.0, 0.0, 0.0),
                    delta_yaw_deg=0.0,
                    gripper_command="close",
                    confidence=policy_action.confidence,
                    target_pixel=policy_action.target_pixel,
                    notes="Tip already within grasp threshold",
                    metadata=policy_action.metadata,
                )
                last_safe_action = action_adapter.adapt(last_policy_action, before_state)
                last_refined_grasp = refined_grasp
                break

            wrist_delta_xyz = (
                image_servo_delta_xy[0] + axis_approach_delta_xyz[0],
                image_servo_delta_xy[1] + axis_approach_delta_xyz[1],
                axis_approach_delta_xyz[2],
            )
            grasp_action = PolicyAction(
                delta_xyz_m=wrist_delta_xyz,
                delta_yaw_deg=policy_action.delta_yaw_deg if args.use_policy_yaw else 0.0,
                gripper_command="open",
                confidence=policy_action.confidence,
                target_pixel=policy_action.target_pixel,
                notes="OpenVLA-guided closed-loop grasp approach",
                metadata={
                    **policy_action.metadata,
                    "control_mode": "openvla_grasp",
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
            safe_action = action_adapter.adapt(grasp_action, before_state)

            print("  Wrist delta xyz:", wrist_delta_xyz)
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
            after_axial_tip_error_m, after_planar_tip_error_m = decompose_tip_error(
                after_tip_to_contact_delta,
                tip_axis_base,
            )
            final_tip_distance_m = after_tip_to_contact_distance_m

            print("  After tip position m:", after_tip_xyz)
            print("  Tip to contact distance after m:", after_tip_to_contact_distance_m)
            print("  Axial tip error after m:", after_axial_tip_error_m)
            print("  Planar tip error after m:", after_planar_tip_error_m)

            step_records.append(
                {
                    "step_index": step_idx + 1,
                    "rgb_path_hint": before_frame.rgb_path_hint,
                    "target_pixel": target_pixel,
                    "target_center_distance_px": target_center_distance_px,
                    "refined_contact_xyz_m": contact_xyz,
                    "tip_to_contact_distance_m_before": tip_to_contact_distance_m,
                    "tip_to_contact_distance_m_after": after_tip_to_contact_distance_m,
                    "axial_tip_error_m_before": axial_tip_error_m,
                    "axial_tip_error_m_after": after_axial_tip_error_m,
                    "planar_tip_error_m_before": planar_tip_error_m,
                    "planar_tip_error_m_after": after_planar_tip_error_m,
                    "image_servo_delta_xy_m": image_servo_delta_xy,
                    "axis_approach_delta_xyz_m": axis_approach_delta_xyz,
                    "safe_delta_xyz_m": safe_action.delta_xyz_m,
                    "safe_delta_yaw_deg": safe_action.delta_yaw_deg,
                    "safe_action_clipped": safe_action.clipped,
                    "policy_metadata": grasp_action.metadata,
                }
            )

            last_policy_action = grasp_action
            last_safe_action = safe_action
            last_refined_grasp = refined_grasp

            grasp_ready_after_move, grasp_trigger_reason_after_move = is_grasp_ready(
                target_center_distance_px=target_center_distance_px,
                tip_to_contact_distance_m=after_tip_to_contact_distance_m,
                axial_tip_error_m=after_axial_tip_error_m,
                planar_tip_error_m=after_planar_tip_error_m,
                success_distance_m=args.success_distance_m,
                grasp_center_distance_px=args.grasp_center_distance_px,
                grasp_axial_error_m=args.grasp_axial_error_m,
                grasp_planar_error_m=args.grasp_planar_error_m,
                grasp_on_center_only=args.grasp_on_center_only,
            )
            consecutive_grasp_ready_steps = (
                consecutive_grasp_ready_steps + 1 if grasp_ready_after_move else 0
            )
            print("  Grasp ready after move:", grasp_ready_after_move)
            print("  Grasp trigger reason after move:", grasp_trigger_reason_after_move or "none")
            print("  Consecutive grasp-ready steps:", consecutive_grasp_ready_steps)

            if consecutive_grasp_ready_steps >= max(args.grasp_stable_steps, 1):
                print("  Reach note: grasp trigger satisfied after move, closing gripper next.")
                break

        approach_success = final_tip_distance_m is not None and final_tip_distance_m <= args.success_distance_m
        if not approach_success and consecutive_grasp_ready_steps >= max(args.grasp_stable_steps, 1):
            approach_success = True
        if approach_success:
            if args.settle_before_close_s > 0.0:
                time.sleep(args.settle_before_close_s)
            close_success = gripper.close()
            print("Gripper close success:", close_success)
            if close_success:
                robot.move_cartesian_delta((0.0, 0.0, lift_height_m), 0.0)
                lift_executed = True
                print("Lift executed height m:", lift_height_m)

        final_state = robot.get_state()
        final_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
            tip_offset_ee_m,
            final_state.ee_position_m,
            final_state.ee_yaw_deg,
            final_state.ee_quaternion_xyzw,
        )
        success = bool(approach_success and close_success)
        failure_reason = ""
        if not approach_success:
            failure_reason = "Gripper tip did not reach the refined target distance threshold"
        elif not close_success:
            failure_reason = "Real gripper close command failed"

        result = ExecutionResult(
            success=success,
            state_trace=[f"openvla_grasp_step_{idx + 1}" for idx in range(len(step_records))],
            message="OpenVLA grasp run completed",
            failure_reason=failure_reason,
            grasp=RefinedGrasp(
                target_xyz_m=final_tip_xyz,
                target_yaw_deg=final_state.ee_yaw_deg,
                grasp_width_m=final_state.gripper_opening_m,
                quality=1.0 if success else 0.0,
                source="openvla_grasp",
                contact_xyz_m=last_refined_grasp.contact_xyz_m if last_refined_grasp is not None else None,
            ),
        )

        print("Final tip-to-contact distance m:", final_tip_distance_m)
        print("Approach success:", approach_success)
        print("Close success:", close_success)
        print("Lift executed:", lift_executed)
        print("Success:", result.success)
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
                    "test_type": "openvla_grasp",
                    "target_color": args.target_color,
                    "requested_steps": args.steps,
                    "success_distance_m": args.success_distance_m,
                    "gripper_mode": config["gripper"].get("mode", "state_only"),
                    "close_success": close_success,
                    "lift_executed": lift_executed,
                    "lift_height_m": lift_height_m,
                    "step_records": step_records,
                },
            )
            print("Trial log:", trial_logger.log_path)
    finally:
        gripper.shutdown()


if __name__ == "__main__":
    main()
