from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

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
        description="Planar coarse approach with optional real gripper close and lift."
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
        help="Consider approach success if target enters this image-center radius.",
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
    parser.add_argument(
        "--enable-forward",
        action="store_true",
        help="When the target is sufficiently centered, add a small forward motion along the camera optical axis.",
    )
    parser.add_argument(
        "--forward-step-m",
        type=float,
        default=0.01,
        help="Camera-forward step size in meters when forward motion is enabled.",
    )
    parser.add_argument(
        "--allow-forward-when-center-distance-below-px",
        type=float,
        default=90.0,
        help="Only add camera-forward motion when the target is already this close to image center.",
    )
    parser.add_argument(
        "--stop-forward-when-depth-below-m",
        type=float,
        default=0.18,
        help="Disable forward motion once the target depth is below this threshold.",
    )
    parser.add_argument(
        "--success-distance-m",
        type=float,
        default=0.03,
        help="Allow grasp once the tip is within this 3D distance of the refined contact point.",
    )
    parser.add_argument(
        "--grasp-center-distance-px",
        type=float,
        default=70.0,
        help="Allow grasp if the target is this close to the image center and local tip pose error is also small.",
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
        "--grasp-on-center-only",
        action="store_true",
        help="Trigger grasp from visual centering only, without requiring 3D axial/planar thresholds.",
    )
    parser.add_argument(
        "--ignore-planar-error-for-grasp",
        action="store_true",
        help="Ignore lateral/side error during grasp readiness and only require image centering plus axial alignment.",
    )
    parser.add_argument(
        "--skip-initial-open",
        action="store_true",
        help="Skip opening the gripper before approach.",
    )
    parser.add_argument(
        "--settle-before-close-s",
        type=float,
        default=0.15,
        help="Pause briefly before closing once the grasp trigger condition is met.",
    )
    parser.add_argument(
        "--pre-close-forward-nudge-m",
        type=float,
        default=0.015,
        help="Base forward nudge after grasp-ready. The script will also add remaining positive axial error and split it across safe steps before closing.",
    )
    parser.add_argument(
        "--lift-height-m",
        type=float,
        default=None,
        help="Override lift height after grasp.",
    )
    parser.add_argument(
        "--contact-planar-servo-activate-below-px",
        type=float,
        default=80.0,
        help="Once the target is this close to image center, start using 3D lateral error to correct grasp-side offset.",
    )
    parser.add_argument(
        "--contact-planar-servo-max-m",
        type=float,
        default=0.015,
        help="Maximum XY correction magnitude from 3D lateral-error servo per step.",
    )
    parser.add_argument(
        "--contact-planar-servo-weight",
        type=float,
        default=0.85,
        help="Blend weight for 3D lateral-error correction after the target is centered enough.",
    )
    parser.add_argument(
        "--contact-planar-servo-axial-window-m",
        type=float,
        default=0.08,
        help="Only trust 3D lateral-error servo strongly when forward/back error is within this window.",
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
            remote_action_gripper_semantics=config["policy"].get("remote_action_gripper_semantics", "open_high"),
        )
    )


def _build_gripper(config: dict, robot: KinovaDriver) -> GripperDriver:
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

    delta_x_m = -error_x_px * x_gain_m_per_px
    delta_y_m = error_y_px * y_gain_m_per_px
    return (delta_x_m, delta_y_m)


def _fmt_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


def _fmt_cm(value_m: float | None, digits: int = 1) -> str:
    if value_m is None:
        return "None"
    return f"{value_m * 100.0:.{digits}f}cm"


def _vector_norm(vector_xyz: tuple[float, float, float] | None) -> float | None:
    if vector_xyz is None:
        return None
    return math.sqrt(sum(component * component for component in vector_xyz))


def _decompose_tip_error(
    delta_xyz: tuple[float, float, float],
    axis_xyz: tuple[float, float, float],
) -> tuple[float, float]:
    axial_error_m = sum(
        delta_component * axis_component
        for delta_component, axis_component in zip(delta_xyz, axis_xyz)
    )
    planar_delta_xyz = tuple(
        delta_component - axial_error_m * axis_component
        for delta_component, axis_component in zip(delta_xyz, axis_xyz)
    )
    planar_error_m = math.sqrt(sum(component * component for component in planar_delta_xyz))
    return (axial_error_m, planar_error_m)


def _planar_component_xyz(
    delta_xyz: tuple[float, float, float],
    axis_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    axial_error_m = sum(
        delta_component * axis_component
        for delta_component, axis_component in zip(delta_xyz, axis_xyz)
    )
    return tuple(
        delta_component - axial_error_m * axis_component
        for delta_component, axis_component in zip(delta_xyz, axis_xyz)
    )


def _clip_xy_magnitude(delta_xy: tuple[float, float], max_m: float) -> tuple[float, float]:
    norm = math.hypot(delta_xy[0], delta_xy[1])
    if max_m <= 0.0 or norm <= max_m or norm <= 1e-9:
        return delta_xy
    scale = max_m / norm
    return (delta_xy[0] * scale, delta_xy[1] * scale)


def _is_grasp_ready(
    target_center_distance_px: float,
    tip_to_contact_distance_m: float | None,
    axial_tip_error_m: float,
    planar_tip_error_m: float,
    success_distance_m: float,
    grasp_center_distance_px: float,
    grasp_axial_error_m: float,
    grasp_planar_error_m: float,
    grasp_on_center_only: bool,
    ignore_planar_error_for_grasp: bool,
) -> tuple[bool, str]:
    if tip_to_contact_distance_m is not None and tip_to_contact_distance_m <= success_distance_m:
        return (True, "3d_distance")
    if target_center_distance_px <= grasp_center_distance_px and grasp_on_center_only:
        return (True, "visual_center_only")
    if (
        ignore_planar_error_for_grasp
        and target_center_distance_px <= grasp_center_distance_px
        and abs(axial_tip_error_m) <= grasp_axial_error_m
    ):
        return (True, "visual_axial_alignment")
    if (
        target_center_distance_px <= grasp_center_distance_px
        and abs(axial_tip_error_m) <= grasp_axial_error_m
        and planar_tip_error_m <= grasp_planar_error_m
    ):
        return (True, "visual_alignment")
    return (False, "")


def _describe_axial_error(axial_m: float) -> str:
    distance_cm = abs(axial_m) * 100.0
    if axial_m > 1e-6:
        return f"目标还在夹爪前方 {distance_cm:.1f}cm"
    if axial_m < -1e-6:
        return f"目标已经到夹爪后方 {distance_cm:.1f}cm"
    return "目标就在夹爪前后方向附近"


def _describe_ready_reason(reason: str) -> str:
    if reason == "3d_distance":
        return "满足抓取：末端已经足够接近目标"
    if reason == "visual_center_only":
        return "满足抓取：目标已经进入图像中心区域"
    if reason == "visual_axial_alignment":
        return "满足抓取：图像居中且前后方向已经对齐（本次忽略横向误差）"
    if reason == "visual_alignment":
        return "满足抓取：图像和3D对齐条件都满足"
    return "未满足抓取条件"


def _measure_alignment(
    *,
    camera: RealSenseDriver,
    policy: OpenVLAWrapper,
    grasp_refiner: GraspRefiner,
    tf_manager: TFManager,
    instruction: str,
    robot_state,
    target_color: str,
    tip_offset_ee_m: tuple[float, float, float],
    tip_axis_ee: tuple[float, float, float],
) -> dict:
    frame = camera.capture_frame()
    centroid = detect_ball_centroid(frame.rgb_path_hint, target_color)
    if centroid is None:
        raise RuntimeError(f"Could not detect a {target_color} ball in {frame.rgb_path_hint}.")

    center_distance = center_distance_px(centroid, frame.width, frame.height)
    observation = Observation(instruction=instruction, frame=frame, robot_state=robot_state)
    policy_action = policy.predict_action(observation)
    policy_action = replace(
        policy_action,
        target_pixel=(int(round(centroid[0])), int(round(centroid[1]))),
    )
    depth_sample = grasp_refiner.depth_filter.sample_target_depth(
        policy_action.target_pixel,
        frame.depth_path_hint,
    )
    refined_grasp = grasp_refiner.refine(policy_action, observation)
    tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
        tip_offset_ee_m,
        robot_state.ee_position_m,
        robot_state.ee_yaw_deg,
        robot_state.ee_quaternion_xyzw,
    )
    tip_axis_base = tf_manager.ee_relative_xyz_to_base_offset(
        tip_axis_ee,
        robot_state.ee_yaw_deg,
        robot_state.ee_quaternion_xyzw,
    )

    tip_to_contact_distance_m = None
    axial_tip_error_m = 0.0
    planar_tip_error_m = 0.0
    tip_to_contact_delta = None
    if refined_grasp.contact_xyz_m is not None:
        tip_to_contact_delta = tuple(
            contact_axis - tip_axis
            for contact_axis, tip_axis in zip(refined_grasp.contact_xyz_m, tip_xyz)
        )
        tip_to_contact_distance_m = _vector_norm(tip_to_contact_delta)
        axial_tip_error_m, planar_tip_error_m = _decompose_tip_error(
            tip_to_contact_delta,
            tip_axis_base,
        )

    return {
        "frame": frame,
        "centroid": centroid,
        "center_distance_px": center_distance,
        "policy_action": policy_action,
        "depth_sample": depth_sample,
        "refined_grasp": refined_grasp,
        "tip_xyz": tip_xyz,
        "tip_to_contact_delta": tip_to_contact_delta,
        "tip_to_contact_distance_m": tip_to_contact_distance_m,
        "axial_tip_error_m": axial_tip_error_m,
        "planar_tip_error_m": planar_tip_error_m,
    }


def main() -> None:
    args = parse_args()
    config = load_config()
    trial_logger = TrialLogger(config["logging"]["log_dir"]) if config["logging"]["enabled"] else None
    instruction = args.instruction.strip() or config["task"]["instruction"]
    lift_height_m = args.lift_height_m if args.lift_height_m is not None else config["task"]["lift_height_m"]

    camera = _build_camera(config)
    robot = _build_robot(config)
    gripper = _build_gripper(config, robot)
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
    print("Image servo x gain m/px:", args.image_servo_x_gain_m_per_px)
    print("Image servo y gain m/px:", args.image_servo_y_gain_m_per_px)
    print("Image servo deadband px:", args.image_servo_deadband_px)
    print("Guarded z enabled:", args.enable_z)
    print("Forward enabled:", args.enable_forward)
    print("Gripper mode:", config["gripper"].get("mode", "state_only"))
    print("Auto grasp enabled: True (close + lift when grasp-ready)")
    print("Ignore planar error for grasp:", args.ignore_planar_error_for_grasp)
    print("Pre-close forward nudge m:", args.pre_close_forward_nudge_m)
    print("3D lateral servo center threshold px:", args.contact_planar_servo_activate_below_px)
    print("3D lateral servo max xy m:", args.contact_planar_servo_max_m)
    print("3D lateral servo weight:", args.contact_planar_servo_weight)

    tip_offset_ee_m = tuple(config["gripper"].get("tip_offset_ee_m", [0.0, 0.0, 0.0]))
    tip_axis_norm = _vector_norm(tip_offset_ee_m)
    if tip_axis_norm is None or tip_axis_norm <= 1e-6:
        tip_axis_ee = (0.0, 0.0, 1.0)
    else:
        tip_axis_ee = tuple(component / tip_axis_norm for component in tip_offset_ee_m)

    step_records = []
    first_observation: Observation | None = None
    last_policy_action = None
    last_safe_action = None
    last_refined_grasp = None
    initial_distance_px: float | None = None
    final_distance_px: float | None = None
    final_tip_distance_m: float | None = None
    grasp_executed = False
    close_success = False
    lift_executed = False
    close_reason = ""
    last_grasp_trigger_reason = ""
    grasp_trigger_axial_error_m: float | None = None
    grasp_trigger_total_error_m: float | None = None
    post_nudge_total_error_m: float | None = None
    post_nudge_axial_error_m: float | None = None
    post_nudge_planar_error_m: float | None = None
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
            depth_sample = grasp_refiner.depth_filter.sample_target_depth(
                policy_action.target_pixel,
                before_frame.depth_path_hint,
            )
            refined_grasp = grasp_refiner.refine(policy_action, observation)
            before_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
                tip_offset_ee_m,
                before_state.ee_position_m,
                before_state.ee_yaw_deg,
                before_state.ee_quaternion_xyzw,
            )
            tip_axis_base = tf_manager.ee_relative_xyz_to_base_offset(
                tip_axis_ee,
                before_state.ee_yaw_deg,
                before_state.ee_quaternion_xyzw,
            )

            before_tip_to_contact_delta = None
            before_tip_to_contact_distance_m = None
            axial_tip_error_m = 0.0
            planar_tip_error_m = 0.0
            contact_planar_delta_xyz = (0.0, 0.0, 0.0)
            if refined_grasp.contact_xyz_m is not None:
                before_tip_to_contact_delta = tuple(
                    contact_axis - tip_axis
                    for contact_axis, tip_axis in zip(refined_grasp.contact_xyz_m, before_tip_xyz)
                )
                before_tip_to_contact_distance_m = _vector_norm(before_tip_to_contact_delta)
                axial_tip_error_m, planar_tip_error_m = _decompose_tip_error(
                    before_tip_to_contact_delta,
                    tip_axis_base,
                )
                contact_planar_delta_xyz = _planar_component_xyz(
                    before_tip_to_contact_delta,
                    tip_axis_base,
                )
            final_tip_distance_m = before_tip_to_contact_distance_m

            grasp_ready_before_move, grasp_trigger_reason = _is_grasp_ready(
                target_center_distance_px=before_distance_px if before_distance_px is not None else float("inf"),
                tip_to_contact_distance_m=before_tip_to_contact_distance_m,
                axial_tip_error_m=axial_tip_error_m,
                planar_tip_error_m=planar_tip_error_m,
                success_distance_m=args.success_distance_m,
                grasp_center_distance_px=args.grasp_center_distance_px,
                grasp_axial_error_m=args.grasp_axial_error_m,
                grasp_planar_error_m=args.grasp_planar_error_m,
                grasp_on_center_only=args.grasp_on_center_only,
                ignore_planar_error_for_grasp=args.ignore_planar_error_for_grasp,
            )
            consecutive_grasp_ready_steps = (
                consecutive_grasp_ready_steps + 1 if grasp_ready_before_move else 0
            )
            if grasp_ready_before_move:
                last_grasp_trigger_reason = grasp_trigger_reason
                grasp_trigger_axial_error_m = axial_tip_error_m
                grasp_trigger_total_error_m = before_tip_to_contact_distance_m

            if consecutive_grasp_ready_steps >= max(args.grasp_stable_steps, 1):
                grasp_executed = True
                close_reason = f"pre_move_{grasp_trigger_reason or 'ready'}"
                grasp_trigger_axial_error_m = axial_tip_error_m
                grasp_trigger_total_error_m = before_tip_to_contact_distance_m
                last_refined_grasp = refined_grasp
                last_policy_action = PolicyAction(
                    delta_xyz_m=(0.0, 0.0, 0.0),
                    delta_yaw_deg=0.0,
                    gripper_command="close",
                    confidence=policy_action.confidence,
                    target_pixel=policy_action.target_pixel,
                    notes="Planar approach reached grasp-ready threshold before move",
                    metadata=policy_action.metadata,
                )
                last_safe_action = action_adapter.adapt(last_policy_action, before_state)
                print(
                    f"[step {step_idx + 1}/{args.steps}]"
                    f" center_px={before_distance_px:.1f}"
                    f" depth_m={_fmt_optional(depth_sample.depth_m, 3)}"
                    f" total_err={_fmt_cm(before_tip_to_contact_distance_m)}"
                    f" forward_err={_fmt_cm(axial_tip_error_m)}"
                    f" side_err={_fmt_cm(planar_tip_error_m)}"
                    f" stable={consecutive_grasp_ready_steps}/{args.grasp_stable_steps}"
                )
                print("  抓取判断:", _describe_ready_reason(grasp_trigger_reason))
                print("  前后方向:", _describe_axial_error(axial_tip_error_m))
                print("  已满足抓取条件，下一步执行闭合夹爪")
                break

            image_servo_delta_xy = image_servo_planar_delta(
                before_centroid,
                before_frame.width,
                before_frame.height,
                x_gain_m_per_px=args.image_servo_x_gain_m_per_px,
                y_gain_m_per_px=args.image_servo_y_gain_m_per_px,
                deadband_px=args.image_servo_deadband_px,
            )
            image_error_x_px = before_centroid[0] - (before_frame.width / 2.0)
            image_error_y_px = before_centroid[1] - (before_frame.height / 2.0)
            if abs(image_error_x_px) <= args.image_servo_deadband_px:
                image_error_x_px = 0.0
            if abs(image_error_y_px) <= args.image_servo_deadband_px:
                image_error_y_px = 0.0

            forward_enabled_this_step = False
            forward_base_offset = (0.0, 0.0, 0.0)
            if (
                args.enable_forward
                and depth_sample.valid
                and before_distance_px is not None
                and before_distance_px <= args.allow_forward_when_center_distance_below_px
                and depth_sample.depth_m > args.stop_forward_when_depth_below_m
            ):
                forward_enabled_this_step = True
                forward_base_offset = tf_manager.camera_vector_to_base_offset(
                    (0.0, 0.0, args.forward_step_m),
                    before_state.ee_yaw_deg,
                    before_state.ee_quaternion_xyzw,
                )

            contact_planar_servo_active = False
            contact_planar_servo_delta_xy = (0.0, 0.0)
            if (
                refined_grasp.contact_xyz_m is not None
                and before_distance_px is not None
                and before_distance_px <= args.contact_planar_servo_activate_below_px
                and abs(axial_tip_error_m) <= args.contact_planar_servo_axial_window_m
            ):
                contact_planar_servo_active = True
                contact_planar_servo_delta_xy = _clip_xy_magnitude(
                    (contact_planar_delta_xyz[0], contact_planar_delta_xyz[1]),
                    args.contact_planar_servo_max_m,
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
                if contact_planar_servo_active:
                    planar_servo_weight = max(0.0, min(args.contact_planar_servo_weight, 1.0))
                    planar_xy_base = (
                        (1.0 - planar_servo_weight) * image_servo_delta_xy[0]
                        + planar_servo_weight * contact_planar_servo_delta_xy[0],
                        (1.0 - planar_servo_weight) * image_servo_delta_xy[1]
                        + planar_servo_weight * contact_planar_servo_delta_xy[1],
                    )
                else:
                    planar_xy_base = image_servo_delta_xy
                planar_delta_xy = (
                    planar_xy_base[0] + forward_base_offset[0],
                    planar_xy_base[1] + forward_base_offset[1],
                )
                planar_notes = "Planar coarse approach using image-servo x/y"
            else:
                refine_xy_delta = (
                    refined_grasp.target_xyz_m[0] - before_state.ee_position_m[0],
                    refined_grasp.target_xyz_m[1] - before_state.ee_position_m[1],
                )
                if contact_planar_servo_active:
                    planar_servo_weight = max(0.0, min(args.contact_planar_servo_weight, 1.0))
                    planar_xy_base = (
                        (1.0 - planar_servo_weight) * refine_xy_delta[0]
                        + planar_servo_weight * contact_planar_servo_delta_xy[0],
                        (1.0 - planar_servo_weight) * refine_xy_delta[1]
                        + planar_servo_weight * contact_planar_servo_delta_xy[1],
                    )
                else:
                    planar_xy_base = refine_xy_delta
                planar_delta_xy = (
                    planar_xy_base[0] + forward_base_offset[0],
                    planar_xy_base[1] + forward_base_offset[1],
                )
                planar_notes = "Planar coarse approach toward refined target"
            if args.enable_forward:
                planar_notes += " with camera-forward motion"

            planar_policy_action = PolicyAction(
                delta_xyz_m=(
                    planar_delta_xy[0],
                    planar_delta_xy[1],
                    guarded_dz + forward_base_offset[2],
                ),
                delta_yaw_deg=0.0,
                gripper_command="open",
                confidence=policy_action.confidence,
                target_pixel=policy_action.target_pixel,
                notes=planar_notes,
                metadata={
                    **policy_action.metadata,
                    "planar_mode": args.planar_mode,
                    "image_servo_delta_xy_m": image_servo_delta_xy,
                    "forward_enabled_this_step": forward_enabled_this_step,
                    "forward_base_offset_m": forward_base_offset,
                    "depth_sample_m": depth_sample.depth_m,
                    "contact_planar_servo_active": contact_planar_servo_active,
                    "contact_planar_servo_delta_xy_m": contact_planar_servo_delta_xy,
                    "refined_target_xyz_m": refined_grasp.target_xyz_m,
                    "source_policy_delta_xyz_m": policy_action.delta_xyz_m,
                    "guarded_z_enabled": args.enable_z,
                    "raw_refined_dz_m": raw_dz,
                    "guarded_dz_m": guarded_dz,
                    "tip_to_contact_distance_m": before_tip_to_contact_distance_m,
                    "axial_tip_error_m": axial_tip_error_m,
                    "planar_tip_error_m": planar_tip_error_m,
                },
            )
            safe_action = action_adapter.adapt(planar_policy_action, before_state)

            print(
                f"[step {step_idx + 1}/{args.steps}]"
                f" 图像距中心={before_distance_px:.1f}px"
                f" 深度={_fmt_optional(depth_sample.depth_m, 3)}m"
                f" 总误差={_fmt_cm(before_tip_to_contact_distance_m)}"
                f" 前后误差={_fmt_cm(axial_tip_error_m)}"
                f" 横向误差={_fmt_cm(planar_tip_error_m)}"
                f" 连续满足={consecutive_grasp_ready_steps}/{args.grasp_stable_steps}"
                f" 前推={'开启' if forward_enabled_this_step else '关闭'}"
                f" 横向对齐={'3D修正' if contact_planar_servo_active else '图像居中'}"
                f" clipped={safe_action.clipped}"
            )
            print(
                "  控制量 "
                "cmd_xyz_m="
                f"({planar_policy_action.delta_xyz_m[0]:+.4f}, {planar_policy_action.delta_xyz_m[1]:+.4f}, {planar_policy_action.delta_xyz_m[2]:+.4f})"
                " safe_xyz_m="
                f"({safe_action.delta_xyz_m[0]:+.4f}, {safe_action.delta_xyz_m[1]:+.4f}, {safe_action.delta_xyz_m[2]:+.4f})"
            )
            print(
                "  图像偏差 "
                f"dx={image_error_x_px:+.1f}px dy={image_error_y_px:+.1f}px"
            )
            if contact_planar_servo_active:
                print(
                    "  3D横向修正:"
                    f" ({contact_planar_servo_delta_xy[0]:+.4f}, {contact_planar_servo_delta_xy[1]:+.4f}) m"
                )
            print("  抓取判断:", _describe_ready_reason(grasp_trigger_reason))
            print("  前后方向:", _describe_axial_error(axial_tip_error_m))
            if safe_action.rejection_reason:
                print("  安全限幅:", safe_action.rejection_reason)

            robot.move_cartesian_delta(safe_action.delta_xyz_m, safe_action.delta_yaw_deg)

            after_state = robot.get_state()
            after_frame = camera.capture_frame()
            after_centroid = detect_ball_centroid(after_frame.rgb_path_hint, args.target_color)
            after_distance_px = center_distance_px(after_centroid, after_frame.width, after_frame.height)
            after_image_error_x_px = None
            after_image_error_y_px = None
            if after_centroid is not None:
                after_image_error_x_px = after_centroid[0] - (after_frame.width / 2.0)
                after_image_error_y_px = after_centroid[1] - (after_frame.height / 2.0)
                if abs(after_image_error_x_px) <= args.image_servo_deadband_px:
                    after_image_error_x_px = 0.0
                if abs(after_image_error_y_px) <= args.image_servo_deadband_px:
                    after_image_error_y_px = 0.0
            final_distance_px = after_distance_px
            after_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
                tip_offset_ee_m,
                after_state.ee_position_m,
                after_state.ee_yaw_deg,
                after_state.ee_quaternion_xyzw,
            )

            after_tip_to_contact_distance_m = None
            after_axial_tip_error_m = axial_tip_error_m
            after_planar_tip_error_m = planar_tip_error_m
            if refined_grasp.contact_xyz_m is not None:
                after_tip_to_contact_delta = tuple(
                    contact_axis - tip_axis
                    for contact_axis, tip_axis in zip(refined_grasp.contact_xyz_m, after_tip_xyz)
                )
                after_tip_to_contact_distance_m = _vector_norm(after_tip_to_contact_delta)
                after_axial_tip_error_m, after_planar_tip_error_m = _decompose_tip_error(
                    after_tip_to_contact_delta,
                    tip_axis_base,
                )
            final_tip_distance_m = after_tip_to_contact_distance_m

            grasp_ready_after_move, grasp_trigger_reason_after_move = _is_grasp_ready(
                target_center_distance_px=after_distance_px if after_distance_px is not None else float("inf"),
                tip_to_contact_distance_m=after_tip_to_contact_distance_m,
                axial_tip_error_m=after_axial_tip_error_m,
                planar_tip_error_m=after_planar_tip_error_m,
                success_distance_m=args.success_distance_m,
                grasp_center_distance_px=args.grasp_center_distance_px,
                grasp_axial_error_m=args.grasp_axial_error_m,
                grasp_planar_error_m=args.grasp_planar_error_m,
                grasp_on_center_only=args.grasp_on_center_only,
                ignore_planar_error_for_grasp=args.ignore_planar_error_for_grasp,
            )
            consecutive_grasp_ready_steps = (
                consecutive_grasp_ready_steps + 1 if grasp_ready_after_move else 0
            )
            if grasp_ready_after_move:
                last_grasp_trigger_reason = grasp_trigger_reason_after_move
                grasp_trigger_axial_error_m = after_axial_tip_error_m
                grasp_trigger_total_error_m = after_tip_to_contact_distance_m

            print(
                "  移动后"
                f" 图像距中心={_fmt_optional(after_distance_px, 1)}px"
                f" 总误差={_fmt_cm(after_tip_to_contact_distance_m)}"
                f" 前后误差={_fmt_cm(after_axial_tip_error_m)}"
                f" 横向误差={_fmt_cm(after_planar_tip_error_m)}"
                f" 连续满足={consecutive_grasp_ready_steps}/{args.grasp_stable_steps}"
            )
            print(
                "  移动后图像偏差 "
                f"dx={_fmt_optional(after_image_error_x_px, 1)}px"
                f" dy={_fmt_optional(after_image_error_y_px, 1)}px"
            )
            print("  移动后抓取判断:", _describe_ready_reason(grasp_trigger_reason_after_move))
            print("  移动后前后方向:", _describe_axial_error(after_axial_tip_error_m))

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
                    "before_image_error_x_px": image_error_x_px,
                    "before_image_error_y_px": image_error_y_px,
                    "after_image_error_x_px": after_image_error_x_px,
                    "after_image_error_y_px": after_image_error_y_px,
                    "before_target_center_distance_px": before_distance_px,
                    "after_target_center_distance_px": after_distance_px,
                    "depth_sample_m": depth_sample.depth_m,
                    "contact_xyz_m": refined_grasp.contact_xyz_m,
                    "before_tip_to_contact_distance_m": before_tip_to_contact_distance_m,
                    "after_tip_to_contact_distance_m": after_tip_to_contact_distance_m,
                    "axial_tip_error_m_before": axial_tip_error_m,
                    "axial_tip_error_m_after": after_axial_tip_error_m,
                    "planar_tip_error_m_before": planar_tip_error_m,
                    "planar_tip_error_m_after": after_planar_tip_error_m,
                    "planar_policy_delta_xyz_m": planar_policy_action.delta_xyz_m,
                    "forward_enabled_this_step": forward_enabled_this_step,
                    "forward_base_offset_m": forward_base_offset,
                    "contact_planar_servo_active": contact_planar_servo_active,
                    "contact_planar_servo_delta_xy_m": contact_planar_servo_delta_xy,
                    "raw_refined_dz_m": raw_dz,
                    "guarded_dz_m": guarded_dz,
                    "safe_delta_xyz_m": safe_action.delta_xyz_m,
                    "safe_action_clipped": safe_action.clipped,
                    "grasp_ready_before_move": grasp_ready_before_move,
                    "grasp_trigger_reason_before_move": grasp_trigger_reason,
                    "grasp_ready_after_move": grasp_ready_after_move,
                    "grasp_trigger_reason_after_move": grasp_trigger_reason_after_move,
                    "consecutive_grasp_ready_steps": consecutive_grasp_ready_steps,
                    "policy_metadata": policy_action.metadata,
                }
            )
            last_policy_action = planar_policy_action
            last_safe_action = safe_action
            last_refined_grasp = refined_grasp

            if consecutive_grasp_ready_steps >= max(args.grasp_stable_steps, 1):
                grasp_executed = True
                close_reason = f"post_move_{grasp_trigger_reason_after_move or 'ready'}"
                print("  已满足抓取条件，下一步执行闭合夹爪")
                break

        if grasp_executed:
            desired_forward_nudge_m = 0.0
            if grasp_trigger_axial_error_m is None:
                desired_forward_nudge_m = max(args.pre_close_forward_nudge_m, 0.0)
            elif grasp_trigger_axial_error_m > 0.0:
                desired_forward_nudge_m = max(
                    args.pre_close_forward_nudge_m + grasp_trigger_axial_error_m,
                    0.0,
                )
            max_safe_step_m = max(config["robot"]["max_translation_step_m"], 1e-6)
            remaining_forward_nudge_m = desired_forward_nudge_m
            pre_close_step_idx = 0
            print(
                "抓取前补偿前推目标:"
                f" 触发时前后误差={_fmt_cm(grasp_trigger_axial_error_m)}"
                f" 基础补偿={_fmt_cm(args.pre_close_forward_nudge_m)}"
                f" 总计划前推={_fmt_cm(desired_forward_nudge_m)}"
            )
            while remaining_forward_nudge_m > 1e-6 and pre_close_step_idx < 6:
                close_prep_state = robot.get_state()
                close_tip_axis_base = tf_manager.ee_relative_xyz_to_base_offset(
                    tip_axis_ee,
                    close_prep_state.ee_yaw_deg,
                    close_prep_state.ee_quaternion_xyzw,
                )
                requested_step_m = min(remaining_forward_nudge_m, max_safe_step_m)
                pre_close_forward_delta_xyz = tuple(
                    requested_step_m * axis_component
                    for axis_component in close_tip_axis_base
                )
                pre_close_action = PolicyAction(
                    delta_xyz_m=pre_close_forward_delta_xyz,
                    delta_yaw_deg=0.0,
                    gripper_command="open",
                    confidence=1.0,
                    target_pixel=None,
                    notes="Pre-close forward nudge",
                    metadata={
                        "control_mode": "pre_close_forward_nudge",
                        "pre_close_forward_nudge_m": args.pre_close_forward_nudge_m,
                        "requested_step_m": requested_step_m,
                        "tip_axis_base": close_tip_axis_base,
                    },
                )
                safe_pre_close_action = action_adapter.adapt(pre_close_action, close_prep_state)
                actual_forward_step_m = max(
                    0.0,
                    sum(
                        delta_component * axis_component
                        for delta_component, axis_component in zip(
                            safe_pre_close_action.delta_xyz_m,
                            close_tip_axis_base,
                        )
                    ),
                )
                pre_close_step_idx += 1
                print(
                    f"  补偿前推第{pre_close_step_idx}步:"
                    f" 请求={_fmt_cm(requested_step_m)}"
                    f" 实际={_fmt_cm(actual_forward_step_m)}"
                    f" 剩余={_fmt_cm(max(remaining_forward_nudge_m - actual_forward_step_m, 0.0))}"
                    f" clipped={safe_pre_close_action.clipped}"
                )
                if safe_pre_close_action.rejection_reason:
                    print("    抓取前补偿限幅:", safe_pre_close_action.rejection_reason)
                if actual_forward_step_m <= 1e-6:
                    print("    抓取前补偿停止: 实际前推过小，避免空转。")
                    break
                robot.move_cartesian_delta(
                    safe_pre_close_action.delta_xyz_m,
                    safe_pre_close_action.delta_yaw_deg,
                )
                remaining_forward_nudge_m = max(
                    remaining_forward_nudge_m - actual_forward_step_m,
                    0.0,
                )
            post_nudge_state = robot.get_state()
            post_nudge_measurement = _measure_alignment(
                camera=camera,
                policy=policy,
                grasp_refiner=grasp_refiner,
                tf_manager=tf_manager,
                instruction=instruction,
                robot_state=post_nudge_state,
                target_color=args.target_color,
                tip_offset_ee_m=tip_offset_ee_m,
                tip_axis_ee=tip_axis_ee,
            )
            post_nudge_total_error_m = post_nudge_measurement["tip_to_contact_distance_m"]
            post_nudge_axial_error_m = post_nudge_measurement["axial_tip_error_m"]
            post_nudge_planar_error_m = post_nudge_measurement["planar_tip_error_m"]
            print(
                "抓取前补偿后复测:"
                f" 图像距中心={_fmt_optional(post_nudge_measurement['center_distance_px'], 1)}px"
                f" 总误差={_fmt_cm(post_nudge_total_error_m)}"
                f" 前后误差={_fmt_cm(post_nudge_axial_error_m)}"
                f" 横向误差={_fmt_cm(post_nudge_planar_error_m)}"
            )
            total_error_improved = False
            if (
                grasp_trigger_total_error_m is not None
                and post_nudge_total_error_m is not None
            ):
                total_error_improved = post_nudge_total_error_m + 1e-4 < grasp_trigger_total_error_m
            print(
                "  补偿后是否更接近目标:",
                total_error_improved,
                f"(触发时总误差={_fmt_cm(grasp_trigger_total_error_m)}, 补偿后总误差={_fmt_cm(post_nudge_total_error_m)})",
            )
            if not total_error_improved:
                close_reason = "post_nudge_not_improved_skip_close"
                print("  补偿后总误差没有变小，跳过闭合夹爪。")
            else:
                if args.settle_before_close_s > 0.0:
                    time.sleep(args.settle_before_close_s)
                close_success = gripper.close()
                print("夹爪闭合结果:", close_success)
                if close_success:
                    robot.move_cartesian_delta((0.0, 0.0, lift_height_m), 0.0)
                    lift_executed = True
                    print("抬升高度:", f"{lift_height_m:.3f}m")

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
        approach_success = bool(reached_radius or improved_enough)
        success = bool(grasp_executed and close_success)

        failure_reason = ""
        if not grasp_executed:
            failure_reason = (
                "Target was centered but never satisfied grasp thresholds "
                f"(last_trigger={last_grasp_trigger_reason or 'none'}, final_tip_err_m={_fmt_optional(final_tip_distance_m, 3)})"
            )
        elif not close_success:
            failure_reason = "Gripper close command failed"

        final_tip_xyz = tf_manager.ee_relative_xyz_to_base_xyz(
            tip_offset_ee_m,
            final_state.ee_position_m,
            final_state.ee_yaw_deg,
            final_state.ee_quaternion_xyzw,
        )
        result = ExecutionResult(
            success=success,
            state_trace=[f"planar_coarse_step_{idx + 1}" for idx in range(len(step_records))],
            message="Planar coarse approach + grasp run completed",
            failure_reason=failure_reason,
            grasp=RefinedGrasp(
                target_xyz_m=final_tip_xyz,
                target_yaw_deg=final_state.ee_yaw_deg,
                grasp_width_m=final_state.gripper_opening_m,
                quality=1.0 if success else 0.0,
                source="planar_coarse_approach_test",
                contact_xyz_m=last_refined_grasp.contact_xyz_m if last_refined_grasp is not None else None,
            ),
        )

        print("[summary]")
        print(f"  初始图像距离中心: {_fmt_optional(initial_distance_px, 1)}px")
        print(f"  最终图像距离中心: {_fmt_optional(final_distance_px, 1)}px")
        print(f"  总改进量: {_fmt_optional(net_improvement_px, 1)}px")
        print(f"  粗接近成功: {approach_success}")
        print(f"  是否触发抓取: {grasp_executed}")
        print(f"  触发原因: {close_reason or 'none'}")
        print(f"  最后一次抓取判断: {_describe_ready_reason(last_grasp_trigger_reason)}")
        print(f"  最终总误差: {_fmt_cm(final_tip_distance_m)}")
        print(f"  补偿后总误差: {_fmt_cm(post_nudge_total_error_m)}")
        print(f"  补偿后前后误差: {_fmt_cm(post_nudge_axial_error_m)}")
        print(f"  补偿后横向误差: {_fmt_cm(post_nudge_planar_error_m)}")
        print(f"  夹爪闭合成功: {close_success}")
        print(f"  是否执行抬升: {lift_executed}")
        print(f"  整体成功: {success}")
        if result.failure_reason:
            print("  失败原因:", result.failure_reason)

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
                    "forward_enabled": args.enable_forward,
                    "forward_step_m": args.forward_step_m,
                    "allow_forward_when_center_distance_below_px": args.allow_forward_when_center_distance_below_px,
                    "stop_forward_when_depth_below_m": args.stop_forward_when_depth_below_m,
                    "max_z_down_step_m": args.max_z_down_step_m,
                    "max_z_up_step_m": args.max_z_up_step_m,
                    "allow_z_when_center_distance_below_px": args.allow_z_when_center_distance_below_px,
                    "success_distance_m": args.success_distance_m,
                    "grasp_center_distance_px": args.grasp_center_distance_px,
                    "grasp_planar_error_m": args.grasp_planar_error_m,
                    "grasp_axial_error_m": args.grasp_axial_error_m,
                    "grasp_stable_steps": args.grasp_stable_steps,
                    "grasp_on_center_only": args.grasp_on_center_only,
                    "ignore_planar_error_for_grasp": args.ignore_planar_error_for_grasp,
                    "pre_close_forward_nudge_m": args.pre_close_forward_nudge_m,
                    "close_reason": close_reason,
                    "close_success": close_success,
                    "lift_executed": lift_executed,
                    "lift_height_m": lift_height_m,
                    "initial_target_center_distance_px": initial_distance_px,
                    "final_target_center_distance_px": final_distance_px,
                    "net_improvement_px": net_improvement_px,
                    "reached_radius": reached_radius,
                    "improved_enough": improved_enough,
                    "final_tip_distance_m": final_tip_distance_m,
                    "last_grasp_trigger_reason": last_grasp_trigger_reason,
                    "grasp_trigger_total_error_m": grasp_trigger_total_error_m,
                    "post_nudge_total_error_m": post_nudge_total_error_m,
                    "post_nudge_axial_error_m": post_nudge_axial_error_m,
                    "post_nudge_planar_error_m": post_nudge_planar_error_m,
                    "step_records": step_records,
                },
            )
            print("  trial_log:", trial_logger.log_path)
    finally:
        gripper.shutdown()


if __name__ == "__main__":
    main()
