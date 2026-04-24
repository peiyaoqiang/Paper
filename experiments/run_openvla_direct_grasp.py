from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.trial_logger import TrialLogger
from adapters.action_adapter import ActionAdapter, ActionAdapterConfig
from common.types import ExecutionResult, Observation, PolicyAction, RefinedGrasp
from drivers.gripper_driver import GripperConfig, GripperDriver
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver
from geometry.depth_filter import DepthFilter
from policy.openvla_wrapper import OpenVLAConfig, OpenVLAWrapper


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closer-to-pure OpenVLA grasp: execute OpenVLA deltas directly, with minimal local logic."
    )
    parser.add_argument("--instruction", type=str, default="", help="Override default instruction.")
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Maximum number of OpenVLA action steps before stopping.",
    )
    parser.add_argument(
        "--approach-min-steps-before-engage",
        type=int,
        default=3,
        help="Do not enter engage from policy close until this many approach motion steps have executed.",
    )
    parser.add_argument(
        "--target-color",
        type=str,
        choices=("red", "green", "none"),
        default="red",
        help="Optional color used only for monitoring and center-based engage trigger.",
    )
    parser.add_argument(
        "--close-trigger",
        type=str,
        choices=("policy_only", "policy_or_center"),
        default="policy_or_center",
        help="Whether engage can be triggered only by policy close, or also by visual centering.",
    )
    parser.add_argument(
        "--close-when-center-distance-below-px",
        type=float,
        default=45.0,
        help="Center-based close threshold when close-trigger includes center fallback.",
    )
    parser.add_argument(
        "--policy-close-engage-max-center-distance-px",
        type=float,
        default=90.0,
        help="Only allow policy-close to enter engage when the target is within this center distance. Set <=0 to disable.",
    )
    parser.add_argument(
        "--center-stable-steps",
        type=int,
        default=1,
        help="Require this many consecutive centered observations before center-triggered engage.",
    )
    parser.add_argument(
        "--use-detected-centroid-as-target-pixel",
        action="store_true",
        help="Override OpenVLA target_pixel with detected ball centroid for logging only.",
    )
    parser.add_argument(
        "--skip-initial-open",
        action="store_true",
        help="Skip opening the gripper before approach.",
    )
    parser.add_argument(
        "--settle-before-close-s",
        type=float,
        default=0.1,
        help="Pause briefly before closing once the grasp trigger condition is met.",
    )
    parser.add_argument(
        "--lift-height-m",
        type=float,
        default=None,
        help="Override lift height after grasp.",
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
        "--continuous-servo",
        action="store_true",
        help="Use continuous twist servo streaming instead of send-and-brake step execution.",
    )
    parser.add_argument(
        "--continuous-servo-hz",
        type=float,
        default=20.0,
        help="Continuous servo publish rate.",
    )
    parser.add_argument(
        "--continuous-servo-control-hz",
        type=float,
        default=12.0,
        help="Async upper-layer control refresh rate for reasserting the latest continuous-servo command.",
    )
    parser.add_argument(
        "--continuous-servo-horizon-s",
        type=float,
        default=0.25,
        help="Interpret each commanded delta over this horizon when converting to velocity.",
    )
    parser.add_argument(
        "--continuous-servo-apply-s",
        type=float,
        default=0.25,
        help="How long to hold each updated command before evaluating observed motion.",
    )
    parser.add_argument(
        "--continuous-servo-stale-timeout-s",
        type=float,
        default=2.0,
        help="Auto-zero continuous servo output if command updates stop for this long.",
    )
    parser.add_argument(
        "--continuous-servo-command-alpha",
        type=float,
        default=0.35,
        help="Blend factor for continuous servo command smoothing. Lower values are smoother.",
    )
    parser.add_argument(
        "--continuous-servo-max-linear-speed-mps",
        type=float,
        default=0.03,
        help="Clamp continuous servo linear speed magnitude to this value.",
    )
    parser.add_argument(
        "--continuous-servo-max-angular-speed-degps",
        type=float,
        default=8.0,
        help="Clamp continuous servo angular speed magnitude to this value.",
    )
    parser.add_argument(
        "--close-on-policy-step",
        action="store_true",
        help="If OpenVLA outputs close, close immediately on that step instead of after another move.",
    )
    parser.add_argument(
        "--pre-close-forward-step-m",
        type=float,
        default=0.015,
        help="Forward distance per step along the gripper axis before closing.",
    )
    parser.add_argument(
        "--engage-forward-sign",
        type=float,
        default=1.0,
        help="Sign for engage forward axis (+1 or -1). Use -1 if engage moves away from target.",
    )
    parser.add_argument(
        "--engage-auto-flip-forward-direction",
        action="store_true",
        help="If engage keeps increasing target depth or worsening centering, flip engage forward direction once.",
    )
    parser.add_argument(
        "--engage-auto-flip-after-steps",
        type=int,
        default=6,
        help="Consider flipping engage forward direction after this many engage steps.",
    )
    parser.add_argument(
        "--engage-auto-flip-max-depth-drop-m",
        type=float,
        default=0.01,
        help="If depth drop stays below this value, engage may be moving the wrong way.",
    )
    parser.add_argument(
        "--engage-auto-flip-center-worsen-px",
        type=float,
        default=6.0,
        help="If center distance worsens by at least this many pixels, engage may be moving the wrong way.",
    )
    parser.add_argument(
        "--engage-auto-flip-max-forward-progress-m",
        type=float,
        default=0.003,
        help="If observed positive forward progress stays below this value after several engage steps, flip engage forward direction once.",
    )
    parser.add_argument(
        "--pre-close-forward-steps",
        type=int,
        default=2,
        help="Number of short forward approach steps to execute before closing.",
    )
    parser.add_argument(
        "--pre-close-servo-x-gain-m-per-px",
        type=float,
        default=0.00045,
        help="Base-frame x correction per image x pixel error during pre-close forward servoing.",
    )
    parser.add_argument(
        "--pre-close-servo-y-gain-m-per-px",
        type=float,
        default=0.00055,
        help="Base-frame y correction per image y pixel error during pre-close forward servoing.",
    )
    parser.add_argument(
        "--pre-close-servo-x-sign",
        type=float,
        default=-1.0,
        help="Image-servo sign for x correction. Use -1 or 1.",
    )
    parser.add_argument(
        "--pre-close-servo-y-sign",
        type=float,
        default=-1.0,
        help="Image-servo sign for y correction. Use -1 or 1.",
    )
    parser.add_argument(
        "--approach-image-servo",
        action="store_true",
        help="During approach, blend in a light local image-servo x/y correction toward the detected target.",
    )
    parser.add_argument(
        "--approach-image-servo-weight",
        type=float,
        default=0.8,
        help="Blend weight for approach image-servo x/y. 1.0 means replace policy x/y with local image-servo x/y.",
    )
    parser.add_argument(
        "--approach-image-servo-max-m",
        type=float,
        default=0.008,
        help="Cap the magnitude of approach image-servo x/y correction.",
    )
    parser.add_argument(
        "--approach-image-servo-min-center-distance-px",
        type=float,
        default=80.0,
        help="Only apply approach image-servo when the target is farther than this center distance.",
    )
    parser.add_argument(
        "--approach-image-servo-x-sign",
        type=float,
        default=-1.0,
        help="Image-servo sign for approach x correction. Use -1 or 1.",
    )
    parser.add_argument(
        "--approach-image-servo-y-sign",
        type=float,
        default=1.0,
        help="Image-servo sign for approach y correction. Use -1 or 1.",
    )
    parser.add_argument(
        "--approach-center-priority",
        action="store_true",
        help="When the target is far from center, let local image-servo dominate approach x/y before engaging.",
    )
    parser.add_argument(
        "--approach-center-priority-threshold-px",
        type=float,
        default=140.0,
        help="Activate center-priority approach when center distance is above this threshold.",
    )
    parser.add_argument(
        "--approach-center-priority-policy-xy-scale",
        type=float,
        default=0.0,
        help="Keep this fraction of OpenVLA x/y during center-priority approach. 0 means local image-servo only.",
    )
    parser.add_argument(
        "--approach-center-priority-z-scale",
        type=float,
        default=0.0,
        help="Keep this fraction of OpenVLA z during center-priority approach. 0 means freeze z while centering.",
    )
    parser.add_argument(
        "--pre-close-servo-deadband-px",
        type=float,
        default=10.0,
        help="Ignore small centroid errors within this deadband during pre-close forward servoing.",
    )
    parser.add_argument(
        "--pre-close-min-forward-steps-before-early-close",
        type=int,
        default=2,
        help="Require at least this many pre-close forward steps before early close can stop the forward approach.",
    )
    parser.add_argument(
        "--engage-min-forward-progress-m",
        type=float,
        default=0.015,
        help="Require at least this observed forward progress along the gripper axis before allowing close.",
    )
    parser.add_argument(
        "--engage-servo-max-m",
        type=float,
        default=0.006,
        help="Limit engage xy servo correction magnitude so forward motion remains dominant.",
    )
    parser.add_argument(
        "--engage-min-forward-command-ratio",
        type=float,
        default=0.6,
        help="Require combined engage command to keep at least this ratio of commanded forward component.",
    )
    parser.add_argument(
        "--close-when-depth-below-m",
        type=float,
        default=0.18,
        help="Allow close when aligned depth at the target pixel is below this value.",
    )
    parser.add_argument(
        "--close-when-depth-drop-m",
        type=float,
        default=0.04,
        help="Allow close when engage-phase target depth has decreased by at least this value.",
    )
    parser.add_argument(
        "--require-depth-ready-for-close",
        action="store_true",
        help="Require depth readiness for close decisions.",
    )
    parser.add_argument(
        "--no-require-depth-ready-for-close",
        dest="require_depth_ready_for_close",
        action="store_false",
        help="Disable depth readiness gating for close decisions.",
    )
    parser.add_argument(
        "--engage-min-observed-step-m",
        type=float,
        default=0.001,
        help="Treat engage steps below this observed translation norm as ineffective.",
    )
    parser.add_argument(
        "--engage-max-stuck-steps",
        type=int,
        default=4,
        help="If this many consecutive engage steps are ineffective, extend engage step budget.",
    )
    parser.add_argument(
        "--engage-extra-steps-on-stuck",
        type=int,
        default=4,
        help="Extra engage steps to append when stuck is detected.",
    )
    parser.add_argument(
        "--engage-max-total-steps",
        type=int,
        default=20,
        help="Hard cap for engage steps including dynamic extensions.",
    )
    parser.add_argument(
        "--engage-abort-center-distance-above-px",
        type=float,
        default=140.0,
        help="Abort engage if the target stays farther than this center distance for several steps. Set <=0 to disable.",
    )
    parser.add_argument(
        "--engage-abort-depth-increase-m",
        type=float,
        default=0.05,
        help="Abort engage if target depth increases by at least this much from engage start. Set <=0 to disable.",
    )
    parser.add_argument(
        "--engage-abort-diverge-steps",
        type=int,
        default=3,
        help="Require this many consecutive diverging engage steps before aborting engage.",
    )
    parser.add_argument(
        "--engage-on-center",
        action="store_true",
        help="Once the target is centered, switch into a forward-engage phase instead of waiting for a close action.",
    )
    parser.add_argument(
        "--engage-at-approach-end",
        action="store_true",
        help="If approach steps are exhausted without trigger, still run engage phase before close.",
    )
    parser.add_argument(
        "--no-engage-at-approach-end",
        dest="engage_at_approach_end",
        action="store_false",
        help="Disable fallback engage at the end of approach.",
    )
    parser.add_argument(
        "--continue-engage-until-interrupt",
        action="store_true",
        help="Keep extending engage instead of stopping at the engage step budget. Stop manually with Ctrl+C.",
    )
    parser.add_argument(
        "--observe-only-no-close",
        action="store_true",
        help="Debug mode: never auto-close the gripper, keep driving during engage to observe whether the arm reaches the target.",
    )
    parser.add_argument(
        "--continuous-engage-extend-steps",
        type=int,
        default=8,
        help="When continue-engage-until-interrupt is enabled, extend engage by this many steps each time the budget is exhausted.",
    )
    parser.add_argument(
        "--compact-console-log",
        action="store_true",
        help="Print concise per-step summaries instead of full verbose blocks.",
    )
    parser.add_argument(
        "--console-log-every-n-steps",
        type=int,
        default=1,
        help="When compact-console-log is enabled, print one summary every N steps.",
    )
    parser.set_defaults(engage_at_approach_end=True)
    parser.set_defaults(require_depth_ready_for_close=True)
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
            ros_node_name=f"{config['camera']['ros_node_name']}_openvla_direct_grasp",
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
            ros_node_name=f"{config['robot']['ros_node_name']}_openvla_direct_grasp",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=twist_command_duration_s,
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=twist_stop_duration_s,
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
            ctag_timeout_s=config["gripper"].get("ctag_timeout_s", 1.0),
            ctag_open_pos_mm=config["gripper"].get("ctag_open_pos_mm", 0.0),
            ctag_close_pos_mm=config["gripper"].get("ctag_close_pos_mm", 120.0),
            ctag_max_stroke_mm=config["gripper"].get("ctag_max_stroke_mm", 120.0),
            ctag_speed=config["gripper"].get("ctag_speed", 30),
            ctag_close_torque=config["gripper"].get("ctag_close_torque", 10),
            ctag_open_torque=config["gripper"].get("ctag_open_torque", 100),
            ctag_acc_dec=config["gripper"].get("ctag_acc_dec", 2000),
            ctag_parity=config["gripper"].get("ctag_parity", "N"),
            ctag_stopbits=config["gripper"].get("ctag_stopbits", 1),
            ctag_enable_rs485_mode=config["gripper"].get("ctag_enable_rs485_mode", False),
            ctag_accept_pos_reached_as_success=config["gripper"].get(
                "ctag_accept_pos_reached_as_success", True
            ),
            ctag_rs485_rts_level_for_tx=config["gripper"].get("ctag_rs485_rts_level_for_tx", True),
            ctag_rs485_rts_level_for_rx=config["gripper"].get("ctag_rs485_rts_level_for_rx", False),
            ctag_rs485_delay_before_tx=config["gripper"].get("ctag_rs485_delay_before_tx", 0.0),
            ctag_rs485_delay_before_rx=config["gripper"].get("ctag_rs485_delay_before_rx", 0.0),
            open_timeout_s=config["gripper"].get("open_timeout_s", 3.0),
            close_timeout_s=config["gripper"].get("close_timeout_s", 5.0),
        ),
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
    elif target_color == "green":
        mask = (g > 80) & (g > r + 20) & (g > b + 20)
    else:
        return None

    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    return (int(round(xs.mean())), int(round(ys.mean())))


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
    x_sign: float = -1.0,
    y_sign: float = -1.0,
) -> tuple[float, float]:
    center_x = width / 2.0
    center_y = height / 2.0
    error_x_px = pixel_xy[0] - center_x
    error_y_px = pixel_xy[1] - center_y

    if abs(error_x_px) <= deadband_px:
        error_x_px = 0.0
    if abs(error_y_px) <= deadband_px:
        error_y_px = 0.0

    x_direction = -1.0 if x_sign < 0.0 else 1.0
    y_direction = -1.0 if y_sign < 0.0 else 1.0
    return (
        x_direction * error_x_px * x_gain_m_per_px,
        y_direction * error_y_px * y_gain_m_per_px,
    )


def vector_norm(vector_xyz: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector_xyz))


def dot_product(vector_a_xyz: tuple[float, float, float], vector_b_xyz: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(vector_a_xyz, vector_b_xyz))


def rotate_ee_vector_to_base(
    ee_vector_xyz: tuple[float, float, float],
    ee_quaternion_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = ee_quaternion_xyzw
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-9:
        return ee_vector_xyz
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
        sum(rotation[row][col] * ee_vector_xyz[col] for col in range(3))
        for row in range(3)
    )


def tip_axis_base_from_state(
    tip_offset_ee_m: tuple[float, float, float],
    ee_quaternion_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    tip_axis_norm = vector_norm(tip_offset_ee_m)
    if tip_axis_norm <= 1e-9:
        tip_axis_ee = (0.0, 0.0, 1.0)
    else:
        tip_axis_ee = tuple(component / tip_axis_norm for component in tip_offset_ee_m)
    return rotate_ee_vector_to_base(tip_axis_ee, ee_quaternion_xyzw)


def sample_depth_m(
    depth_filter: DepthFilter,
    frame_depth_path_hint: str,
    target_pixel: tuple[int, int] | None,
) -> tuple[float | None, bool, tuple[int, int] | None]:
    if target_pixel is None:
        return (None, False, None)
    depth_sample = depth_filter.sample_target_depth(target_pixel, frame_depth_path_hint)
    if not depth_sample.valid:
        return (None, False, depth_sample.pixel_xy)
    return (depth_sample.depth_m, True, depth_sample.pixel_xy)


def format_optional_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


def should_print_compact_step(step_index: int, every_n_steps: int) -> bool:
    return step_index <= 1 or step_index % max(every_n_steps, 1) == 0


def main() -> None:
    args = parse_args()
    config = load_config()
    trial_logger = TrialLogger(config["logging"]["log_dir"]) if config["logging"]["enabled"] else None
    instruction = args.instruction.strip() or config["task"]["instruction"]
    lift_height_m = args.lift_height_m if args.lift_height_m is not None else config["task"]["lift_height_m"]
    max_translation_step_m = (
        args.max_translation_step_m
        if args.max_translation_step_m is not None
        else config["robot"]["max_translation_step_m"]
    )
    tip_offset_ee_m = tuple(config["gripper"].get("tip_offset_ee_m", [0.0, 0.0, 0.0]))

    camera = build_camera(config)
    robot = build_robot(config, args)
    gripper = build_gripper(config, robot)
    policy = build_policy(config)
    depth_filter = DepthFilter()
    action_adapter = ActionAdapter(
        ActionAdapterConfig(
            max_translation_step_m=max_translation_step_m,
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            workspace_xyz_min=tuple(config["robot"]["workspace_xyz_min"]),
            workspace_xyz_max=tuple(config["robot"]["workspace_xyz_max"]),
            workspace_enforced=config["robot"].get("workspace_enforced", True),
        )
    )

    print("Instruction:", instruction)
    print("Requested direct grasp steps:", args.steps)
    print("Approach min steps before engage:", args.approach_min_steps_before_engage)
    print("Target color monitor:", args.target_color)
    print("Engage trigger mode:", args.close_trigger)
    print("Center engage threshold px:", args.close_when_center_distance_below_px)
    print("Policy-close engage max center distance px:", args.policy_close_engage_max_center_distance_px)
    print("Max translation step m:", max_translation_step_m)
    print("Twist command duration s:", robot.config.twist_command_duration_s)
    print("Twist stop duration s:", robot.config.twist_stop_duration_s)
    print("Sequential axis commands:", robot.config.sequential_axis_commands)
    print("Continuous servo:", args.continuous_servo)
    print("Continuous servo hz:", args.continuous_servo_hz)
    print("Continuous servo control hz:", args.continuous_servo_control_hz)
    print("Continuous servo horizon s:", args.continuous_servo_horizon_s)
    print("Continuous servo apply s:", args.continuous_servo_apply_s)
    print("Continuous servo stale timeout s:", args.continuous_servo_stale_timeout_s)
    print("Continuous servo command alpha:", args.continuous_servo_command_alpha)
    print("Continuous servo max linear speed mps:", args.continuous_servo_max_linear_speed_mps)
    print("Continuous servo max angular speed degps:", args.continuous_servo_max_angular_speed_degps)
    print("Engage forward sign:", args.engage_forward_sign)
    print("Engage auto flip forward direction:", args.engage_auto_flip_forward_direction)
    print("Engage auto flip after steps:", args.engage_auto_flip_after_steps)
    print("Engage auto flip max depth drop m:", args.engage_auto_flip_max_depth_drop_m)
    print("Engage auto flip center worsen px:", args.engage_auto_flip_center_worsen_px)
    print("Engage auto flip max forward progress m:", args.engage_auto_flip_max_forward_progress_m)
    print("Pre-close forward step m:", args.pre_close_forward_step_m)
    print("Pre-close forward steps:", args.pre_close_forward_steps)
    print("Pre-close servo x gain:", args.pre_close_servo_x_gain_m_per_px)
    print("Pre-close servo y gain:", args.pre_close_servo_y_gain_m_per_px)
    print("Pre-close servo x sign:", args.pre_close_servo_x_sign)
    print("Pre-close servo y sign:", args.pre_close_servo_y_sign)
    print("Approach image servo:", args.approach_image_servo)
    print("Approach image servo weight:", args.approach_image_servo_weight)
    print("Approach image servo max m:", args.approach_image_servo_max_m)
    print(
        "Approach image servo min center distance px:",
        args.approach_image_servo_min_center_distance_px,
    )
    print("Approach image servo x sign:", args.approach_image_servo_x_sign)
    print("Approach image servo y sign:", args.approach_image_servo_y_sign)
    print("Approach center priority:", args.approach_center_priority)
    print("Approach center priority threshold px:", args.approach_center_priority_threshold_px)
    print(
        "Approach center priority policy xy scale:",
        args.approach_center_priority_policy_xy_scale,
    )
    print("Approach center priority z scale:", args.approach_center_priority_z_scale)
    print("Pre-close servo deadband px:", args.pre_close_servo_deadband_px)
    print(
        "Pre-close min forward steps before early close:",
        args.pre_close_min_forward_steps_before_early_close,
    )
    print("Engage min forward progress m:", args.engage_min_forward_progress_m)
    print("Engage servo max m:", args.engage_servo_max_m)
    print("Engage min forward command ratio:", args.engage_min_forward_command_ratio)
    print("Require depth ready for close:", args.require_depth_ready_for_close)
    print("Close depth threshold m:", args.close_when_depth_below_m)
    print("Close depth drop threshold m:", args.close_when_depth_drop_m)
    print("Engage min observed step m:", args.engage_min_observed_step_m)
    print("Engage max stuck steps:", args.engage_max_stuck_steps)
    print("Engage extra steps on stuck:", args.engage_extra_steps_on_stuck)
    print("Engage max total steps:", args.engage_max_total_steps)
    print("Engage abort center distance px:", args.engage_abort_center_distance_above_px)
    print("Engage abort depth increase m:", args.engage_abort_depth_increase_m)
    print("Engage abort diverge steps:", args.engage_abort_diverge_steps)
    print("Engage on center:", args.engage_on_center)
    print("Engage at approach end:", args.engage_at_approach_end)
    print("Continue engage until interrupt:", args.continue_engage_until_interrupt)
    print("Observe only no close:", args.observe_only_no_close)
    print("Continuous engage extend steps:", args.continuous_engage_extend_steps)
    print("Compact console log:", args.compact_console_log)
    print("Console log every n steps:", args.console_log_every_n_steps)

    step_records = []
    first_observation: Observation | None = None
    last_policy_action = None
    last_safe_action = None
    close_success = False
    lift_executed = False
    close_reason = ""
    engage_trigger_reason = ""
    approach_steps_executed = 0
    engage_steps_executed = 0
    engage_forward_progress_m = 0.0
    engage_forward_progress_positive_m = 0.0
    engage_depth_start_m: float | None = None
    engage_depth_latest_m: float | None = None
    engage_depth_drop_m = 0.0
    engage_stuck_steps = 0
    engage_depth_ready = not args.require_depth_ready_for_close
    engage_seed_target_pixel: tuple[int, int] | None = None
    last_known_target_pixel: tuple[int, int] | None = None
    phase = "approach"
    manual_stop_requested = False
    continuous_servo_active = False
    continuous_servo_command_thread: threading.Thread | None = None
    continuous_servo_command_running = False
    continuous_servo_command_lock = threading.Lock()
    continuous_servo_command_target = {
        "delta_xyz_m": (0.0, 0.0, 0.0),
        "delta_yaw_deg": 0.0,
    }

    try:
        remote_ok, remote_message = policy.check_remote_health()
        print("OpenVLA remote health:", remote_message)
        if not remote_ok:
            raise RuntimeError(remote_message)

        if not args.skip_initial_open:
            open_success = gripper.open()
            print("Initial gripper open success:", open_success)
            if not open_success:
                raise RuntimeError("Failed to open the gripper before direct grasp.")

        if args.continuous_servo:
            robot.start_continuous_twist_servo(
                publish_rate_hz=args.continuous_servo_hz,
                stale_timeout_s=args.continuous_servo_stale_timeout_s,
                command_alpha=args.continuous_servo_command_alpha,
                max_linear_speed_mps=args.continuous_servo_max_linear_speed_mps,
                max_angular_speed_degps=args.continuous_servo_max_angular_speed_degps,
            )
            continuous_servo_active = True
            print("Continuous servo note: background twist streaming started.")

            def set_async_servo_target(delta_xyz_m: tuple[float, float, float], delta_yaw_deg: float) -> None:
                with continuous_servo_command_lock:
                    continuous_servo_command_target["delta_xyz_m"] = delta_xyz_m
                    continuous_servo_command_target["delta_yaw_deg"] = delta_yaw_deg
                robot.set_continuous_twist_delta(
                    delta_xyz_m,
                    delta_yaw_deg,
                    horizon_s=args.continuous_servo_horizon_s,
                )

            def continuous_servo_command_loop() -> None:
                control_period_s = 1.0 / max(args.continuous_servo_control_hz, 1.0)
                while continuous_servo_command_running:
                    robot.refresh_continuous_twist_watchdog()
                    time.sleep(control_period_s)

            continuous_servo_command_running = True
            continuous_servo_command_thread = threading.Thread(
                target=continuous_servo_command_loop,
                name="openvla_direct_async_servo_command",
                daemon=True,
            )
            continuous_servo_command_thread.start()
            print("Continuous servo note: async command refresh thread started.")
        else:
            def set_async_servo_target(delta_xyz_m: tuple[float, float, float], delta_yaw_deg: float) -> None:
                return None

        def execute_safe_action(safe_action) -> None:
            if args.continuous_servo:
                set_async_servo_target(
                    safe_action.delta_xyz_m,
                    safe_action.delta_yaw_deg,
                )
                if args.continuous_servo_apply_s > 0.0:
                    time.sleep(args.continuous_servo_apply_s)
            else:
                robot.move_cartesian_delta(safe_action.delta_xyz_m, safe_action.delta_yaw_deg)

        center_ready_steps = 0
        for step_idx in range(max(args.steps, 0)):
            before_state = robot.get_state()
            frame = camera.capture_frame()
            observation = Observation(instruction=instruction, frame=frame, robot_state=before_state)
            if first_observation is None:
                first_observation = observation

            policy_action = policy.predict_action(observation)
            detected_centroid = None
            center_distance = None
            if args.target_color != "none":
                detected_centroid = detect_ball_centroid(frame.rgb_path_hint, args.target_color)
                if detected_centroid is not None:
                    center_distance = center_distance_px(detected_centroid, frame.width, frame.height)

            if detected_centroid is not None and args.use_detected_centroid_as_target_pixel:
                policy_action = PolicyAction(
                    delta_xyz_m=policy_action.delta_xyz_m,
                    delta_yaw_deg=policy_action.delta_yaw_deg,
                    gripper_command=policy_action.gripper_command,
                    confidence=policy_action.confidence,
                    target_pixel=detected_centroid,
                    notes=policy_action.notes,
                    metadata=policy_action.metadata,
                )

            depth_target_pixel = detected_centroid if detected_centroid is not None else policy_action.target_pixel
            if depth_target_pixel is not None:
                last_known_target_pixel = depth_target_pixel
            approach_depth_m, approach_depth_valid, approach_depth_pixel = sample_depth_m(
                depth_filter,
                frame.depth_path_hint,
                depth_target_pixel,
            )
            approach_depth_ready = False
            if approach_depth_valid and approach_depth_m is not None:
                approach_depth_ready = approach_depth_m <= max(args.close_when_depth_below_m, 0.0)

            policy_requests_close = policy_action.gripper_command == "close"
            center_requests_engage = (
                args.close_trigger == "policy_or_center"
                and center_distance is not None
                and center_distance <= args.close_when_center_distance_below_px
            )
            center_ready_steps = center_ready_steps + 1 if center_requests_engage else 0
            center_ready = center_ready_steps >= max(args.center_stable_steps, 1)
            center_can_trigger_engage = args.engage_on_center and center_ready
            policy_center_ready_for_engage = (
                args.policy_close_engage_max_center_distance_px <= 0.0
                or center_distance is None
                or center_distance <= args.policy_close_engage_max_center_distance_px
            )
            policy_can_trigger_engage = (
                policy_requests_close
                and approach_steps_executed >= max(args.approach_min_steps_before_engage, 0)
                and policy_center_ready_for_engage
            )
            policy_close_waiting = policy_requests_close and not policy_can_trigger_engage

            approach_step_index = step_idx + 1
            compact_approach_step = (
                args.compact_console_log
                and should_print_compact_step(approach_step_index, args.console_log_every_n_steps)
            )
            if args.compact_console_log:
                if compact_approach_step:
                    print(
                        "[approach"
                        f" {approach_step_index}]"
                        f" center_px={format_optional_float(center_distance, 1)}"
                        f" depth_m={format_optional_float(approach_depth_m, 3)}"
                        f" close_req={policy_requests_close}"
                        f" policy_gate={policy_can_trigger_engage}"
                        f" center_gate={center_can_trigger_engage}"
                        f" waiting={policy_close_waiting}"
                        f" action=({policy_action.delta_xyz_m[0]:+.3f},{policy_action.delta_xyz_m[1]:+.3f},{policy_action.delta_xyz_m[2]:+.3f})"
                    )
            else:
                print(f"Approach step {approach_step_index}")
                print("  Phase:", phase)
                print("  RGB path:", frame.rgb_path_hint)
                print("  Before ee_position_m:", before_state.ee_position_m)
                print("  Policy target_pixel:", policy_action.target_pixel)
                print("  Detected centroid:", detected_centroid)
                print("  Center distance px:", center_distance)
                print("  Policy delta_xyz_m:", policy_action.delta_xyz_m)
                print("  Policy delta_yaw_deg:", policy_action.delta_yaw_deg)
                print("  Policy gripper_command:", policy_action.gripper_command)
                print("  Policy requests close:", policy_requests_close)
                print("  Center requests engage:", center_requests_engage)
                print("  Consecutive centered steps:", center_ready_steps)
                print("  Center can trigger engage:", center_can_trigger_engage)
                print("  Policy center ready for engage:", policy_center_ready_for_engage)
                print("  Policy can trigger engage:", policy_can_trigger_engage)
                print("  Policy close waiting:", policy_close_waiting)
                print("  Depth target pixel:", approach_depth_pixel)
                print("  Depth m:", approach_depth_m)
                print("  Depth ready:", approach_depth_ready)

            if policy_requests_close and args.close_on_policy_step and not policy_can_trigger_engage:
                print(
                    "  Close note: policy asked close but approach-min-steps gate is not met, deferring close."
                )
            if (
                policy_requests_close
                and args.close_on_policy_step
                and policy_can_trigger_engage
                and (not args.require_depth_ready_for_close or approach_depth_ready)
            ):
                print("  Close note: close_on_policy_step enabled, skipping engage phase.")
                if args.continuous_servo:
                    set_async_servo_target((0.0, 0.0, 0.0), 0.0)
                last_policy_action = policy_action
                last_safe_action = action_adapter.adapt(policy_action, before_state)
                close_reason = "policy_close_immediate"
                engage_trigger_reason = "policy_close_immediate"
                phase = "close"
                break

            if center_can_trigger_engage or policy_can_trigger_engage:
                engage_trigger_reason = "center_ready" if center_can_trigger_engage else "policy_close"
                print("  Engage note: transition to engage phase, no extra approach move on this step.")
                engage_stuck_steps = 0
                engage_seed_target_pixel = depth_target_pixel
                if approach_depth_valid and approach_depth_m is not None:
                    engage_depth_start_m = approach_depth_m
                    engage_depth_latest_m = approach_depth_m
                    engage_depth_drop_m = 0.0
                    engage_depth_ready = approach_depth_ready
                else:
                    engage_depth_start_m = None
                    engage_depth_latest_m = None
                    engage_depth_drop_m = 0.0
                    engage_depth_ready = not args.require_depth_ready_for_close
                last_policy_action = policy_action
                last_safe_action = action_adapter.adapt(policy_action, before_state)
                phase = "engage"
                break

            approach_servo_delta_xy = (0.0, 0.0)
            approach_servo_active = False
            approach_center_priority_active = False
            effective_policy_action = policy_action
            if (
                args.approach_image_servo
                and detected_centroid is not None
                and center_distance is not None
                and center_distance >= max(args.approach_image_servo_min_center_distance_px, 0.0)
            ):
                approach_servo_delta_xy = image_servo_planar_delta(
                    detected_centroid,
                    frame.width,
                    frame.height,
                    x_gain_m_per_px=args.pre_close_servo_x_gain_m_per_px,
                    y_gain_m_per_px=args.pre_close_servo_y_gain_m_per_px,
                    deadband_px=args.pre_close_servo_deadband_px,
                    x_sign=args.approach_image_servo_x_sign,
                    y_sign=args.approach_image_servo_y_sign,
                )
                approach_servo_norm = math.hypot(approach_servo_delta_xy[0], approach_servo_delta_xy[1])
                if approach_servo_norm > max(args.approach_image_servo_max_m, 0.0) > 0.0:
                    approach_servo_scale = args.approach_image_servo_max_m / max(approach_servo_norm, 1e-9)
                    approach_servo_delta_xy = (
                        approach_servo_delta_xy[0] * approach_servo_scale,
                        approach_servo_delta_xy[1] * approach_servo_scale,
                    )
                if (
                    args.approach_center_priority
                    and center_distance >= max(args.approach_center_priority_threshold_px, 0.0)
                ):
                    approach_center_priority_active = True
                    policy_xy_scale = max(0.0, min(args.approach_center_priority_policy_xy_scale, 1.0))
                    policy_z_scale = max(0.0, min(args.approach_center_priority_z_scale, 1.0))
                    blended_delta_xyz = (
                        approach_servo_delta_xy[0] + policy_xy_scale * policy_action.delta_xyz_m[0],
                        approach_servo_delta_xy[1] + policy_xy_scale * policy_action.delta_xyz_m[1],
                        policy_z_scale * policy_action.delta_xyz_m[2],
                    )
                else:
                    blend_weight = max(0.0, min(args.approach_image_servo_weight, 1.0))
                    blended_delta_xyz = (
                        (1.0 - blend_weight) * policy_action.delta_xyz_m[0]
                        + blend_weight * approach_servo_delta_xy[0],
                        (1.0 - blend_weight) * policy_action.delta_xyz_m[1]
                        + blend_weight * approach_servo_delta_xy[1],
                        policy_action.delta_xyz_m[2],
                    )
                effective_policy_action = PolicyAction(
                    delta_xyz_m=blended_delta_xyz,
                    delta_yaw_deg=policy_action.delta_yaw_deg,
                    gripper_command=policy_action.gripper_command,
                    confidence=policy_action.confidence,
                    target_pixel=policy_action.target_pixel,
                    notes=policy_action.notes,
                    metadata={
                        **(policy_action.metadata or {}),
                        "approach_image_servo_active": True,
                        "approach_image_servo_delta_xy": approach_servo_delta_xy,
                        "approach_image_servo_weight": max(0.0, min(args.approach_image_servo_weight, 1.0)),
                        "approach_center_priority_active": approach_center_priority_active,
                    },
                )
                approach_servo_active = True

            safe_action = action_adapter.adapt(effective_policy_action, before_state)
            if args.compact_console_log:
                if compact_approach_step:
                    print(
                        "  safe="
                        f"({safe_action.delta_xyz_m[0]:+.3f},{safe_action.delta_xyz_m[1]:+.3f},{safe_action.delta_xyz_m[2]:+.3f})"
                        f" clipped={safe_action.clipped}"
                    )
                    if approach_servo_active:
                        print(
                            "  approach_servo="
                            f"({approach_servo_delta_xy[0]:+.3f},{approach_servo_delta_xy[1]:+.3f})"
                            f" priority={approach_center_priority_active}"
                        )
                if safe_action.rejection_reason:
                    print("  Safe action note:", safe_action.rejection_reason)
            else:
                print("  Effective policy delta_xyz_m:", effective_policy_action.delta_xyz_m)
                print("  Approach image servo active:", approach_servo_active)
                if approach_servo_active:
                    print("  Approach image servo delta_xy:", approach_servo_delta_xy)
                    print("  Approach center priority active:", approach_center_priority_active)
                print("  Safe delta_xyz_m:", safe_action.delta_xyz_m)
                print("  Safe delta_yaw_deg:", safe_action.delta_yaw_deg)
                print("  Safe clipped:", safe_action.clipped)
                if safe_action.rejection_reason:
                    print("  Safe action note:", safe_action.rejection_reason)

            execute_safe_action(safe_action)
            after_state = robot.get_state()
            observed_delta = tuple(
                after - before for after, before in zip(after_state.ee_position_m, before_state.ee_position_m)
            )
            observed_yaw_delta = after_state.ee_yaw_deg - before_state.ee_yaw_deg

            if args.compact_console_log:
                if compact_approach_step:
                    observed_step_norm = vector_norm(observed_delta)
                    print(
                        "  observed="
                        f"({observed_delta[0]:+.4f},{observed_delta[1]:+.4f},{observed_delta[2]:+.4f})"
                        f" step_norm={observed_step_norm:.4f}"
                    )
            else:
                print("  After ee_position_m:", after_state.ee_position_m)
                print("  Observed ee delta:", observed_delta)
                print("  Observed yaw delta:", observed_yaw_delta)

            approach_steps_executed += 1
            step_records.append(
                {
                    "step_index": len(step_records) + 1,
                    "phase": "approach",
                    "phase_step_index": approach_steps_executed,
                    "rgb_path_hint": frame.rgb_path_hint,
                    "before_ee_position_m": before_state.ee_position_m,
                    "after_ee_position_m": after_state.ee_position_m,
                    "policy_target_pixel": policy_action.target_pixel,
                    "detected_centroid": detected_centroid,
                    "center_distance_px": center_distance,
                    "policy_delta_xyz_m": policy_action.delta_xyz_m,
                    "effective_policy_delta_xyz_m": effective_policy_action.delta_xyz_m,
                    "policy_delta_yaw_deg": policy_action.delta_yaw_deg,
                    "policy_gripper_command": policy_action.gripper_command,
                    "approach_image_servo_active": approach_servo_active,
                    "approach_image_servo_delta_xy": approach_servo_delta_xy,
                    "approach_center_priority_active": approach_center_priority_active,
                    "safe_delta_xyz_m": safe_action.delta_xyz_m,
                    "safe_delta_yaw_deg": safe_action.delta_yaw_deg,
                    "safe_action_clipped": safe_action.clipped,
                    "observed_ee_delta": observed_delta,
                    "observed_yaw_delta": observed_yaw_delta,
                    "policy_requests_close": policy_requests_close,
                    "policy_close_waiting": policy_close_waiting,
                    "center_requests_engage": center_requests_engage,
                    "center_ready_steps": center_ready_steps,
                    "policy_center_ready_for_engage": policy_center_ready_for_engage,
                    "depth_target_pixel": approach_depth_pixel,
                    "depth_m": approach_depth_m,
                    "depth_valid": approach_depth_valid,
                    "depth_ready": approach_depth_ready,
                    "engage_trigger_reason": engage_trigger_reason,
                    "policy_metadata": policy_action.metadata,
                }
            )
            last_policy_action = policy_action
            last_safe_action = safe_action
        else:
            if (
                args.engage_at_approach_end
                and args.pre_close_forward_steps > 0
                and args.pre_close_forward_step_m > 0.0
            ):
                phase = "engage"
                engage_trigger_reason = "approach_exhausted"
                engage_seed_target_pixel = last_known_target_pixel
                print("Approach note: steps exhausted, entering engage fallback phase.")
            else:
                phase = "done_without_close"

        if phase == "engage":
            engage_center_ready_steps = 0
            engage_diverging_steps = 0
            engage_initial_center_distance: float | None = None
            engage_best_center_distance: float | None = None
            engage_forward_flip_count = 0
            engage_steps_budget = max(args.pre_close_forward_steps, 0)
            if args.pre_close_forward_step_m > 1e-9:
                min_steps_for_progress = int(
                    math.ceil(max(args.engage_min_forward_progress_m, 0.0) / args.pre_close_forward_step_m)
                )
                engage_steps_budget = max(engage_steps_budget, min_steps_for_progress)
            engage_hard_cap: int | None = None
            if not args.continue_engage_until_interrupt:
                engage_hard_cap = max(args.engage_max_total_steps, 0)
                engage_steps_budget = min(engage_steps_budget, engage_hard_cap)
            print("Engage steps budget:", engage_steps_budget)
            if args.continue_engage_until_interrupt:
                print("Engage note: continue-engage-until-interrupt enabled, engage budget will auto-extend.")
            if args.observe_only_no_close:
                print(
                    "Engage note: observe-only-no-close enabled, close readiness will be logged but not executed."
                )

            engage_idx = 0
            forward_progress_ready = False
            engage_forward_sign = 1.0 if args.engage_forward_sign >= 0.0 else -1.0
            while engage_idx < engage_steps_budget:
                close_prep_state = robot.get_state()
                tip_axis_base = tip_axis_base_from_state(
                    tip_offset_ee_m,
                    close_prep_state.ee_quaternion_xyzw,
                )
                engage_axis_base = tuple(engage_forward_sign * component for component in tip_axis_base)
                servo_frame = camera.capture_frame()
                servo_centroid = None
                servo_center_distance = None
                servo_delta_xy = (0.0, 0.0)
                if args.target_color != "none":
                    servo_centroid = detect_ball_centroid(servo_frame.rgb_path_hint, args.target_color)
                    if servo_centroid is not None:
                        engage_seed_target_pixel = servo_centroid
                        servo_center_distance = center_distance_px(
                            servo_centroid,
                            servo_frame.width,
                            servo_frame.height,
                        )
                        if engage_initial_center_distance is None:
                            engage_initial_center_distance = servo_center_distance
                            engage_best_center_distance = servo_center_distance
                        elif engage_best_center_distance is None:
                            engage_best_center_distance = servo_center_distance
                        else:
                            engage_best_center_distance = min(
                                engage_best_center_distance,
                                servo_center_distance,
                            )
                        servo_delta_xy = image_servo_planar_delta(
                            servo_centroid,
                            servo_frame.width,
                            servo_frame.height,
                            x_gain_m_per_px=args.pre_close_servo_x_gain_m_per_px,
                            y_gain_m_per_px=args.pre_close_servo_y_gain_m_per_px,
                            deadband_px=args.pre_close_servo_deadband_px,
                            x_sign=args.pre_close_servo_x_sign,
                            y_sign=args.pre_close_servo_y_sign,
                        )
                engage_depth_m, engage_depth_valid, engage_depth_pixel = sample_depth_m(
                    depth_filter,
                    servo_frame.depth_path_hint,
                    servo_centroid if servo_centroid is not None else engage_seed_target_pixel,
                )
                if engage_depth_valid and engage_depth_m is not None:
                    if engage_depth_start_m is None:
                        engage_depth_start_m = engage_depth_m
                    engage_depth_latest_m = engage_depth_m
                    engage_depth_drop_m = max(0.0, engage_depth_start_m - engage_depth_m)
                engage_depth_increase_m = 0.0
                if engage_depth_start_m is not None and engage_depth_latest_m is not None:
                    engage_depth_increase_m = max(0.0, engage_depth_latest_m - engage_depth_start_m)

                servo_norm = math.hypot(servo_delta_xy[0], servo_delta_xy[1])
                if servo_norm > max(args.engage_servo_max_m, 0.0) > 0.0:
                    servo_scale = args.engage_servo_max_m / max(servo_norm, 1e-9)
                    servo_delta_xy = (
                        servo_delta_xy[0] * servo_scale,
                        servo_delta_xy[1] * servo_scale,
                    )

                center_requests_engage = (
                    args.close_trigger == "policy_or_center"
                    and servo_center_distance is not None
                    and servo_center_distance <= args.close_when_center_distance_below_px
                )
                engage_center_ready_steps = (
                    engage_center_ready_steps + 1 if center_requests_engage else 0
                )
                engage_center_ready = engage_center_ready_steps >= max(args.center_stable_steps, 1)

                depth_ready_step = not args.require_depth_ready_for_close
                if args.require_depth_ready_for_close:
                    if engage_depth_valid and engage_depth_m is not None:
                        depth_ready_step = (
                            engage_depth_m <= max(args.close_when_depth_below_m, 0.0)
                            or engage_depth_drop_m >= max(args.close_when_depth_drop_m, 0.0)
                        )
                    elif engage_depth_latest_m is not None:
                        depth_ready_step = (
                            engage_depth_latest_m <= max(args.close_when_depth_below_m, 0.0)
                            or engage_depth_drop_m >= max(args.close_when_depth_drop_m, 0.0)
                        )
                    else:
                        depth_ready_step = False
                engage_depth_ready = depth_ready_step

                forward_delta_xyz = tuple(
                    args.pre_close_forward_step_m * axis_component
                    for axis_component in engage_axis_base
                )
                combined_delta_xyz = (
                    forward_delta_xyz[0] + servo_delta_xy[0],
                    forward_delta_xyz[1] + servo_delta_xy[1],
                    forward_delta_xyz[2],
                )
                forward_command_projection = dot_product(combined_delta_xyz, engage_axis_base)
                min_forward_command_m = (
                    max(args.pre_close_forward_step_m, 0.0)
                    * max(min(args.engage_min_forward_command_ratio, 1.0), 0.0)
                )
                if forward_command_projection < min_forward_command_m:
                    servo_projection = dot_product((servo_delta_xy[0], servo_delta_xy[1], 0.0), engage_axis_base)
                    min_servo_projection = min_forward_command_m - max(args.pre_close_forward_step_m, 0.0)
                    if servo_projection < min_servo_projection and abs(servo_projection) > 1e-9:
                        keep_ratio = min_servo_projection / servo_projection
                        keep_ratio = max(0.0, min(1.0, keep_ratio))
                        servo_delta_xy = (
                            servo_delta_xy[0] * keep_ratio,
                            servo_delta_xy[1] * keep_ratio,
                        )
                        combined_delta_xyz = (
                            forward_delta_xyz[0] + servo_delta_xy[0],
                            forward_delta_xyz[1] + servo_delta_xy[1],
                            forward_delta_xyz[2],
                        )
                        forward_command_projection = dot_product(combined_delta_xyz, engage_axis_base)
                engage_action = PolicyAction(
                    delta_xyz_m=combined_delta_xyz,
                    delta_yaw_deg=0.0,
                    gripper_command="open",
                    confidence=1.0,
                    target_pixel=None,
                    notes="Direct grasp engage phase forward motion",
                    metadata={
                        "control_mode": "engage",
                        "engage_trigger_reason": engage_trigger_reason,
                        "engage_step_index": engage_idx + 1,
                        "engage_steps_total": engage_steps_budget,
                        "engage_servo_delta_xy": servo_delta_xy,
                        "engage_forward_delta_xyz": forward_delta_xyz,
                        "engage_forward_command_projection_m": forward_command_projection,
                        "engage_forward_sign": engage_forward_sign,
                        "engage_center_ready": engage_center_ready,
                        "engage_depth_ready": engage_depth_ready,
                        "engage_depth_m": engage_depth_m,
                        "engage_depth_drop_m": engage_depth_drop_m,
                        "tip_axis_base": tip_axis_base,
                        "engage_axis_base": engage_axis_base,
                    },
                )
                safe_engage_action = action_adapter.adapt(engage_action, close_prep_state)
                engage_step_index = engage_idx + 1
                compact_engage_step = (
                    args.compact_console_log
                    and should_print_compact_step(engage_step_index, args.console_log_every_n_steps)
                )
                if args.compact_console_log:
                    if compact_engage_step:
                        print(
                            "[engage"
                            f" {engage_step_index}]"
                            f" trigger={engage_trigger_reason}"
                            f" sign={engage_forward_sign:+.0f}"
                            f" center_px={format_optional_float(servo_center_distance, 1)}"
                            f" depth_m={format_optional_float(engage_depth_m, 3)}"
                            f" drop_m={engage_depth_drop_m:.3f}"
                            f" safe=({safe_engage_action.delta_xyz_m[0]:+.3f},{safe_engage_action.delta_xyz_m[1]:+.3f},{safe_engage_action.delta_xyz_m[2]:+.3f})"
                            f" clipped={safe_engage_action.clipped}"
                        )
                else:
                    print(f"Engage step {engage_step_index}")
                    print("  Phase:", phase)
                    print("  Engage trigger reason:", engage_trigger_reason)
                    print("  Servo RGB path:", servo_frame.rgb_path_hint)
                    print("  Servo centroid:", servo_centroid)
                    print("  Servo center distance px:", servo_center_distance)
                    print("  Servo center requests engage:", center_requests_engage)
                    print("  Servo centered steps:", engage_center_ready_steps)
                    print("  Servo delta xy:", servo_delta_xy)
                    print("  Depth target pixel:", engage_depth_pixel)
                    print("  Depth m:", engage_depth_m)
                    print("  Depth drop m:", engage_depth_drop_m)
                    print("  Depth ready:", engage_depth_ready)
                    print("  Engage forward sign:", engage_forward_sign)
                    print("  Tip axis base:", tip_axis_base)
                    print("  Engage axis base:", engage_axis_base)
                    print("  Forward delta xyz:", forward_delta_xyz)
                    print("  Servo+forward delta xyz:", combined_delta_xyz)
                    print("  Forward command projection m:", forward_command_projection)
                    print("  Safe engage delta xyz:", safe_engage_action.delta_xyz_m)
                    print("  Safe engage clipped:", safe_engage_action.clipped)
                    if safe_engage_action.rejection_reason:
                        print("  Safe engage note:", safe_engage_action.rejection_reason)

                execute_safe_action(safe_engage_action)
                after_state = robot.get_state()
                observed_delta = tuple(
                    after - before
                    for after, before in zip(after_state.ee_position_m, close_prep_state.ee_position_m)
                )
                observed_yaw_delta = after_state.ee_yaw_deg - close_prep_state.ee_yaw_deg
                observed_forward_progress_step = dot_product(observed_delta, engage_axis_base)
                engage_forward_progress_m += observed_forward_progress_step
                if observed_forward_progress_step > 0.0:
                    engage_forward_progress_positive_m += observed_forward_progress_step
                forward_progress_ready = (
                    engage_forward_progress_positive_m >= max(args.engage_min_forward_progress_m, 0.0)
                )
                observed_step_norm = vector_norm(observed_delta)
                if observed_step_norm < max(args.engage_min_observed_step_m, 0.0):
                    engage_stuck_steps += 1
                else:
                    engage_stuck_steps = 0
                close_ready = (
                    engage_idx + 1 >= max(args.pre_close_min_forward_steps_before_early_close, 0)
                    and forward_progress_ready
                    and engage_depth_ready
                )
                center_diverging = (
                    args.engage_abort_center_distance_above_px > 0.0
                    and servo_center_distance is not None
                    and servo_center_distance >= args.engage_abort_center_distance_above_px
                )
                depth_diverging = (
                    args.engage_abort_depth_increase_m > 0.0
                    and engage_depth_increase_m >= args.engage_abort_depth_increase_m
                )
                if center_diverging or depth_diverging:
                    engage_diverging_steps += 1
                else:
                    engage_diverging_steps = 0
                if args.compact_console_log:
                    if compact_engage_step:
                        print(
                            "  observed="
                            f"({observed_delta[0]:+.4f},{observed_delta[1]:+.4f},{observed_delta[2]:+.4f})"
                            f" step_norm={observed_step_norm:.4f}"
                            f" fwd+={engage_forward_progress_positive_m:.4f}"
                            f" close_ready={close_ready}"
                            f" stuck={engage_stuck_steps}"
                            f" diverge={engage_diverging_steps}"
                        )
                    if safe_engage_action.rejection_reason:
                        print("  Safe engage note:", safe_engage_action.rejection_reason)
                else:
                    print("  After ee_position_m:", after_state.ee_position_m)
                    print("  Observed ee delta:", observed_delta)
                    print("  Observed yaw delta:", observed_yaw_delta)
                    print("  Observed step norm m:", observed_step_norm)
                    print("  Engage stuck steps:", engage_stuck_steps)
                    print("  Observed forward progress step m:", observed_forward_progress_step)
                    print("  Observed forward progress cumulative m:", engage_forward_progress_m)
                    print(
                        "  Observed forward progress cumulative positive m:",
                        engage_forward_progress_positive_m,
                    )
                    print("  Forward progress ready:", forward_progress_ready)
                    print("  Close ready:", close_ready)
                    print("  Depth increase m:", engage_depth_increase_m)
                    print("  Center diverging:", center_diverging)
                    print("  Depth diverging:", depth_diverging)
                    print("  Engage diverging steps:", engage_diverging_steps)

                engage_steps_executed += 1
                step_records.append(
                    {
                        "step_index": len(step_records) + 1,
                        "phase": "engage",
                        "phase_step_index": engage_steps_executed,
                        "rgb_path_hint": servo_frame.rgb_path_hint,
                        "before_ee_position_m": close_prep_state.ee_position_m,
                        "after_ee_position_m": after_state.ee_position_m,
                        "policy_target_pixel": None,
                        "detected_centroid": servo_centroid,
                        "center_distance_px": servo_center_distance,
                        "policy_delta_xyz_m": engage_action.delta_xyz_m,
                        "policy_delta_yaw_deg": engage_action.delta_yaw_deg,
                        "policy_gripper_command": engage_action.gripper_command,
                        "safe_delta_xyz_m": safe_engage_action.delta_xyz_m,
                        "safe_delta_yaw_deg": safe_engage_action.delta_yaw_deg,
                        "safe_action_clipped": safe_engage_action.clipped,
                        "observed_ee_delta": observed_delta,
                        "observed_yaw_delta": observed_yaw_delta,
                        "observed_step_norm_m": observed_step_norm,
                        "engage_stuck_steps": engage_stuck_steps,
                        "observed_forward_progress_step_m": observed_forward_progress_step,
                        "observed_forward_progress_cumulative_m": engage_forward_progress_m,
                        "observed_forward_progress_cumulative_positive_m": engage_forward_progress_positive_m,
                        "policy_requests_close": False,
                        "center_requests_engage": center_requests_engage,
                        "center_ready_steps": engage_center_ready_steps,
                        "depth_target_pixel": engage_depth_pixel,
                        "depth_m": engage_depth_m,
                        "depth_valid": engage_depth_valid,
                        "depth_ready": engage_depth_ready,
                        "depth_drop_m": engage_depth_drop_m,
                        "depth_increase_m": engage_depth_increase_m,
                        "close_ready": close_ready,
                        "center_diverging": center_diverging,
                        "depth_diverging": depth_diverging,
                        "engage_diverging_steps": engage_diverging_steps,
                        "engage_trigger_reason": engage_trigger_reason,
                        "policy_metadata": engage_action.metadata,
                    }
                )
                last_policy_action = engage_action
                last_safe_action = safe_engage_action
                center_worsen_px = 0.0
                if (
                    servo_center_distance is not None
                    and engage_best_center_distance is not None
                ):
                    center_worsen_px = max(0.0, servo_center_distance - engage_best_center_distance)
                auto_flip_requested = (
                    args.engage_auto_flip_forward_direction
                    and engage_forward_flip_count == 0
                    and not close_ready
                    and engage_idx + 1 >= max(args.engage_auto_flip_after_steps, 1)
                    and (
                        (
                            engage_depth_drop_m <= max(args.engage_auto_flip_max_depth_drop_m, 0.0)
                            and center_worsen_px >= max(args.engage_auto_flip_center_worsen_px, 0.0)
                        )
                        or (
                            engage_forward_progress_positive_m
                            <= max(args.engage_auto_flip_max_forward_progress_m, 0.0)
                        )
                    )
                )
                if auto_flip_requested:
                    engage_forward_sign *= -1.0
                    engage_forward_flip_count += 1
                    engage_forward_progress_m = 0.0
                    engage_forward_progress_positive_m = 0.0
                    forward_progress_ready = False
                    engage_stuck_steps = 0
                    engage_diverging_steps = 0
                    engage_depth_start_m = engage_depth_latest_m
                    engage_depth_drop_m = 0.0
                    print(
                        "  Engage note: depth/centering indicate wrong forward direction, flipping engage forward sign."
                    )
                    if args.compact_console_log:
                        print(
                            "  engage_auto_flip:"
                            f" sign={engage_forward_sign:+.0f}"
                            f" center_worsen_px={center_worsen_px:.1f}"
                            f" depth_drop_m={engage_depth_drop_m:.3f}"
                            f" fwd+={engage_forward_progress_positive_m:.4f}"
                        )
                    engage_idx += 1
                    continue
                if (
                    engage_diverging_steps >= max(args.engage_abort_diverge_steps, 1)
                    and not close_ready
                ):
                    phase = "done_without_close"
                    close_reason = "engage_diverged_center_or_depth"
                    print(
                        "  Engage note: target is diverging during engage, aborting engage to avoid driving away."
                    )
                    break
                if close_ready:
                    if args.observe_only_no_close:
                        close_reason = f"observe_only_ready_step_{engage_idx + 1}"
                        print(
                            "  Engage note: forward/depth readiness reached, but observe-only-no-close keeps driving."
                        )
                    else:
                        close_reason = f"engage_ready_step_{engage_idx + 1}"
                        print("  Engage note: forward/depth readiness reached, closing.")
                        break
                if engage_center_ready and not close_ready:
                    print(
                        "  Engage note: center is ready but forward/depth readiness is still insufficient, continue engage."
                    )
                if (
                    engage_stuck_steps >= max(args.engage_max_stuck_steps, 1)
                    and engage_hard_cap is not None
                    and engage_steps_budget < engage_hard_cap
                ):
                    old_budget = engage_steps_budget
                    engage_steps_budget = min(
                        engage_hard_cap,
                        engage_steps_budget + max(args.engage_extra_steps_on_stuck, 0),
                    )
                    engage_stuck_steps = 0
                    if engage_steps_budget > old_budget:
                        print(
                            f"  Engage note: detected ineffective motion streak, extending step budget {old_budget} -> {engage_steps_budget}."
                        )
                if (
                    args.continue_engage_until_interrupt
                    and engage_idx + 1 >= engage_steps_budget
                    and (args.observe_only_no_close or not close_ready)
                ):
                    extend_steps = max(args.continuous_engage_extend_steps, 1)
                    old_budget = engage_steps_budget
                    engage_steps_budget += extend_steps
                    print(
                        f"  Engage note: budget exhausted without stop condition, extending engage budget {old_budget} -> {engage_steps_budget}."
                    )
                engage_idx += 1

            if args.observe_only_no_close:
                phase = "observe_drive"
                if not close_reason:
                    close_reason = "observe_only_engage_continued"
            elif not close_reason and forward_progress_ready and engage_depth_ready:
                close_reason = (
                    "engage_complete_ready"
                    f"_progress_{engage_forward_progress_positive_m:.4f}m"
                    f"_depth_drop_{engage_depth_drop_m:.4f}m"
                )
            elif not close_reason:
                phase = "done_without_close"
                close_reason = "engage_not_ready_for_close"
                print(
                    "Engage note: engage finished without forward/depth readiness, skipping close to avoid planar-only grasp."
                )
            if args.observe_only_no_close:
                print("Observe note: engage loop ended in observe-only mode, gripper close remains disabled.")
            elif phase != "done_without_close":
                phase = "close"

        if phase == "close":
            if args.continuous_servo and continuous_servo_active:
                set_async_servo_target((0.0, 0.0, 0.0), 0.0)
                continuous_servo_command_running = False
                if continuous_servo_command_thread is not None:
                    continuous_servo_command_thread.join(timeout=1.0)
                    continuous_servo_command_thread = None
                    print("Continuous servo note: async command refresh thread stopped.")
                robot.stop_continuous_twist_servo(stop_duration_s=0.2)
                continuous_servo_active = False
                print("Continuous servo note: stopped before gripper close.")
            if args.settle_before_close_s > 0.0:
                time.sleep(args.settle_before_close_s)
            close_success = gripper.close()
            print("Gripper close success:", close_success)
            print("Close reason:", close_reason)
            print("Engage trigger reason:", engage_trigger_reason)
            if close_success:
                robot.move_cartesian_delta((0.0, 0.0, lift_height_m), 0.0)
                lift_executed = True
                print("Lift executed height m:", lift_height_m)

        final_state = robot.get_state()
        failure_reason = ""
        if not close_success:
            failure_reason = (
                "Direct grasp did not complete close phase successfully "
                f"(phase={phase}, engage_trigger={engage_trigger_reason or 'none'}, close_reason={close_reason or 'none'})"
            )
        result = ExecutionResult(
            success=bool(close_success),
            state_trace=[f"openvla_direct_step_{idx + 1}" for idx in range(len(step_records))],
            message="OpenVLA direct grasp run completed",
            failure_reason=failure_reason,
            grasp=RefinedGrasp(
                target_xyz_m=final_state.ee_position_m,
                target_yaw_deg=final_state.ee_yaw_deg,
                grasp_width_m=final_state.gripper_opening_m,
                quality=1.0 if close_success else 0.0,
                source="openvla_direct_grasp",
            ),
        )

        print("Final ee_position_m:", final_state.ee_position_m)
        print("Final phase:", phase)
        print("Approach steps executed:", approach_steps_executed)
        print("Engage steps executed:", engage_steps_executed)
        print("Engage trigger reason:", engage_trigger_reason)
        print("Engage observed forward progress m:", engage_forward_progress_m)
        print("Engage observed forward progress positive m:", engage_forward_progress_positive_m)
        print("Engage depth start m:", engage_depth_start_m)
        print("Engage depth latest m:", engage_depth_latest_m)
        print("Engage depth drop m:", engage_depth_drop_m)
        print("Engage depth ready:", engage_depth_ready)
        print("Close reason:", close_reason)
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
        ):
            trial_logger.log_trial(
                instruction=instruction,
                observation=first_observation,
                policy_action=last_policy_action,
                safe_action=last_safe_action,
                refined_grasp=result.grasp,
                result=result,
                final_robot_state=final_state,
                metadata={
                    "test_type": "openvla_direct_grasp",
                    "requested_steps": args.steps,
                    "approach_min_steps_before_engage": args.approach_min_steps_before_engage,
                    "target_color": args.target_color,
                    "close_trigger": args.close_trigger,
                    "close_when_center_distance_below_px": args.close_when_center_distance_below_px,
                    "policy_close_engage_max_center_distance_px": args.policy_close_engage_max_center_distance_px,
                    "close_when_depth_below_m": args.close_when_depth_below_m,
                    "close_when_depth_drop_m": args.close_when_depth_drop_m,
                    "require_depth_ready_for_close": args.require_depth_ready_for_close,
                    "center_stable_steps": args.center_stable_steps,
                    "close_on_policy_step": args.close_on_policy_step,
                    "engage_forward_sign": 1.0 if args.engage_forward_sign >= 0.0 else -1.0,
                    "engage_auto_flip_forward_direction": args.engage_auto_flip_forward_direction,
                    "engage_auto_flip_after_steps": args.engage_auto_flip_after_steps,
                    "engage_auto_flip_max_depth_drop_m": args.engage_auto_flip_max_depth_drop_m,
                    "engage_auto_flip_center_worsen_px": args.engage_auto_flip_center_worsen_px,
                    "engage_auto_flip_max_forward_progress_m": args.engage_auto_flip_max_forward_progress_m,
                    "continuous_servo": args.continuous_servo,
                    "continuous_servo_hz": args.continuous_servo_hz,
                    "continuous_servo_control_hz": args.continuous_servo_control_hz,
                    "continuous_servo_horizon_s": args.continuous_servo_horizon_s,
                    "continuous_servo_apply_s": args.continuous_servo_apply_s,
                    "continuous_servo_stale_timeout_s": args.continuous_servo_stale_timeout_s,
                    "continuous_servo_command_alpha": args.continuous_servo_command_alpha,
                    "continuous_servo_max_linear_speed_mps": args.continuous_servo_max_linear_speed_mps,
                    "continuous_servo_max_angular_speed_degps": args.continuous_servo_max_angular_speed_degps,
                    "engage_on_center": args.engage_on_center,
                    "engage_at_approach_end": args.engage_at_approach_end,
                    "continue_engage_until_interrupt": args.continue_engage_until_interrupt,
                    "observe_only_no_close": args.observe_only_no_close,
                    "continuous_engage_extend_steps": args.continuous_engage_extend_steps,
                    "pre_close_servo_x_sign": args.pre_close_servo_x_sign,
                    "pre_close_servo_y_sign": args.pre_close_servo_y_sign,
                    "approach_image_servo": args.approach_image_servo,
                    "approach_image_servo_weight": args.approach_image_servo_weight,
                    "approach_image_servo_max_m": args.approach_image_servo_max_m,
                    "approach_image_servo_min_center_distance_px": args.approach_image_servo_min_center_distance_px,
                    "approach_image_servo_x_sign": args.approach_image_servo_x_sign,
                    "approach_image_servo_y_sign": args.approach_image_servo_y_sign,
                    "approach_center_priority": args.approach_center_priority,
                    "approach_center_priority_threshold_px": args.approach_center_priority_threshold_px,
                    "approach_center_priority_policy_xy_scale": args.approach_center_priority_policy_xy_scale,
                    "approach_center_priority_z_scale": args.approach_center_priority_z_scale,
                    "phase": phase,
                    "approach_steps_executed": approach_steps_executed,
                    "engage_steps_executed": engage_steps_executed,
                    "engage_trigger_reason": engage_trigger_reason,
                    "engage_forward_progress_m": engage_forward_progress_m,
                    "engage_forward_progress_positive_m": engage_forward_progress_positive_m,
                    "engage_min_forward_progress_m": args.engage_min_forward_progress_m,
                    "engage_depth_start_m": engage_depth_start_m,
                    "engage_depth_latest_m": engage_depth_latest_m,
                    "engage_depth_drop_m": engage_depth_drop_m,
                    "engage_depth_ready": engage_depth_ready,
                    "engage_stuck_steps": engage_stuck_steps,
                    "engage_min_observed_step_m": args.engage_min_observed_step_m,
                    "engage_max_stuck_steps": args.engage_max_stuck_steps,
                    "engage_extra_steps_on_stuck": args.engage_extra_steps_on_stuck,
                    "engage_max_total_steps": args.engage_max_total_steps,
                    "engage_abort_center_distance_above_px": args.engage_abort_center_distance_above_px,
                    "engage_abort_depth_increase_m": args.engage_abort_depth_increase_m,
                    "engage_abort_diverge_steps": args.engage_abort_diverge_steps,
                    "close_reason": close_reason,
                    "close_success": close_success,
                    "lift_executed": lift_executed,
                    "lift_height_m": lift_height_m,
                    "step_records": step_records,
                },
            )
            print("Trial log:", trial_logger.log_path)
    except KeyboardInterrupt:
        manual_stop_requested = True
        phase = "interrupted" if phase == "approach" else f"interrupted_{phase}"
        close_reason = close_reason or "manual_interrupt"
        print("Manual stop requested: stopping direct grasp run and continuous servo.")
    finally:
        if args.continuous_servo and continuous_servo_command_running:
            continuous_servo_command_running = False
            if continuous_servo_command_thread is not None:
                try:
                    continuous_servo_command_thread.join(timeout=1.0)
                    print("Continuous servo note: async command refresh thread stopped in finally.")
                except Exception as command_stop_exc:
                    print("Continuous servo command thread stop warning:", command_stop_exc)
        if args.continuous_servo and continuous_servo_active:
            try:
                set_async_servo_target((0.0, 0.0, 0.0), 0.0)
                robot.stop_continuous_twist_servo(stop_duration_s=0.2)
                print("Continuous servo note: stopped in finally.")
            except Exception as stop_exc:
                print("Continuous servo stop warning:", stop_exc)
        if manual_stop_requested:
            print("Run note: exited by manual interrupt.")
        gripper.shutdown()


if __name__ == "__main__":
    main()
