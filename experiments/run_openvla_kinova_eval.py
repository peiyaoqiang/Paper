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
from common.types import ExecutionResult, Observation, PolicyAction, RefinedGrasp
from drivers.gripper_driver import GripperConfig, GripperDriver
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver
from policy.openvla_wrapper import OpenVLAConfig, OpenVLAWrapper


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route-A OpenVLA on Kinova: execute OpenVLA actions directly."
    )
    parser.add_argument("--instruction", type=str, default="", help="Override default instruction.")
    parser.add_argument("--steps", type=int, default=20, help="Maximum number of OpenVLA steps to execute.")
    parser.add_argument(
        "--target-color",
        type=str,
        choices=("red", "green", "none"),
        default="none",
        help="Optional monitoring target color. Used only for logging, never for control.",
    )
    parser.add_argument(
        "--stop-when-monitor-center-below-px",
        type=float,
        default=None,
        help="Stop the rollout when the monitored target is this close to image center.",
    )
    parser.add_argument(
        "--stop-when-monitor-center-worsens-n-steps",
        type=int,
        default=0,
        help="Stop the rollout after the monitored center distance worsens for N consecutive visible steps.",
    )
    parser.add_argument(
        "--skip-initial-open",
        action="store_true",
        help="Skip opening the gripper before evaluation.",
    )
    parser.add_argument(
        "--policy-gripper-mode",
        type=str,
        choices=("ignore", "close_only", "open_close"),
        default="close_only",
        help="How to apply OpenVLA gripper outputs.",
    )
    parser.add_argument(
        "--close-on-policy-step",
        action="store_true",
        help="If policy outputs close, close immediately and stop the rollout.",
    )
    parser.add_argument(
        "--observe-only-no-close",
        action="store_true",
        help="Debug mode: never close the gripper even if policy asks to close.",
    )
    parser.add_argument(
        "--approach-only",
        action="store_true",
        help=(
            "Run OpenVLA closed-loop approach only: ignore all gripper outputs, never connect or close "
            "the external gripper, and optionally stop from monitor thresholds."
        ),
    )
    parser.add_argument(
        "--monitor-assisted-grasp",
        action="store_true",
        help="When the monitored target is stably centered, execute a final assisted grasp.",
    )
    parser.add_argument(
        "--monitor-grasp-center-below-px",
        type=float,
        default=70.0,
        help="Trigger monitor-assisted grasp when center distance stays below this threshold.",
    )
    parser.add_argument(
        "--monitor-grasp-stable-steps",
        type=int,
        default=2,
        help="Require this many consecutive centered steps before monitor-assisted grasp.",
    )
    parser.add_argument(
        "--monitor-pre-close-delta-x-m",
        type=float,
        default=0.0,
        help="Optional base-frame x offset before assisted close.",
    )
    parser.add_argument(
        "--monitor-pre-close-delta-y-m",
        type=float,
        default=0.0,
        help="Optional base-frame y offset before assisted close.",
    )
    parser.add_argument(
        "--monitor-pre-close-delta-z-m",
        type=float,
        default=-0.02,
        help="Optional base-frame z offset before assisted close. Negative means descend.",
    )
    parser.add_argument(
        "--settle-before-close-s",
        type=float,
        default=0.1,
        help="Pause briefly before closing when policy requests close.",
    )
    parser.add_argument(
        "--lift-height-m",
        type=float,
        default=None,
        help="Optional lift height after a successful close.",
    )
    parser.add_argument(
        "--max-translation-step-m",
        type=float,
        default=None,
        help="Override ActionAdapter/Kinova per-step translation limit.",
    )
    parser.add_argument(
        "--max-rotation-step-deg",
        type=float,
        default=None,
        help="Override ActionAdapter/Kinova per-step yaw limit.",
    )
    parser.add_argument(
        "--disable-safety-clipping",
        action="store_true",
        help="Execute adapted OpenVLA xyz/yaw without per-step translation or yaw clipping.",
    )
    parser.add_argument(
        "--twist-command-duration-s",
        type=float,
        default=None,
        help="Override Kinova twist command duration.",
    )
    parser.add_argument(
        "--twist-stop-duration-s",
        type=float,
        default=None,
        help="Override Kinova twist braking duration.",
    )
    parser.add_argument(
        "--combined-axis-commands",
        action="store_true",
        help="Send xyz together instead of sequential axis commands.",
    )
    parser.add_argument(
        "--twist-command-frame",
        type=str,
        default=None,
        help="Override Kinova twist_command_frame, e.g. base_link or tool_frame.",
    )
    parser.add_argument(
        "--policy-axis-order",
        type=str,
        default="xyz",
        help="How to map remote policy xyz onto Kinova xyz. Example: xyz, xzy, yxz.",
    )
    parser.add_argument(
        "--policy-overall-scale",
        type=float,
        default=1.0,
        help="Global scale applied to OpenVLA delta_xyz before safety clipping.",
    )
    parser.add_argument("--policy-x-sign", type=float, default=1.0, help="Sign for adapted policy x.")
    parser.add_argument("--policy-y-sign", type=float, default=1.0, help="Sign for adapted policy y.")
    parser.add_argument("--policy-z-sign", type=float, default=1.0, help="Sign for adapted policy z.")
    parser.add_argument("--policy-x-scale", type=float, default=1.0, help="Per-axis scale for adapted policy x.")
    parser.add_argument("--policy-y-scale", type=float, default=1.0, help="Per-axis scale for adapted policy y.")
    parser.add_argument("--policy-z-scale", type=float, default=1.0, help="Per-axis scale for adapted policy z.")
    parser.add_argument("--policy-yaw-sign", type=float, default=1.0, help="Sign for adapted policy yaw.")
    parser.add_argument("--policy-yaw-scale", type=float, default=1.0, help="Scale for adapted policy yaw.")
    parser.add_argument(
        "--zero-policy-z",
        action="store_true",
        help="Force adapted policy z to zero for safer early calibration.",
    )
    return parser.parse_args()


def _fmt_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


def _fmt_vector(vector_xyz: tuple[float, float, float]) -> str:
    return f"({vector_xyz[0]:+.4f}, {vector_xyz[1]:+.4f}, {vector_xyz[2]:+.4f})"


def _vector_norm(vector_xyz: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector_xyz))


def _normalize_sign(value: float) -> float:
    return -1.0 if value < 0.0 else 1.0


def _validate_axis_order(axis_order: str) -> tuple[int, int, int]:
    normalized = axis_order.strip().lower()
    if len(normalized) != 3 or set(normalized) != {"x", "y", "z"}:
        raise ValueError("--policy-axis-order must be a permutation of xyz, for example xyz or xzy.")
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    return tuple(axis_to_index[axis_name] for axis_name in normalized)


def _adapt_policy_action(
    policy_action: PolicyAction,
    *,
    axis_order: tuple[int, int, int],
    overall_scale: float,
    axis_signs: tuple[float, float, float],
    axis_scales: tuple[float, float, float],
    yaw_sign: float,
    yaw_scale: float,
    zero_policy_z: bool,
) -> PolicyAction:
    raw_xyz = policy_action.delta_xyz_m
    reordered_xyz = tuple(raw_xyz[source_idx] for source_idx in axis_order)
    adapted_xyz = tuple(
        overall_scale * axis_sign * axis_scale * axis_value
        for axis_value, axis_sign, axis_scale in zip(reordered_xyz, axis_signs, axis_scales)
    )
    if zero_policy_z:
        adapted_xyz = (adapted_xyz[0], adapted_xyz[1], 0.0)

    adapted_yaw_deg = yaw_sign * yaw_scale * policy_action.delta_yaw_deg
    return PolicyAction(
        delta_xyz_m=adapted_xyz,
        delta_yaw_deg=adapted_yaw_deg,
        gripper_command=policy_action.gripper_command,
        confidence=policy_action.confidence,
        target_pixel=policy_action.target_pixel,
        notes=policy_action.notes,
        metadata={
            **policy_action.metadata,
            "raw_policy_delta_xyz_m": policy_action.delta_xyz_m,
            "raw_policy_delta_yaw_deg": policy_action.delta_yaw_deg,
            "policy_axis_order": axis_order,
            "policy_overall_scale": overall_scale,
            "policy_axis_signs": axis_signs,
            "policy_axis_scales": axis_scales,
            "policy_yaw_sign": yaw_sign,
            "policy_yaw_scale": yaw_scale,
            "zero_policy_z": zero_policy_z,
        },
    )


def detect_ball_centroid(rgb_path: str, target_color: str) -> tuple[int, int] | None:
    if target_color == "none":
        return None

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


def center_distance_px(pixel_xy: tuple[int, int] | None, width: int, height: int) -> float | None:
    if pixel_xy is None:
        return None
    center_x = width / 2.0
    center_y = height / 2.0
    return math.hypot(pixel_xy[0] - center_x, pixel_xy[1] - center_y)


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
            ros_node_name=f"{config['camera']['ros_node_name']}_openvla_kinova_eval",
        )
    )


def _build_robot(config: dict, args: argparse.Namespace) -> KinovaDriver:
    max_translation_step_m = (
        args.max_translation_step_m
        if args.max_translation_step_m is not None
        else config["robot"]["max_translation_step_m"]
    )
    max_rotation_step_deg = (
        args.max_rotation_step_deg
        if args.max_rotation_step_deg is not None
        else config["robot"]["max_rotation_step_deg"]
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
    twist_command_frame = (
        args.twist_command_frame
        if args.twist_command_frame is not None
        else config["robot"].get("twist_command_frame", "tool_frame")
    )

    return KinovaDriver(
        KinovaConfig(
            max_translation_step_m=max_translation_step_m,
            max_rotation_step_deg=max_rotation_step_deg,
            mode=config["robot"]["mode"],
            joint_state_topic=config["robot"]["joint_state_topic"],
            twist_command_topic=config["robot"]["twist_command_topic"],
            base_frame=config["robot"]["base_frame"],
            ee_frame=config["robot"]["ee_frame"],
            twist_command_frame=twist_command_frame,
            sequential_axis_commands=sequential_axis_commands,
            ros_node_name=f"{config['robot']['ros_node_name']}_openvla_kinova_eval",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=twist_command_duration_s,
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=twist_stop_duration_s,
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


def main() -> None:
    args = parse_args()
    config = load_config()
    trial_logger = TrialLogger(config["logging"]["log_dir"]) if config["logging"]["enabled"] else None
    instruction = args.instruction.strip() or config["task"]["instruction"]
    lift_height_m = args.lift_height_m if args.lift_height_m is not None else config["task"]["lift_height_m"]

    if args.approach_only:
        args.policy_gripper_mode = "ignore"
        args.observe_only_no_close = True
        args.monitor_assisted_grasp = False
        args.close_on_policy_step = False

    camera = _build_camera(config)
    robot = _build_robot(config, args)
    needs_gripper = (
        not args.observe_only_no_close
        and (
            args.policy_gripper_mode != "ignore"
            or args.monitor_assisted_grasp
        )
    )
    gripper: GripperDriver | None = _build_gripper(config, robot) if needs_gripper else None
    policy = _build_policy(config)
    action_adapter = ActionAdapter(
        ActionAdapterConfig(
            max_translation_step_m=robot.config.max_translation_step_m,
            max_rotation_step_deg=robot.config.max_rotation_step_deg,
            workspace_xyz_min=tuple(config["robot"]["workspace_xyz_min"]),
            workspace_xyz_max=tuple(config["robot"]["workspace_xyz_max"]),
            safety_clipping_enabled=not args.disable_safety_clipping,
            workspace_enforced=config["robot"].get("workspace_enforced", True),
        )
    )

    print("Instruction:", instruction)
    print("Requested policy steps:", args.steps)
    print("Policy mode:", config["policy"]["mode"])
    print("Policy remote URL:", config["policy"]["remote_url"])
    print("Approach only:", args.approach_only)
    print("Policy gripper mode:", args.policy_gripper_mode)
    print("Observe only no close:", args.observe_only_no_close)
    print("External gripper connected:", gripper is not None)
    print("Monitor assisted grasp:", args.monitor_assisted_grasp)
    print("Monitor grasp center below px:", args.monitor_grasp_center_below_px)
    print("Monitor grasp stable steps:", args.monitor_grasp_stable_steps)
    print(
        "Monitor pre-close delta xyz m:",
        (args.monitor_pre_close_delta_x_m, args.monitor_pre_close_delta_y_m, args.monitor_pre_close_delta_z_m),
    )
    print("Close on policy step:", args.close_on_policy_step)
    print("Target color monitor:", args.target_color)
    print("Stop when monitor center below px:", args.stop_when_monitor_center_below_px)
    print("Stop when monitor center worsens n steps:", args.stop_when_monitor_center_worsens_n_steps)
    print("Max translation step m:", robot.config.max_translation_step_m)
    print("Max rotation step deg:", robot.config.max_rotation_step_deg)
    print("Safety clipping enabled:", not args.disable_safety_clipping)
    print("Twist command frame:", robot.config.twist_command_frame)
    print("Policy axis order:", args.policy_axis_order)
    print("Policy overall scale:", args.policy_overall_scale)
    print(
        "Policy axis signs:",
        (_normalize_sign(args.policy_x_sign), _normalize_sign(args.policy_y_sign), _normalize_sign(args.policy_z_sign)),
    )
    print(
        "Policy axis scales:",
        (args.policy_x_scale, args.policy_y_scale, args.policy_z_scale),
    )
    print("Policy yaw sign:", _normalize_sign(args.policy_yaw_sign))
    print("Policy yaw scale:", args.policy_yaw_scale)
    print("Zero policy z:", args.zero_policy_z)

    axis_order = _validate_axis_order(args.policy_axis_order)
    axis_signs = (
        _normalize_sign(args.policy_x_sign),
        _normalize_sign(args.policy_y_sign),
        _normalize_sign(args.policy_z_sign),
    )
    axis_scales = (
        args.policy_x_scale,
        args.policy_y_scale,
        args.policy_z_scale,
    )
    yaw_sign = _normalize_sign(args.policy_yaw_sign)

    first_observation: Observation | None = None
    last_policy_action: PolicyAction | None = None
    last_safe_action = None
    last_frame = None
    step_records = []
    state_trace = []
    close_success = False
    lift_executed = False
    close_reason = ""
    stop_reason = ""
    best_monitored_center_distance_px: float | None = None
    monitor_worsen_streak = 0
    monitor_center_stable_streak = 0

    try:
        remote_ok, remote_message = policy.check_remote_health()
        print("OpenVLA remote health:", remote_message)
        if not remote_ok:
            raise RuntimeError(remote_message)

        if not args.skip_initial_open and gripper is not None and args.policy_gripper_mode != "ignore":
            open_success = gripper.open()
            print("Initial gripper open success:", open_success)
            if not open_success:
                raise RuntimeError("Failed to open gripper before evaluation.")

        for step_idx in range(args.steps):
            before_state = robot.get_state()
            frame = camera.capture_frame()
            last_frame = frame
            observation = Observation(instruction=instruction, frame=frame, robot_state=before_state)
            if first_observation is None:
                first_observation = observation

            monitored_centroid = detect_ball_centroid(frame.rgb_path_hint, args.target_color)
            monitored_center_distance_px = center_distance_px(monitored_centroid, frame.width, frame.height)

            raw_policy_action = policy.predict_action(observation)
            adapted_policy_action = _adapt_policy_action(
                raw_policy_action,
                axis_order=axis_order,
                overall_scale=args.policy_overall_scale,
                axis_signs=axis_signs,
                axis_scales=axis_scales,
                yaw_sign=yaw_sign,
                yaw_scale=args.policy_yaw_scale,
                zero_policy_z=args.zero_policy_z,
            )
            safe_action = action_adapter.adapt(adapted_policy_action, before_state)

            print(
                f"[step {step_idx + 1}/{args.steps}]"
                f" target_px={raw_policy_action.target_pixel}"
                f" 图像距中心={_fmt_optional(monitored_center_distance_px, 1)}px"
                f" confidence={raw_policy_action.confidence:.3f}"
                f" clipped={safe_action.clipped}"
            )
            print(
                "  原始Policy动作 "
                f"delta_xyz_m={_fmt_vector(raw_policy_action.delta_xyz_m)}"
                f" delta_yaw_deg={raw_policy_action.delta_yaw_deg:+.3f}"
                f" gripper={raw_policy_action.gripper_command}"
            )
            print(
                "  适配后Policy动作 "
                f"delta_xyz_m={_fmt_vector(adapted_policy_action.delta_xyz_m)}"
                f" delta_yaw_deg={adapted_policy_action.delta_yaw_deg:+.3f}"
            )
            print(
                "  执行动作 "
                f"delta_xyz_m={_fmt_vector(safe_action.delta_xyz_m)}"
                f" delta_yaw_deg={safe_action.delta_yaw_deg:+.3f}"
            )
            if monitored_centroid is not None:
                print("  监控目标像素:", monitored_centroid)
            if safe_action.rejection_reason:
                print("  安全限幅:", safe_action.rejection_reason)
            if raw_policy_action.notes:
                print("  Policy备注:", raw_policy_action.notes)

            executed_gripper_command = "none"
            if (
                args.policy_gripper_mode == "open_close"
                and raw_policy_action.gripper_command == "open"
                and not args.observe_only_no_close
            ):
                if gripper is None:
                    raise RuntimeError("Gripper command requested but external gripper is not connected.")
                gripper.open()
                executed_gripper_command = "open"

            if monitored_center_distance_px is not None and monitored_centroid is not None:
                if monitored_center_distance_px <= args.monitor_grasp_center_below_px:
                    monitor_center_stable_streak += 1
                else:
                    monitor_center_stable_streak = 0
            else:
                monitor_center_stable_streak = 0

            if args.monitor_assisted_grasp:
                print(
                    "  抓取监控:",
                    f"居中稳定 {monitor_center_stable_streak}/{args.monitor_grasp_stable_steps}",
                )

            if (
                args.monitor_assisted_grasp
                and monitored_center_distance_px is not None
                and monitored_centroid is not None
                and monitor_center_stable_streak >= args.monitor_grasp_stable_steps
            ):
                close_reason = "monitor_assisted_close"
                print(
                    "  抓取触发: 监控目标已稳定居中",
                    f"{monitor_center_stable_streak}/{args.monitor_grasp_stable_steps}",
                    f"(当前 {monitored_center_distance_px:.1f}px <= {args.monitor_grasp_center_below_px:.1f}px)",
                )
                pre_close_delta_xyz_m = (
                    args.monitor_pre_close_delta_x_m,
                    args.monitor_pre_close_delta_y_m,
                    args.monitor_pre_close_delta_z_m,
                )
                if any(abs(value) > 1e-9 for value in pre_close_delta_xyz_m):
                    pre_close_action = action_adapter.adapt(
                        PolicyAction(
                            delta_xyz_m=pre_close_delta_xyz_m,
                            delta_yaw_deg=0.0,
                            gripper_command="none",
                            confidence=1.0,
                            target_pixel=raw_policy_action.target_pixel,
                            notes="monitor_assisted_pre_close",
                            metadata={"source": "monitor_assisted_pre_close"},
                        ),
                        before_state,
                    )
                    print(
                        "  抓前调整 "
                        f"delta_xyz_m={_fmt_vector(pre_close_action.delta_xyz_m)}"
                        f" clipped={pre_close_action.clipped}"
                    )
                    if pre_close_action.rejection_reason:
                        print("  抓前调整限幅:", pre_close_action.rejection_reason)
                    robot.move_cartesian_delta(pre_close_action.delta_xyz_m, 0.0)
                    time.sleep(max(args.settle_before_close_s, 0.0))

                if args.observe_only_no_close:
                    print("  抓取触发到了，但当前是 observe-only，不执行夹爪闭合。")
                else:
                    if gripper is None:
                        raise RuntimeError("Monitor-assisted close requested but external gripper is not connected.")
                    if args.settle_before_close_s > 0.0:
                        time.sleep(args.settle_before_close_s)
                    close_success = gripper.close()
                    executed_gripper_command = "close"
                    print("  夹爪闭合结果:", close_success)
                    if close_success and lift_height_m > 0.0:
                        robot.move_cartesian_delta((0.0, 0.0, lift_height_m), 0.0)
                        lift_executed = True
                        print("  抬升高度:", f"{lift_height_m:.3f}m")

                step_records.append(
                    {
                        "step_index": step_idx + 1,
                        "rgb_path_hint": frame.rgb_path_hint,
                        "before_ee_position_m": before_state.ee_position_m,
                        "raw_policy_delta_xyz_m": raw_policy_action.delta_xyz_m,
                        "raw_policy_delta_yaw_deg": raw_policy_action.delta_yaw_deg,
                        "adapted_policy_delta_xyz_m": adapted_policy_action.delta_xyz_m,
                        "adapted_policy_delta_yaw_deg": adapted_policy_action.delta_yaw_deg,
                        "policy_gripper_command": raw_policy_action.gripper_command,
                        "safe_delta_xyz_m": safe_action.delta_xyz_m,
                        "safe_delta_yaw_deg": safe_action.delta_yaw_deg,
                        "safe_action_clipped": safe_action.clipped,
                        "monitored_centroid": monitored_centroid,
                        "monitored_center_distance_px": monitored_center_distance_px,
                        "executed_gripper_command": executed_gripper_command,
                        "policy_metadata": raw_policy_action.metadata,
                    }
                )
                last_policy_action = adapted_policy_action
                last_safe_action = safe_action
                state_trace.extend(
                    [
                        f"observe_{step_idx + 1}",
                        f"policy_predict_{step_idx + 1}",
                        f"monitor_assisted_grasp_{step_idx + 1}",
                    ]
                )
                break

            if (
                args.policy_gripper_mode in ("close_only", "open_close")
                and raw_policy_action.gripper_command == "close"
                and args.close_on_policy_step
            ):
                close_reason = "policy_close"
                if args.observe_only_no_close:
                    print("  Policy要求闭合，但当前是 observe-only，不执行夹爪闭合。")
                else:
                    if gripper is None:
                        raise RuntimeError("Policy close requested but external gripper is not connected.")
                    if args.settle_before_close_s > 0.0:
                        time.sleep(args.settle_before_close_s)
                    close_success = gripper.close()
                    executed_gripper_command = "close"
                    print("  夹爪闭合结果:", close_success)
                    if close_success and lift_height_m > 0.0:
                        robot.move_cartesian_delta((0.0, 0.0, lift_height_m), 0.0)
                        lift_executed = True
                        print("  抬升高度:", f"{lift_height_m:.3f}m")

                step_records.append(
                    {
                        "step_index": step_idx + 1,
                        "rgb_path_hint": frame.rgb_path_hint,
                        "before_ee_position_m": before_state.ee_position_m,
                        "raw_policy_delta_xyz_m": raw_policy_action.delta_xyz_m,
                        "raw_policy_delta_yaw_deg": raw_policy_action.delta_yaw_deg,
                        "adapted_policy_delta_xyz_m": adapted_policy_action.delta_xyz_m,
                        "adapted_policy_delta_yaw_deg": adapted_policy_action.delta_yaw_deg,
                        "policy_gripper_command": raw_policy_action.gripper_command,
                        "safe_delta_xyz_m": safe_action.delta_xyz_m,
                        "safe_delta_yaw_deg": safe_action.delta_yaw_deg,
                        "safe_action_clipped": safe_action.clipped,
                        "monitored_centroid": monitored_centroid,
                        "monitored_center_distance_px": monitored_center_distance_px,
                        "executed_gripper_command": executed_gripper_command,
                        "policy_metadata": raw_policy_action.metadata,
                    }
                )
                last_policy_action = adapted_policy_action
                last_safe_action = safe_action
                state_trace.extend(
                    [
                        f"observe_{step_idx + 1}",
                        f"policy_predict_{step_idx + 1}",
                        f"policy_close_{step_idx + 1}",
                    ]
                )
                break

            robot.move_cartesian_delta(safe_action.delta_xyz_m, safe_action.delta_yaw_deg)
            after_state = robot.get_state()
            observed_delta_xyz = tuple(
                after_axis - before_axis
                for after_axis, before_axis in zip(after_state.ee_position_m, before_state.ee_position_m)
            )
            observed_delta_yaw = after_state.ee_yaw_deg - before_state.ee_yaw_deg

            print(
                "  观测位移 "
                f"delta_xyz_m={_fmt_vector(observed_delta_xyz)}"
                f" delta_yaw_deg={observed_delta_yaw:+.3f}"
                f" norm={_vector_norm(observed_delta_xyz):.4f}m"
            )

            step_records.append(
                {
                    "step_index": step_idx + 1,
                    "rgb_path_hint": frame.rgb_path_hint,
                    "before_ee_position_m": before_state.ee_position_m,
                    "after_ee_position_m": after_state.ee_position_m,
                    "raw_policy_delta_xyz_m": raw_policy_action.delta_xyz_m,
                    "raw_policy_delta_yaw_deg": raw_policy_action.delta_yaw_deg,
                    "adapted_policy_delta_xyz_m": adapted_policy_action.delta_xyz_m,
                    "adapted_policy_delta_yaw_deg": adapted_policy_action.delta_yaw_deg,
                    "policy_gripper_command": raw_policy_action.gripper_command,
                    "safe_delta_xyz_m": safe_action.delta_xyz_m,
                    "safe_delta_yaw_deg": safe_action.delta_yaw_deg,
                    "safe_action_clipped": safe_action.clipped,
                    "observed_delta_xyz_m": observed_delta_xyz,
                    "observed_delta_yaw_deg": observed_delta_yaw,
                    "monitored_centroid": monitored_centroid,
                    "monitored_center_distance_px": monitored_center_distance_px,
                    "executed_gripper_command": executed_gripper_command,
                    "policy_metadata": raw_policy_action.metadata,
                }
            )
            last_policy_action = adapted_policy_action
            last_safe_action = safe_action
            state_trace.extend(
                [
                    f"observe_{step_idx + 1}",
                    f"policy_predict_{step_idx + 1}",
                    f"action_adapt_{step_idx + 1}",
                    f"robot_move_{step_idx + 1}",
                ]
            )

            if monitored_center_distance_px is not None:
                if (
                    args.stop_when_monitor_center_below_px is not None
                    and monitored_center_distance_px <= args.stop_when_monitor_center_below_px
                ):
                    stop_reason = f"monitor_center_below_{args.stop_when_monitor_center_below_px:.1f}px"
                    print(
                        "  监控停机: 图像距中心已到",
                        f"{monitored_center_distance_px:.1f}px <= {args.stop_when_monitor_center_below_px:.1f}px",
                    )
                    break

                if best_monitored_center_distance_px is None:
                    best_monitored_center_distance_px = monitored_center_distance_px
                    monitor_worsen_streak = 0
                elif monitored_center_distance_px < best_monitored_center_distance_px:
                    best_monitored_center_distance_px = monitored_center_distance_px
                    monitor_worsen_streak = 0
                else:
                    monitor_worsen_streak += 1
                    if args.stop_when_monitor_center_worsens_n_steps > 0:
                        print(
                            "  监控趋势: 连续变差",
                            f"{monitor_worsen_streak}/{args.stop_when_monitor_center_worsens_n_steps}",
                            f"(当前 {monitored_center_distance_px:.1f}px, 最优 {best_monitored_center_distance_px:.1f}px)",
                        )
                    if (
                        args.stop_when_monitor_center_worsens_n_steps > 0
                        and monitor_worsen_streak >= args.stop_when_monitor_center_worsens_n_steps
                    ):
                        stop_reason = (
                            "monitor_center_worsened_"
                            f"{args.stop_when_monitor_center_worsens_n_steps}_steps"
                        )
                        print("  监控停机: 图像距中心已连续变差，停止当前 rollout。")
                        break
            else:
                monitor_worsen_streak = 0

        final_state = robot.get_state()
        result = ExecutionResult(
            success=close_success if not args.observe_only_no_close else True,
            state_trace=state_trace,
            message="OpenVLA Kinova route-A evaluation completed",
            failure_reason="" if close_success or args.observe_only_no_close else "Policy never triggered close",
            grasp=RefinedGrasp(
                target_xyz_m=final_state.ee_position_m,
                target_yaw_deg=final_state.ee_yaw_deg,
                grasp_width_m=final_state.gripper_opening_m,
                quality=1.0 if close_success else 0.0,
                source="openvla_kinova_eval",
            ),
        )

        print("[summary]")
        print("  步数:", len(step_records))
        print("  最终末端位置:", final_state.ee_position_m)
        print("  最终末端yaw:", f"{final_state.ee_yaw_deg:.3f}")
        print("  是否执行闭合:", close_reason == "policy_close")
        print("  夹爪闭合成功:", close_success)
        print("  是否执行抬升:", lift_executed)
        print("  监控停机原因:", stop_reason if stop_reason else "none")
        print("  整体成功:", result.success)
        if result.failure_reason:
            print("  失败原因:", result.failure_reason)

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
                    "test_type": "openvla_kinova_eval",
                    "approach_only": args.approach_only,
                    "requested_steps": args.steps,
                    "policy_mode": config["policy"]["mode"],
                    "policy_remote_url": config["policy"]["remote_url"],
                    "policy_gripper_mode": args.policy_gripper_mode,
                    "close_on_policy_step": args.close_on_policy_step,
                    "observe_only_no_close": args.observe_only_no_close,
                    "target_color_monitor": args.target_color,
                    "twist_command_frame": robot.config.twist_command_frame,
                    "policy_axis_order": args.policy_axis_order,
                    "policy_overall_scale": args.policy_overall_scale,
                    "policy_x_sign": axis_signs[0],
                    "policy_y_sign": axis_signs[1],
                    "policy_z_sign": axis_signs[2],
                    "policy_x_scale": axis_scales[0],
                    "policy_y_scale": axis_scales[1],
                    "policy_z_scale": axis_scales[2],
                    "policy_yaw_sign": yaw_sign,
                    "policy_yaw_scale": args.policy_yaw_scale,
                    "zero_policy_z": args.zero_policy_z,
                    "max_translation_step_m": robot.config.max_translation_step_m,
                    "max_rotation_step_deg": robot.config.max_rotation_step_deg,
                    "twist_command_duration_s": robot.config.twist_command_duration_s,
                    "twist_stop_duration_s": robot.config.twist_stop_duration_s,
                    "sequential_axis_commands": robot.config.sequential_axis_commands,
                    "close_reason": close_reason,
                    "stop_reason": stop_reason,
                    "close_success": close_success,
                    "lift_executed": lift_executed,
                    "lift_height_m": lift_height_m,
                    "monitor_assisted_grasp": args.monitor_assisted_grasp,
                    "monitor_grasp_center_below_px": args.monitor_grasp_center_below_px,
                    "monitor_grasp_stable_steps": args.monitor_grasp_stable_steps,
                    "monitor_pre_close_delta_xyz_m": (
                        args.monitor_pre_close_delta_x_m,
                        args.monitor_pre_close_delta_y_m,
                        args.monitor_pre_close_delta_z_m,
                    ),
                    "last_rgb_path_hint": None if last_frame is None else last_frame.rgb_path_hint,
                    "step_records": step_records,
                },
            )
            print("  trial_log:", trial_logger.log_path)
    finally:
        if gripper is not None:
            gripper.shutdown()


if __name__ == "__main__":
    main()
