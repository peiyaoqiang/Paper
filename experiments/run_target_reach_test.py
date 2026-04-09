from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from analysis.trial_logger import TrialLogger
from adapters.action_adapter import ActionAdapter, ActionAdapterConfig
from common.types import ExecutionResult, Observation, RefinedGrasp
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver
from policy.openvla_wrapper import OpenVLAConfig, OpenVLAWrapper


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct target-reach test using image-space ball centroid as the evaluation signal."
    )
    parser.add_argument("--instruction", type=str, default="", help="Override default instruction.")
    parser.add_argument(
        "--target-color",
        type=str,
        choices=("red", "green"),
        required=True,
        help="Ball color to track in image space.",
    )
    parser.add_argument("--steps", type=int, default=3, help="Number of safe control steps.")
    parser.add_argument(
        "--success-radius-px",
        type=float,
        default=120.0,
        help="Consider the target reached if its centroid is within this many pixels of image center.",
    )
    parser.add_argument(
        "--min-improvement-px",
        type=float,
        default=15.0,
        help="Require at least this many pixels of net improvement when the target already starts near the center.",
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
            ros_node_name=f"{config['camera']['ros_node_name']}_target_reach_test",
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
            ros_node_name=f"{config['robot']['ros_node_name']}_target_reach_test",
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


def _detect_ball_centroid(rgb_path: str, target_color: str) -> tuple[float, float] | None:
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


def _center_distance_px(centroid: tuple[float, float] | None, width: int, height: int) -> float | None:
    if centroid is None:
        return None
    cx = width / 2.0
    cy = height / 2.0
    return math.hypot(centroid[0] - cx, centroid[1] - cy)


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

    print("Instruction:", instruction)
    print("Tracked target color:", args.target_color)
    print("Requested safe steps:", args.steps)

    step_records = []
    first_observation: Observation | None = None
    last_policy_action = None
    last_safe_action = None
    initial_distance_px: float | None = None
    final_distance_px: float | None = None

    for step_idx in range(args.steps):
        before_state = robot.get_state()
        before_frame = camera.capture_frame()
        before_centroid = _detect_ball_centroid(before_frame.rgb_path_hint, args.target_color)
        before_distance_px = _center_distance_px(before_centroid, before_frame.width, before_frame.height)
        if initial_distance_px is None:
            initial_distance_px = before_distance_px

        observation = Observation(instruction=instruction, frame=before_frame, robot_state=before_state)
        if first_observation is None:
            first_observation = observation

        policy_action = policy.predict_action(observation)
        safe_action = action_adapter.adapt(policy_action, before_state)

        print(f"Step {step_idx + 1}")
        print("  RGB path:", before_frame.rgb_path_hint)
        print("  Before ee_position_m:", before_state.ee_position_m)
        print("  Target centroid before:", before_centroid)
        print("  Target center distance before px:", before_distance_px)
        print("  Policy delta_xyz_m:", policy_action.delta_xyz_m)
        print("  Policy delta_yaw_deg:", policy_action.delta_yaw_deg)
        print("  Safe delta_xyz_m:", safe_action.delta_xyz_m)
        print("  Safe delta_yaw_deg:", safe_action.delta_yaw_deg)
        print("  Safe clipped:", safe_action.clipped)
        if safe_action.rejection_reason:
            print("  Safe action note:", safe_action.rejection_reason)

        robot.move_cartesian_delta(safe_action.delta_xyz_m, safe_action.delta_yaw_deg)

        after_state = robot.get_state()
        after_frame = camera.capture_frame()
        after_centroid = _detect_ball_centroid(after_frame.rgb_path_hint, args.target_color)
        after_distance_px = _center_distance_px(after_centroid, after_frame.width, after_frame.height)
        final_distance_px = after_distance_px

        print("  After ee_position_m:", after_state.ee_position_m)
        print("  Target centroid after:", after_centroid)
        print("  Target center distance after px:", after_distance_px)

        step_records.append(
            {
                "step_index": step_idx + 1,
                "before_rgb_path_hint": before_frame.rgb_path_hint,
                "after_rgb_path_hint": after_frame.rgb_path_hint,
                "before_ee_position_m": before_state.ee_position_m,
                "after_ee_position_m": after_state.ee_position_m,
                "before_target_centroid": before_centroid,
                "after_target_centroid": after_centroid,
                "before_target_center_distance_px": before_distance_px,
                "after_target_center_distance_px": after_distance_px,
                "policy_delta_xyz_m": policy_action.delta_xyz_m,
                "policy_delta_yaw_deg": policy_action.delta_yaw_deg,
                "safe_delta_xyz_m": safe_action.delta_xyz_m,
                "safe_delta_yaw_deg": safe_action.delta_yaw_deg,
                "safe_action_clipped": safe_action.clipped,
                "policy_metadata": policy_action.metadata,
            }
        )
        last_policy_action = policy_action
        last_safe_action = safe_action

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
    result = ExecutionResult(
        success=success,
        state_trace=[f"target_reach_step_{idx + 1}" for idx in range(args.steps)],
        message="Target reach test completed",
        failure_reason="" if success else "Target did not get closer to image center",
        grasp=RefinedGrasp(
            target_xyz_m=final_state.ee_position_m,
            target_yaw_deg=final_state.ee_yaw_deg,
            grasp_width_m=final_state.gripper_opening_m,
            quality=1.0,
            source="target_reach_test",
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

    if trial_logger is not None and first_observation is not None and last_policy_action is not None and last_safe_action is not None:
        trial_logger.log_trial(
            instruction=instruction,
            observation=first_observation,
            policy_action=last_policy_action,
            safe_action=last_safe_action,
            refined_grasp=result.grasp,
            result=result,
            final_robot_state=final_state,
            metadata={
                "test_type": "target_reach_test",
                "target_color": args.target_color,
                "requested_steps": args.steps,
                "success_radius_px": args.success_radius_px,
                "min_improvement_px": args.min_improvement_px,
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
