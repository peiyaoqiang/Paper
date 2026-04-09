from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from analysis.trial_logger import TrialLogger
from common.types import ExecutionResult, Observation, PolicyAction, RefinedGrasp, SafeAction
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate how image-space target motion responds to small robot Cartesian moves."
    )
    parser.add_argument(
        "--target-color",
        type=str,
        choices=("red", "green"),
        required=True,
        help="Ball color to track in image space.",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=0.015,
        help="Translation step for x/y calibration moves.",
    )
    parser.add_argument(
        "--z-step-m",
        type=float,
        default=0.005,
        help="Translation step for z calibration moves when enabled.",
    )
    parser.add_argument(
        "--include-z",
        action="store_true",
        help="Also run +z/-z calibration moves.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.8,
        help="Non-zero twist command duration in seconds.",
    )
    parser.add_argument(
        "--stop-duration",
        type=float,
        default=0.8,
        help="Zero-twist braking duration in seconds.",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.5,
        help="Extra wait time after motion before capturing the after-frame.",
    )
    return parser.parse_args()


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
            ros_node_name=f"{config['camera']['ros_node_name']}_image_motion_calibration",
        )
    )


def build_robot(config: dict, args: argparse.Namespace) -> KinovaDriver:
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
            ros_node_name=f"{config['robot']['ros_node_name']}_image_motion_calibration",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=args.duration,
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=args.stop_duration,
        )
    )


def build_move_sequence(args: argparse.Namespace) -> list[tuple[str, tuple[float, float, float]]]:
    moves = [
        ("plus_x", (args.step_m, 0.0, 0.0)),
        ("minus_x", (-args.step_m, 0.0, 0.0)),
        ("plus_y", (0.0, args.step_m, 0.0)),
        ("minus_y", (0.0, -args.step_m, 0.0)),
    ]
    if args.include_z:
        moves.extend(
            [
                ("plus_z", (0.0, 0.0, args.z_step_m)),
                ("minus_z", (0.0, 0.0, -args.z_step_m)),
            ]
        )
    return moves


def main() -> None:
    args = parse_args()
    config = load_config()
    trial_logger = TrialLogger(config["logging"]["log_dir"]) if config["logging"]["enabled"] else None
    camera = build_camera(config)
    robot = build_robot(config, args)
    moves = build_move_sequence(args)

    print("Tracked target color:", args.target_color)
    print("Move sequence:", [name for name, _ in moves])
    print("Step magnitude x/y (m):", args.step_m)
    if args.include_z:
        print("Step magnitude z (m):", args.z_step_m)
    print("Command duration s:", args.duration)
    print("Stop duration s:", args.stop_duration)

    step_records = []
    first_observation: Observation | None = None

    for idx, (move_name, delta_xyz_m) in enumerate(moves, start=1):
        before_state = robot.get_state()
        before_frame = camera.capture_frame()
        before_centroid = detect_ball_centroid(before_frame.rgb_path_hint, args.target_color)
        if before_centroid is None:
            raise RuntimeError(
                f"Could not detect a {args.target_color} ball in {before_frame.rgb_path_hint}."
            )
        before_distance_px = center_distance_px(before_centroid, before_frame.width, before_frame.height)
        observation = Observation(
            instruction=f"image_motion_calibration:{move_name}",
            frame=before_frame,
            robot_state=before_state,
        )
        if first_observation is None:
            first_observation = observation

        print(f"Move {idx}: {move_name}")
        print("  RGB path before:", before_frame.rgb_path_hint)
        print("  Before ee_position_m:", before_state.ee_position_m)
        print("  Before target centroid:", before_centroid)
        print("  Before center distance px:", before_distance_px)
        print("  Sending delta_xyz_m:", delta_xyz_m)

        robot.move_cartesian_delta(delta_xyz_m, 0.0)
        time.sleep(args.settle)

        after_state = robot.get_state()
        after_frame = camera.capture_frame()
        after_centroid = detect_ball_centroid(after_frame.rgb_path_hint, args.target_color)
        after_distance_px = center_distance_px(after_centroid, after_frame.width, after_frame.height)
        if after_centroid is None:
            raise RuntimeError(
                f"Could not detect a {args.target_color} ball in {after_frame.rgb_path_hint}."
            )

        centroid_delta = (
            after_centroid[0] - before_centroid[0],
            after_centroid[1] - before_centroid[1],
        )
        ee_delta = tuple(
            after_value - before_value
            for after_value, before_value in zip(after_state.ee_position_m, before_state.ee_position_m)
        )
        distance_change_px = None
        if before_distance_px is not None and after_distance_px is not None:
            distance_change_px = after_distance_px - before_distance_px

        print("  RGB path after:", after_frame.rgb_path_hint)
        print("  After ee_position_m:", after_state.ee_position_m)
        print("  After target centroid:", after_centroid)
        print("  After center distance px:", after_distance_px)
        print("  Image centroid delta px:", centroid_delta)
        print("  Observed ee delta:", ee_delta)
        print("  Center distance change px:", distance_change_px)

        step_records.append(
            {
                "move_name": move_name,
                "command_delta_xyz_m": delta_xyz_m,
                "before_rgb_path_hint": before_frame.rgb_path_hint,
                "after_rgb_path_hint": after_frame.rgb_path_hint,
                "before_ee_position_m": before_state.ee_position_m,
                "after_ee_position_m": after_state.ee_position_m,
                "observed_ee_delta": ee_delta,
                "before_target_centroid": before_centroid,
                "after_target_centroid": after_centroid,
                "image_centroid_delta_px": centroid_delta,
                "before_center_distance_px": before_distance_px,
                "after_center_distance_px": after_distance_px,
                "center_distance_change_px": distance_change_px,
            }
        )

    if trial_logger is not None and first_observation is not None:
        final_state = robot.get_state()
        last_record = step_records[-1]
        policy_action = PolicyAction(
            delta_xyz_m=last_record["command_delta_xyz_m"],
            delta_yaw_deg=0.0,
            gripper_command="open",
            confidence=1.0,
            target_pixel=None,
            notes="Image-motion calibration final move",
            metadata={"move_name": last_record["move_name"]},
        )
        safe_action = SafeAction(
            delta_xyz_m=last_record["command_delta_xyz_m"],
            delta_yaw_deg=0.0,
            gripper_command="open",
            clipped=False,
            rejection_reason="",
        )
        result = ExecutionResult(
            success=True,
            state_trace=[record["move_name"] for record in step_records],
            message="Image-motion calibration completed",
            grasp=RefinedGrasp(
                target_xyz_m=final_state.ee_position_m,
                target_yaw_deg=final_state.ee_yaw_deg,
                grasp_width_m=final_state.gripper_opening_m,
                quality=1.0,
                source="image_motion_calibration",
            ),
        )
        trial_logger.log_trial(
            instruction=f"image_motion_calibration:{args.target_color}",
            observation=first_observation,
            policy_action=policy_action,
            safe_action=safe_action,
            refined_grasp=result.grasp,
            result=result,
            final_robot_state=final_state,
            metadata={
                "test_type": "image_motion_calibration",
                "target_color": args.target_color,
                "step_m": args.step_m,
                "z_step_m": args.z_step_m,
                "include_z": args.include_z,
                "command_duration_s": args.duration,
                "stop_duration_s": args.stop_duration,
                "settle_s": args.settle,
                "step_records": step_records,
            },
        )
        print("Trial log:", trial_logger.log_path)


if __name__ == "__main__":
    main()
