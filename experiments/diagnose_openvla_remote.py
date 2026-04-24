from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.types import Observation
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver
from policy.openvla_wrapper import OpenVLAConfig, OpenVLAWrapper


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze one real observation and query OpenVLA multiple times to diagnose remote output stability."
    )
    parser.add_argument(
        "--instruction",
        action="append",
        dest="instructions",
        default=[],
        help="Instruction to evaluate. Repeat this flag to compare multiple instructions on the same frozen observation.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated remote queries for each instruction.",
    )
    parser.add_argument(
        "--unnorm-key",
        type=str,
        default=None,
        help="Override policy unnorm_key for this diagnostic run.",
    )
    parser.add_argument(
        "--image-input-key",
        type=str,
        default=None,
        help="Override policy image_input_key for this diagnostic run.",
    )
    parser.add_argument(
        "--print-raw-response",
        action="store_true",
        help="Print the raw JSON response returned by the OpenVLA remote service.",
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
            ros_node_name=f"{config['camera']['ros_node_name']}_diagnose_openvla_remote",
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
            ros_node_name=f"{config['robot']['ros_node_name']}_diagnose_openvla_remote",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=config["robot"]["twist_command_duration_s"],
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=config["robot"]["twist_stop_duration_s"],
        )
    )


def _build_policy(config: dict, args: argparse.Namespace) -> OpenVLAWrapper:
    return OpenVLAWrapper(
        OpenVLAConfig(
            model_name=config["policy"]["model_name"],
            mode=config["policy"]["mode"],
            remote_url=config["policy"]["remote_url"],
            remote_timeout_s=config["policy"]["remote_timeout_s"],
            unnorm_key=args.unnorm_key if args.unnorm_key is not None else config["policy"]["unnorm_key"],
            image_input_key=(
                args.image_input_key if args.image_input_key is not None else config["policy"]["image_input_key"]
            ),
            remote_action_gripper_semantics=config["policy"].get("remote_action_gripper_semantics", "open_high"),
        )
    )


def _fmt_vector(values: tuple[float, float, float]) -> str:
    return f"({values[0]:+.4f}, {values[1]:+.4f}, {values[2]:+.4f})"


def _vector_norm(values: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in values))


def _axis_stats(axis_values: list[float]) -> tuple[float, float, float, float]:
    mean_value = statistics.fmean(axis_values)
    std_value = statistics.pstdev(axis_values) if len(axis_values) > 1 else 0.0
    min_value = min(axis_values)
    max_value = max(axis_values)
    return (mean_value, std_value, min_value, max_value)


def main() -> None:
    args = parse_args()
    if not args.instructions:
        args.instructions = ["pick up the red ball"]
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    config = load_config()
    camera = _build_camera(config)
    robot = _build_robot(config)
    policy = _build_policy(config, args)

    remote_ok, remote_message = policy.check_remote_health()
    print("OpenVLA remote health:", remote_message)
    if not remote_ok:
        raise RuntimeError(remote_message)

    robot_state = robot.get_state()
    frame = camera.capture_frame()

    print("[frozen_observation]")
    print("  rgb_path:", frame.rgb_path_hint)
    print("  depth_path:", frame.depth_path_hint)
    print("  image_size:", (frame.width, frame.height))
    print("  ee_position_m:", robot_state.ee_position_m)
    print("  ee_yaw_deg:", robot_state.ee_yaw_deg)
    print("  unnorm_key:", policy.config.unnorm_key)
    print("  image_input_key:", policy.config.image_input_key)

    for instruction in args.instructions:
        print()
        print(f"[instruction] {instruction}")

        actions = []
        close_votes = 0
        target_pixels = []
        confidences = []

        for repeat_idx in range(args.repeats):
            observation = Observation(
                instruction=instruction,
                frame=frame,
                robot_state=robot_state,
            )
            raw_response = None
            if args.print_raw_response and policy.config.mode == "remote_api":
                payload = policy._build_remote_payload(observation)
                raw_response = policy._post_json(policy.config.remote_url, payload)
                policy_action = policy._policy_action_from_remote_response(raw_response, observation)
            else:
                policy_action = policy.predict_action(observation)
            actions.append(policy_action.delta_xyz_m)
            confidences.append(policy_action.confidence)
            target_pixels.append(policy_action.target_pixel)
            if policy_action.gripper_command == "close":
                close_votes += 1

            print(
                f"  [repeat {repeat_idx + 1}/{args.repeats}]"
                f" delta_xyz_m={_fmt_vector(policy_action.delta_xyz_m)}"
                f" norm={_vector_norm(policy_action.delta_xyz_m):.4f}m"
                f" yaw={policy_action.delta_yaw_deg:+.4f}"
                f" gripper={policy_action.gripper_command}"
                f" confidence={policy_action.confidence:.3f}"
                f" target_px={policy_action.target_pixel}"
            )
            if raw_response is not None:
                print(f"    raw_response={json.dumps(raw_response, ensure_ascii=False)}")

        xs = [action[0] for action in actions]
        ys = [action[1] for action in actions]
        zs = [action[2] for action in actions]
        norms = [_vector_norm(action) for action in actions]
        x_stats = _axis_stats(xs)
        y_stats = _axis_stats(ys)
        z_stats = _axis_stats(zs)
        norm_stats = _axis_stats(norms)

        unique_target_pixels = sorted({str(pixel) for pixel in target_pixels})
        unique_gripper_ratio = close_votes / float(len(actions))

        print("  [summary]")
        print(
            "    x:"
            f" mean={x_stats[0]:+.4f}"
            f" std={x_stats[1]:.4f}"
            f" min={x_stats[2]:+.4f}"
            f" max={x_stats[3]:+.4f}"
        )
        print(
            "    y:"
            f" mean={y_stats[0]:+.4f}"
            f" std={y_stats[1]:.4f}"
            f" min={y_stats[2]:+.4f}"
            f" max={y_stats[3]:+.4f}"
        )
        print(
            "    z:"
            f" mean={z_stats[0]:+.4f}"
            f" std={z_stats[1]:.4f}"
            f" min={z_stats[2]:+.4f}"
            f" max={z_stats[3]:+.4f}"
        )
        print(
            "    norm:"
            f" mean={norm_stats[0]:.4f}"
            f" std={norm_stats[1]:.4f}"
            f" min={norm_stats[2]:.4f}"
            f" max={norm_stats[3]:.4f}"
        )
        print(
            "    gripper_close_ratio:"
            f" {unique_gripper_ratio:.2f}"
            f" ({close_votes}/{len(actions)})"
        )
        print(
            "    confidence_mean:"
            f" {statistics.fmean(confidences):.3f}"
        )
        print("    unique_target_pixels:", unique_target_pixels)


if __name__ == "__main__":
    main()
