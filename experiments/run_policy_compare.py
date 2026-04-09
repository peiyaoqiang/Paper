from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.types import Observation, RobotState
from drivers.kinova_driver import KinovaConfig, KinovaDriver
from drivers.realsense_driver import RealSenseConfig, RealSenseDriver
from policy.openvla_wrapper import OpenVLAConfig, OpenVLAWrapper


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one real observation and compare OpenVLA outputs across multiple instructions."
    )
    parser.add_argument(
        "--instruction",
        action="append",
        dest="instructions",
        default=[],
        help="Instruction to evaluate. Repeat this flag to compare multiple instructions on the same captured frame.",
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
            ros_node_name=f"{config['camera']['ros_node_name']}_policy_compare",
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
            ros_node_name=f"{config['robot']['ros_node_name']}_policy_compare",
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


def _print_robot_state(robot_state: RobotState) -> None:
    print("Captured robot state:")
    print("  ee_position_m:", robot_state.ee_position_m)
    print("  ee_yaw_deg:", robot_state.ee_yaw_deg)
    print("  joint_positions:", robot_state.joint_positions)


def main() -> None:
    args = parse_args()
    if len(args.instructions) < 2:
        raise SystemExit("Please provide at least two --instruction values for comparison.")

    config = load_config()
    camera = _build_camera(config)
    robot = _build_robot(config)
    policy = _build_policy(config)

    robot_state = robot.get_state()
    frame = camera.capture_frame()

    print("Frozen observation for comparison")
    print("  RGB path:", frame.rgb_path_hint)
    print("  Depth path:", frame.depth_path_hint)
    print("  Image size:", (frame.width, frame.height))
    _print_robot_state(robot_state)
    print()

    for index, instruction in enumerate(args.instructions, start=1):
        observation = Observation(
            instruction=instruction,
            frame=frame,
            robot_state=robot_state,
        )
        policy_action = policy.predict_action(observation)
        print(f"Instruction {index}: {instruction}")
        print("  Policy delta_xyz_m:", policy_action.delta_xyz_m)
        print("  Policy delta_yaw_deg:", policy_action.delta_yaw_deg)
        print("  Policy gripper_command:", policy_action.gripper_command)
        print("  Policy confidence:", policy_action.confidence)
        print("  Policy target_pixel:", policy_action.target_pixel)
        print("  Policy metadata:", policy_action.metadata)
        print()


if __name__ == "__main__":
    main()
