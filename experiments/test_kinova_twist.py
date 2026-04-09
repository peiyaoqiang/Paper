from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.kinova_driver import KinovaConfig, KinovaDriver


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe minimal Kinova twist test.")
    parser.add_argument("--dx", type=float, default=0.0, help="Delta x in meters.")
    parser.add_argument("--dy", type=float, default=0.0, help="Delta y in meters.")
    parser.add_argument("--dz", type=float, default=0.0, help="Delta z in meters.")
    parser.add_argument("--dyaw", type=float, default=0.0, help="Delta yaw in degrees.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.4,
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
        help="Extra wait time after command before reading the final state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    robot = KinovaDriver(
        KinovaConfig(
            max_translation_step_m=config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            mode=config["robot"]["mode"],
            joint_state_topic=config["robot"]["joint_state_topic"],
            twist_command_topic=config["robot"]["twist_command_topic"],
            base_frame=config["robot"]["base_frame"],
            ee_frame=config["robot"]["ee_frame"],
            twist_command_frame=config["robot"].get("twist_command_frame", "tool_frame"),
            ros_node_name=f"{config['robot']['ros_node_name']}_twist_test",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=args.duration,
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=args.stop_duration,
        )
    )

    before = robot.get_state()
    print("Before ee_position_m:", before.ee_position_m)
    print("Before ee_yaw_deg:", before.ee_yaw_deg)
    print("Before joint_positions:", before.joint_positions)

    delta_xyz_m = (args.dx, args.dy, args.dz)
    delta_yaw_deg = args.dyaw
    print("Sending delta_xyz_m:", delta_xyz_m)
    print("Sending delta_yaw_deg:", delta_yaw_deg)
    print("Command duration s:", args.duration)
    print("Stop duration s:", args.stop_duration)
    robot.move_cartesian_delta(delta_xyz_m, delta_yaw_deg)

    time.sleep(args.settle)
    after = robot.get_state()
    print("After ee_position_m:", after.ee_position_m)
    print("After ee_yaw_deg:", after.ee_yaw_deg)
    print("After joint_positions:", after.joint_positions)
    print(
        "Observed ee delta:",
        tuple(after_v - before_v for after_v, before_v in zip(after.ee_position_m, before.ee_position_m)),
    )


if __name__ == "__main__":
    main()
