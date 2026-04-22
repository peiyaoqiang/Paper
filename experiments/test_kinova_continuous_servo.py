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
    parser = argparse.ArgumentParser(description="Manual continuous-servo Kinova twist test.")
    parser.add_argument("--dx", type=float, default=0.0, help="Delta x in meters per update.")
    parser.add_argument("--dy", type=float, default=0.0, help="Delta y in meters per update.")
    parser.add_argument("--dz", type=float, default=0.0, help="Delta z in meters per update.")
    parser.add_argument("--dyaw", type=float, default=0.0, help="Delta yaw in degrees per update.")
    parser.add_argument("--steps", type=int, default=10, help="Number of repeated updates to send.")
    parser.add_argument("--apply-s", type=float, default=0.10, help="Hold time after each update.")
    parser.add_argument("--horizon-s", type=float, default=1.0, help="Delta-to-velocity horizon.")
    parser.add_argument("--servo-hz", type=float, default=20.0, help="Background servo publish rate.")
    parser.add_argument("--stale-timeout-s", type=float, default=0.35, help="Auto-zero timeout.")
    parser.add_argument("--command-alpha", type=float, default=0.25, help="Servo smoothing alpha.")
    parser.add_argument(
        "--max-linear-speed-mps",
        type=float,
        default=0.02,
        help="Clamp continuous linear speed magnitude.",
    )
    parser.add_argument(
        "--max-angular-speed-degps",
        type=float,
        default=6.0,
        help="Clamp continuous angular speed magnitude.",
    )
    parser.add_argument("--settle-s", type=float, default=0.3, help="Wait after stop before final read.")
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
            sequential_axis_commands=False,
            ros_node_name=f"{config['robot']['ros_node_name']}_continuous_servo_test",
            state_timeout_s=config["robot"]["state_timeout_s"],
            twist_command_duration_s=config["robot"]["twist_command_duration_s"],
            twist_publish_rate_hz=config["robot"]["twist_publish_rate_hz"],
            twist_stop_duration_s=config["robot"]["twist_stop_duration_s"],
        )
    )

    delta_xyz_m = (args.dx, args.dy, args.dz)
    delta_yaw_deg = args.dyaw
    print("Continuous servo test")
    print("Delta per update:", delta_xyz_m)
    print("Yaw per update deg:", delta_yaw_deg)
    print("Steps:", args.steps)
    print("Apply s:", args.apply_s)
    print("Horizon s:", args.horizon_s)
    print("Servo hz:", args.servo_hz)
    print("Stale timeout s:", args.stale_timeout_s)
    print("Command alpha:", args.command_alpha)
    print("Max linear speed mps:", args.max_linear_speed_mps)
    print("Max angular speed degps:", args.max_angular_speed_degps)

    before = robot.get_state()
    print("Before ee_position_m:", before.ee_position_m)
    print("Before ee_yaw_deg:", before.ee_yaw_deg)

    robot.start_continuous_twist_servo(
        publish_rate_hz=args.servo_hz,
        stale_timeout_s=args.stale_timeout_s,
        command_alpha=args.command_alpha,
        max_linear_speed_mps=args.max_linear_speed_mps,
        max_angular_speed_degps=args.max_angular_speed_degps,
    )
    try:
        previous_state = before
        for step_idx in range(args.steps):
            robot.set_continuous_twist_delta(
                delta_xyz_m,
                delta_yaw_deg,
                horizon_s=args.horizon_s,
            )
            time.sleep(args.apply_s)
            current_state = robot.get_state()
            observed_delta = tuple(
                current - previous for current, previous in zip(current_state.ee_position_m, previous_state.ee_position_m)
            )
            print(f"Step {step_idx + 1}")
            print("  Current ee_position_m:", current_state.ee_position_m)
            print("  Observed ee delta:", observed_delta)
            print("  Observed yaw delta:", current_state.ee_yaw_deg - previous_state.ee_yaw_deg)
            previous_state = current_state
    finally:
        robot.stop_continuous_twist_servo(stop_duration_s=0.2)

    time.sleep(args.settle_s)
    after = robot.get_state()
    total_delta = tuple(after_v - before_v for after_v, before_v in zip(after.ee_position_m, before.ee_position_m))
    print("After ee_position_m:", after.ee_position_m)
    print("After ee_yaw_deg:", after.ee_yaw_deg)
    print("Total observed ee delta:", total_delta)
    print("Total observed yaw delta:", after.ee_yaw_deg - before.ee_yaw_deg)


if __name__ == "__main__":
    main()
