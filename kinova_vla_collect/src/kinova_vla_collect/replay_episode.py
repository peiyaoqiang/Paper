from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kinova_vla_collect.config import AppConfig, load_config
from kinova_vla_collect.kinova_robot import KinovaRobot
from kinova_vla_collect.modbus_gripper import ModbusGripper

FloatArray = NDArray[np.float32]


def load_episode(episode_dir: Path) -> tuple[dict[str, Any], FloatArray]:
    meta_path = episode_dir / "meta.json"
    steps_path = episode_dir / "steps.npz"
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory does not exist: {episode_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing steps.npz: {steps_path}")

    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)
    steps = np.load(steps_path, allow_pickle=False)
    try:
        if "actions" not in steps.files:
            raise KeyError(f"steps.npz missing actions key: {steps_path}")
        actions = steps["actions"].astype(np.float32)
    finally:
        steps.close()

    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"Expected actions shape [T, 4], got {actions.shape}")
    return meta, actions


def replay_episode(
    episode_dir: Path,
    config: AppConfig,
    dry_run: bool | None = None,
    speed_scale: float = 1.0,
    max_steps: int | None = None,
    assume_yes: bool = False,
) -> None:
    if speed_scale < 0.0:
        raise ValueError("speed_scale must be non-negative")

    meta, actions = load_episode(episode_dir)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        actions = actions[:max_steps]

    control_hz = float(meta.get("control_hz", config.control.hz))
    if control_hz <= 0.0:
        raise ValueError(f"Invalid control_hz: {control_hz}")
    dt = 1.0 / control_hz
    scaled_actions = actions.copy()
    scaled_actions[:, :3] *= float(speed_scale)

    print_replay_summary(episode_dir, meta, actions, scaled_actions, control_hz, speed_scale)
    if not assume_yes and not confirm_replay():
        print("Replay cancelled.")
        return

    use_dry_run = config.hardware.dry_run if dry_run is None else dry_run
    robot = KinovaRobot(
        ip=config.kinova.ip,
        username=config.kinova.username,
        password=config.kinova.password,
        dry_run=use_dry_run,
        max_linear_speed=config.kinova.max_linear_speed,
        mode=config.kinova.mode,
        joint_state_topic=config.kinova.joint_state_topic,
        twist_command_topic=config.kinova.twist_command_topic,
        base_frame=config.kinova.base_frame,
        ee_frame=config.kinova.ee_frame,
        twist_command_frame=config.kinova.twist_command_frame,
        sequential_axis_commands=config.kinova.sequential_axis_commands,
        state_timeout_s=config.kinova.state_timeout_s,
        twist_publish_rate_hz=config.kinova.twist_publish_rate_hz,
        twist_stop_duration_s=config.kinova.twist_stop_duration_s,
    )
    gripper = ModbusGripper(
        host=config.gripper.host,
        port=config.gripper.port,
        unit_id=config.gripper.unit_id,
        dry_run=use_dry_run,
        mode=config.gripper.mode,
        serial_port=config.gripper.serial_port,
        baudrate=config.gripper.baudrate,
        timeout_s=config.gripper.timeout_s,
        open_pos_mm=config.gripper.open_pos_mm,
        close_pos_mm=config.gripper.close_pos_mm,
        max_stroke_mm=config.gripper.max_stroke_mm,
        speed=config.gripper.speed,
        close_torque=config.gripper.close_torque,
        open_torque=config.gripper.open_torque,
        acc_dec=config.gripper.acc_dec,
        parity=config.gripper.parity,
        stopbits=config.gripper.stopbits,
        enable_rs485_mode=config.gripper.enable_rs485_mode,
        accept_pos_reached_as_success=config.gripper.accept_pos_reached_as_success,
        open_timeout_s=config.gripper.open_timeout_s,
        close_timeout_s=config.gripper.close_timeout_s,
    )

    try:
        robot.connect()
        gripper.connect()
        next_tick = time.monotonic()
        for step_index, action in enumerate(scaled_actions):
            robot.step_delta_action(action, dt)
            gripper.apply_action(float(action[3]))
            if step_index % max(1, int(control_hz)) == 0:
                print(
                    f"step={step_index:06d}/{len(scaled_actions):06d} "
                    f"action={action.tolist()} gripper={float(action[3]):+.1f}"
                )
            next_tick += dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
        robot.stop()
        gripper.hold()
        print("Replay complete.")
    except Exception:
        try:
            robot.stop()
        except Exception as exc:
            print(f"Warning: robot.stop() failed during exception cleanup: {exc}")
        try:
            gripper.hold()
        except Exception as exc:
            print(f"Warning: gripper.hold() failed during exception cleanup: {exc}")
        raise
    finally:
        try:
            robot.stop()
        except Exception as exc:
            print(f"Warning: robot.stop() failed during shutdown: {exc}")
        try:
            gripper.hold()
        except Exception as exc:
            print(f"Warning: gripper.hold() failed during shutdown: {exc}")
        try:
            gripper.disconnect()
        except Exception as exc:
            print(f"Warning: gripper.disconnect() failed: {exc}")
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"Warning: robot.disconnect() failed: {exc}")


def print_replay_summary(
    episode_dir: Path,
    meta: dict[str, Any],
    original_actions: FloatArray,
    scaled_actions: FloatArray,
    control_hz: float,
    speed_scale: float,
) -> None:
    print("Replay safety summary")
    print(f"  episode: {episode_dir}")
    print(f"  task: {meta.get('task')}")
    print(f"  recorded_success: {meta.get('success')}")
    print(f"  steps: {len(scaled_actions)}")
    print(f"  control_hz: {control_hz}")
    print(f"  speed_scale: {speed_scale}")
    print("  WARNING: Confirm the robot is in the same or a safe initial pose before replay.")
    print_action_stats("  original action", original_actions)
    print_action_stats("  replay action", scaled_actions)


def print_action_stats(prefix: str, actions: FloatArray) -> None:
    if actions.size == 0:
        print(f"{prefix}: empty")
        return
    labels = ["dx", "dy", "dz", "gripper"]
    mins = actions.min(axis=0)
    maxs = actions.max(axis=0)
    means = actions.mean(axis=0)
    stds = actions.std(axis=0)
    print(prefix)
    for index, label in enumerate(labels):
        print(
            f"    {label}: min={float(mins[index]):+.6f} max={float(maxs[index]):+.6f} "
            f"mean={float(means[index]):+.6f} std={float(stds[index]):.6f}"
        )
    gripper = actions[:, 3]
    total = max(1, actions.shape[0])
    print(
        "    gripper ratio: "
        f"open={float(np.sum(gripper < -0.5) / total):.3f} "
        f"hold={float(np.sum(np.abs(gripper) <= 0.5) / total):.3f} "
        f"close={float(np.sum(gripper > 0.5) / total):.3f}"
    )


def confirm_replay() -> bool:
    answer = input(
        "Type 'yes' to replay these actions on the robot/gripper "
        "after verifying the initial pose is safe: "
    )
    return answer.strip().lower() == "yes"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded Kinova VLA episode.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collect_pick_red_block.yaml"),
        help="Path to replay hardware config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run replay.")
    parser.add_argument("--real", action="store_true", help="Force real hardware replay.")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("-y", "--yes", action="store_true", help="Skip interactive safety confirmation.")
    args = parser.parse_args(argv)

    if args.dry_run and args.real:
        raise ValueError("Use at most one of --dry-run or --real")
    config = load_config(args.config)
    dry_run_override = True if args.dry_run else False if args.real else None
    replay_episode(
        episode_dir=args.episode_dir,
        config=config,
        dry_run=dry_run_override,
        speed_scale=args.speed_scale,
        max_steps=args.max_steps,
        assume_yes=args.yes,
    )


if __name__ == "__main__":
    main()
