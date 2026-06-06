from __future__ import annotations

import argparse
import math
import time

import numpy as np

from kinova_vla_collect.kinova_robot import KinovaRobot
from kinova_vla_collect.spacemouse_controller import SpaceMouseController


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Control the dry-run Kinova simulator with a SpaceMouse.")
    parser.add_argument("--device", type=str, default=None, help="pyspacemouse device name override.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-path", type=str, default=None, help="Open a specific /dev/hidraw* path.")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--max-delta-m", type=float, default=0.005)
    parser.add_argument("--max-delta-rad", type=float, default=math.radians(2.0))
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--max-linear-speed", type=float, default=0.08)
    parser.add_argument("--require-enable-button", action="store_true")
    parser.add_argument("--print-period", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl+C or stop button.")
    parser.add_argument("--mock-input", action="store_true", help="Use a synthetic SpaceMouse-like signal.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")

    dt = 1.0 / args.hz
    robot = KinovaRobot(
        ip="0.0.0.0",
        username="",
        password="",
        dry_run=True,
        mode="ros2_twist",
        max_linear_speed=args.max_linear_speed,
    )
    controller = (
        _MockSpaceMouseController(args.max_delta_m, args.max_delta_rad)
        if args.mock_input
        else SpaceMouseController(
            device=args.device,
            device_index=args.device_index,
            device_path=args.device_path,
            deadzone=args.deadzone,
            max_delta_m=args.max_delta_m,
            max_delta_rad=args.max_delta_rad,
            require_enable_button=args.require_enable_button,
            debug=args.debug,
        )
    )

    robot.connect()
    controller.connect()

    print("\033[36mSpaceMouse 仿真遥操作已启动。Ctrl+C 或 SpaceMouse stop 按钮退出。\033[0m")
    if args.require_enable_button:
        print("\033[36m按住 enable 按钮时才会更新仿真臂位姿。\033[0m")

    next_tick = time.monotonic()
    last_print_time = 0.0
    step = 0

    try:
        while args.steps <= 0 or step < args.steps:
            action, buttons = controller.read()
            if buttons["stop"]:
                print("\033[33m收到 SpaceMouse stop 按钮，停止仿真。\033[0m")
                break

            robot.step_delta_action(action, dt=dt)
            state = robot.get_state()

            now = time.monotonic()
            if now - last_print_time >= args.print_period:
                _print_status(step, state, action, buttons)
                last_print_time = now

            step += 1
            next_tick += dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\033[33m用户中断，停止仿真。\033[0m")
    finally:
        robot.stop()
        controller.disconnect()


def _print_status(step: int, state: np.ndarray, action: np.ndarray, buttons: dict[str, bool]) -> None:
    pos = state[:3]
    rpy = state[3:6]
    action_xyz = action[:3]
    gripper = float(action[-1])
    print(
        "\r\033[2K"
        f"step={step:06d} "
        f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
        f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f}) "
        f"dxyz=({action_xyz[0]:+.4f},{action_xyz[1]:+.4f},{action_xyz[2]:+.4f}) "
        f"g={gripper:+.0f}",
        end="",
        flush=True,
    )


class _MockSpaceMouseController:
    def __init__(self, max_delta_m: float, max_delta_rad: float) -> None:
        self.max_delta_m = max_delta_m
        self.max_delta_rad = max_delta_rad
        self._step = 0

    def connect(self) -> None:
        print("\033[33m使用 mock SpaceMouse 输入，仅用于脚本自检。\033[0m")

    def disconnect(self) -> None:
        return

    def read(self) -> tuple[np.ndarray, dict[str, bool]]:
        phase = self._step * 0.08
        self._step += 1
        action = np.array(
            [
                math.sin(phase) * self.max_delta_m * 0.4,
                math.cos(phase * 0.7) * self.max_delta_m * 0.2,
                math.sin(phase * 0.5) * self.max_delta_m * 0.2,
                0.0,
                0.0,
                math.sin(phase * 0.4) * self.max_delta_rad * 0.3,
                -1.0,
            ],
            dtype=np.float32,
        )
        return action, {
            "enable": True,
            "start": True,
            "success": False,
            "abort": False,
            "stop": False,
        }


if __name__ == "__main__":
    main()
