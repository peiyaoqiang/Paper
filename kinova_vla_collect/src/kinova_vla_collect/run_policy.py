from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kinova_vla_collect.config import AppConfig, load_config
from kinova_vla_collect.kinova_robot import KinovaRobot
from kinova_vla_collect.modbus_gripper import ModbusGripper
from kinova_vla_collect.policy_client import PolicyClient
from kinova_vla_collect.realsense_camera import RealSenseCamera
from kinova_vla_collect.utils.safety import SafetyLimiter, WorkspaceLimits

FloatArray = NDArray[np.float32]


@dataclass
class KeyboardStopper:
    enabled: bool = True
    _old_settings: list[object] | None = field(default=None, init=False)

    def __enter__(self) -> "KeyboardStopper":
        if self.enabled and sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def should_stop(self) -> bool:
        if not self.enabled or not sys.stdin.isatty():
            return False
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return False
        return sys.stdin.read(1).lower() == "q"


class PolicyRunner:
    def __init__(
        self,
        config: AppConfig,
        server_url: str,
        task_prompt: str | None = None,
        dry_run: bool | None = None,
        max_steps: int | None = None,
        policy_timeout_s: float = 10.0,
    ) -> None:
        self.config = config
        self.task_prompt = task_prompt or config.task.prompt
        self.hz = 5.0
        self.dt = 1.0 / self.hz
        self.max_steps = max_steps
        self.use_dry_run = config.hardware.dry_run if dry_run is None else dry_run
        self.action_queue: list[FloatArray] = []
        self.running = False

        self.camera = RealSenseCamera(
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            serial=config.camera.serial,
            dry_run=self.use_dry_run,
        )
        self.robot = KinovaRobot(
            ip=config.kinova.ip,
            username=config.kinova.username,
            password=config.kinova.password,
            dry_run=self.use_dry_run,
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
        self.gripper = ModbusGripper(
            host=config.gripper.host,
            port=config.gripper.port,
            unit_id=config.gripper.unit_id,
            dry_run=self.use_dry_run,
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
        self.policy = PolicyClient(
            server_url=server_url,
            timeout_s=policy_timeout_s,
            dry_run=self.use_dry_run,
        )
        workspace = config.control.workspace
        self.safety = SafetyLimiter(
            max_delta_m=0.01,
            workspace=WorkspaceLimits(
                x_min=workspace.x_min,
                x_max=workspace.x_max,
                y_min=workspace.y_min,
                y_max=workspace.y_max,
                z_min=workspace.z_min,
                z_max=workspace.z_max,
            ),
        )

    def run(self) -> None:
        self._connect()
        self.running = True
        print("Policy deployment running at 5Hz. Press q to stop. Xbox is not used.")
        next_tick = time.monotonic()
        step_index = 0
        try:
            with KeyboardStopper() as keyboard:
                while self.running:
                    if keyboard.should_stop():
                        print("Stop requested by keyboard.")
                        break
                    if self.max_steps is not None and step_index >= self.max_steps:
                        print(f"Reached max_steps={self.max_steps}.")
                        break

                    image = self.camera.get_rgb()
                    state = self.robot.get_state()
                    action = self._next_action(image=image, state=state)
                    clipped_action = self.safety.limit_action(action, current_position=state[:3])
                    self.robot.step_delta_action(clipped_action, self.dt)
                    self.gripper.apply_action(float(clipped_action[3]))

                    if step_index % 5 == 0:
                        print(
                            f"step={step_index:06d} action={clipped_action.tolist()} "
                            f"queued={len(self.action_queue)}"
                        )
                    step_index += 1
                    next_tick += self.dt
                    sleep_s = next_tick - time.monotonic()
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    else:
                        next_tick = time.monotonic()
        except Exception:
            self._emergency_cleanup()
            raise
        finally:
            self._shutdown()

    def _next_action(self, image: NDArray[np.uint8], state: FloatArray) -> FloatArray:
        if not self.action_queue:
            action_chunk = self.policy.predict_chunk(
                wrist_rgb=image,
                robot_state=state,
                task_prompt=self.task_prompt,
            )
            self.action_queue.extend(action_chunk[index].copy() for index in range(action_chunk.shape[0]))
        return self.action_queue.pop(0)

    def _connect(self) -> None:
        self.camera.start()
        self.robot.connect()
        self.gripper.connect()

    def _emergency_cleanup(self) -> None:
        try:
            self.robot.stop()
        except Exception as exc:
            print(f"Emergency cleanup warning: robot.stop() failed: {exc}")
        try:
            self.gripper.hold()
        except Exception as exc:
            print(f"Emergency cleanup warning: gripper.hold() failed: {exc}")
        try:
            self.camera.stop()
        except Exception as exc:
            print(f"Emergency cleanup warning: camera.stop() failed: {exc}")

    def _shutdown(self) -> None:
        try:
            self.robot.stop()
        except Exception as exc:
            print(f"Warning: robot.stop() failed: {exc}")
        try:
            self.gripper.hold()
        except Exception as exc:
            print(f"Warning: gripper.hold() failed: {exc}")
        try:
            self.gripper.disconnect()
        except Exception as exc:
            print(f"Warning: gripper.disconnect() failed: {exc}")
        try:
            self.camera.stop()
        except Exception as exc:
            print(f"Warning: camera.stop() failed: {exc}")
        try:
            self.robot.disconnect()
        except Exception as exc:
            print(f"Warning: robot.disconnect() failed: {exc}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run OpenPI/VLA policy on Kinova Gen3.")
    parser.add_argument("--config", type=Path, default=Path("configs/collect_pick_red_block.yaml"))
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000/act")
    parser.add_argument("--task-prompt", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run without hardware or server.")
    parser.add_argument("--real", action="store_true", help="Force real hardware and real policy server.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--policy-timeout-s", type=float, default=10.0)
    args = parser.parse_args(argv)

    if args.dry_run and args.real:
        raise ValueError("Use at most one of --dry-run or --real")
    config = load_config(args.config)
    dry_run_override = True if args.dry_run else False if args.real else None
    PolicyRunner(
        config=config,
        server_url=args.server_url,
        task_prompt=args.task_prompt,
        dry_run=dry_run_override,
        max_steps=args.max_steps,
        policy_timeout_s=args.policy_timeout_s,
    ).run()


if __name__ == "__main__":
    main()
