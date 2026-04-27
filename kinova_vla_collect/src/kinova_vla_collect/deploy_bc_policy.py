from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from kinova_vla_collect.bc_policy_client import KinovaBCRemoteClient
from kinova_vla_collect.config import AppConfig, load_config
from kinova_vla_collect.kinova_robot import KinovaRobot
from kinova_vla_collect.modbus_gripper import ModbusGripper
from kinova_vla_collect.realsense_camera import RealSenseCamera
from kinova_vla_collect.utils.safety import SafetyLimiter, WorkspaceLimits

FloatArray = NDArray[np.float32]
ImageArray = NDArray[np.uint8]


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


class KinovaBCDeploymentRunner:
    def __init__(
        self,
        config: AppConfig,
        server_url: str,
        task_prompt: str | None = None,
        hardware_dry_run: bool | None = None,
        policy_dry_run: bool = False,
        hz: float = 3.0,
        max_steps: int = 30,
        max_delta_m: float = 0.003,
        xyz_scale: float = 1.0,
        open_gripper_on_start: bool = True,
        startup_open_s: float = 1.0,
        policy_timeout_s: float = 10.0,
        log_dir: Path = Path("outputs/deploy_runs"),
        save_image_every: int = 10,
        camera_color_order: str = "rgb",
    ) -> None:
        if hz <= 0.0:
            raise ValueError("hz must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_delta_m <= 0.0:
            raise ValueError("max_delta_m must be positive")
        self.config = config
        self.server_url = server_url
        self.task_prompt = task_prompt or config.task.prompt
        self.use_dry_run = config.hardware.dry_run if hardware_dry_run is None else hardware_dry_run
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.max_steps = int(max_steps)
        self.max_delta_m = float(max_delta_m)
        self.xyz_scale = float(xyz_scale)
        self.open_gripper_on_start = open_gripper_on_start
        self.startup_open_s = max(0.0, float(startup_open_s))
        self.save_image_every = max(0, int(save_image_every))
        self.camera_color_order = camera_color_order.lower()
        if self.camera_color_order not in {"rgb", "bgr"}:
            raise ValueError("camera_color_order must be 'rgb' or 'bgr'")
        self.run_dir = log_dir / datetime.now().strftime("%Y%m%d_%H%M%S_bc_deploy")
        self.jsonl_path = self.run_dir / "steps.jsonl"

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
        self.policy = KinovaBCRemoteClient(
            base_url=server_url,
            timeout_s=policy_timeout_s,
            dry_run=policy_dry_run,
        )
        workspace = config.control.workspace
        self.safety = SafetyLimiter(
            max_delta_m=self.max_delta_m,
            workspace=WorkspaceLimits(
                x_min=workspace.x_min,
                x_max=workspace.x_max,
                y_min=workspace.y_min,
                y_max=workspace.y_max,
                z_min=workspace.z_min,
                z_max=workspace.z_max,
            ),
        )

    def check_server(self) -> None:
        print(f"Checking BC policy server: {self.server_url}")
        print("healthz:", json.dumps(self.policy.healthz(), ensure_ascii=False))
        print("metadata:", json.dumps(self.policy.metadata(), ensure_ascii=False))

    def smoke_test(self) -> None:
        self._connect()
        try:
            image, state = self._observe()
            result = self.policy.act(image, state, self.task_prompt)
            raw_action = KinovaBCRemoteClient.parse_action(result)
            safe_action = self._make_safe_action(raw_action, state)
            self._save_smoke_artifacts(image, state, result, safe_action)
            print("Smoke test OK.")
            print(f"  state shape: {state.shape}")
            print(f"  image shape: {image.shape}")
            print(f"  raw action:  {raw_action.tolist()}")
            print(f"  safe action: {safe_action.tolist()}")
            print(f"  log dir: {self.run_dir}")
        finally:
            self._shutdown()

    def run_closed_loop(self) -> None:
        self._connect()
        self._prepare_gripper_for_grasp()
        self.check_server()
        self.policy.reset()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config()
        print(
            f"BC deployment running: hz={self.hz:.2f}, max_steps={self.max_steps}, "
            f"max_delta_m={self.max_delta_m:.4f}, hardware_dry_run={self.use_dry_run}. Press q to stop."
        )
        print(f"Task prompt sent with each /act request: {self.task_prompt!r}")
        next_tick = time.monotonic()
        try:
            with KeyboardStopper() as keyboard:
                for step_index in range(self.max_steps):
                    if keyboard.should_stop():
                        print("Stop requested by keyboard.")
                        break
                    loop_start = time.monotonic()
                    image, state = self._observe()
                    result = self.policy.act(image, state, self.task_prompt)
                    raw_action = KinovaBCRemoteClient.parse_action(result)
                    safe_action = self._make_safe_action(raw_action, state)
                    self.robot.step_delta_action(safe_action, self.dt)
                    gripper_command = self.gripper.apply_action(float(safe_action[-1]))
                    elapsed_ms = (time.monotonic() - loop_start) * 1000.0
                    self._log_step(
                        step_index=step_index,
                        state=state,
                        raw_action=raw_action,
                        safe_action=safe_action,
                        response=result,
                        elapsed_ms=elapsed_ms,
                        gripper_name=gripper_command.name,
                        image=image,
                    )
                    if step_index % max(1, int(self.hz)) == 0:
                        print(
                            f"step={step_index:04d}/{self.max_steps} "
                            f"raw={raw_action.tolist()} safe={safe_action.tolist()} "
                            f"ee_z={float(state[2]):.4f} "
                            f"gripper={gripper_command.name} loop_ms={elapsed_ms:.1f}"
                        )
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
            print(f"Deployment log: {self.run_dir}")

    def _connect(self) -> None:
        self.camera.start()
        self.robot.connect()
        self.gripper.connect()

    def _observe(self) -> tuple[ImageArray, FloatArray]:
        image = self.camera.get_rgb()
        if self.camera_color_order == "bgr":
            image = image[..., ::-1].copy()
        state = self.robot.get_state().astype(np.float32)
        state = state.copy()
        state[6] = self.gripper.get_position()
        if state.shape != (14,):
            raise RuntimeError(f"Robot state must have shape (14,), got {state.shape}")
        return image, state

    def _make_safe_action(self, raw_action: FloatArray, state: FloatArray) -> FloatArray:
        action = np.asarray(raw_action, dtype=np.float32).copy()
        action[:3] *= self.xyz_scale
        return self.safety.limit_action(action, current_position=state[:3])

    def _write_run_config(self) -> None:
        summary = {
            "server_url": self.server_url,
            "task_prompt": self.task_prompt,
            "hardware_dry_run": self.use_dry_run,
            "hz": self.hz,
            "max_steps": self.max_steps,
            "max_delta_m": self.max_delta_m,
            "xyz_scale": self.xyz_scale,
            "open_gripper_on_start": self.open_gripper_on_start,
            "startup_open_s": self.startup_open_s,
            "camera_color_order": self.camera_color_order,
            "action_definition": "[dx, dy, dz, droll, dpitch, dyaw, gripper]",
            "config_workspace": self.config.control.workspace.__dict__,
        }
        with (self.run_dir / "run_config.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

    def _save_smoke_artifacts(
        self,
        image: ImageArray,
        state: FloatArray,
        result: dict[str, Any],
        safe_action: FloatArray,
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(self.run_dir / "smoke_wrist.jpg", quality=95)
        artifact = {
            "state": state.astype(float).tolist(),
            "response": result,
            "safe_action": safe_action.astype(float).tolist(),
        }
        with (self.run_dir / "smoke_result.json").open("w", encoding="utf-8") as file:
            json.dump(artifact, file, indent=2, ensure_ascii=False)

    def _prepare_gripper_for_grasp(self) -> None:
        if not self.open_gripper_on_start:
            return
        print(f"Opening gripper before grasp for {self.startup_open_s:.1f}s...")
        self.gripper.open_gripper()
        if self.startup_open_s > 0.0:
            time.sleep(self.startup_open_s)
        self.gripper.hold()

    def _log_step(
        self,
        step_index: int,
        state: FloatArray,
        raw_action: FloatArray,
        safe_action: FloatArray,
        response: dict[str, Any],
        elapsed_ms: float,
        gripper_name: str,
        image: ImageArray,
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        image_path = None
        if self.save_image_every > 0 and step_index % self.save_image_every == 0:
            image_dir = self.run_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{step_index:06d}.jpg"
            Image.fromarray(image, mode="RGB").save(image_path, quality=90)
        record = {
            "step": step_index,
            "time": time.time(),
            "state": state.astype(float).tolist(),
            "raw_action": raw_action.astype(float).tolist(),
            "safe_action": safe_action.astype(float).tolist(),
            "gripper_command": gripper_name,
            "elapsed_ms": elapsed_ms,
            "policy_timing_ms": response.get("timing_ms"),
            "gripper_prob": response.get("gripper_prob"),
            "gripper_value": response.get("gripper_value"),
            "gripper_label": response.get("gripper_label"),
            "gripper_probs": response.get("gripper_probs"),
            "image": str(image_path) if image_path is not None else None,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

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


def _confirm_real_run(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    print("Real hardware deployment requested. Keep one hand near E-stop and verify the initial pose.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deploy the Kinova BC baseline on the real control PC.")
    parser.add_argument("--config", type=Path, default=Path("configs/collect_pick_red_block.yaml"))
    parser.add_argument("--server-url", type=str, required=True, help="Base URL, for example http://GPU_IP:8001")
    parser.add_argument("--task-prompt", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use fake camera/robot/gripper.")
    parser.add_argument("--real", action="store_true", help="Force real camera/robot/gripper.")
    parser.add_argument("--policy-dry-run", action="store_true", help="Do not call the GPU policy server.")
    parser.add_argument("--check-server", action="store_true", help="Only call /healthz and /metadata.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one observation -> policy action, without motion.")
    parser.add_argument("--hz", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-delta-m", type=float, default=0.003)
    parser.add_argument("--xyz-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-open-gripper-on-start",
        action="store_true",
        help="Do not send an initial open command before closed-loop grasp execution.",
    )
    parser.add_argument(
        "--startup-open-s",
        type=float,
        default=1.0,
        help="Seconds to keep the initial gripper open command active before policy execution.",
    )
    parser.add_argument("--policy-timeout-s", type=float, default=10.0)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/deploy_runs"))
    parser.add_argument("--save-image-every", type=int, default=10)
    parser.add_argument(
        "--camera-color-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="Use bgr when the camera source is an OpenCV BGR frame; it will be converted to RGB before /act.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.dry_run and args.real:
        raise ValueError("Use at most one of --dry-run or --real")

    config = load_config(args.config)
    hardware_dry_run = True if args.dry_run else False if args.real else None
    runner = KinovaBCDeploymentRunner(
        config=config,
        server_url=args.server_url,
        task_prompt=args.task_prompt,
        hardware_dry_run=hardware_dry_run,
        policy_dry_run=args.policy_dry_run,
        hz=args.hz,
        max_steps=args.max_steps,
        max_delta_m=args.max_delta_m,
        xyz_scale=args.xyz_scale,
        open_gripper_on_start=not args.no_open_gripper_on_start,
        startup_open_s=args.startup_open_s,
        policy_timeout_s=args.policy_timeout_s,
        log_dir=args.log_dir,
        save_image_every=args.save_image_every,
        camera_color_order=args.camera_color_order,
    )

    if args.check_server:
        runner.check_server()
        return
    if args.smoke_test:
        runner.smoke_test()
        return

    _confirm_real_run(args)
    runner.run_closed_loop()


if __name__ == "__main__":
    main()
