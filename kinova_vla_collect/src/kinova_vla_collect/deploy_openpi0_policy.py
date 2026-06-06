from __future__ import annotations

import argparse
import json
import math
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from kinova_vla_collect.config import AppConfig, load_config
from kinova_vla_collect.kinova_robot import KinovaRobot
from kinova_vla_collect.modbus_gripper import ModbusGripper
from kinova_vla_collect.realsense_camera import RealSenseCamera
from kinova_vla_collect.utils.safety import SafetyLimiter, WorkspaceLimits

FloatArray = NDArray[np.float32]
ImageArray = NDArray[np.uint8]
ObservationFormat = Literal["kinova_lerobot", "droid"]
GripperMode = Literal["passthrough", "ignore", "close_only", "open_close"]


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


class OpenPI0WebsocketClient:
    """Small OpenPI-compatible WebSocket client for the local robot laptop.

    The remote GPU server is expected to load the checkpoint and expose the
    normal OpenPI WebSocket policy protocol:
    - server sends metadata once after connection
    - client sends a msgpack-packed observation dict containing NumPy arrays
    - server replies with a dict/array containing an action or action chunk
    """

    def __init__(
        self,
        server_uri: str,
        api_key: str | None = None,
        dry_run: bool = False,
        connect_retry_s: float = 2.0,
        ping_interval_s: float | None = None,
        ping_timeout_s: float | None = None,
        close_timeout_s: float = 10.0,
    ) -> None:
        self.server_uri = _normalize_server_uri(server_uri)
        self.api_key = api_key
        self.dry_run = dry_run
        self.connect_retry_s = connect_retry_s
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self.close_timeout_s = close_timeout_s
        self.metadata: dict[str, Any] = {}
        self._ws: Any | None = None
        self._websockets_client: Any | None = None
        self._packer: Any | None = None
        self._dry_step = 0

    def connect(self) -> dict[str, Any]:
        if self.dry_run:
            self.metadata = {"dry_run": True, "action_dim": 7, "action_horizon": 1}
            return self.metadata
        self._ensure_deps()
        assert self._websockets_client is not None
        assert self._packer is not None
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        while True:
            try:
                connect_kwargs = {
                    "compression": None,
                    "max_size": None,
                    "ping_interval": self.ping_interval_s,
                    "ping_timeout": self.ping_timeout_s,
                    "close_timeout": self.close_timeout_s,
                }
                try:
                    self._ws = self._websockets_client.connect(
                        self.server_uri,
                        **connect_kwargs,
                        additional_headers=headers,
                    )
                except TypeError:
                    self._ws = self._websockets_client.connect(
                        self.server_uri,
                        **connect_kwargs,
                        extra_headers=headers,
                    )
                self.metadata = _unpackb(self._ws.recv())
                if not isinstance(self.metadata, dict):
                    self.metadata = {"metadata": self.metadata}
                return self.metadata
            except ConnectionRefusedError:
                print(f"OpenPI0 server not ready; retrying in {self.connect_retry_s:.1f}s")
                time.sleep(self.connect_retry_s)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.dry_run:
            actions = self._dry_run_actions()
            return {"actions": actions, "dry_run": True}
        if self._ws is None:
            self.connect()
        assert self._ws is not None
        assert self._packer is not None
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI0 server returned text error:\n{response}")
        decoded = _unpackb(response)
        if isinstance(decoded, dict):
            return decoded
        return {"actions": decoded}

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def __enter__(self) -> "OpenPI0WebsocketClient":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _ensure_deps(self) -> None:
        if self._websockets_client is not None:
            return
        try:
            import msgpack
            import websockets.sync.client
        except ImportError as exc:
            raise RuntimeError(
                "OpenPI0 WebSocket mode requires dependencies: pip install websockets msgpack"
            ) from exc
        self._websockets_client = websockets.sync.client
        self._packer = msgpack.Packer(default=_pack_numpy_array)

    def _dry_run_actions(self) -> FloatArray:
        self._dry_step += 1
        phase = self._dry_step / 12.0
        return np.array(
            [
                [
                    0.001 * math.sin(phase),
                    0.001 * math.cos(phase),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0 if self._dry_step < 20 else 0.0 if self._dry_step < 40 else 1.0,
                ]
            ],
            dtype=np.float32,
        )


class OpenPI0DeploymentRunner:
    def __init__(
        self,
        config: AppConfig,
        server_uri: str,
        task_prompt: str | None = None,
        hardware_dry_run: bool | None = None,
        policy_dry_run: bool = False,
        api_key: str | None = None,
        hz: float = 3.0,
        max_steps: int = 30,
        chunk_steps: int = 1,
        max_delta_m: float = 0.003,
        max_delta_rad: float | None = None,
        xyz_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_mode: GripperMode = "passthrough",
        invert_gripper: bool = False,
        min_steps_before_close: int = 5,
        lift_after_close_m: float = 0.08,
        open_gripper_on_start: bool = True,
        startup_open_s: float = 1.0,
        observation_format: ObservationFormat = "kinova_lerobot",
        image_size: int = 224,
        policy_state_mode: Literal["real", "zero"] = "real",
        preview: bool = False,
        preview_scale: float = 0.75,
        save_image_every: int = 10,
        log_dir: Path = Path("outputs/deploy_runs"),
        connect_retry_s: float = 2.0,
        camera_color_order: str = "rgb",
    ) -> None:
        if hz <= 0.0:
            raise ValueError("hz must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if chunk_steps <= 0:
            raise ValueError("chunk_steps must be positive")
        if max_delta_m <= 0.0:
            raise ValueError("max_delta_m must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.config = config
        self.server_uri = _normalize_server_uri(server_uri)
        self.task_prompt = task_prompt or config.task.prompt
        self.use_dry_run = config.hardware.dry_run if hardware_dry_run is None else hardware_dry_run
        self.policy_dry_run = policy_dry_run
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.max_steps = int(max_steps)
        self.chunk_steps = int(chunk_steps)
        self.max_delta_m = float(max_delta_m)
        self.max_delta_rad = float(max_delta_rad if max_delta_rad is not None else config.control.max_delta_rad)
        self.xyz_scale = float(xyz_scale)
        self.rotation_scale = float(rotation_scale)
        self.gripper_mode = gripper_mode
        self.invert_gripper = invert_gripper
        self.min_steps_before_close = max(0, int(min_steps_before_close))
        self.lift_after_close_m = max(0.0, float(lift_after_close_m))
        self.open_gripper_on_start = open_gripper_on_start
        self.startup_open_s = max(0.0, float(startup_open_s))
        self.observation_format = observation_format
        self.image_size = int(image_size)
        self.policy_state_mode = policy_state_mode
        self.preview = preview
        self.preview_scale = max(0.1, float(preview_scale))
        self.save_image_every = max(0, int(save_image_every))
        self.camera_color_order = camera_color_order.lower()
        if self.camera_color_order not in {"rgb", "bgr"}:
            raise ValueError("camera_color_order must be 'rgb' or 'bgr'")

        self.log_dir = log_dir
        self.run_dir = self._make_run_dir("openpi0_deploy")
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
        self.policy = OpenPI0WebsocketClient(
            server_uri=self.server_uri,
            api_key=api_key,
            dry_run=policy_dry_run,
            connect_retry_s=connect_retry_s,
        )
        self.preview_window = ImagePreview(enabled=preview, scale=self.preview_scale)
        workspace = config.control.workspace
        self.safety = SafetyLimiter(
            max_delta_m=self.max_delta_m,
            max_delta_rad=self.max_delta_rad,
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
        print(f"Checking OpenPI0 policy server: {self.server_uri}")
        try:
            metadata = self.policy.connect()
            print("metadata:", json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2))
        finally:
            self.policy.close()

    def smoke_test(self) -> None:
        self._connect_hardware()
        try:
            self.policy.connect()
            image, state = self._observe()
            if self.preview_window.show(image, "smoke-test"):
                print("Preview stop requested.")
            obs = self._make_observation(image, state)
            started = time.monotonic()
            response = self.policy.infer(obs)
            raw_chunk = parse_action_chunk(response)
            raw_action = raw_chunk[0]
            safe_action = self._make_safe_action(raw_action, state)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._save_smoke_artifacts(image, state, obs, response, raw_chunk, safe_action, elapsed_ms)
            print("Smoke test OK.")
            print(f"  state shape: {state.shape}")
            print(f"  image shape: {image.shape}")
            print(f"  observation keys: {sorted(obs.keys())}")
            print(f"  chunk shape: {raw_chunk.shape}")
            print(f"  raw action:  {raw_action.tolist()}")
            print(f"  safe action: {safe_action.tolist()}")
            print(f"  elapsed ms:  {elapsed_ms:.1f}")
            print(f"  log dir: {self.run_dir}")
        finally:
            self._shutdown()

    def run_closed_loop(self) -> None:
        self._connect_hardware()
        self._prepare_gripper_for_grasp()
        self.policy.connect()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config()
        print(
            f"OpenPI0 deployment running: hz={self.hz:.2f}, max_steps={self.max_steps}, "
            f"chunk_steps={self.chunk_steps}, max_delta_m={self.max_delta_m:.4f}, "
            f"hardware_dry_run={self.use_dry_run}. Press q to stop."
        )
        print(f"Task prompt in observation: {self.task_prompt!r}")
        next_tick = time.monotonic()
        step_index = 0
        try:
            with KeyboardStopper() as keyboard:
                while step_index < self.max_steps:
                    if keyboard.should_stop():
                        print("Stop requested by keyboard.")
                        break

                    loop_start = time.monotonic()
                    image, state = self._observe()
                    if self.preview_window.show(image, f"step {step_index}/{self.max_steps}"):
                        print("Stop requested by preview window.")
                        break
                    obs = self._make_observation(image, state)
                    response = self.policy.infer(obs)
                    raw_chunk = parse_action_chunk(response)
                    executed_this_query = 0

                    for chunk_index, raw_action in enumerate(raw_chunk[: self.chunk_steps]):
                        if step_index >= self.max_steps:
                            break
                        current_state = self.robot.get_state().astype(np.float32)
                        current_state = current_state.copy()
                        current_state[6] = self.gripper.get_position()
                        safe_action = self._make_safe_action(raw_action, current_state)
                        self.robot.step_delta_action(safe_action, self.dt)
                        gripper_command = self._apply_gripper_action(safe_action, step_index)
                        elapsed_ms = (time.monotonic() - loop_start) * 1000.0
                        self._log_step(
                            step_index=step_index,
                            chunk_index=chunk_index,
                            state=current_state,
                            raw_action=raw_action,
                            safe_action=safe_action,
                            response=response,
                            elapsed_ms=elapsed_ms,
                            gripper_name=gripper_command,
                            image=image,
                        )
                        if step_index % max(1, int(self.hz)) == 0:
                            print(
                                f"step={step_index:04d}/{self.max_steps} chunk={chunk_index} "
                                f"raw={raw_action.tolist()} safe={safe_action.tolist()} "
                                f"ee_z={float(current_state[2]):.4f} "
                                f"gripper={gripper_command} loop_ms={elapsed_ms:.1f}"
                            )
                        step_index += 1
                        executed_this_query += 1

                        next_tick += self.dt
                        sleep_s = next_tick - time.monotonic()
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)
                        else:
                            next_tick = time.monotonic()

                        if self.gripper_mode == "close_only" and gripper_command == "close":
                            self._lift_after_close()
                            print("close_only grasp rollout finished.")
                            return

                    if executed_this_query == 0:
                        raise RuntimeError("OpenPI0 policy returned no executable actions")
        except Exception:
            self._emergency_cleanup()
            raise
        finally:
            self._shutdown()
            print(f"Deployment log: {self.run_dir}")

    def run_prompt_once(self, task_prompt: str, prepare_gripper: bool = True) -> None:
        previous_prompt = self.task_prompt
        self.task_prompt = task_prompt
        self._reset_run_log("openpi0_deploy")
        if prepare_gripper:
            self._prepare_gripper_for_grasp()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config()
        print(
            f"OpenPI0 rollout: prompt={self.task_prompt!r}, hz={self.hz:.2f}, "
            f"max_steps={self.max_steps}, chunk_steps={self.chunk_steps}. Press q to stop."
        )
        next_tick = time.monotonic()
        step_index = 0
        try:
            with KeyboardStopper() as keyboard:
                while step_index < self.max_steps:
                    if keyboard.should_stop():
                        print("Stop requested by keyboard.")
                        break

                    loop_start = time.monotonic()
                    image, state = self._observe()
                    if self.preview_window.show(image, f"step {step_index}/{self.max_steps}"):
                        print("Stop requested by preview window.")
                        break
                    obs = self._make_observation(image, state)
                    response = self.policy.infer(obs)
                    raw_chunk = parse_action_chunk(response)
                    executed_this_query = 0

                    for chunk_index, raw_action in enumerate(raw_chunk[: self.chunk_steps]):
                        if step_index >= self.max_steps:
                            break
                        current_state = self.robot.get_state().astype(np.float32)
                        current_state = current_state.copy()
                        current_state[6] = self.gripper.get_position()
                        safe_action = self._make_safe_action(raw_action, current_state)
                        self.robot.step_delta_action(safe_action, self.dt)
                        gripper_command = self._apply_gripper_action(safe_action, step_index)
                        elapsed_ms = (time.monotonic() - loop_start) * 1000.0
                        self._log_step(
                            step_index=step_index,
                            chunk_index=chunk_index,
                            state=current_state,
                            raw_action=raw_action,
                            safe_action=safe_action,
                            response=response,
                            elapsed_ms=elapsed_ms,
                            gripper_name=gripper_command,
                            image=image,
                        )
                        if step_index % max(1, int(self.hz)) == 0:
                            print(
                                f"step={step_index:04d}/{self.max_steps} chunk={chunk_index} "
                                f"safe={safe_action.tolist()} ee_z={float(current_state[2]):.4f} "
                                f"gripper={gripper_command} loop_ms={elapsed_ms:.1f}"
                            )
                        step_index += 1
                        executed_this_query += 1

                        next_tick += self.dt
                        sleep_s = next_tick - time.monotonic()
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)
                        else:
                            next_tick = time.monotonic()

                        if self.gripper_mode == "close_only" and gripper_command == "close":
                            self._lift_after_close()
                            print("close_only grasp rollout finished.")
                            return

                    if executed_this_query == 0:
                        raise RuntimeError("OpenPI0 policy returned no executable actions")
        except Exception:
            self._emergency_cleanup()
            raise
        finally:
            self.robot.stop()
            self.task_prompt = previous_prompt
            print(f"Deployment log: {self.run_dir}")

    def reset_to_pose(
        self,
        target_pose: FloatArray,
        hz: float = 10.0,
        timeout_s: float = 45.0,
        position_tolerance_m: float = 0.01,
        rotation_tolerance_rad: float = 0.08,
        max_delta_m: float | None = None,
        max_delta_rad: float | None = None,
        open_gripper: bool = True,
    ) -> None:
        target = np.asarray(target_pose, dtype=np.float32)
        if target.shape != (6,):
            raise ValueError(f"reset target pose must have shape (6,), got {target.shape}")
        if hz <= 0.0:
            raise ValueError("reset hz must be positive")
        if timeout_s <= 0.0:
            raise ValueError("reset timeout must be positive")
        if not self.safety.workspace.contains_position(target[:3]):
            raise ValueError(f"reset target position is outside workspace: {target[:3].tolist()}")

        if open_gripper:
            print("Opening gripper for reset...")
            self.gripper.open_gripper()
            if self.startup_open_s > 0.0:
                time.sleep(self.startup_open_s)
            self.gripper.hold()

        dt = 1.0 / hz
        limiter = SafetyLimiter(
            max_delta_m=float(max_delta_m if max_delta_m is not None else self.max_delta_m),
            max_delta_rad=float(max_delta_rad if max_delta_rad is not None else self.max_delta_rad),
            workspace=self.safety.workspace,
        )
        deadline = time.monotonic() + timeout_s
        step_index = 0
        last_print_s = 0.0
        print(f"Resetting to pose xyz/rpy={target.tolist()} at {hz:.1f} Hz...")
        try:
            while time.monotonic() < deadline:
                state = self.robot.get_state().astype(np.float32)
                position_error = target[:3] - state[:3]
                rotation_error = _wrap_angle_array(target[3:6] - state[3:6])
                position_norm = float(np.linalg.norm(position_error))
                rotation_norm = float(np.linalg.norm(rotation_error))
                if position_norm <= position_tolerance_m and rotation_norm <= rotation_tolerance_rad:
                    self.robot.stop()
                    print(
                        f"Reset reached: pos_err={position_norm:.4f} m, "
                        f"rot_err={rotation_norm:.4f} rad, steps={step_index}"
                    )
                    return

                action = np.zeros(7, dtype=np.float32)
                action[:3] = position_error
                action[3:6] = rotation_error
                safe_action = limiter.limit_action(action, current_position=state[:3])
                if position_norm > position_tolerance_m and np.allclose(safe_action[:3], 0.0):
                    raise RuntimeError(
                        "Reset motion was blocked by workspace limits. "
                        f"current={state[:3].tolist()} target={target[:3].tolist()}"
                    )
                self.robot.step_delta_action(safe_action, dt)

                now = time.monotonic()
                if now - last_print_s >= 1.0:
                    print(
                        f"reset step={step_index:04d} pos_err={position_norm:.4f} m "
                        f"rot_err={rotation_norm:.4f} rad action={safe_action[:6].tolist()}"
                    )
                    last_print_s = now
                step_index += 1
                time.sleep(dt)
        finally:
            self.robot.stop()
            self.gripper.hold()
        raise TimeoutError(f"Timed out resetting to pose after {timeout_s:.1f}s")

    def _connect_hardware(self) -> None:
        self.camera.start()
        self.robot.connect()
        self.gripper.connect()

    def _make_run_dir(self, suffix: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.log_dir / f"{stamp}_{suffix}"
        counter = 1
        while candidate.exists():
            candidate = self.log_dir / f"{stamp}_{counter:02d}_{suffix}"
            counter += 1
        return candidate

    def _reset_run_log(self, suffix: str) -> None:
        self.run_dir = self._make_run_dir(suffix)
        self.jsonl_path = self.run_dir / "steps.jsonl"

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

    def _make_observation(self, image: ImageArray, state: FloatArray) -> dict[str, Any]:
        image_for_policy = resize_with_pad(image, self.image_size, self.image_size)
        policy_state = state if self.policy_state_mode == "real" else np.zeros_like(state)
        if self.observation_format == "kinova_lerobot":
            return {
                "observation.images.wrist": image_for_policy,
                "observation.state": np.asarray(policy_state, dtype=np.float32),
                "task": self.task_prompt,
                "prompt": self.task_prompt,
            }
        joints = np.asarray(policy_state[7:14], dtype=np.float32)
        gripper_position = np.asarray([policy_state[6]], dtype=np.float32)
        return {
            "observation/exterior_image_1_left": image_for_policy,
            "observation/wrist_image_left": image_for_policy,
            "observation/joint_position": joints,
            "observation/gripper_position": gripper_position,
            "prompt": self.task_prompt,
        }

    def _make_safe_action(self, raw_action: FloatArray, state: FloatArray) -> FloatArray:
        action = np.asarray(raw_action[:7], dtype=np.float32).copy()
        action[:3] *= self.xyz_scale
        action[3:6] *= self.rotation_scale
        if self.invert_gripper:
            action[6] *= -1.0
        if self.gripper_mode == "ignore":
            action[6] = 0.0
        return self.safety.limit_action(action, current_position=state[:3])

    def _apply_gripper_action(self, safe_action: FloatArray, step_index: int) -> str:
        if self.gripper_mode == "ignore":
            self.gripper.hold()
            return "ignore"
        value = float(safe_action[6])
        if self.gripper_mode == "close_only":
            if value <= 0.5:
                self.gripper.hold()
                return "hold"
            if step_index + 1 < self.min_steps_before_close:
                self.gripper.hold()
                return "hold_close_blocked"
            self.gripper.close_gripper()
            return "close"
        if self.gripper_mode == "open_close":
            command = self.gripper.apply_action(1.0 if value > 0.5 else -1.0)
            return command.name
        command = self.gripper.apply_action(value)
        return command.name

    def _lift_after_close(self) -> None:
        if self.lift_after_close_m <= 0.0:
            return
        action = np.array([0.0, 0.0, self.lift_after_close_m, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state = self.robot.get_state().astype(np.float32)
        safe = self.safety.limit_action(action, current_position=state[:3])
        self.robot.step_delta_action(safe, self.dt)
        print(f"lift executed with safe dz={float(safe[2]):.4f} m")

    def _prepare_gripper_for_grasp(self) -> None:
        if not self.open_gripper_on_start:
            return
        print(f"Opening gripper before grasp for {self.startup_open_s:.1f}s...")
        self.gripper.open_gripper()
        if self.startup_open_s > 0.0:
            time.sleep(self.startup_open_s)
        self.gripper.hold()

    def _write_run_config(self) -> None:
        summary = {
            "server_uri": self.server_uri,
            "task_prompt": self.task_prompt,
            "hardware_dry_run": self.use_dry_run,
            "policy_dry_run": self.policy_dry_run,
            "hz": self.hz,
            "max_steps": self.max_steps,
            "chunk_steps": self.chunk_steps,
            "max_delta_m": self.max_delta_m,
            "max_delta_rad": self.max_delta_rad,
            "xyz_scale": self.xyz_scale,
            "rotation_scale": self.rotation_scale,
            "gripper_mode": self.gripper_mode,
            "invert_gripper": self.invert_gripper,
            "observation_format": self.observation_format,
            "image_size": self.image_size,
            "policy_state_mode": self.policy_state_mode,
            "preview": self.preview,
            "preview_scale": self.preview_scale,
            "action_definition": "[dx, dy, dz, droll, dpitch, dyaw, gripper]",
            "openpi_metadata": _jsonable(self.policy.metadata),
            "config_workspace": self.config.control.workspace.__dict__,
        }
        with (self.run_dir / "run_config.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

    def _save_smoke_artifacts(
        self,
        image: ImageArray,
        state: FloatArray,
        observation: dict[str, Any],
        response: dict[str, Any],
        raw_chunk: FloatArray,
        safe_action: FloatArray,
        elapsed_ms: float,
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(self.run_dir / "smoke_wrist.jpg", quality=95)
        artifact = {
            "state": state.astype(float).tolist(),
            "observation_keys": sorted(observation.keys()),
            "response": _jsonable(response),
            "raw_chunk": raw_chunk.astype(float).tolist(),
            "safe_action": safe_action.astype(float).tolist(),
            "elapsed_ms": elapsed_ms,
        }
        with (self.run_dir / "smoke_result.json").open("w", encoding="utf-8") as file:
            json.dump(artifact, file, indent=2, ensure_ascii=False)

    def _log_step(
        self,
        step_index: int,
        chunk_index: int,
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
            "chunk": chunk_index,
            "time": time.time(),
            "state": state.astype(float).tolist(),
            "raw_action": raw_action.astype(float).tolist(),
            "safe_action": safe_action.astype(float).tolist(),
            "gripper_command": gripper_name,
            "elapsed_ms": elapsed_ms,
            "policy_timing_ms": _jsonable(response.get("timing_ms")),
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
        try:
            self.policy.close()
        except Exception as exc:
            print(f"Warning: policy.close() failed: {exc}")
        try:
            self.preview_window.close()
        except Exception as exc:
            print(f"Warning: preview_window.close() failed: {exc}")


def parse_action_chunk(policy_response: dict[str, Any] | FloatArray) -> FloatArray:
    candidate: Any
    if isinstance(policy_response, dict):
        if "actions" in policy_response:
            candidate = policy_response["actions"]
        elif "action" in policy_response:
            candidate = policy_response["action"]
        elif "predicted_actions" in policy_response:
            candidate = policy_response["predicted_actions"]
        else:
            raise RuntimeError("OpenPI0 response must contain one of: actions, action, predicted_actions")
    else:
        candidate = policy_response

    actions = np.asarray(candidate, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError(f"OpenPI0 action chunk must have shape (T, >=7), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError(f"OpenPI0 action chunk contains NaN or Inf: {actions}")
    return actions[:, :7].astype(np.float32)


def resize_with_pad(image: ImageArray, height: int, width: int) -> ImageArray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H, W, 3], got {image.shape}")
    image = image.astype(np.uint8, copy=False)
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)

    pil_image = Image.fromarray(image, mode="RGB")
    cur_width, cur_height = pil_image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_width = max(1, int(cur_width / ratio))
    resized_height = max(1, int(cur_height / ratio))
    resized = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    padded = Image.new("RGB", (width, height), 0)
    pad_x = max(0, (width - resized_width) // 2)
    pad_y = max(0, (height - resized_height) // 2)
    padded.paste(resized, (pad_x, pad_y))
    return np.asarray(padded, dtype=np.uint8)


class ImagePreview:
    def __init__(self, enabled: bool, scale: float = 0.75, window_name: str = "OpenPI0 Kinova wrist RGB") -> None:
        self.enabled = enabled
        self.scale = scale
        self.window_name = window_name
        self._cv2: Any | None = None

    def show(self, image_rgb: ImageArray, status: str = "") -> bool:
        if not self.enabled:
            return False
        cv2 = self._ensure_cv2()
        frame_bgr = image_rgb[..., ::-1].copy()
        if status:
            cv2.putText(
                frame_bgr,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if abs(self.scale - 1.0) > 1e-6:
            frame_bgr = cv2.resize(
                frame_bgr,
                None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow(self.window_name, frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        return key in {ord("q"), 27}

    def close(self) -> None:
        if self._cv2 is not None:
            self._cv2.destroyWindow(self.window_name)

    def _ensure_cv2(self) -> Any:
        if self._cv2 is not None:
            return self._cv2
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Image preview requires OpenCV. Install with: pip install opencv-python") from exc
        self._cv2 = cv2
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        return cv2


def _normalize_server_uri(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("ws://") or stripped.startswith("wss://"):
        return stripped
    if stripped.startswith("http://"):
        return "ws://" + stripped[len("http://") :]
    if stripped.startswith("https://"):
        return "wss://" + stripped[len("https://") :]
    return "ws://" + stripped


def _wrap_angle_array(values: FloatArray) -> FloatArray:
    return ((np.asarray(values, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _pack_numpy_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_numpy_array(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _unpackb(payload: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(payload, object_hook=_unpack_numpy_array)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _confirm_real_run(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    print("Real hardware deployment requested. Keep one hand near E-stop and verify the initial pose.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deploy OpenPI0/pi0 on the Kinova real-robot control PC.")
    parser.add_argument("--config", type=Path, default=Path("configs/collect_pick_red_block.yaml"))
    parser.add_argument("--server-uri", type=str, required=True, help="ws://GPU_IP:8000 or http://GPU_IP:8000")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--task-prompt", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use fake camera/robot/gripper.")
    parser.add_argument("--real", action="store_true", help="Force real camera/robot/gripper.")
    parser.add_argument("--policy-dry-run", action="store_true", help="Do not call the GPU policy server.")
    parser.add_argument("--check-server", action="store_true", help="Only connect and print OpenPI metadata.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one observation -> policy action, without motion.")
    parser.add_argument("--hz", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--chunk-steps", type=int, default=1)
    parser.add_argument("--max-delta-m", type=float, default=0.003)
    parser.add_argument("--max-delta-rad", type=float, default=None)
    parser.add_argument("--xyz-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument(
        "--observation-format",
        choices=("kinova_lerobot", "droid"),
        default="kinova_lerobot",
        help="Use kinova_lerobot for this repo's converted dataset keys.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--policy-state-mode", choices=("real", "zero"), default="real")
    parser.add_argument("--preview", action="store_true", help="Show live wrist RGB in an OpenCV window.")
    parser.add_argument("--preview-scale", type=float, default=0.75)
    parser.add_argument(
        "--gripper-mode",
        choices=("passthrough", "ignore", "close_only", "open_close"),
        default="passthrough",
    )
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--min-steps-before-close", type=int, default=5)
    parser.add_argument("--lift-after-close-m", type=float, default=0.08)
    parser.add_argument(
        "--no-open-gripper-on-start",
        action="store_true",
        help="Do not send an initial open command before closed-loop grasp execution.",
    )
    parser.add_argument("--startup-open-s", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/deploy_runs"))
    parser.add_argument("--save-image-every", type=int, default=10)
    parser.add_argument("--connect-retry-s", type=float, default=2.0)
    parser.add_argument("--camera-color-order", choices=("rgb", "bgr"), default="rgb")
    args = parser.parse_args(argv)

    if args.dry_run and args.real:
        raise ValueError("Use at most one of --dry-run or --real")

    config = load_config(args.config)
    hardware_dry_run = True if args.dry_run else False if args.real else None
    runner = OpenPI0DeploymentRunner(
        config=config,
        server_uri=args.server_uri,
        task_prompt=args.task_prompt,
        hardware_dry_run=hardware_dry_run,
        policy_dry_run=args.policy_dry_run,
        api_key=args.api_key,
        hz=args.hz,
        max_steps=args.max_steps,
        chunk_steps=args.chunk_steps,
        max_delta_m=args.max_delta_m,
        max_delta_rad=args.max_delta_rad,
        xyz_scale=args.xyz_scale,
        rotation_scale=args.rotation_scale,
        gripper_mode=args.gripper_mode,
        invert_gripper=args.invert_gripper,
        min_steps_before_close=args.min_steps_before_close,
        lift_after_close_m=args.lift_after_close_m,
        open_gripper_on_start=not args.no_open_gripper_on_start,
        startup_open_s=args.startup_open_s,
        observation_format=args.observation_format,
        image_size=args.image_size,
        policy_state_mode=args.policy_state_mode,
        preview=args.preview,
        preview_scale=args.preview_scale,
        save_image_every=args.save_image_every,
        log_dir=args.log_dir,
        connect_retry_s=args.connect_retry_s,
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
