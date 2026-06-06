from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kinova_vla_collect.config import AppConfig, load_config
from kinova_vla_collect.kinova_robot import KinovaRobot
from kinova_vla_collect.modbus_gripper import ModbusGripper
from kinova_vla_collect.realsense_camera import RealSenseCamera
from kinova_vla_collect.recorder import EpisodeRecorder
from kinova_vla_collect.spacemouse_controller import (
    SpaceMouseController,
    SpaceMouseMapping,
    SpaceMouseSigns,
)
from kinova_vla_collect.teleop_spacemouse_real import (
    DirectTwistPublisher,
    _action_to_twist,
    _decouple_spacemouse_groups,
    _spacemouse_to_xbox_action,
)
from kinova_vla_collect.utils.safety import SafetyLimiter, WorkspaceLimits


class CollectorMode(Enum):
    NOT_RECORDING = auto()
    RECORDING = auto()


@dataclass
class LoopStats:
    last_print_time: float = 0.0
    last_state_time: float = 0.0
    last_record_time: float = 0.0
    last_loop_time: float = 0.0
    fps: float = 0.0


class KeyboardCommands:
    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def drain(self) -> list[str]:
        commands: list[str] = []
        while True:
            try:
                commands.append(self._queue.get_nowait())
            except queue.Empty:
                return commands

    def _read_loop(self) -> None:
        while self._running:
            line = sys.stdin.readline()
            if line == "":
                return
            command = line.strip().lower()
            if command:
                self._queue.put(command)


class SpaceMouseDatasetCollector:
    def __init__(self, config: AppConfig, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.dry_run = config.hardware.dry_run
        self.teleop_dt = 1.0 / args.teleop_hz
        self.record_dt = 1.0 / config.control.hz
        self.mode = CollectorMode.NOT_RECORDING
        self.episode_index = self._next_episode_index()
        self.current_episode_steps = 0
        self.running = False
        self.stats = LoopStats()
        self.latest_state = np.zeros(14, dtype=np.float32)
        self.last_gripper_target = -1.0
        self.last_gripper_name = "open"
        self.button1_down = False
        self.button1_down_time = 0.0
        self.button1_long_handled = False

        self.camera = RealSenseCamera(
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            serial=config.camera.serial,
            dry_run=self.dry_run,
        )
        self.robot_state = KinovaRobot(
            ip=config.kinova.ip,
            username=config.kinova.username,
            password=config.kinova.password,
            dry_run=self.dry_run,
            max_linear_speed=args.max_linear_speed,
            mode=config.kinova.mode,
            joint_state_topic=config.kinova.joint_state_topic,
            twist_command_topic=config.kinova.twist_command_topic,
            base_frame=config.kinova.base_frame,
            ee_frame=config.kinova.ee_frame,
            twist_command_frame=config.kinova.twist_command_frame,
            state_timeout_s=config.kinova.state_timeout_s,
            twist_publish_rate_hz=args.teleop_hz,
            twist_stop_duration_s=config.kinova.twist_stop_duration_s,
            enable_motion_commands=False,
        )
        self.twist = DirectTwistPublisher(config.kinova.twist_command_topic)
        self.gripper = ModbusGripper(
            host=config.gripper.host,
            port=config.gripper.port,
            unit_id=config.gripper.unit_id,
            dry_run=self.dry_run,
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
        self.spacemouse = SpaceMouseController(
            device=args.device,
            device_index=args.device_index,
            device_path=args.device_path,
            deadzone=args.deadzone,
            max_delta_m=args.max_delta_m,
            max_delta_rad=args.max_delta_rad,
            require_enable_button=args.require_enable_button,
            mapping=SpaceMouseMapping(
                signs=SpaceMouseSigns(dx=1.0, dy=1.0, dz=1.0, droll=1.0, dpitch=1.0, dyaw=1.0)
            ),
            debug=args.debug,
            calibrate_on_connect=not args.no_calibrate,
            calibration_duration_s=args.calibration_duration,
        )
        self.teleop_args = SimpleNamespace(
            max_delta_m=args.max_delta_m,
            max_delta_rad=args.max_delta_rad,
            max_linear_speed=args.max_linear_speed,
            max_angular_speed=args.max_angular_speed,
            linear_scale=args.linear_scale,
            angular_scale=args.angular_scale,
            sign_x=args.sign_x,
            sign_y=args.sign_y,
            sign_z=args.sign_z,
            sign_roll=args.sign_roll,
            sign_pitch=args.sign_pitch,
            sign_yaw=args.sign_yaw,
            control_layout=args.control_layout,
            allow_mixed_motion=args.allow_mixed_motion,
            translation_only=False,
            rotation_only=False,
            dominance_ratio=args.dominance_ratio,
        )
        self.safety = SafetyLimiter(
            max_delta_m=config.control.max_delta_m,
            max_delta_rad=config.control.max_delta_rad,
            workspace=WorkspaceLimits(
                x_min=config.control.workspace.x_min,
                x_max=config.control.workspace.x_max,
                y_min=config.control.workspace.y_min,
                y_max=config.control.workspace.y_max,
                z_min=config.control.workspace.z_min,
                z_max=config.control.workspace.z_max,
            ),
        )
        self.recorder = EpisodeRecorder(
            dataset_root=config.dataset.root,
            task_name=config.task.name,
            task_prompt=config.task.prompt,
            robot_name=config.dataset.robot,
            camera_name=config.dataset.camera,
            control_hz=config.control.hz,
            action_dim=7,
            action_space="delta_ee_pose_rpy_with_gripper_spacemouse",
        )
        self.keyboard = KeyboardCommands()

    def run(self) -> None:
        self._connect()
        self.running = True
        self.keyboard.start()
        self._print_help()

        next_tick = time.monotonic()
        last_twist = np.zeros(6, dtype=np.float32)
        try:
            while self.running:
                loop_start = time.monotonic()
                last_twist = self._run_one_step(last_twist, loop_start)
                self._update_fps(loop_start)
                next_tick += self.teleop_dt
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
        except KeyboardInterrupt:
            print(_color("\n用户中断，正在保存/清理。", "yellow"))
        except Exception:
            self._emergency_cleanup()
            raise
        finally:
            self._shutdown()

    def _run_one_step(self, last_twist: np.ndarray, now: float) -> np.ndarray:
        self._handle_keyboard_commands()
        raw_action, buttons = self.spacemouse.read()
        self._handle_button1(buttons["stop"], now)
        if not self.running:
            return np.zeros(6, dtype=np.float32)
        if buttons["stop"]:
            twist = np.zeros(6, dtype=np.float32)
            self.twist.publish(twist)
            self._print_status(np.zeros(7, dtype=np.float32), twist)
            return twist

        self._maybe_send_gripper(float(raw_action[6]))

        if now - self.stats.last_state_time >= self.args.state_update_period:
            self.latest_state = self.robot_state.get_state()
            self.stats.last_state_time = now

        filtered_raw = _decouple_spacemouse_groups(raw_action, self.teleop_args)
        action = _spacemouse_to_xbox_action(filtered_raw, self.teleop_args)
        action[-1] = self.last_gripper_target

        if self.mode is CollectorMode.RECORDING or self.config.control.allow_motion_when_not_recording:
            safe_action = self.safety.limit_action(action, self.latest_state[:3])
            twist = _action_to_twist(safe_action, self.teleop_dt, self.teleop_args)
            self.twist.publish(twist)
        else:
            safe_action = action.copy()
            safe_action[:6] = 0.0
            twist = np.zeros(6, dtype=np.float32)
            self.twist.publish(twist)

        if self.mode is CollectorMode.RECORDING and now - self.stats.last_record_time >= self.record_dt:
            self._record_frame(twist)
            self.stats.last_record_time = now

        self._print_status(safe_action, twist)
        return twist if np.any(twist) else last_twist

    def _handle_keyboard_commands(self) -> None:
        for command in self.keyboard.drain():
            if command in {"h", "help", "帮助"}:
                self._print_help()
            elif command in {"s", "start", "开始"}:
                self._start_episode()
            elif command in {"y", "success", "成功"}:
                self._finish_episode(success=True, reason="operator_success")
            elif command in {"n", "fail", "failure", "失败"}:
                self._finish_episode(success=False, reason="operator_failure")
            elif command in {"d", "discard", "丢弃"}:
                self._discard_episode()
            elif command in {"q", "quit", "exit", "退出"}:
                self.running = False
            else:
                print(_color(f"未知命令: {command!r}，输入 h 查看帮助。", "yellow"))

    def _handle_button1(self, is_down: bool, now: float) -> None:
        if is_down and not self.button1_down:
            self.button1_down = True
            self.button1_down_time = now
            self.button1_long_handled = False
            return

        if is_down and self.button1_down:
            held_s = now - self.button1_down_time
            if not self.button1_long_handled and held_s >= self.args.button1_hold_exit_s:
                self.button1_long_handled = True
                print(_color("\nSpaceMouse button1 长按，停止并退出。", "yellow"))
                self.running = False
            return

        if not is_down and self.button1_down:
            held_s = now - self.button1_down_time
            was_long = self.button1_long_handled
            self.button1_down = False
            self.button1_down_time = 0.0
            self.button1_long_handled = False
            if was_long or held_s < self.args.button1_debounce_s:
                return
            self._toggle_episode_success()

    def _toggle_episode_success(self) -> None:
        if self.mode is CollectorMode.NOT_RECORDING:
            self._start_episode()
        else:
            self._finish_episode(success=True, reason="spacemouse_button1_success")

    def _start_episode(self) -> None:
        if self.mode is CollectorMode.RECORDING:
            print(_color("当前已经在录制，先 y/n/d 结束当前 episode。", "yellow"))
            return
        self.spacemouse.reset_gripper_target(-1.0)
        self.last_gripper_target = -1.0
        self.last_gripper_name = "open"
        if self.args.open_gripper_on_start:
            self.gripper.open_gripper()
            time.sleep(min(1.0, max(0.0, self.config.gripper.open_timeout_s)))
            self.gripper.hold()
        self.episode_index = self._next_episode_index()
        episode_dir = self.recorder.start_episode(self.episode_index)
        self.current_episode_steps = 0
        self.stats.last_record_time = 0.0
        self.mode = CollectorMode.RECORDING
        print(_color(f"开始录制 episode_{self.episode_index:06d}: {episode_dir}", "green"))

    def _finish_episode(self, *, success: bool, reason: str) -> None:
        if self.mode is not CollectorMode.RECORDING:
            print(_color("当前没有正在录制的 episode。", "yellow"))
            return
        shard = self.recorder.save_episode(
            success=success,
            extra_meta={
                "episode_index": self.episode_index,
                "end_reason": reason,
                "collector": "spacemouse",
                "teleop_hz": self.args.teleop_hz,
                "record_hz": self.config.control.hz,
            },
        )
        label = "成功" if success else "失败"
        print(_color(f"已保存{label} episode_{self.episode_index:06d}: {shard}", "green" if success else "red"))
        self.mode = CollectorMode.NOT_RECORDING
        self.current_episode_steps = 0
        self.episode_index = self._next_episode_index()

    def _discard_episode(self) -> None:
        if self.mode is not CollectorMode.RECORDING:
            print(_color("当前没有正在录制的 episode。", "yellow"))
            return
        self.recorder.discard_episode()
        print(_color(f"已丢弃 episode_{self.episode_index:06d}", "yellow"))
        self.mode = CollectorMode.NOT_RECORDING
        self.current_episode_steps = 0
        self.episode_index = self._next_episode_index()

    def _record_frame(self, twist: np.ndarray) -> None:
        image = self.camera.get_rgb()
        state = self.robot_state.get_state().copy()
        state[6] = self.gripper.get_position()
        action = _twist_to_record_action(
            twist=twist,
            record_dt=self.record_dt,
            gripper_target=self.last_gripper_target,
            config=self.config,
        )
        self.recorder.append(image, state, action)
        self.current_episode_steps += 1
        if self.current_episode_steps >= self.config.control.max_steps:
            self._finish_episode(success=False, reason="max_steps_exceeded")

    def _maybe_send_gripper(self, gripper_target: float) -> None:
        gripper_target = 1.0 if float(gripper_target) > 0.0 else -1.0
        if gripper_target == self.last_gripper_target:
            return
        command = self.gripper.apply_action(gripper_target)
        self.last_gripper_target = gripper_target
        self.last_gripper_name = command.name
        print(_color(f"\n夹爪命令: {command.name}", "cyan"))

    def _connect(self) -> None:
        self.camera.start()
        self.robot_state.connect()
        self.twist.connect()
        self.gripper.connect()
        self.spacemouse.connect()
        self.latest_state = self.robot_state.get_state()

    def _shutdown(self) -> None:
        if self.mode is CollectorMode.RECORDING:
            try:
                self._finish_episode(success=False, reason="shutdown_while_recording")
            except Exception as exc:
                print(f"Warning: failed to save active episode during shutdown: {exc}")
        try:
            self.twist.stop_for(self.config.kinova.twist_stop_duration_s)
        except Exception as exc:
            print(f"Warning: twist stop failed: {exc}")
        for name, fn in [
            ("spacemouse.disconnect()", self.spacemouse.disconnect),
            ("gripper.hold()", self.gripper.hold),
            ("gripper.disconnect()", self.gripper.disconnect),
            ("camera.stop()", self.camera.stop),
            ("robot.disconnect()", self.robot_state.disconnect),
            ("twist.disconnect()", self.twist.disconnect),
        ]:
            try:
                fn()
            except Exception as exc:
                print(f"Warning: {name} failed: {exc}")
        self._print_dataset_summary()

    def _emergency_cleanup(self) -> None:
        try:
            self.twist.stop_for(self.config.kinova.twist_stop_duration_s)
        except Exception as exc:
            print(f"Emergency cleanup warning: twist stop failed: {exc}")
        try:
            self.gripper.hold()
        except Exception as exc:
            print(f"Emergency cleanup warning: gripper.hold() failed: {exc}")

    def _next_episode_index(self) -> int:
        task_dir = self.config.dataset.root / self.config.task.name
        index = 0
        while (task_dir / "data" / f"episode_{index:06d}.npz").exists() or (
            task_dir / "images" / f"episode_{index:06d}"
        ).exists():
            index += 1
        return index

    def _update_fps(self, loop_start: float) -> None:
        if self.stats.last_loop_time > 0.0:
            period = loop_start - self.stats.last_loop_time
            if period > 1e-6:
                instant_fps = 1.0 / period
                self.stats.fps = instant_fps if self.stats.fps <= 0.0 else 0.8 * self.stats.fps + 0.2 * instant_fps
        self.stats.last_loop_time = loop_start

    def _print_status(self, action: np.ndarray, twist: np.ndarray) -> None:
        now = time.monotonic()
        if now - self.stats.last_print_time < 0.5:
            return
        self.stats.last_print_time = now
        mode_text = "录制中" if self.mode is CollectorMode.RECORDING else "待机"
        color = "green" if self.mode is CollectorMode.RECORDING else "blue"
        print(
            _color(
                f"episode={self.episode_index:06d} 模式={mode_text} "
                f"步数={self.current_episode_steps}/{self.config.control.max_steps} "
                f"action={_format_action(action)} "
                f"v={np.linalg.norm(twist[:3]):.3f}m/s w={np.linalg.norm(twist[3:6]):.2f}rad/s "
                f"gripper={self.last_gripper_name} loop={self.stats.fps:.1f}Hz",
                color,
            )
        )

    def _print_help(self) -> None:
        print(_color("SpaceMouse 采集控制：", "cyan"))
        print("  SpaceMouse button0: 夹爪开/合切换")
        print("  SpaceMouse button1 短按: 开始录制 / 保存成功")
        print("  SpaceMouse button1 长按: 停止并退出")
        print("  备用键盘: n=保存失败, d=丢弃, q=退出, h=帮助")

    def _print_dataset_summary(self) -> None:
        try:
            summary = self.recorder.write_summary()
        except Exception as exc:
            print(f"Warning: failed to write dataset summary: {exc}")
            return
        print(_color("采集 summary", "cyan"))
        print(f"  episode 数量: {summary['episode_count']}")
        print(f"  总帧数: {summary['total_frames']}")
        print(f"  平均 fps: {summary['average_fps']:.3f}")
        print(f"  action min:  {summary['action_min']}")
        print(f"  action max:  {summary['action_max']}")
        print(f"  gripper open frame 数量: {summary['gripper_open_frames']}")
        print(f"  gripper close frame 数量: {summary['gripper_close_frames']}")


def _twist_to_record_action(
    *,
    twist: np.ndarray,
    record_dt: float,
    gripper_target: float,
    config: AppConfig,
) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip(
        twist[:3] * record_dt,
        -config.control.max_delta_m,
        config.control.max_delta_m,
    )
    action[3:6] = np.clip(
        twist[3:6] * record_dt,
        -config.control.max_delta_rad,
        config.control.max_delta_rad,
    )
    action[6] = 1.0 if float(gripper_target) > 0.0 else -1.0
    return action


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect Kinova VLA episodes with SpaceMouse.")
    parser.add_argument("--config", type=Path, default=Path("kinova_vla_collect/configs/collect_pick_red_block.yaml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-path", type=str, default=None)
    parser.add_argument("--teleop-hz", type=float, default=100.0)
    parser.add_argument("--deadzone", type=float, default=0.03)
    parser.add_argument("--calibration-duration", type=float, default=0.8)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--require-enable-button", action="store_true")
    parser.add_argument("--max-delta-m", type=float, default=0.006)
    parser.add_argument("--max-delta-rad", type=float, default=0.05)
    parser.add_argument("--max-linear-speed", type=float, default=0.07)
    parser.add_argument("--max-angular-speed", type=float, default=10.0)
    parser.add_argument("--linear-scale", type=float, default=1.2)
    parser.add_argument("--angular-scale", type=float, default=16.0)
    parser.add_argument("--sign-x", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-y", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-z", type=float, choices=[-1.0, 1.0], default=-1.0)
    parser.add_argument("--sign-roll", type=float, choices=[-1.0, 1.0], default=-1.0)
    parser.add_argument("--sign-pitch", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-yaw", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--control-layout", choices=["normal", "swapped"], default="normal")
    parser.add_argument("--allow-mixed-motion", action="store_true")
    parser.add_argument("--dominance-ratio", type=float, default=1.15)
    parser.add_argument("--state-update-period", type=float, default=0.05)
    parser.add_argument("--open-gripper-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--button1-debounce-s", type=float, default=0.15)
    parser.add_argument("--button1-hold-exit-s", type=float, default=2.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    if args.teleop_hz <= 0.0:
        raise ValueError("--teleop-hz must be positive")
    config = load_config(args.config)
    SpaceMouseDatasetCollector(config, args).run()


def _format_action(action: np.ndarray) -> str:
    labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    return "{" + ", ".join(f"{label}={float(value):+.4f}" for label, value in zip(labels, action)) + "}"


def _color(text: str, color: str) -> str:
    codes = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
    }
    code = codes.get(color)
    if code is None:
        return text
    return f"\033[{code}m{text}\033[0m"


if __name__ == "__main__":
    main()
