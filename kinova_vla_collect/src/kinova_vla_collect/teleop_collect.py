from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

from kinova_vla_collect.config import AppConfig, load_config
from kinova_vla_collect.kinova_robot import KinovaRobot
from kinova_vla_collect.modbus_gripper import ModbusGripper
from kinova_vla_collect.realsense_camera import RealSenseCamera
from kinova_vla_collect.recorder import EpisodeRecorder
from kinova_vla_collect.utils.safety import SafetyLimiter, WorkspaceLimits
from kinova_vla_collect.xbox_controller import XboxController


class CollectorMode(Enum):
    NOT_RECORDING = auto()
    RECORDING = auto()


@dataclass
class LoopStats:
    last_print_time: float = 0.0
    last_loop_time: float = 0.0
    fps: float = 0.0


class TeleopCollector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        dry_run = config.hardware.dry_run

        self.dt = 1.0 / config.control.hz
        self.mode = CollectorMode.NOT_RECORDING
        self.episode_index = self._next_episode_index()
        self.current_episode_steps = 0
        self.running = False
        self.stats = LoopStats()

        self.camera = RealSenseCamera(
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            serial=config.camera.serial,
            dry_run=dry_run,
        )

        self.robot = KinovaRobot(
            ip=config.kinova.ip,
            username=config.kinova.username,
            password=config.kinova.password,
            dry_run=dry_run,
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
            dry_run=dry_run,
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

        self.xbox = XboxController(
            device_index=config.xbox.device_index,
            deadzone=config.control.deadzone,
            max_delta_m=config.control.max_delta_m,
            max_delta_rad=config.control.max_delta_rad,
            action_dim=config.control.action_dim,
            dry_run=dry_run,
            mapping=config.xbox.mapping,
            debug=config.xbox.debug,
            dry_run_mode=config.xbox.dry_run_mode,  # type: ignore[arg-type]
            gripper_action_mode="persistent_target",
        )

        self.recorder = EpisodeRecorder(
            dataset_root=config.dataset.root,
            task_name=config.task.name,
            task_prompt=config.task.prompt,
            robot_name=config.dataset.robot,
            camera_name=config.dataset.camera,
            control_hz=config.control.hz,
            action_dim=config.control.action_dim,
            action_space=(
                "delta_ee_pose_rpy_with_gripper"
                if config.control.action_dim == 7
                else "delta_ee_position_with_gripper"
            ),
        )

        workspace = config.control.workspace
        self.safety = SafetyLimiter(
            max_delta_m=config.control.max_delta_m,
            max_delta_rad=config.control.max_delta_rad,
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

        print(
            _color(
                "采集器已就绪：Start=开始录制，A=保存成功，B=保存失败，Back=停止程序。",
                "cyan",
            )
        )

        next_tick = time.monotonic()

        try:
            while self.running:
                loop_start = time.monotonic()

                self._run_one_step()
                self._update_fps(loop_start)

                next_tick += self.dt
                sleep_s = next_tick - time.monotonic()

                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()

        except KeyboardInterrupt:
            print(_color("用户中断。", "yellow"))

        except Exception:
            self._emergency_cleanup()
            raise

        finally:
            self._shutdown()

    def _run_one_step(self) -> None:
        try:
            image = self.camera.get_rgb()
            state = self.robot.get_state().copy()
            state[6] = self.gripper.get_position()
            raw_action, buttons = self.xbox.read()
        except Exception as exc:
            self.robot.stop()
            self.gripper.hold()
            print(_color(f"跳过当前帧：图像/state/遥控输入读取失败：{exc}", "yellow"))
            return

        if not self.xbox.input_available:
            self.robot.stop()
            self.gripper.hold()
            print(_color("跳过当前帧：遥控器输入不可用，等待重连。", "yellow"))
            return
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            self.robot.stop()
            self.gripper.hold()
            print(_color("跳过当前帧：wrist RGB 图像无效。", "yellow"))
            return
        if state.shape != (14,) or not np.all(np.isfinite(state)):
            self.robot.stop()
            self.gripper.hold()
            print(_color("跳过当前帧：robot state 无效。", "yellow"))
            return
        if raw_action.shape != (self.config.control.action_dim,) or not np.all(np.isfinite(raw_action)):
            self.robot.stop()
            self.gripper.hold()
            print(_color("跳过当前帧：遥控器 action 无效。", "yellow"))
            return

        action = self.safety.limit_action(raw_action, current_position=state[:3])
        if float(action[-1]) not in {-1.0, 1.0}:
            raise RuntimeError(
                f"Invalid gripper target in action[-1]: {float(action[-1]):+.3f}. "
                "VLA/OpenPI collection requires -1=open target or +1=close target; 0 hold is forbidden."
            )

        if buttons["stop"]:
            self.robot.stop()
            self.running = False
            print(_color("收到停止指令，正在安全停止。", "yellow"))
            self._print_status(action)
            return

        if self.mode is CollectorMode.NOT_RECORDING and buttons["start"]:
            print(_color("开始新 episode：先打开夹爪，并把 gripper target 置为 -1。", "cyan"))
            self.xbox.reset_gripper_target(-1.0)
            self.gripper.open_gripper()
            time.sleep(min(1.0, max(0.0, self.config.gripper.open_timeout_s)))
            self.gripper.hold()

            self.episode_index = self._next_episode_index()
            episode_dir = self.recorder.start_episode(self.episode_index)
            self.current_episode_steps = 0
            self.mode = CollectorMode.RECORDING

            print(_color(f"开始录制 episode_{self.episode_index:06d}: {episode_dir}", "green"))
            return

        if self.mode is CollectorMode.RECORDING or self.config.control.allow_motion_when_not_recording:
            self.robot.step_delta_action(action, self.dt)
            self.gripper.apply_action(float(action[-1]))
        else:
            self.robot.stop()
            self.gripper.hold()

        if self.mode is CollectorMode.RECORDING:
            self.recorder.append(image, state, action)
            self.current_episode_steps += 1

        if self.mode is CollectorMode.RECORDING:
            if buttons["success"]:
                self._save_current_episode(success=True, reason="operator_success")
            elif buttons["abort"]:
                self._save_current_episode(success=False, reason="operator_failure")
            elif self.current_episode_steps >= self.config.control.max_steps:
                self._save_current_episode(success=False, reason="max_steps_exceeded")

        self._print_status(action)

    def _save_current_episode(self, success: bool, reason: str) -> None:
        episode_dir = self.recorder.save_episode(
            success=success,
            extra_meta={
                "episode_index": self.episode_index,
                "end_reason": reason,
                "max_steps": self.config.control.max_steps,
            },
        )

        label = "成功" if success else "失败"
        color = "green" if success else "red"
        print(_color(f"已保存{label} episode_{self.episode_index:06d}: {episode_dir} 原因={reason}", color))

        self.mode = CollectorMode.NOT_RECORDING
        self.current_episode_steps = 0
        self.episode_index = self._next_episode_index()

    def _connect(self) -> None:
        self.camera.start()
        self.robot.connect()
        self.gripper.connect()
        self.xbox.connect()

    def _shutdown(self) -> None:
        if self.recorder.episode_dir is not None:
            try:
                self._save_current_episode(success=False, reason="shutdown_while_recording")
            except Exception as exc:
                print(f"Warning: failed to save active episode during shutdown: {exc}")

        try:
            self.robot.stop()
        except Exception as exc:
            print(f"Warning: robot.stop() failed: {exc}")

        try:
            self.gripper.hold()
        except Exception as exc:
            print(f"Warning: gripper.hold() failed: {exc}")

        try:
            self.xbox.disconnect()
        except Exception as exc:
            print(f"Warning: xbox.disconnect() failed: {exc}")

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

        self._print_dataset_summary()

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

    def _update_fps(self, loop_start: float) -> None:
        if self.stats.last_loop_time > 0.0:
            period = loop_start - self.stats.last_loop_time

            if period > 1e-6:
                instant_fps = 1.0 / period
                self.stats.fps = (
                    instant_fps
                    if self.stats.fps <= 0.0
                    else 0.8 * self.stats.fps + 0.2 * instant_fps
                )

        self.stats.last_loop_time = loop_start

    def _print_status(self, action: object) -> None:
        now = time.monotonic()

        if now - self.stats.last_print_time < 0.5:
            return

        self.stats.last_print_time = now
        recording = self.mode is CollectorMode.RECORDING
        mode_text = "录制中" if recording else "待机"
        mode_color = "green" if recording else "blue"
        status_text = (
            f"episode={self.episode_index:06d} "
            f"模式={mode_text} "
            f"步数={self.current_episode_steps}/{self.config.control.max_steps} "
            f"动作={_format_action(action)} "
            f"夹爪命令={float(action[-1]):+.1f} "  # type: ignore[index]
            f"频率={self.stats.fps:.2f}Hz"
        )

        print(_color(status_text, mode_color))

    def _next_episode_index(self) -> int:
        task_dir = self.config.dataset.root / self.config.task.name
        index = 0

        while (task_dir / "data" / f"episode_{index:06d}.npz").exists() or (
            task_dir / "images" / f"episode_{index:06d}"
        ).exists():
            index += 1

        return index

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
        print(f"  action mean: {summary['action_mean']}")
        print(f"  action std:  {summary['action_std']}")
        print(f"  gripper open frame 数量: {summary['gripper_open_frames']}")
        print(f"  gripper close frame 数量: {summary['gripper_close_frames']}")
        print(f"  每个 episode 的帧数: {summary['episode_frame_counts']}")
        print(f"  是否存在 NaN / inf: {summary['has_nan_or_inf']}")
        print(f"  非 -1/+1 gripper 数量: {summary['invalid_gripper_value_count']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect Kinova VLA episodes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collect_pick_red_block.yaml"),
        help="Path to collection YAML config.",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)

    TeleopCollector(config).run()


def _format_action(action: object) -> object:
    values = getattr(action, "tolist", lambda: action)()
    if not isinstance(values, list):
        return values
    labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"] if len(values) == 7 else [
        "dx",
        "dy",
        "dz",
        "gripper",
    ]
    return "{" + ", ".join(f"{label}={float(value):+.4f}" for label, value in zip(labels, values)) + "}"


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
