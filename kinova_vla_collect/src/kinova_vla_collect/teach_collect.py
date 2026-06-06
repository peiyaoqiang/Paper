from __future__ import annotations

import argparse
import math
import select
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from kinova_vla_collect.config import AppConfig, load_config
from kinova_vla_collect.kinova_robot import KinovaRobot
from kinova_vla_collect.modbus_gripper import ModbusGripper
from kinova_vla_collect.realsense_camera import RealSenseCamera
from kinova_vla_collect.recorder import EpisodeRecorder


class AutoBringupManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.started_by_this_process = False
        self._output_handle: Any | None = None

    def start(self) -> None:
        if self.config.hardware.dry_run or not self.config.teach.bringup.enabled:
            return

        if self._joint_states_available():
            print(_color("检测到已有 /joint_states，跳过自动启动 Kortex bringup。", "cyan"))
            self.deactivate_motion_controllers()
            return

        bringup = self.config.teach.bringup
        cmd = [
            "ros2",
            "launch",
            bringup.launch_package,
            bringup.launch_file,
            f"robot_ip:={bringup.robot_ip}",
            f"dof:={bringup.dof}",
            f"use_internal_bus_gripper_comm:={str(bringup.use_internal_bus_gripper_comm).lower()}",
            f"robot_controller:={bringup.robot_controller}",
            f"robot_pos_controller:={bringup.robot_pos_controller}",
            f"launch_rviz:={str(bringup.launch_rviz).lower()}",
        ]

        stdout: Any = None
        stderr: Any = None
        if not bringup.log_output:
            stdout = subprocess.DEVNULL
            stderr = subprocess.STDOUT

        print(_color("自动启动 Kortex bringup，用于发布 /joint_states 和 TF。", "cyan"))
        self.process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        self.started_by_this_process = True
        self._wait_for_controller_manager()
        self.deactivate_motion_controllers()

    def deactivate_motion_controllers(self) -> None:
        bringup = self.config.teach.bringup
        if self.config.hardware.dry_run or not bringup.enabled:
            return
        controllers = [name for name in bringup.deactivate_controllers if name]
        if not controllers:
            return

        deadline = time.monotonic() + bringup.startup_timeout_s
        last_error = ""
        last_output = ""
        while time.monotonic() < deadline:
            output = self._list_controllers()
            if output is not None:
                last_output = output.strip()
                active_targets = [
                    name
                    for name in controllers
                    if _controller_state_from_list_output(output, name) == "active"
                ]
                loaded_targets = [
                    name
                    for name in controllers
                    if _controller_state_from_list_output(output, name) is not None
                ]
                if not loaded_targets:
                    time.sleep(0.5)
                    continue
                if not active_targets:
                    print(_color("运动控制器已处于 inactive；只保留 joint_state_broadcaster。", "cyan"))
                    return
                result = subprocess.run(
                    [
                        "ros2",
                        "control",
                        "switch_controllers",
                        "--deactivate",
                        *active_targets,
                        "--activate-asap",
                        "--switch-timeout",
                        "5.0",
                        "-c",
                        bringup.controller_manager,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    print(
                        _color(
                            "已自动停用运动控制器："
                            + ", ".join(active_targets)
                            + "。现在可以用 Web/末端按钮进入示教模式。",
                            "green",
                        )
                    )
                    return
                last_error = result.stdout.strip()
            time.sleep(0.5)

        if self._joint_states_available():
            print(
                _color(
                    "未在 controller 列表中识别到待停用的运动控制器，但 /joint_states 已可用；"
                    "继续以只读状态采集模式运行。",
                    "yellow",
                )
            )
            if last_output:
                print("最后一次 controller 列表：\n" + last_output)
            return

        raise TimeoutError(
            "自动停用运动控制器超时。请检查 `ros2 control list_controllers`。"
            + (f" 最后 controller 列表：\n{last_output}" if last_output else "")
            + (f" 最后错误：{last_error}" if last_error else "")
        )

    def shutdown(self) -> None:
        if self.config.teach.bringup.keep_process_on_exit:
            return
        if not self.started_by_this_process or self.process is None:
            return
        if self.process.poll() is not None:
            return
        print(_color("正在停止脚本自动启动的 Kortex bringup。", "yellow"))
        self.process.terminate()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2.0)

    def _wait_for_controller_manager(self) -> None:
        deadline = time.monotonic() + self.config.teach.bringup.startup_timeout_s
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("Kortex bringup 进程提前退出。请手动运行 ros2 launch 查看错误日志。")
            if self._list_controllers() is not None:
                return
            time.sleep(0.5)
        raise TimeoutError("等待 /controller_manager 超时，Kortex bringup 可能没有正常启动。")

    def _list_controllers(self) -> str | None:
        result = subprocess.run(
            [
                "ros2",
                "control",
                "list_controllers",
                "-c",
                self.config.teach.bringup.controller_manager,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout

    def _joint_states_available(self) -> bool:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        return self.config.kinova.joint_state_topic in result.stdout.splitlines()


class CollectorMode(Enum):
    NOT_RECORDING = auto()
    RECORDING = auto()


@dataclass
class LoopStats:
    last_print_time: float = 0.0
    last_loop_time: float = 0.0
    fps: float = 0.0


@dataclass
class TeachCommands:
    start: bool = False
    success: bool = False
    failure: bool = False
    stop: bool = False
    open_gripper: bool = False
    close_gripper: bool = False
    toggle_gripper: bool = False


@dataclass
class PendingSample:
    image: np.ndarray
    state: np.ndarray
    gripper_target: float
    timestamp: float


class TerminalCommandReader:
    """Small non-blocking keyboard reader for episode lifecycle commands."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._old_settings: list[Any] | None = None

    def connect(self) -> None:
        if not self.enabled or not sys.stdin.isatty():
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def disconnect(self) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

    def read(self) -> TeachCommands:
        commands = TeachCommands()
        if not self.enabled or not sys.stdin.isatty():
            return commands

        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            key = sys.stdin.read(1).lower()
            if key == "r":
                commands.start = True
            elif key == "v":
                commands.success = True
            elif key == "b":
                commands.failure = True
            elif key == "q":
                commands.stop = True
            elif key == "o":
                commands.open_gripper = True
            elif key == "c":
                commands.close_gripper = True
            elif key == "g":
                commands.toggle_gripper = True
        return commands


class ROSButtonReader:
    """Optional adapter for an end-effector button exposed as a ROS2 topic."""

    def __init__(
        self,
        topic: str,
        message_type: str,
        joy_button_index: int,
        toggle_on_press: bool,
        close_on_press: bool,
        debounce_s: float,
    ) -> None:
        self.topic = topic
        self.message_type = message_type
        self.joy_button_index = joy_button_index
        self.toggle_on_press = toggle_on_press
        self.close_on_press = close_on_press
        self.debounce_s = max(0.0, float(debounce_s))
        self._rclpy: Any | None = None
        self._node: Any | None = None
        self._pressed = False
        self._last_pressed = False
        self._last_edge_time = 0.0

    def connect(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
        except ImportError as exc:  # pragma: no cover - depends on ROS2 runtime
            raise RuntimeError("ROS2 button input requires rclpy") from exc

        msg_cls = self._resolve_message_type()
        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._node = Node("kinova_vla_teach_button")
        self._node.create_subscription(msg_cls, self.topic, self._on_message, 10)

    def disconnect(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        self._node = None

    def read(self) -> TeachCommands:
        commands = TeachCommands()
        if self._rclpy is None or self._node is None:
            return commands
        self._rclpy.spin_once(self._node, timeout_sec=0.0)
        now = time.monotonic()
        pressed_edge = self._pressed and not self._last_pressed
        self._last_pressed = self._pressed
        if not pressed_edge or (now - self._last_edge_time) < self.debounce_s:
            return commands
        self._last_edge_time = now
        if self.toggle_on_press:
            commands.toggle_gripper = True
        elif self.close_on_press:
            commands.close_gripper = True
        else:
            commands.open_gripper = True
        return commands

    def _on_message(self, msg: Any) -> None:
        msg_type = self.message_type.lower()
        if msg_type == "std_msgs/bool":
            self._pressed = bool(msg.data)
        elif msg_type == "std_msgs/int32":
            self._pressed = int(msg.data) != 0
        elif msg_type == "sensor_msgs/joy":
            buttons = getattr(msg, "buttons", [])
            self._pressed = (
                0 <= self.joy_button_index < len(buttons)
                and int(buttons[self.joy_button_index]) != 0
            )

    def _resolve_message_type(self) -> Any:
        msg_type = self.message_type.lower()
        if msg_type == "std_msgs/bool":
            from std_msgs.msg import Bool

            return Bool
        if msg_type == "std_msgs/int32":
            from std_msgs.msg import Int32

            return Int32
        if msg_type == "sensor_msgs/joy":
            from sensor_msgs.msg import Joy

            return Joy
        raise ValueError(
            "teach.gripper_button.message_type must be one of "
            "std_msgs/Bool, std_msgs/Int32, or sensor_msgs/Joy"
        )


class TeachCollector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        dry_run = config.hardware.dry_run

        self.dt = 1.0 / config.control.hz
        self.mode = CollectorMode.NOT_RECORDING
        self.episode_index = self._next_episode_index()
        self.current_episode_steps = 0
        self.running = False
        self.stats = LoopStats()
        self._pending_sample: PendingSample | None = None
        self._episode_start_time = 0.0
        self._gripper_target = config.teach.gripper_button.initial_target
        self._last_clip_warning_time = 0.0
        self.bringup = AutoBringupManager(config)

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
            enable_motion_commands=False,
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
        self.recorder = EpisodeRecorder(
            dataset_root=config.dataset.root,
            task_name=config.task.name,
            task_prompt=config.task.prompt,
            robot_name=config.dataset.robot,
            camera_name=config.dataset.camera,
            control_hz=config.control.hz,
            action_dim=config.control.action_dim,
            action_space="delta_ee_pose_rpy_with_gripper_state_delta_teach",
        )
        self.keyboard = TerminalCommandReader(enabled=config.teach.command_source == "keyboard")
        self.button_reader = self._build_button_reader()

    def run(self) -> None:
        try:
            self._connect()
            self.running = True

            print(_color("示教采集器已就绪。请先在 Kinova 端开启示教/拖拽模式。", "cyan"))
            print(_color("键盘：r=开始，v=保存成功，b=保存失败，q=退出，o/c/g=夹爪备用控制。", "cyan"))
            print(_color("拖拽机械臂完成：接近红球 -> 夹爪闭合 -> 移动到黑色 X -> 打开夹爪 -> 后退。", "cyan"))

            next_tick = time.monotonic()
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
        commands = self._read_commands()
        self._apply_gripper_commands(commands)

        if commands.stop:
            self.running = False
            print(_color("收到退出指令，正在安全停止。", "yellow"))
            return

        try:
            image = self.camera.get_rgb()
            state = self.robot.get_state().copy()
            state[6] = self.gripper.get_position()
        except Exception as exc:
            self.robot.stop()
            self.gripper.hold()
            print(_color(f"跳过当前帧：图像/state 读取失败：{exc}", "yellow"))
            return

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            print(_color("跳过当前帧：wrist RGB 图像无效。", "yellow"))
            return
        if state.shape != (14,) or not np.all(np.isfinite(state)):
            print(_color("跳过当前帧：robot state 无效。", "yellow"))
            return

        now = time.monotonic()
        if self.mode is CollectorMode.NOT_RECORDING and commands.start:
            self._start_episode(image=image, state=state, timestamp=now)
            return

        if self.mode is CollectorMode.RECORDING:
            self._append_state_delta_sample(image=image, state=state, timestamp=now)
            if commands.success:
                self._save_current_episode(success=True, reason="operator_success")
            elif commands.failure:
                self._save_current_episode(success=False, reason="operator_failure")
            elif self.current_episode_steps >= self.config.control.max_steps:
                self._save_current_episode(success=False, reason="max_steps_exceeded")

        self._print_status()

    def _start_episode(self, image: np.ndarray, state: np.ndarray, timestamp: float) -> None:
        if self.config.teach.open_gripper_on_start:
            print(_color("开始新 episode：打开夹爪，并把 gripper target 置为 -1。", "cyan"))
            self._gripper_target = -1.0
            self.gripper.open_gripper()
            time.sleep(min(1.0, max(0.0, self.config.gripper.open_timeout_s)))
            self.gripper.hold()

        self.episode_index = self._next_episode_index()
        episode_dir = self.recorder.start_episode(self.episode_index)
        self.current_episode_steps = 0
        self._episode_start_time = timestamp
        self._pending_sample = PendingSample(
            image=image.copy(),
            state=state.astype(np.float32, copy=True),
            gripper_target=self._gripper_target,
            timestamp=timestamp,
        )
        self.mode = CollectorMode.RECORDING
        print(_color(f"开始示教录制 episode_{self.episode_index:06d}: {episode_dir}", "green"))

    def _append_state_delta_sample(self, image: np.ndarray, state: np.ndarray, timestamp: float) -> None:
        if self._pending_sample is None:
            self._pending_sample = PendingSample(
                image=image.copy(),
                state=state.astype(np.float32, copy=True),
                gripper_target=self._gripper_target,
                timestamp=timestamp,
            )
            return

        action = self._state_delta_to_action(self._pending_sample.state, state, self._pending_sample.gripper_target)
        self.recorder.append(
            image=self._pending_sample.image,
            state=self._pending_sample.state,
            action=action,
            timestamp=self._pending_sample.timestamp - self._episode_start_time,
        )
        self.current_episode_steps += 1
        self._pending_sample = PendingSample(
            image=image.copy(),
            state=state.astype(np.float32, copy=True),
            gripper_target=self._gripper_target,
            timestamp=timestamp,
        )

    def _state_delta_to_action(self, previous_state: np.ndarray, current_state: np.ndarray, gripper_target: float) -> np.ndarray:
        action = np.zeros((self.config.control.action_dim,), dtype=np.float32)
        delta_position = current_state[:3] - previous_state[:3]
        delta_rpy = np.array(
            [_wrap_angle(float(cur - prev)) for prev, cur in zip(previous_state[3:6], current_state[3:6])],
            dtype=np.float32,
        )
        action[:3] = delta_position.astype(np.float32)
        action[3:6] = delta_rpy
        action[-1] = 1.0 if gripper_target > 0.0 else -1.0

        if self.config.teach.record_clipped_deltas:
            before = action.copy()
            action[:3] = np.clip(action[:3], -self.config.control.max_delta_m, self.config.control.max_delta_m)
            action[3:6] = np.clip(action[3:6], -self.config.control.max_delta_rad, self.config.control.max_delta_rad)
            self._warn_if_clipped(before, action)

        return action

    def _warn_if_clipped(self, before: np.ndarray, after: np.ndarray) -> None:
        if np.allclose(before[:6], after[:6]):
            return
        now = time.monotonic()
        if now - self._last_clip_warning_time < 2.0:
            return
        self._last_clip_warning_time = now
        print(
            _color(
                "提示：示教拖拽速度超过单步 action 上限，已裁剪标签。可以放慢拖拽或调大 max_delta_m/max_delta_rad。",
                "yellow",
            )
        )

    def _read_commands(self) -> TeachCommands:
        commands = self.keyboard.read()
        if self.button_reader is None:
            return commands
        button_commands = self.button_reader.read()
        return TeachCommands(
            start=commands.start,
            success=commands.success,
            failure=commands.failure,
            stop=commands.stop,
            open_gripper=commands.open_gripper or button_commands.open_gripper,
            close_gripper=commands.close_gripper or button_commands.close_gripper,
            toggle_gripper=commands.toggle_gripper or button_commands.toggle_gripper,
        )

    def _apply_gripper_commands(self, commands: TeachCommands) -> None:
        target: float | None = None
        if commands.toggle_gripper:
            target = -1.0 if self._gripper_target > 0.0 else 1.0
        elif commands.close_gripper:
            target = 1.0
        elif commands.open_gripper:
            target = -1.0
        if target is None:
            return
        self._gripper_target = target
        if self.config.teach.gripper_button.apply_gripper_command:
            self.gripper.apply_action(self._gripper_target)
            self.gripper.hold()
        name = "close" if self._gripper_target > 0.0 else "open"
        print(_color(f"夹爪目标切换为 {name} ({self._gripper_target:+.0f})", "magenta"))

    def _save_current_episode(self, success: bool, reason: str) -> None:
        episode_dir = self.recorder.save_episode(
            success=success,
            extra_meta={
                "episode_index": self.episode_index,
                "end_reason": reason,
                "max_steps": self.config.control.max_steps,
                "collection_mode": "kinova_teach_state_delta",
                "motion_source": "kinova_teach_manual_drag",
                "gripper_source": self.config.teach.gripper_button.mode,
                "task_instruction": self.config.task.prompt,
            },
        )
        label = "成功" if success else "失败"
        color = "green" if success else "red"
        print(_color(f"已保存{label}示教 episode_{self.episode_index:06d}: {episode_dir} 原因={reason}", color))
        self.mode = CollectorMode.NOT_RECORDING
        self.current_episode_steps = 0
        self._pending_sample = None
        self._episode_start_time = 0.0
        self.episode_index = self._next_episode_index()

    def _build_button_reader(self) -> ROSButtonReader | None:
        button = self.config.teach.gripper_button
        if button.mode == "keyboard":
            return None
        if button.mode == "ros_topic":
            return ROSButtonReader(
                topic=button.topic,
                message_type=button.message_type,
                joy_button_index=button.joy_button_index,
                toggle_on_press=button.toggle_on_press,
                close_on_press=button.close_on_press,
                debounce_s=button.debounce_s,
            )
        raise ValueError("teach.gripper_button.mode must be 'keyboard' or 'ros_topic'")

    def _connect(self) -> None:
        self.bringup.start()
        self.camera.start()
        self.robot.connect()
        self.gripper.connect()
        self.keyboard.connect()
        if self.button_reader is not None:
            self.button_reader.connect()

    def _shutdown(self) -> None:
        if self.recorder.episode_dir is not None:
            try:
                self._save_current_episode(success=False, reason="shutdown_while_recording")
            except Exception as exc:
                print(f"Warning: failed to save active episode during shutdown: {exc}")
        for name, action in [
            ("robot.stop()", self.robot.stop),
            ("gripper.hold()", self._safe_gripper_hold),
            ("keyboard.disconnect()", self.keyboard.disconnect),
            ("button_reader.disconnect()", lambda: self.button_reader.disconnect() if self.button_reader else None),
            ("gripper.disconnect()", self.gripper.disconnect),
            ("camera.stop()", self.camera.stop),
            ("robot.disconnect()", self.robot.disconnect),
            ("bringup.shutdown()", self.bringup.shutdown),
        ]:
            try:
                action()
            except Exception as exc:
                print(f"Warning: {name} failed: {exc}")
        self._print_dataset_summary()

    def _emergency_cleanup(self) -> None:
        for name, action in [
            ("robot.stop()", self.robot.stop),
            ("gripper.hold()", self._safe_gripper_hold),
            ("camera.stop()", self.camera.stop),
        ]:
            try:
                action()
            except Exception as exc:
                print(f"Emergency cleanup warning: {name} failed: {exc}")

    def _safe_gripper_hold(self) -> None:
        if not getattr(self.gripper, "_connected", False):
            return
        self.gripper.hold()

    def _update_fps(self, loop_start: float) -> None:
        if self.stats.last_loop_time > 0.0:
            period = loop_start - self.stats.last_loop_time
            if period > 1e-6:
                instant_fps = 1.0 / period
                self.stats.fps = instant_fps if self.stats.fps <= 0.0 else 0.8 * self.stats.fps + 0.2 * instant_fps
        self.stats.last_loop_time = loop_start

    def _print_status(self) -> None:
        now = time.monotonic()
        if now - self.stats.last_print_time < 0.5:
            return
        self.stats.last_print_time = now
        recording = self.mode is CollectorMode.RECORDING
        mode_text = "录制中" if recording else "待机"
        mode_color = "green" if recording else "blue"
        grip = "close" if self._gripper_target > 0.0 else "open"
        print(
            _color(
                f"episode={self.episode_index:06d} 模式={mode_text} "
                f"步数={self.current_episode_steps}/{self.config.control.max_steps} "
                f"夹爪目标={grip}({self._gripper_target:+.0f}) 频率={self.stats.fps:.2f}Hz",
                mode_color,
            )
        )

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
        print(f"  gripper open frame 数量: {summary['gripper_open_frames']}")
        print(f"  gripper close frame 数量: {summary['gripper_close_frames']}")
        print(f"  非 -1/+1 gripper 数量: {summary['invalid_gripper_value_count']}")


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _controller_state_from_list_output(output: str, controller_name: str) -> str | None:
    expected = _normalize_controller_name(controller_name)
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and _normalize_controller_name(parts[0]) == expected:
            if "active" in parts:
                return "active"
            if "inactive" in parts:
                return "inactive"
            return parts[1]
    return None


def _normalize_controller_name(name: str) -> str:
    clean = name.strip().strip("/")
    return clean.split("/")[-1]


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect Kinova VLA episodes with Kinova teach/manual-drag mode.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/collect_place_red_ball_on_black_x.yaml"),
        help="Path to collection YAML config.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    TeachCollector(config).run()


if __name__ == "__main__":
    main()
