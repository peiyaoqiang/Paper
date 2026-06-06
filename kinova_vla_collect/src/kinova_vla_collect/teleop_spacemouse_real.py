from __future__ import annotations

import argparse
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from kinova_vla_collect.kinova_robot import KinovaRobot
from kinova_vla_collect.modbus_gripper import ModbusGripper
from kinova_vla_collect.spacemouse_controller import (
    SpaceMouseController,
    SpaceMouseMapping,
    SpaceMouseSigns,
)
from kinova_vla_collect.utils.safety import SafetyLimiter, WorkspaceLimits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Low-speed SpaceMouse teleoperation test for a real Kinova Gen3."
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-path", type=str, default=None)
    parser.add_argument("--backend", choices=["direct_ros_twist", "kinova_robot"], default="direct_ros_twist")
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument("--deadzone", type=float, default=0.03)
    parser.add_argument("--calibration-duration", type=float, default=0.8)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--max-delta-m", type=float, default=0.006)
    parser.add_argument("--max-delta-rad", type=float, default=0.05)
    parser.add_argument("--max-linear-speed", type=float, default=0.07)
    parser.add_argument("--max-angular-speed", type=float, default=10.0)
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=1.2,
        help="Extra SpaceMouse linear sensitivity before safety clipping.",
    )
    parser.add_argument(
        "--angular-scale",
        type=float,
        default=16.0,
        help="Extra SpaceMouse angular sensitivity before safety clipping.",
    )
    parser.add_argument(
        "--require-enable-button",
        dest="require_enable_button",
        action="store_true",
        default=False,
        help="Require holding SpaceMouse button 0 before sending motion commands.",
    )
    parser.add_argument(
        "--no-enable-button",
        dest="require_enable_button",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--sign-x", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-y", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-z", type=float, choices=[-1.0, 1.0], default=-1.0)
    parser.add_argument("--sign-roll", type=float, choices=[-1.0, 1.0], default=-1.0)
    parser.add_argument("--sign-pitch", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-yaw", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument(
        "--control-layout",
        choices=["swapped", "normal"],
        default="normal",
        help=(
            "swapped: xyz<-SpaceMouse roll/pitch/yaw and rpy<-SpaceMouse x/y/z. "
            "normal: xyz<-SpaceMouse x/y/z and rpy<-SpaceMouse roll/pitch/yaw."
        ),
    )
    parser.add_argument("--translation-only", action="store_true", help="Ignore roll/pitch/yaw commands.")
    parser.add_argument("--rotation-only", action="store_true", help="Ignore xyz commands.")
    parser.add_argument(
        "--allow-mixed-motion",
        action="store_true",
        help="Allow simultaneous xyz and roll/pitch/yaw. Default decouples SpaceMouse crosstalk.",
    )
    parser.add_argument(
        "--dominance-ratio",
        type=float,
        default=1.15,
        help="When decoupling, the stronger group must exceed the weaker group by this ratio.",
    )

    parser.add_argument("--mode", type=str, default="ros2_twist", choices=["ros2_twist", "kortex_twist"])
    parser.add_argument("--ip", type=str, default="192.168.1.10")
    parser.add_argument("--username", type=str, default="admin")
    parser.add_argument("--password", type=str, default="admin")
    parser.add_argument("--joint-state-topic", type=str, default="/joint_states")
    parser.add_argument("--twist-command-topic", type=str, default="/twist_controller/commands")
    parser.add_argument("--base-frame", type=str, default="base_link")
    parser.add_argument("--ee-frame", type=str, default="end_effector_link")
    parser.add_argument("--twist-command-frame", type=str, default="tool_frame")
    parser.add_argument("--twist-publish-rate-hz", type=float, default=100.0)
    parser.add_argument("--twist-stop-duration-s", type=float, default=0.2)
    parser.add_argument("--state-timeout-s", type=float, default=5.0)

    parser.add_argument("--x-min", type=float, default=0.20)
    parser.add_argument("--x-max", type=float, default=0.80)
    parser.add_argument("--y-min", type=float, default=-0.55)
    parser.add_argument("--y-max", type=float, default=0.40)
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max", type=float, default=0.65)
    parser.add_argument("--no-workspace-limit", action="store_true")

    parser.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl+C or SpaceMouse button 1.")
    parser.add_argument("--print-period", type=float, default=0.1)
    parser.add_argument(
        "--state-update-period",
        type=float,
        default=0.2,
        help="Seconds between ROS state/TF reads during teleop. Lower is fresher; higher is more responsive.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-gripper", action="store_true", help="Disable button0 gripper open/close commands.")
    parser.add_argument("--gripper-mode", type=str, default="ctag_rtu")
    parser.add_argument("--gripper-host", type=str, default="192.168.1.20")
    parser.add_argument("--gripper-port", type=int, default=502)
    parser.add_argument("--gripper-unit-id", type=int, default=1)
    parser.add_argument("--gripper-serial-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--gripper-baudrate", type=int, default=115200)
    parser.add_argument("--gripper-timeout-s", type=float, default=4.0)
    parser.add_argument("--gripper-open-pos-mm", type=float, default=0.0)
    parser.add_argument("--gripper-close-pos-mm", type=float, default=120.0)
    parser.add_argument("--gripper-max-stroke-mm", type=float, default=120.0)
    parser.add_argument("--gripper-speed", type=int, default=30)
    parser.add_argument("--gripper-close-torque", type=int, default=10)
    parser.add_argument("--gripper-open-torque", type=int, default=100)
    parser.add_argument("--gripper-acc-dec", type=int, default=2000)
    args = parser.parse_args(argv)

    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")

    dt = 1.0 / args.hz
    controller = SpaceMouseController(
        device=args.device,
        device_index=args.device_index,
        device_path=args.device_path,
        deadzone=args.deadzone,
        max_delta_m=args.max_delta_m,
        max_delta_rad=args.max_delta_rad,
        require_enable_button=args.require_enable_button,
        mapping=SpaceMouseMapping(
            signs=SpaceMouseSigns(
                dx=1.0,
                dy=1.0,
                dz=1.0,
                droll=1.0,
                dpitch=1.0,
                dyaw=1.0,
            )
        ),
        debug=args.debug,
        calibrate_on_connect=not args.no_calibrate,
        calibration_duration_s=args.calibration_duration,
    )
    robot: KinovaRobot | None = None
    direct_twist: DirectTwistPublisher | None = None
    gripper: ModbusGripper | None = None
    if args.backend == "kinova_robot":
        robot = KinovaRobot(
            ip=args.ip,
            username=args.username,
            password=args.password,
            dry_run=False,
            max_linear_speed=args.max_linear_speed,
            mode=args.mode,
            joint_state_topic=args.joint_state_topic,
            twist_command_topic=args.twist_command_topic,
            base_frame=args.base_frame,
            ee_frame=args.ee_frame,
            twist_command_frame=args.twist_command_frame,
            state_timeout_s=args.state_timeout_s,
            twist_publish_rate_hz=args.twist_publish_rate_hz,
            twist_stop_duration_s=args.twist_stop_duration_s,
        )
    else:
        direct_twist = DirectTwistPublisher(args.twist_command_topic)
    if not args.no_gripper:
        gripper = ModbusGripper(
            host=args.gripper_host,
            port=args.gripper_port,
            unit_id=args.gripper_unit_id,
            dry_run=False,
            mode=args.gripper_mode,
            serial_port=args.gripper_serial_port,
            baudrate=args.gripper_baudrate,
            timeout_s=args.gripper_timeout_s,
            open_pos_mm=args.gripper_open_pos_mm,
            close_pos_mm=args.gripper_close_pos_mm,
            max_stroke_mm=args.gripper_max_stroke_mm,
            speed=args.gripper_speed,
            close_torque=args.gripper_close_torque,
            open_torque=args.gripper_open_torque,
            acc_dec=args.gripper_acc_dec,
        )
    safety = SafetyLimiter(
        max_delta_m=args.max_delta_m,
        max_delta_rad=args.max_delta_rad,
        workspace=WorkspaceLimits(
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            z_min=args.z_min,
            z_max=args.z_max,
        ),
    )

    if robot is not None:
        robot.connect()
    if direct_twist is not None:
        direct_twist.connect()
    if gripper is not None:
        gripper.connect()
    controller.connect()

    print("\033[36m真实 Kinova SpaceMouse 低速测试已启动。\033[0m")
    if args.require_enable_button:
        print("\033[36m安全模式：按住 SpaceMouse 按钮0 才运动；松手立即发零速度；按钮1退出。\033[0m")
    else:
        print("\033[33m按钮0 deadman 未启用：推动 SpaceMouse 会直接运动；按钮1退出。\033[0m")
    print(
        "\033[36m"
        f"backend={args.backend} mode={args.mode} topic={args.twist_command_topic} "
        f"max_delta_m={args.max_delta_m:.4f} max_delta_rad={args.max_delta_rad:.4f} "
        f"linear_scale={args.linear_scale:.2f} angular_scale={args.angular_scale:.2f} "
        f"max_v={args.max_linear_speed:.2f}m/s max_w={args.max_angular_speed:.2f}rad/s "
        f"layout={args.control_layout}"
        "\033[0m"
    )

    latest_state = robot.get_state() if robot is not None else np.zeros(14, dtype=np.float32)
    last_state_time = time.monotonic()
    next_tick = time.monotonic()
    last_print_time = 0.0
    last_gripper_target: float | None = None if gripper is None else -1.0
    last_gripper_name = "disabled" if gripper is None else "ready"
    step = 0
    try:
        while args.steps <= 0 or step < args.steps:
            raw_action, buttons = controller.read()
            if buttons["stop"]:
                print("\n\033[33m收到按钮1 stop，停止机械臂。\033[0m")
                break
            if gripper is not None:
                gripper_target = float(raw_action[6])
                if last_gripper_target is None:
                    last_gripper_target = gripper_target
                elif gripper_target != last_gripper_target:
                    command = gripper.apply_action(gripper_target)
                    last_gripper_target = gripper_target
                    last_gripper_name = command.name
                    print(f"\n\033[36m夹爪命令: {command.name}\033[0m")

            now = time.monotonic()
            if robot is not None and now - last_state_time >= args.state_update_period:
                latest_state = robot.get_state()
                last_state_time = now
            state = latest_state
            filtered_raw_action = _decouple_spacemouse_groups(raw_action, args)
            action = _spacemouse_to_xbox_action(filtered_raw_action, args)
            if args.translation_only:
                action[3:6] = 0.0
            if args.rotation_only:
                action[:3] = 0.0
            if direct_twist is not None:
                safe_action = action
                twist = _action_to_twist(safe_action, dt, args)
                direct_twist.publish(twist)
            else:
                if robot is None:
                    raise RuntimeError("Internal error: no teleop backend initialized")
                safe_action = action if args.no_workspace_limit else safety.limit_action(action, state[:3])
                robot.step_delta_action(safe_action, dt=dt)
                twist = _action_to_twist(safe_action, dt, args)

            now = time.monotonic()
            if now - last_print_time >= args.print_period:
                _print_status(
                    step,
                    state,
                    raw_action,
                    filtered_raw_action,
                    safe_action,
                    twist,
                    buttons,
                    last_gripper_name,
                )
                last_print_time = now

            step += 1
            next_tick += dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\n\033[33m用户中断，停止机械臂。\033[0m")
    finally:
        if direct_twist is not None:
            direct_twist.stop_for(args.twist_stop_duration_s)
            direct_twist.disconnect()
        if robot is not None:
            robot.stop()
            robot.disconnect()
        if gripper is not None:
            gripper.disconnect()
        controller.disconnect()


class DirectTwistPublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.node: Node | None = None
        self.publisher = None

    def connect(self) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("spacemouse_direct_twist_teleop")
        self.publisher = self.node.create_publisher(Twist, self.topic, 10)

    def publish(self, twist_array: np.ndarray) -> None:
        if self.node is None or self.publisher is None:
            raise RuntimeError("DirectTwistPublisher is not connected")
        msg = Twist()
        msg.linear.x = float(twist_array[0])
        msg.linear.y = float(twist_array[1])
        msg.linear.z = float(twist_array[2])
        msg.angular.x = float(twist_array[3])
        msg.angular.y = float(twist_array[4])
        msg.angular.z = float(twist_array[5])
        self.publisher.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def stop_for(self, duration_s: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_s)
        zero = np.zeros(6, dtype=np.float32)
        while time.monotonic() < deadline:
            self.publish(zero)
            time.sleep(0.01)

    def disconnect(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
        self.node = None
        if rclpy.ok():
            rclpy.shutdown()


def _spacemouse_to_xbox_action(raw_action: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Map SpaceMouse physical axes into the same action layout used by Xbox.

    XboxController._axes_to_action uses:
      dx <- forward/back stick axis
      dy <- left/right stick axis
      dz <- vertical stick axis
      droll, dpitch, dyaw <- angular controls

    Normal layout:
      dx/dy/dz <- SpaceMouse y/x/z
      roll/pitch/yaw <- SpaceMouse roll/pitch/yaw

    Swapped layout:
      dx/dy/dz <- SpaceMouse pitch/roll/yaw
      roll/pitch/yaw <- SpaceMouse x/y/z
    """
    if args.control_layout == "swapped":
        return np.array(
            [
                args.sign_x * raw_action[4] * args.linear_scale,
                args.sign_y * raw_action[3] * args.linear_scale,
                args.sign_z * raw_action[5] * args.linear_scale,
                args.sign_roll * raw_action[0] * args.angular_scale,
                args.sign_pitch * raw_action[1] * args.angular_scale,
                args.sign_yaw * raw_action[2] * args.angular_scale,
                raw_action[6],
            ],
            dtype=np.float32,
        )

    return np.array(
        [
            args.sign_x * raw_action[1] * args.linear_scale,
            args.sign_y * raw_action[0] * args.linear_scale,
            args.sign_z * raw_action[2] * args.linear_scale,
            args.sign_roll * raw_action[3] * args.angular_scale,
            args.sign_pitch * raw_action[4] * args.angular_scale,
            args.sign_yaw * raw_action[5] * args.angular_scale,
            raw_action[6],
        ],
        dtype=np.float32,
    )


def _decouple_spacemouse_groups(raw_action: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.allow_mixed_motion or args.translation_only or args.rotation_only:
        return raw_action

    filtered = np.array(raw_action, dtype=np.float32, copy=True)
    linear_level = float(np.linalg.norm(filtered[:3]) / max(args.max_delta_m, 1e-9))
    angular_level = float(np.linalg.norm(filtered[3:6]) / max(args.max_delta_rad, 1e-9))
    ratio = max(1.0, float(args.dominance_ratio))

    if angular_level > linear_level * ratio:
        filtered[:3] = 0.0
    elif linear_level > angular_level * ratio:
        filtered[3:6] = 0.0
    return filtered


def _action_to_twist(action: np.ndarray, dt: float, args: argparse.Namespace) -> np.ndarray:
    twist = np.zeros(6, dtype=np.float32)
    twist[:3] = action[:3] / max(dt, 1e-9)
    linear_speed = float(np.linalg.norm(twist[:3]))
    if linear_speed > args.max_linear_speed:
        twist[:3] *= args.max_linear_speed / max(linear_speed, 1e-9)

    twist[3:6] = action[3:6] / max(dt, 1e-9)
    angular_speed = float(np.linalg.norm(twist[3:6]))
    if angular_speed > args.max_angular_speed:
        twist[3:6] *= args.max_angular_speed / max(angular_speed, 1e-9)
    return twist


def _print_status(
    step: int,
    state: np.ndarray,
    raw_action: np.ndarray,
    filtered_raw_action: np.ndarray,
    action: np.ndarray,
    twist: np.ndarray,
    buttons: dict[str, bool],
    gripper_name: str,
) -> None:
    pos = state[:3]
    rpy = state[3:6]
    linear_speed = float(np.linalg.norm(twist[:3]))
    angular_speed = float(np.linalg.norm(twist[3:6]))
    print(
        "\r\033[2K"
        f"step={step:06d} enable={int(buttons['enable'])} "
        f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
        f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f}) "
        f"dxyz=({action[0]:+.4f},{action[1]:+.4f},{action[2]:+.4f}) "
        f"drpy=({action[3]:+.4f},{action[4]:+.4f},{action[5]:+.4f}) "
        f"raw_xyz=({raw_action[0]:+.4f},{raw_action[1]:+.4f},{raw_action[2]:+.4f}) "
        f"raw_rpy=({raw_action[3]:+.4f},{raw_action[4]:+.4f},{raw_action[5]:+.4f}) "
        f"flt_xyz=({filtered_raw_action[0]:+.4f},{filtered_raw_action[1]:+.4f},{filtered_raw_action[2]:+.4f}) "
        f"flt_rpy=({filtered_raw_action[3]:+.4f},{filtered_raw_action[4]:+.4f},{filtered_raw_action[5]:+.4f}) "
        f"v={linear_speed:.3f}m/s w={angular_speed:.2f}rad/s "
        f"gripper={gripper_name}",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    main()
