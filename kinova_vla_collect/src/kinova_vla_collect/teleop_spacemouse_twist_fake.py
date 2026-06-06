from __future__ import annotations

import argparse
import math
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node

from kinova_vla_collect.spacemouse_controller import (
    SpaceMouseController,
    SpaceMouseMapping,
    SpaceMouseSigns,
)


class SpaceMouseTwistFakeTeleop(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("spacemouse_twist_fake_teleop")
        self.args = args
        self.dt = 1.0 / args.hz

        self.controller = SpaceMouseController(
            device=args.device,
            device_index=args.device_index,
            device_path=args.device_path,
            deadzone=args.deadzone,
            max_delta_m=args.max_linear_speed / args.hz,
            max_delta_rad=args.max_angular_speed / args.hz,
            require_enable_button=False,
            mapping=SpaceMouseMapping(
                signs=SpaceMouseSigns(
                    dx=args.sign_x,
                    dy=args.sign_y,
                    dz=args.sign_z,
                    droll=args.sign_roll,
                    dpitch=args.sign_pitch,
                    dyaw=args.sign_yaw,
                )
            ),
            debug=args.debug,
            calibrate_on_connect=not args.no_calibrate,
            calibration_duration_s=args.calibration_duration,
        )

        self.twist_pub = self.create_publisher(Twist, args.twist_topic, 10)
        self.switch_controller_client = self.create_client(
            SwitchController,
            args.switch_controller_service,
        )
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            args.gripper_action,
        )

        self.step = 0
        self.last_print_time = 0.0
        self.last_gripper_target: float | None = None
        self.pending_gripper_goal = False
        self.last_nonzero_command_time = 0.0
        self.filtered_linear = np.zeros(3, dtype=np.float64)
        self.filtered_angular = np.zeros(3, dtype=np.float64)

    def connect(self) -> None:
        self.controller.connect()
        if self.args.auto_switch_controllers:
            self._activate_twist_controller()
        if not self.gripper_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warning(
                f"Gripper action {self.args.gripper_action} is not available; arm teleop will still run."
            )

    def disconnect(self) -> None:
        self._publish_zero_burst()
        self.controller.disconnect()

    def run(self) -> None:
        print("\033[36mSpaceMouse Twist 遥操作已启动：直接推动旋钮控制 RViz Kinova。按钮0夹爪开/合，按钮1退出。\033[0m")
        next_tick = time.monotonic()
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)
                self._run_one_step()
                self.step += 1

                next_tick += self.dt
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
        except KeyboardInterrupt:
            print("\033[33m停止 SpaceMouse Twist 遥操作。\033[0m")

    def _run_one_step(self) -> None:
        action, buttons = self.controller.read()
        if buttons["stop"]:
            raise KeyboardInterrupt

        self._maybe_send_gripper_goal(float(action[-1]))

        linear = np.asarray(action[:3], dtype=np.float64) / self.dt
        angular = np.asarray(action[3:6], dtype=np.float64) / self.dt

        linear[np.abs(linear) < self.args.min_linear_speed] = 0.0
        angular[np.abs(angular) < self.args.min_angular_speed] = 0.0

        linear = self._limit_norm(linear, self.args.max_linear_speed)
        angular = self._limit_norm(angular, self.args.max_angular_speed)

        alpha = float(np.clip(self.args.command_alpha, 0.0, 1.0))
        self.filtered_linear = (1.0 - alpha) * self.filtered_linear + alpha * linear
        self.filtered_angular = (1.0 - alpha) * self.filtered_angular + alpha * angular

        if (
            np.linalg.norm(self.filtered_linear) < self.args.min_linear_speed
            and np.linalg.norm(self.filtered_angular) < self.args.min_angular_speed
        ):
            self.filtered_linear[:] = 0.0
            self.filtered_angular[:] = 0.0

        twist = Twist()
        twist.linear.x = float(self.filtered_linear[0])
        twist.linear.y = float(self.filtered_linear[1])
        twist.linear.z = float(self.filtered_linear[2])
        twist.angular.x = float(self.filtered_angular[0])
        twist.angular.y = float(self.filtered_angular[1])
        twist.angular.z = float(self.filtered_angular[2])
        self.twist_pub.publish(twist)

        nonzero = np.linalg.norm(self.filtered_linear) > 0.0 or np.linalg.norm(self.filtered_angular) > 0.0
        if nonzero:
            self.last_nonzero_command_time = time.monotonic()
        self._print_status(nonzero=nonzero)

    def _activate_twist_controller(self) -> None:
        self.get_logger().info("Switching controllers: activate twist_controller, deactivate joint_trajectory_controller")
        if not self.switch_controller_client.wait_for_service(timeout_sec=self.args.startup_timeout):
            raise TimeoutError(f"Timed out waiting for {self.args.switch_controller_service}")

        request = SwitchController.Request()
        request.activate_controllers = [self.args.twist_controller]
        request.deactivate_controllers = [self.args.trajectory_controller]
        request.strictness = SwitchController.Request.BEST_EFFORT
        request.activate_asap = True
        request.timeout = _duration(5.0)

        future = self.switch_controller_client.call_async(request)
        deadline = time.monotonic() + 8.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done() or future.result() is None or not future.result().ok:
            self.get_logger().warning(
                "Controller switch did not report success. "
                "Check `ros2 control list_controllers`; twist_controller must be active."
            )

    def _maybe_send_gripper_goal(self, gripper_target: float) -> None:
        if self.pending_gripper_goal:
            return
        if self.last_gripper_target is not None and gripper_target == self.last_gripper_target:
            return
        self.last_gripper_target = gripper_target
        if not self.gripper_client.server_is_ready():
            return

        goal = GripperCommand.Goal()
        goal.command.position = (
            self.args.gripper_close_position if gripper_target > 0.0 else self.args.gripper_open_position
        )
        goal.command.max_effort = self.args.gripper_max_effort
        self.pending_gripper_goal = True
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(self._on_gripper_goal_response)

    def _on_gripper_goal_response(self, future: object) -> None:
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.pending_gripper_goal = False
                self.get_logger().warning("Gripper goal rejected")
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._on_gripper_goal_result)
        except Exception as exc:
            self.pending_gripper_goal = False
            self.get_logger().warning(f"Gripper goal failed before acceptance: {exc}")

    def _on_gripper_goal_result(self, future: object) -> None:
        self.pending_gripper_goal = False
        try:
            future.result()
        except Exception as exc:
            self.get_logger().warning(f"Gripper result failed: {exc}")

    def _publish_zero_burst(self) -> None:
        zero = Twist()
        for _ in range(10):
            self.twist_pub.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.02)

    def _print_status(self, *, nonzero: bool) -> None:
        now = time.monotonic()
        if now - self.last_print_time < self.args.print_period:
            return
        self.last_print_time = now
        print(
            "\r\033[2K"
            f"step={self.step:06d} {'move' if nonzero else 'zero':<4} "
            f"v=({self.filtered_linear[0]:+.3f},{self.filtered_linear[1]:+.3f},{self.filtered_linear[2]:+.3f})m/s "
            f"w=({self.filtered_angular[0]:+.2f},{self.filtered_angular[1]:+.2f},{self.filtered_angular[2]:+.2f})rad/s",
            end="",
            flush=True,
        )

    @staticmethod
    def _limit_norm(values: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(values))
        if norm <= max_norm or norm <= 1e-12:
            return values
        return values * (max_norm / norm)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Teleoperate Kinova Gen3 fake hardware with SpaceMouse using twist_controller."
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-path", type=str, default=None)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--deadzone", type=float, default=0.10)
    parser.add_argument("--calibration-duration", type=float, default=0.8)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--max-linear-speed", type=float, default=0.18)
    parser.add_argument("--max-angular-speed", type=float, default=math.radians(45.0))
    parser.add_argument("--min-linear-speed", type=float, default=0.004)
    parser.add_argument("--min-angular-speed", type=float, default=math.radians(1.0))
    parser.add_argument("--command-alpha", type=float, default=0.45)
    parser.add_argument("--sign-x", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-y", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-z", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-roll", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-pitch", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-yaw", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--twist-topic", type=str, default="/twist_controller/commands")
    parser.add_argument("--switch-controller-service", type=str, default="/controller_manager/switch_controller")
    parser.add_argument("--twist-controller", type=str, default="twist_controller")
    parser.add_argument("--trajectory-controller", type=str, default="joint_trajectory_controller")
    parser.add_argument("--auto-switch-controllers", action="store_true", default=True)
    parser.add_argument("--no-auto-switch-controllers", dest="auto_switch_controllers", action="store_false")
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--gripper-action", type=str, default="/robotiq_gripper_controller/gripper_cmd")
    parser.add_argument("--gripper-open-position", type=float, default=0.0)
    parser.add_argument("--gripper-close-position", type=float, default=0.8)
    parser.add_argument("--gripper-max-effort", type=float, default=40.0)
    parser.add_argument("--print-period", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")

    rclpy.init(args=None)
    node = SpaceMouseTwistFakeTeleop(args)
    try:
        node.connect()
        node.run()
    finally:
        node.disconnect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _duration(seconds: float) -> Duration:
    whole = int(seconds)
    return Duration(sec=whole, nanosec=int((seconds - whole) * 1e9))


if __name__ == "__main__":
    main()
