from __future__ import annotations

import argparse
import math
import time
from typing import Sequence

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PoseStamped, Quaternion
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply
from trajectory_msgs.msg import JointTrajectoryPoint

from kinova_vla_collect.spacemouse_controller import (
    SpaceMouseController,
    SpaceMouseMapping,
    SpaceMouseSigns,
)


MANIPULATOR_JOINTS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]


class SpaceMouseMoveItFakeTeleop(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("spacemouse_moveit_fake_teleop")
        self.args = args
        self.dt = 1.0 / args.hz

        self.controller = SpaceMouseController(
            device=args.device,
            device_index=args.device_index,
            device_path=args.device_path,
            deadzone=args.deadzone,
            max_delta_m=args.max_delta_m,
            max_delta_rad=args.max_delta_rad,
            require_enable_button=args.require_enable_button,
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

        self.ik_client = self.create_client(GetPositionIK, args.ik_service)
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            args.trajectory_action,
        )
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            args.gripper_action,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            args.joint_state_topic,
            self._on_joint_state,
            10,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_joint_state: JointState | None = None
        self.target_position: np.ndarray | None = None
        self.target_quaternion: np.ndarray | None = None
        self.last_print_time = 0.0
        self.last_goal_time = 0.0
        self.pending_goal = False
        self.pending_gripper_goal = False
        self.last_gripper_target: float | None = None
        self.step = 0

    def connect(self) -> None:
        self.controller.connect()
        self.get_logger().info(f"Waiting for IK service {self.args.ik_service} ...")
        if not self.ik_client.wait_for_service(timeout_sec=self.args.startup_timeout):
            raise TimeoutError(
                f"Timed out waiting for {self.args.ik_service}. Start MoveIt fake stack first."
            )
        self.get_logger().info(f"Waiting for trajectory action {self.args.trajectory_action} ...")
        if not self.trajectory_client.wait_for_server(timeout_sec=self.args.startup_timeout):
            raise TimeoutError(
                f"Timed out waiting for {self.args.trajectory_action}. "
                "Make sure joint_trajectory_controller is active."
            )
        self.get_logger().info(f"Waiting for gripper action {self.args.gripper_action} ...")
        if not self.gripper_client.wait_for_server(timeout_sec=self.args.startup_timeout):
            self.get_logger().warning(
                f"Timed out waiting for {self.args.gripper_action}; arm teleop will still run."
            )
        self._wait_for_joint_state()
        self._initialize_target_pose()

    def disconnect(self) -> None:
        self.controller.disconnect()

    def run(self) -> None:
        print("\033[36m正式版已启动：SpaceMouse -> MoveIt IK -> fake joint_trajectory_controller -> RViz。\033[0m")
        print("\033[36m直接推动 SpaceMouse 控制末端；按钮0切换夹爪开/合，按钮1退出。\033[0m")
        if self.args.require_enable_button:
            print("\033[36m按住 enable 按钮时才发送 IK 轨迹。\033[0m")

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
            print("\033[33m用户中断，停止 MoveIt fake 遥操作。\033[0m")

    def _run_one_step(self) -> None:
        action, buttons = self.controller.read()
        if buttons["stop"]:
            raise KeyboardInterrupt
        if self.target_position is None or self.target_quaternion is None:
            return
        if self.latest_joint_state is None:
            return
        if self.pending_goal:
            return

        self._maybe_send_gripper_goal(float(action[-1]))
        if self.args.require_enable_button and not buttons["enable"]:
            self._print_status(action, sent=False, note="idle")
            return

        action = self._suppress_tiny_action(action)
        if np.linalg.norm(action[:6]) < 1e-7:
            self._print_status(action, sent=False, note="zero")
            return

        candidate_position = self.target_position + action[:3].astype(np.float64)
        candidate_position = np.array(
            [
                np.clip(candidate_position[0], self.args.x_min, self.args.x_max),
                np.clip(candidate_position[1], self.args.y_min, self.args.y_max),
                np.clip(candidate_position[2], self.args.z_min, self.args.z_max),
            ],
            dtype=np.float64,
        )
        delta_quat = np.array(
            quaternion_from_euler(float(action[3]), float(action[4]), float(action[5])),
            dtype=np.float64,
        )
        candidate_quaternion = _normalize_quaternion(
            np.array(quaternion_multiply(self.target_quaternion, delta_quat), dtype=np.float64)
        )

        solution = self._compute_ik(candidate_position, candidate_quaternion)
        if solution is None:
            self._print_status(action, sent=False, note="no_ik")
            return

        self._send_joint_goal(solution)
        self.target_position = candidate_position
        self.target_quaternion = candidate_quaternion
        self._print_status(action, sent=True, note="sent")

    def _compute_ik(self, position: np.ndarray, quaternion_xyzw: np.ndarray) -> dict[str, float] | None:
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.args.group_name
        request.ik_request.ik_link_name = self.args.ee_link
        request.ik_request.avoid_collisions = False
        request.ik_request.timeout.sec = 0
        request.ik_request.timeout.nanosec = int(self.args.ik_timeout * 1e9)
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = self.args.base_frame
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose.position.x = float(position[0])
        request.ik_request.pose_stamped.pose.position.y = float(position[1])
        request.ik_request.pose_stamped.pose.position.z = float(position[2])
        request.ik_request.pose_stamped.pose.orientation = _quaternion_msg(quaternion_xyzw)

        seed = self.latest_joint_state
        if seed is not None:
            request.ik_request.robot_state.joint_state.name = list(seed.name)
            request.ik_request.robot_state.joint_state.position = list(seed.position)

        future = self.ik_client.call_async(request)
        deadline = time.monotonic() + self.args.ik_timeout + 0.2
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.002)

        if not future.done():
            return None
        response = future.result()
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        names = list(response.solution.joint_state.name)
        positions = list(response.solution.joint_state.position)
        by_name = dict(zip(names, positions))
        if not all(name in by_name for name in MANIPULATOR_JOINTS):
            return None
        return {name: float(by_name[name]) for name in MANIPULATOR_JOINTS}

    def _send_joint_goal(self, joint_positions: dict[str, float]) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(MANIPULATOR_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [joint_positions[name] for name in MANIPULATOR_JOINTS]
        point.velocities = [0.0] * len(MANIPULATOR_JOINTS)
        point.time_from_start = _duration(self.args.goal_duration)
        goal.trajectory.points = [point]

        self.pending_goal = True
        send_future = self.trajectory_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)

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

    def _on_goal_response(self, future: object) -> None:
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.pending_goal = False
                self.get_logger().warning("Trajectory goal rejected")
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._on_goal_result)
        except Exception as exc:
            self.pending_goal = False
            self.get_logger().warning(f"Trajectory goal failed before acceptance: {exc}")

    def _on_goal_result(self, future: object) -> None:
        self.pending_goal = False
        try:
            result = future.result().result
            if result.error_code != 0:
                self.get_logger().warning(f"Trajectory result error_code={result.error_code}")
        except Exception as exc:
            self.get_logger().warning(f"Trajectory result failed: {exc}")

    def _wait_for_joint_state(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_joint_state is not None:
                return
        raise TimeoutError(f"Timed out waiting for {self.args.joint_state_topic}")

    def _initialize_target_pose(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.base_frame,
                    self.args.ee_link,
                    rclpy.time.Time(),
                )
            except Exception:
                continue
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            self.target_position = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
            self.target_quaternion = _normalize_quaternion(
                np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)
            )
            return
        raise TimeoutError(f"Timed out looking up TF {self.args.base_frame} -> {self.args.ee_link}")

    def _on_joint_state(self, msg: JointState) -> None:
        if all(name in msg.name for name in MANIPULATOR_JOINTS):
            self.latest_joint_state = msg

    def _suppress_tiny_action(self, action: np.ndarray) -> np.ndarray:
        filtered = np.array(action, dtype=np.float32, copy=True)
        filtered[:3] = np.where(np.abs(filtered[:3]) < self.args.min_delta_m, 0.0, filtered[:3])
        filtered[3:6] = np.where(np.abs(filtered[3:6]) < self.args.min_delta_rad, 0.0, filtered[3:6])
        return filtered

    def _print_status(self, action: np.ndarray, *, sent: bool, note: str) -> None:
        now = time.monotonic()
        if now - self.last_print_time < self.args.print_period:
            return
        self.last_print_time = now
        if self.target_position is None or self.target_quaternion is None:
            return
        rpy = euler_from_quaternion(self.target_quaternion.tolist())
        print(
            "\r\033[2K"
            f"step={self.step:06d} {note:<5} "
            f"target=({self.target_position[0]:+.3f},{self.target_position[1]:+.3f},{self.target_position[2]:+.3f}) "
            f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f}) "
            f"dxyz=({action[0]:+.4f},{action[1]:+.4f},{action[2]:+.4f}) "
            f"sent={int(sent)}",
            end="",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Teleoperate Kinova Gen3 in MoveIt fake hardware using SpaceMouse IK."
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-path", type=str, default=None)
    parser.add_argument("--hz", type=float, default=15.0)
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--max-delta-m", type=float, default=0.008)
    parser.add_argument("--max-delta-rad", type=float, default=math.radians(2.0))
    parser.add_argument("--min-delta-m", type=float, default=0.0002)
    parser.add_argument("--min-delta-rad", type=float, default=math.radians(0.05))
    parser.add_argument("--calibration-duration", type=float, default=0.8)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--sign-x", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-y", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-z", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-roll", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-pitch", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--sign-yaw", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument(
        "--require-enable-button",
        dest="require_enable_button",
        action="store_true",
        help="Require holding SpaceMouse button 0 before sending IK goals.",
    )
    parser.add_argument(
        "--no-enable-button",
        dest="require_enable_button",
        action="store_false",
        help="Allow motion without holding SpaceMouse button 0.",
    )
    parser.set_defaults(require_enable_button=False)
    parser.add_argument("--group-name", type=str, default="manipulator")
    parser.add_argument("--base-frame", type=str, default="base_link")
    parser.add_argument("--ee-link", type=str, default="end_effector_link")
    parser.add_argument("--joint-state-topic", type=str, default="/joint_states")
    parser.add_argument("--ik-service", type=str, default="/compute_ik")
    parser.add_argument(
        "--trajectory-action",
        type=str,
        default="/joint_trajectory_controller/follow_joint_trajectory",
    )
    parser.add_argument(
        "--gripper-action",
        type=str,
        default="/robotiq_gripper_controller/gripper_cmd",
    )
    parser.add_argument("--gripper-open-position", type=float, default=0.0)
    parser.add_argument("--gripper-close-position", type=float, default=0.8)
    parser.add_argument("--gripper-max-effort", type=float, default=40.0)
    parser.add_argument("--ik-timeout", type=float, default=0.08)
    parser.add_argument("--goal-duration", type=float, default=0.08)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--print-period", type=float, default=0.1)
    parser.add_argument("--x-min", type=float, default=-0.8)
    parser.add_argument("--x-max", type=float, default=0.8)
    parser.add_argument("--y-min", type=float, default=-0.8)
    parser.add_argument("--y-max", type=float, default=0.8)
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=1.3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")

    rclpy.init(args=None)
    node = SpaceMouseMoveItFakeTeleop(args)
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


def _quaternion_msg(values: Sequence[float]) -> Quaternion:
    msg = Quaternion()
    msg.x = float(values[0])
    msg.y = float(values[1])
    msg.z = float(values[2])
    msg.w = float(values[3])
    return msg


def _normalize_quaternion(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return values / norm


if __name__ == "__main__":
    main()
