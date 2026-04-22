from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import List, Tuple

from common.types import Quaternion, RobotState

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from tf2_ros import Buffer, TransformListener
    from tf_transformations import euler_from_quaternion
except ImportError:  # pragma: no cover - depends on ROS2 runtime
    rclpy = None
    Twist = None
    Node = object
    qos_profile_sensor_data = None
    JointState = object
    Buffer = None
    TransformListener = None
    euler_from_quaternion = None


Vector3 = Tuple[float, float, float]


@dataclass
class KinovaConfig:
    max_translation_step_m: float
    max_rotation_step_deg: float
    mode: str = "mock"
    joint_state_topic: str = "/joint_states"
    twist_command_topic: str = "/twist_controller/commands"
    base_frame: str = "base_link"
    ee_frame: str = "tool_frame"
    twist_command_frame: str = "tool_frame"
    sequential_axis_commands: bool = True
    ros_node_name: str = "paper_kinova_driver"
    state_timeout_s: float = 2.0
    twist_command_duration_s: float = 0.2
    twist_publish_rate_hz: float = 20.0
    twist_stop_duration_s: float = 0.6


class _KinovaROSInterface(Node):
    def __init__(self, config: KinovaConfig) -> None:
        super().__init__(config.ros_node_name)
        self.config = config
        self.latest_joint_state: JointState | None = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            config.joint_state_topic,
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.twist_pub = self.create_publisher(Twist, config.twist_command_topic, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

    def _on_joint_state(self, msg: JointState) -> None:
        self.latest_joint_state = msg


@dataclass
class KinovaDriver:
    """
    Minimal Kinova arm interface.

    Supports:
    - `mock`: in-memory state only
    - `ros2_twist`: read `/joint_states` + TF, publish Cartesian `Twist`
    """

    config: KinovaConfig
    ee_position_m: Vector3 = (0.45, 0.00, 0.25)
    ee_yaw_deg: float = 0.0
    ee_quaternion_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)
    joint_positions: List[float] = field(default_factory=lambda: [0.0] * 7)
    gripper_opening_m: float = 0.08

    def __post_init__(self) -> None:
        self.ros_interface: _KinovaROSInterface | None = None
        self._continuous_servo_lock = threading.Lock()
        self._continuous_servo_thread: threading.Thread | None = None
        self._continuous_servo_running = False
        self._continuous_servo_publish_rate_hz = 20.0
        self._continuous_servo_stale_timeout_s = 2.0
        self._continuous_servo_command_alpha = 1.0
        self._continuous_servo_max_linear_speed_mps = 0.08
        self._continuous_servo_max_angular_speed_rps = math.radians(20.0)
        self._continuous_servo_last_update_time = 0.0
        self._continuous_servo_target_linear_command: Vector3 = (0.0, 0.0, 0.0)
        self._continuous_servo_target_angular_command: Vector3 = (0.0, 0.0, 0.0)
        self._continuous_servo_linear_command: Vector3 = (0.0, 0.0, 0.0)
        self._continuous_servo_angular_command: Vector3 = (0.0, 0.0, 0.0)
        if self.config.mode == "ros2_twist":
            self._init_ros2()

    def _init_ros2(self) -> None:
        if (
            rclpy is None
            or Twist is None
            or JointState is None
            or Buffer is None
            or euler_from_quaternion is None
            or qos_profile_sensor_data is None
        ):
            raise RuntimeError(
                "ROS2 Kinova mode requires `rclpy`, `sensor_msgs`, `geometry_msgs`, `tf2_ros`, and `tf_transformations`."
            )
        if not rclpy.ok():
            rclpy.init(args=None)
        self.ros_interface = _KinovaROSInterface(self.config)

    def get_state(self) -> RobotState:
        if self.config.mode == "ros2_twist":
            return self._get_state_ros2()
        return RobotState(
            joint_positions=list(self.joint_positions),
            ee_position_m=self.ee_position_m,
            ee_yaw_deg=self.ee_yaw_deg,
            gripper_opening_m=self.gripper_opening_m,
            ee_quaternion_xyzw=self.ee_quaternion_xyzw,
        )

    def move_cartesian_delta(self, delta_xyz_m: Vector3, delta_yaw_deg: float) -> None:
        if self.config.mode == "ros2_twist":
            self._move_cartesian_delta_ros2(delta_xyz_m, delta_yaw_deg)
            return
        self.ee_position_m = tuple(
            current + delta for current, delta in zip(self.ee_position_m, delta_xyz_m)
        )
        self.ee_yaw_deg += delta_yaw_deg

    def set_gripper_opening(self, width_m: float) -> None:
        self.gripper_opening_m = width_m

    def _spin_until_ready(self) -> None:
        if self.ros_interface is None:
            raise RuntimeError("ROS2 Kinova interface is not initialized.")
        # Fresh ROS2 processes can miss the first few best-effort sensor packets.
        # Give the joint state stream a few short windows before declaring failure.
        for _ in range(3):
            deadline = time.monotonic() + self.config.state_timeout_s
            while time.monotonic() < deadline:
                rclpy.spin_once(self.ros_interface, timeout_sec=0.1)
                if self.ros_interface.latest_joint_state is not None:
                    return
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for joint state on {self.config.joint_state_topic}")

    def _lookup_frame_pose(self, frame_name: str) -> tuple[Vector3, float, Quaternion]:
        if self.ros_interface is None:
            raise RuntimeError("ROS2 Kinova interface is not initialized.")
        deadline = time.monotonic() + self.config.state_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.ros_interface, timeout_sec=0.05)
            try:
                transform = self.ros_interface.tf_buffer.lookup_transform(
                    self.config.base_frame,
                    frame_name,
                    rclpy.time.Time(),
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                _, _, yaw = euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
                return (
                    (translation.x, translation.y, translation.z),
                    math.degrees(yaw),
                    (rotation.x, rotation.y, rotation.z, rotation.w),
                )
            except Exception:
                continue
        raise TimeoutError(
            f"Timed out looking up TF from {self.config.base_frame} to {frame_name}"
        )

    def _lookup_ee_pose(self) -> tuple[Vector3, float, Quaternion]:
        return self._lookup_frame_pose(self.config.ee_frame)

    def _rotation_matrix_from_quaternion(self, quaternion_xyzw: Quaternion) -> tuple[Vector3, Vector3, Vector3]:
        x, y, z, w = quaternion_xyzw
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        )

    def _base_vector_to_command_frame(self, vector_base: Vector3) -> Vector3:
        command_frame = self.config.twist_command_frame
        if not command_frame or command_frame == self.config.base_frame:
            return vector_base

        _, _, quaternion_xyzw = self._lookup_frame_pose(command_frame)
        rotation_base_from_command = self._rotation_matrix_from_quaternion(quaternion_xyzw)

        # The Kinova driver consumes twists in the configured command frame.
        # Convert our desired base-frame vector into that local frame via R^T.
        return (
            rotation_base_from_command[0][0] * vector_base[0]
            + rotation_base_from_command[1][0] * vector_base[1]
            + rotation_base_from_command[2][0] * vector_base[2],
            rotation_base_from_command[0][1] * vector_base[0]
            + rotation_base_from_command[1][1] * vector_base[1]
            + rotation_base_from_command[2][1] * vector_base[2],
            rotation_base_from_command[0][2] * vector_base[0]
            + rotation_base_from_command[1][2] * vector_base[1]
            + rotation_base_from_command[2][2] * vector_base[2],
        )

    def _publish_twist_command_ros2(
        self,
        linear_velocity_command: Vector3,
        angular_velocity_command: Vector3,
    ) -> None:
        if self.ros_interface is None:
            raise RuntimeError("ROS2 Kinova interface is not initialized.")

        twist = Twist()
        twist.linear.x = linear_velocity_command[0]
        twist.linear.y = linear_velocity_command[1]
        twist.linear.z = linear_velocity_command[2]
        twist.angular.x = angular_velocity_command[0]
        twist.angular.y = angular_velocity_command[1]
        twist.angular.z = angular_velocity_command[2]

        publish_period = 1.0 / max(self.config.twist_publish_rate_hz, 1.0)
        duration = max(self.config.twist_command_duration_s, 1e-3)
        publish_count = max(1, int(duration / publish_period))
        for _ in range(publish_count):
            self.ros_interface.twist_pub.publish(twist)
            rclpy.spin_once(self.ros_interface, timeout_sec=0.0)
            time.sleep(publish_period)

        zero_twist = Twist()
        stop_duration = max(self.config.twist_stop_duration_s, publish_period)
        stop_count = max(1, int(stop_duration / publish_period))
        for _ in range(stop_count):
            self.ros_interface.twist_pub.publish(zero_twist)
            time.sleep(publish_period)

    def _get_state_ros2(self) -> RobotState:
        self._spin_until_ready()
        ee_position_m, ee_yaw_deg, ee_quaternion_xyzw = self._lookup_ee_pose()
        joint_state = self.ros_interface.latest_joint_state
        joint_positions = []
        for name, position in zip(joint_state.name, joint_state.position):
            if name.startswith("joint_"):
                joint_positions.append(float(position))

        self.ee_position_m = ee_position_m
        self.ee_yaw_deg = ee_yaw_deg
        self.ee_quaternion_xyzw = ee_quaternion_xyzw
        if joint_positions:
            self.joint_positions = joint_positions

        return RobotState(
            joint_positions=list(self.joint_positions),
            ee_position_m=self.ee_position_m,
            ee_yaw_deg=self.ee_yaw_deg,
            gripper_opening_m=self.gripper_opening_m,
            ee_quaternion_xyzw=self.ee_quaternion_xyzw,
        )

    def _move_cartesian_delta_ros2(self, delta_xyz_m: Vector3, delta_yaw_deg: float) -> None:
        if self.ros_interface is None:
            raise RuntimeError("ROS2 Kinova interface is not initialized.")

        duration = max(self.config.twist_command_duration_s, 1e-3)
        command_segments: list[tuple[Vector3, Vector3]] = []

        if self.config.sequential_axis_commands:
            for axis_idx in range(3):
                axis_delta = delta_xyz_m[axis_idx]
                if abs(axis_delta) <= 1e-9:
                    continue
                linear_velocity_base = [0.0, 0.0, 0.0]
                linear_velocity_base[axis_idx] = float(axis_delta / duration)
                command_segments.append(
                    (self._base_vector_to_command_frame(tuple(linear_velocity_base)), (0.0, 0.0, 0.0))
                )
            if abs(delta_yaw_deg) > 1e-9:
                angular_velocity_base = (0.0, 0.0, math.radians(delta_yaw_deg) / duration)
                command_segments.append(((0.0, 0.0, 0.0), self._base_vector_to_command_frame(angular_velocity_base)))
        else:
            linear_velocity_base = tuple(float(delta / duration) for delta in delta_xyz_m)
            angular_velocity_base = (0.0, 0.0, math.radians(delta_yaw_deg) / duration)
            command_segments.append(
                (
                    self._base_vector_to_command_frame(linear_velocity_base),
                    self._base_vector_to_command_frame(angular_velocity_base),
                )
            )

        for linear_velocity_command, angular_velocity_command in command_segments:
            self._publish_twist_command_ros2(linear_velocity_command, angular_velocity_command)
        self._get_state_ros2()

    def start_continuous_twist_servo(
        self,
        publish_rate_hz: float | None = None,
        stale_timeout_s: float = 2.0,
        command_alpha: float = 1.0,
        max_linear_speed_mps: float = 0.08,
        max_angular_speed_degps: float = 20.0,
    ) -> None:
        if self.config.mode != "ros2_twist":
            return
        if self.ros_interface is None:
            raise RuntimeError("ROS2 Kinova interface is not initialized.")
        if self._continuous_servo_running:
            return

        with self._continuous_servo_lock:
            self._continuous_servo_publish_rate_hz = (
                float(publish_rate_hz)
                if publish_rate_hz is not None
                else float(self.config.twist_publish_rate_hz)
            )
            self._continuous_servo_publish_rate_hz = max(self._continuous_servo_publish_rate_hz, 1.0)
            self._continuous_servo_stale_timeout_s = max(float(stale_timeout_s), 0.1)
            self._continuous_servo_command_alpha = max(0.0, min(float(command_alpha), 1.0))
            self._continuous_servo_max_linear_speed_mps = max(float(max_linear_speed_mps), 1e-4)
            self._continuous_servo_max_angular_speed_rps = max(
                math.radians(float(max_angular_speed_degps)),
                1e-4,
            )
            self._continuous_servo_last_update_time = time.monotonic()
            self._continuous_servo_target_linear_command = (0.0, 0.0, 0.0)
            self._continuous_servo_target_angular_command = (0.0, 0.0, 0.0)
            self._continuous_servo_linear_command = (0.0, 0.0, 0.0)
            self._continuous_servo_angular_command = (0.0, 0.0, 0.0)
            self._continuous_servo_running = True

        self._continuous_servo_thread = threading.Thread(
            target=self._continuous_twist_servo_loop,
            name="kinova_continuous_twist_servo",
            daemon=True,
        )
        self._continuous_servo_thread.start()

    def set_continuous_twist_delta(
        self,
        delta_xyz_m: Vector3,
        delta_yaw_deg: float,
        horizon_s: float = 0.25,
    ) -> None:
        if self.config.mode != "ros2_twist":
            self.move_cartesian_delta(delta_xyz_m, delta_yaw_deg)
            return
        if self.ros_interface is None:
            raise RuntimeError("ROS2 Kinova interface is not initialized.")
        if not self._continuous_servo_running:
            raise RuntimeError("Continuous twist servo is not running. Call start_continuous_twist_servo first.")

        duration = max(float(horizon_s), 1e-3)
        linear_velocity_base_unclipped = tuple(float(delta / duration) for delta in delta_xyz_m)
        angular_velocity_base_unclipped = (0.0, 0.0, math.radians(float(delta_yaw_deg)) / duration)
        linear_speed = math.sqrt(sum(component * component for component in linear_velocity_base_unclipped))
        if linear_speed > self._continuous_servo_max_linear_speed_mps:
            linear_scale = self._continuous_servo_max_linear_speed_mps / max(linear_speed, 1e-9)
            linear_velocity_base = tuple(component * linear_scale for component in linear_velocity_base_unclipped)
        else:
            linear_velocity_base = linear_velocity_base_unclipped

        angular_speed = abs(angular_velocity_base_unclipped[2])
        if angular_speed > self._continuous_servo_max_angular_speed_rps:
            angular_velocity_base = (
                0.0,
                0.0,
                math.copysign(self._continuous_servo_max_angular_speed_rps, angular_velocity_base_unclipped[2]),
            )
        else:
            angular_velocity_base = angular_velocity_base_unclipped
        linear_velocity_command = self._base_vector_to_command_frame(linear_velocity_base)
        angular_velocity_command = self._base_vector_to_command_frame(angular_velocity_base)

        with self._continuous_servo_lock:
            self._continuous_servo_target_linear_command = linear_velocity_command
            self._continuous_servo_target_angular_command = angular_velocity_command
            self._continuous_servo_last_update_time = time.monotonic()

    def stop_continuous_twist_servo(self, stop_duration_s: float = 0.3) -> None:
        if self.config.mode != "ros2_twist":
            return
        if self.ros_interface is None:
            return
        if not self._continuous_servo_running and self._continuous_servo_thread is None:
            return

        with self._continuous_servo_lock:
            self._continuous_servo_running = False
            self._continuous_servo_target_linear_command = (0.0, 0.0, 0.0)
            self._continuous_servo_target_angular_command = (0.0, 0.0, 0.0)
            self._continuous_servo_linear_command = (0.0, 0.0, 0.0)
            self._continuous_servo_angular_command = (0.0, 0.0, 0.0)

        if self._continuous_servo_thread is not None:
            self._continuous_servo_thread.join(timeout=1.0)
            self._continuous_servo_thread = None

        zero_twist = Twist()
        publish_period = 1.0 / max(self.config.twist_publish_rate_hz, 1.0)
        stop_count = max(1, int(max(stop_duration_s, publish_period) / publish_period))
        for _ in range(stop_count):
            self.ros_interface.twist_pub.publish(zero_twist)
            time.sleep(publish_period)

    def refresh_continuous_twist_watchdog(self) -> None:
        if self.config.mode != "ros2_twist":
            return
        if not self._continuous_servo_running:
            return
        with self._continuous_servo_lock:
            self._continuous_servo_last_update_time = time.monotonic()

    def _continuous_twist_servo_loop(self) -> None:
        if self.ros_interface is None:
            return
        publish_period = 1.0 / max(self._continuous_servo_publish_rate_hz, 1.0)
        zero_twist = Twist()
        while True:
            with self._continuous_servo_lock:
                running = self._continuous_servo_running
                target_linear_velocity_command = self._continuous_servo_target_linear_command
                target_angular_velocity_command = self._continuous_servo_target_angular_command
                linear_velocity_command = self._continuous_servo_linear_command
                angular_velocity_command = self._continuous_servo_angular_command
                last_update_time = self._continuous_servo_last_update_time
                stale_timeout_s = self._continuous_servo_stale_timeout_s
                command_alpha = self._continuous_servo_command_alpha
            if not running:
                break

            stale = (time.monotonic() - last_update_time) > stale_timeout_s
            if stale:
                target_linear_velocity_command = (0.0, 0.0, 0.0)
                target_angular_velocity_command = (0.0, 0.0, 0.0)
                linear_velocity_command = (0.0, 0.0, 0.0)
                angular_velocity_command = (0.0, 0.0, 0.0)
                twist = zero_twist
            else:
                linear_velocity_command = tuple(
                    current + command_alpha * (target - current)
                    for current, target in zip(linear_velocity_command, target_linear_velocity_command)
                )
                angular_velocity_command = tuple(
                    current + command_alpha * (target - current)
                    for current, target in zip(angular_velocity_command, target_angular_velocity_command)
                )
                twist = Twist()
                twist.linear.x = linear_velocity_command[0]
                twist.linear.y = linear_velocity_command[1]
                twist.linear.z = linear_velocity_command[2]
                twist.angular.x = angular_velocity_command[0]
                twist.angular.y = angular_velocity_command[1]
                twist.angular.z = angular_velocity_command[2]

            with self._continuous_servo_lock:
                self._continuous_servo_linear_command = linear_velocity_command
                self._continuous_servo_angular_command = angular_velocity_command
                if stale:
                    self._continuous_servo_target_linear_command = (0.0, 0.0, 0.0)
                    self._continuous_servo_target_angular_command = (0.0, 0.0, 0.0)

            self.ros_interface.twist_pub.publish(twist)
            time.sleep(publish_period)
