from __future__ import annotations

import argparse
import math
import threading
import time
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

try:
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2, Session_pb2
except ImportError:  # pragma: no cover - depends on Kinova Kortex runtime
    RouterClient = None  # type: ignore[assignment]
    SessionManager = None  # type: ignore[assignment]
    TCPTransport = None  # type: ignore[assignment]
    BaseClient = None  # type: ignore[assignment]
    BaseCyclicClient = None  # type: ignore[assignment]
    Base_pb2 = None  # type: ignore[assignment]
    Session_pb2 = None  # type: ignore[assignment]

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from tf2_ros import Buffer, TransformListener
    from tf_transformations import euler_from_quaternion
except ImportError:  # pragma: no cover - depends on ROS2 runtime
    rclpy = None  # type: ignore[assignment]
    Twist = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment]
    qos_profile_sensor_data = None  # type: ignore[assignment]
    JointState = object  # type: ignore[assignment]
    Buffer = None  # type: ignore[assignment]
    TransformListener = None  # type: ignore[assignment]
    euler_from_quaternion = None  # type: ignore[assignment]


class _KinovaROSInterface(Node):  # type: ignore[misc, valid-type]
    def __init__(self, node_name: str, joint_state_topic: str, twist_command_topic: str) -> None:
        super().__init__(node_name)
        self.latest_joint_state: Any | None = None
        self.joint_state_sub = self.create_subscription(
            JointState,
            joint_state_topic,
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.twist_pub = self.create_publisher(Twist, twist_command_topic, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

    def _on_joint_state(self, msg: Any) -> None:
        self.latest_joint_state = msg


class KinovaRobot:
    """
    Kinova Gen3 interface with two real backends:

    - `ros2_twist`: reuse the previous project stack. Reads `/joint_states` and
      TF, publishes `geometry_msgs/Twist` continuously in the configured command
      frame.
    - `kortex_twist`: direct Kortex placeholder backend kept for future use.

    State layout, shape `(14,)`:
    [eef_x, eef_y, eef_z,
     eef_roll, eef_pitch, eef_yaw,
     gripper_pos,
     joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]
    """

    def __init__(
        self,
        ip: str,
        username: str,
        password: str,
        dry_run: bool = False,
        max_linear_speed: float = 0.05,
        mode: str = "kortex_twist",
        joint_state_topic: str = "/joint_states",
        twist_command_topic: str = "/twist_controller/commands",
        base_frame: str = "base_link",
        ee_frame: str = "end_effector_link",
        twist_command_frame: str = "tool_frame",
        sequential_axis_commands: bool = True,
        ros_node_name: str = "kinova_vla_collect_robot",
        state_timeout_s: float = 5.0,
        twist_publish_rate_hz: float = 20.0,
        twist_stop_duration_s: float = 0.6,
    ) -> None:
        self.ip = ip
        self.username = username
        self.password = password
        self.dry_run = dry_run
        self.mode = mode
        self.max_linear_speed = float(max_linear_speed)
        if self.max_linear_speed <= 0.0:
            raise ValueError("max_linear_speed must be positive")

        self.joint_state_topic = joint_state_topic
        self.twist_command_topic = twist_command_topic
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.twist_command_frame = twist_command_frame
        self.sequential_axis_commands = sequential_axis_commands
        self.ros_node_name = ros_node_name
        self.state_timeout_s = state_timeout_s
        self.twist_publish_rate_hz = max(1.0, twist_publish_rate_hz)
        self.twist_stop_duration_s = max(0.0, twist_stop_duration_s)

        self._connected = False

        self._transport: Any | None = None
        self._router: Any | None = None
        self._session_manager: Any | None = None
        self._base: Any | None = None
        self._base_cyclic: Any | None = None

        self._ros_interface: _KinovaROSInterface | None = None
        self._ros_lock = threading.Lock()
        self._ros_running = False
        self._ros_thread: threading.Thread | None = None
        self._ros_target_twist = np.zeros(6, dtype=np.float32)
        self._ros_last_command_time = 0.0
        self._ros_stale_timeout_s = 0.5

        self._simulated_state = np.zeros(14, dtype=np.float32)
        self._simulated_state[0:3] = np.array([0.40, 0.0, 0.30], dtype=np.float32)
        self._last_twist = np.zeros(6, dtype=np.float32)

    def __enter__(self) -> "KinovaRobot":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.disconnect()

    def connect(self) -> None:
        if self._connected:
            return
        if self.dry_run:
            self._connected = True
            return
        if self.mode == "ros2_twist":
            self._connect_ros2()
        else:
            self._connect_kortex()
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected and self._transport is None and self._ros_interface is None:
            return
        try:
            self.stop()
        except Exception as exc:
            print(f"Warning: Kinova stop during disconnect failed: {exc}")
        self._cleanup_ros2()
        self._cleanup_kortex()
        self._connected = False

    def get_state(self) -> FloatArray:
        self._ensure_connected()
        if self.dry_run:
            return self._simulated_state.copy()
        if self.mode == "ros2_twist":
            return self._get_state_ros2()
        return self._get_state_kortex()

    def step_delta_action(self, action: FloatArray, dt: float) -> None:
        self._ensure_connected()
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape not in {(4,), (7,)}:
            raise ValueError(f"Kinova action must have shape (4,) or (7,), got {action_array.shape}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        linear_velocity = action_array[:3] / float(dt)
        limited_linear_velocity = self._limit_linear_velocity(linear_velocity)
        angular_velocity = np.zeros(3, dtype=np.float32)
        if action_array.shape == (7,):
            angular_velocity = action_array[3:6] / float(dt)
        twist_base = np.array(
            [
                limited_linear_velocity[0],
                limited_linear_velocity[1],
                limited_linear_velocity[2],
                angular_velocity[0],
                angular_velocity[1],
                angular_velocity[2],
            ],
            dtype=np.float32,
        )

        if self.dry_run:
            self._last_twist = twist_base
            self._simulated_state[0:3] += limited_linear_velocity * float(dt)
            self._simulated_state[3:6] += angular_velocity * float(dt)
            self._simulated_state[7:10] += limited_linear_velocity * float(dt) * 0.1
            return

        if self.mode == "ros2_twist":
            self._set_ros_twist_command(twist_base, stale_timeout_s=max(2.5 * dt, 0.3))
        else:
            self._send_cartesian_twist_velocity_kortex(twist_base)

    def command_ee_delta(self, dx: float, dy: float, dz: float, dt: float = 0.2) -> None:
        action = np.array([dx, dy, dz, 0.0], dtype=np.float32)
        self.step_delta_action(action, dt=dt)

    def stop(self) -> None:
        if not self._connected:
            return
        zero_twist = np.zeros(6, dtype=np.float32)
        if self.dry_run:
            self._last_twist = zero_twist
            return
        if self.mode == "ros2_twist":
            self._set_ros_twist_command(zero_twist, stale_timeout_s=0.0)
            self._publish_ros_twist(zero_twist)
            self._publish_zero_for_duration(self.twist_stop_duration_s)
        else:
            self._send_cartesian_twist_velocity_kortex(zero_twist)

    def emergency_stop(self) -> None:
        self._ensure_connected()
        if self.mode == "ros2_twist" or self.dry_run:
            self.stop()
            return
        if self._base is None:
            raise RuntimeError("Kinova BaseClient is not initialized")
        try:
            self._base.Stop()
        except Exception as exc:
            raise RuntimeError("Failed to send Kinova emergency stop command") from exc

    def _connect_ros2(self) -> None:
        self._ensure_ros2_available()
        if not rclpy.ok():
            rclpy.init(args=None)
        self._ros_interface = _KinovaROSInterface(
            node_name=self.ros_node_name,
            joint_state_topic=self.joint_state_topic,
            twist_command_topic=self.twist_command_topic,
        )
        self._wait_for_ros_state()
        self._start_ros_publisher()

    def _start_ros_publisher(self) -> None:
        if self._ros_running:
            return
        self._ros_running = True
        self._ros_last_command_time = time.monotonic()
        self._ros_thread = threading.Thread(
            target=self._ros_publish_loop,
            name="kinova_vla_ros_twist_publisher",
            daemon=True,
        )
        self._ros_thread.start()

    def _ros_publish_loop(self) -> None:
        period = 1.0 / self.twist_publish_rate_hz
        zero_twist = np.zeros(6, dtype=np.float32)
        while self._ros_running:
            with self._ros_lock:
                target = self._ros_target_twist.copy()
                last_command_time = self._ros_last_command_time
                stale_timeout_s = self._ros_stale_timeout_s
            if stale_timeout_s > 0.0 and (time.monotonic() - last_command_time) > stale_timeout_s:
                target = zero_twist
            try:
                self._publish_ros_twist(target)
            except Exception as exc:
                print(f"Warning: failed to publish ROS2 twist: {exc}")
            time.sleep(period)

    def _set_ros_twist_command(self, twist_base: FloatArray, stale_timeout_s: float) -> None:
        command_twist = twist_base.copy()
        command_twist[:3] = self._base_vector_to_command_frame(twist_base[:3])
        command_twist[3:6] = self._base_vector_to_command_frame(twist_base[3:6])
        with self._ros_lock:
            self._ros_target_twist = command_twist.astype(np.float32)
            self._ros_last_command_time = time.monotonic()
            self._ros_stale_timeout_s = stale_timeout_s

    def _publish_ros_twist(self, twist_array: FloatArray) -> None:
        if self._ros_interface is None or Twist is None:
            return
        twist = Twist()
        twist.linear.x = float(twist_array[0])
        twist.linear.y = float(twist_array[1])
        twist.linear.z = float(twist_array[2])
        twist.angular.x = float(twist_array[3])
        twist.angular.y = float(twist_array[4])
        twist.angular.z = float(twist_array[5])
        self._ros_interface.twist_pub.publish(twist)
        rclpy.spin_once(self._ros_interface, timeout_sec=0.0)

    def _publish_zero_for_duration(self, duration_s: float) -> None:
        if duration_s <= 0.0:
            return
        period = 1.0 / self.twist_publish_rate_hz
        zero_twist = np.zeros(6, dtype=np.float32)
        end_time = time.monotonic() + duration_s
        while time.monotonic() < end_time:
            self._publish_ros_twist(zero_twist)
            time.sleep(period)

    def _get_state_ros2(self) -> FloatArray:
        self._wait_for_ros_state()
        position, rpy = self._lookup_ee_pose()
        joint_positions = self._read_joint_positions()
        state = np.zeros(14, dtype=np.float32)
        state[0:3] = np.array(position, dtype=np.float32)
        state[3:6] = np.array(rpy, dtype=np.float32)
        state[6] = 0.0
        state[7:14] = np.array(joint_positions[:7], dtype=np.float32)
        return state

    def _wait_for_ros_state(self) -> None:
        if self._ros_interface is None:
            raise RuntimeError("ROS2 Kinova interface is not initialized")
        deadline = time.monotonic() + self.state_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self._ros_interface, timeout_sec=0.05)
            if self._ros_interface.latest_joint_state is not None:
                return
        raise TimeoutError(f"Timed out waiting for joint states on {self.joint_state_topic}")

    def _lookup_ee_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self._ros_interface is None or euler_from_quaternion is None:
            raise RuntimeError("ROS2 Kinova interface is not initialized")
        deadline = time.monotonic() + self.state_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self._ros_interface, timeout_sec=0.02)
            try:
                transform = self._ros_interface.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.ee_frame,
                    rclpy.time.Time(),
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                roll, pitch, yaw = euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
                return (
                    (float(translation.x), float(translation.y), float(translation.z)),
                    (float(roll), float(pitch), float(yaw)),
                )
            except Exception as exc:
                last_error = exc
                time.sleep(0.02)
        raise TimeoutError(f"Timed out looking up TF {self.base_frame} -> {self.ee_frame}: {last_error}")

    def _read_joint_positions(self) -> list[float]:
        if self._ros_interface is None or self._ros_interface.latest_joint_state is None:
            return [0.0] * 7
        joint_state = self._ros_interface.latest_joint_state
        pairs = [
            (str(name), float(position))
            for name, position in zip(joint_state.name, joint_state.position)
            if str(name).startswith("joint_")
        ]
        pairs.sort(key=lambda item: _joint_sort_key(item[0]))
        positions = [position for _, position in pairs[:7]]
        while len(positions) < 7:
            positions.append(0.0)
        return positions

    def _base_vector_to_command_frame(self, vector_base: FloatArray) -> FloatArray:
        if not self.twist_command_frame or self.twist_command_frame == self.base_frame:
            return vector_base.astype(np.float32)
        if self._ros_interface is None:
            return vector_base.astype(np.float32)
        try:
            transform = self._ros_interface.tf_buffer.lookup_transform(
                self.base_frame,
                self.twist_command_frame,
                rclpy.time.Time(),
            )
            rotation = transform.transform.rotation
            matrix = _rotation_matrix_from_quaternion((rotation.x, rotation.y, rotation.z, rotation.w))
            command_vector = np.array(
                [
                    matrix[0][0] * vector_base[0] + matrix[1][0] * vector_base[1] + matrix[2][0] * vector_base[2],
                    matrix[0][1] * vector_base[0] + matrix[1][1] * vector_base[1] + matrix[2][1] * vector_base[2],
                    matrix[0][2] * vector_base[0] + matrix[1][2] * vector_base[1] + matrix[2][2] * vector_base[2],
                ],
                dtype=np.float32,
            )
            return command_vector
        except Exception as exc:
            raise RuntimeError(
                f"Failed to transform twist from {self.base_frame} to {self.twist_command_frame}"
            ) from exc

    def _cleanup_ros2(self) -> None:
        self._ros_running = False
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=1.0)
            self._ros_thread = None
        if self._ros_interface is not None:
            try:
                self._ros_interface.destroy_node()
            except Exception as exc:
                print(f"Warning: failed to destroy ROS2 Kinova node: {exc}")
        self._ros_interface = None

    def _connect_kortex(self) -> None:
        self._ensure_kortex_available()
        try:
            self._transport = TCPTransport()
            self._transport.connect(self.ip, 10000)
            self._router = RouterClient(self._transport, self._kortex_error_callback)
            self._session_manager = SessionManager(self._router)
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.username
            session_info.password = self.password
            session_info.session_inactivity_timeout = 60000
            session_info.connection_inactivity_timeout = 2000
            self._session_manager.CreateSession(session_info)
            self._base = BaseClient(self._router)
            self._base_cyclic = BaseCyclicClient(self._router)
        except Exception as exc:
            self._cleanup_kortex()
            raise RuntimeError(f"Failed to connect Kinova Gen3 at {self.ip}: {exc}") from exc

    def _get_state_kortex(self) -> FloatArray:
        if self._base_cyclic is None:
            raise RuntimeError("Kinova BaseCyclic client is not initialized")
        try:
            feedback = self._base_cyclic.RefreshFeedback()
            state = np.zeros(14, dtype=np.float32)
            state[0] = float(feedback.base.tool_pose_x)
            state[1] = float(feedback.base.tool_pose_y)
            state[2] = float(feedback.base.tool_pose_z)
            state[3] = math.radians(float(feedback.base.tool_pose_theta_x))
            state[4] = math.radians(float(feedback.base.tool_pose_theta_y))
            state[5] = math.radians(float(feedback.base.tool_pose_theta_z))
            state[6] = 0.0
            for index in range(min(7, len(feedback.actuators))):
                state[7 + index] = math.radians(float(feedback.actuators[index].position))
            return state
        except Exception as exc:
            raise RuntimeError("Failed to read/parse Kinova cyclic feedback") from exc

    def _send_cartesian_twist_velocity_kortex(self, twist: FloatArray) -> None:
        if self._base is None or Base_pb2 is None:
            raise RuntimeError("Kinova BaseClient is not initialized")
        try:
            command = Base_pb2.TwistCommand()
            if hasattr(Base_pb2, "CARTESIAN_REFERENCE_FRAME_BASE"):
                command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            command.duration = 0
            command.twist.linear_x = float(twist[0])
            command.twist.linear_y = float(twist[1])
            command.twist.linear_z = float(twist[2])
            command.twist.angular_x = float(twist[3])
            command.twist.angular_y = float(twist[4])
            command.twist.angular_z = float(twist[5])
            self._base.SendTwistCommand(command)
        except Exception as exc:
            raise RuntimeError("Failed to send Kinova Kortex Cartesian twist command") from exc

    def _limit_linear_velocity(self, linear_velocity: FloatArray) -> FloatArray:
        velocity = np.asarray(linear_velocity, dtype=np.float32)
        speed = float(np.linalg.norm(velocity))
        if speed <= self.max_linear_speed:
            return velocity
        return (velocity * (self.max_linear_speed / max(speed, 1e-9))).astype(np.float32)

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("KinovaRobot is not connected. Call connect() first.")

    @staticmethod
    def _ensure_ros2_available() -> None:
        missing = []
        for name, value in {
            "rclpy": rclpy,
            "geometry_msgs.msg.Twist": Twist,
            "sensor_msgs.msg.JointState": JointState,
            "tf2_ros.Buffer": Buffer,
            "tf2_ros.TransformListener": TransformListener,
            "tf_transformations.euler_from_quaternion": euler_from_quaternion,
        }.items():
            if value is None:
                missing.append(name)
        if missing:
            raise RuntimeError(
                "ROS2 Kinova mode requires sourced ROS2/Kinova environment. Missing: "
                + ", ".join(missing)
            )

    @staticmethod
    def _ensure_kortex_available() -> None:
        missing = [
            name
            for name, value in {
                "kortex_api.RouterClient": RouterClient,
                "kortex_api.SessionManager": SessionManager,
                "kortex_api.TCPTransport": TCPTransport,
                "kortex_api BaseClient": BaseClient,
                "kortex_api BaseCyclicClient": BaseCyclicClient,
                "kortex_api Base_pb2": Base_pb2,
                "kortex_api Session_pb2": Session_pb2,
            }.items()
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "Kinova Kortex API import failed. Install Kinova Kortex Python API "
                "or run with dry_run=True. Missing: " + ", ".join(missing)
            )

    def _cleanup_kortex(self) -> None:
        if self._session_manager is not None:
            try:
                self._session_manager.CloseSession()
            except Exception as exc:
                print(f"Warning: failed to close Kinova session: {exc}")
        if self._transport is not None:
            try:
                self._transport.disconnect()
            except Exception as exc:
                print(f"Warning: failed to disconnect Kinova transport: {exc}")
        self._base = None
        self._base_cyclic = None
        self._session_manager = None
        self._router = None
        self._transport = None

    @staticmethod
    def _kortex_error_callback(error: Any) -> None:
        print(f"Kinova Kortex router error: {error}")


def _joint_sort_key(name: str) -> tuple[int, str]:
    suffix = name.split("_")[-1]
    try:
        return int(suffix), name
    except ValueError:
        return 999, name


def _rotation_matrix_from_quaternion(quaternion_xyzw: tuple[float, float, float, float]) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    x, y, z, w = quaternion_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Test KinovaRobot delta action interface.")
    parser.add_argument("--ip", type=str, default="192.168.1.10")
    parser.add_argument("--username", type=str, default="admin")
    parser.add_argument("--password", type=str, default="admin")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--mode", type=str, default="ros2_twist")
    parser.add_argument("--max-linear-speed", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=0.2)
    args = parser.parse_args(argv)

    actions = [
        np.array([0.005, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.005, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, -0.005, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    ]

    with KinovaRobot(
        ip=args.ip,
        username=args.username,
        password=args.password,
        dry_run=args.dry_run,
        mode=args.mode,
        max_linear_speed=args.max_linear_speed,
    ) as robot:
        print(f"initial state shape={robot.get_state().shape} state={robot.get_state().tolist()}")
        for index, action in enumerate(actions):
            robot.step_delta_action(action, dt=args.dt)
            print(f"step={index} action={action.tolist()} state={robot.get_state().tolist()}")
        robot.stop()
        print("stopped")


if __name__ == "__main__":
    main()
