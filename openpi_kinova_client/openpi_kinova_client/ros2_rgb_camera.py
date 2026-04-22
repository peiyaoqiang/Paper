from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo
    from sensor_msgs.msg import Image as ROSImage
except ImportError:  # pragma: no cover - depends on ROS2 runtime
    rclpy = None
    Node = object
    CameraInfo = object
    ROSImage = object


@dataclass(frozen=True)
class RGBFrame:
    rgb_path_hint: str
    width: int
    height: int
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None


@dataclass(frozen=True)
class ROS2RGBCameraConfig:
    width: int
    height: int
    mode: str = "ros2"
    color_topic: str = "/camera/camera/color/image_raw"
    camera_info_topic: str = "/camera/camera/color/camera_info"
    capture_timeout_s: float = 3.0
    output_dir: str = "analysis/captures"
    ros_node_name: str = "openpi_ros2_rgb_camera"


class _ROS2RGBSubscriber(Node):
    def __init__(self, config: ROS2RGBCameraConfig) -> None:
        super().__init__(config.ros_node_name)
        self.latest_rgb: np.ndarray | None = None
        self.latest_camera_info: CameraInfo | None = None
        self.create_subscription(ROSImage, config.color_topic, self._on_color, 10)
        self.create_subscription(CameraInfo, config.camera_info_topic, self._on_camera_info, 10)

    def _on_color(self, msg: ROSImage) -> None:
        self.latest_rgb = ros_image_to_rgb(msg)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg


def ros_image_to_rgb(msg) -> np.ndarray:  # type: ignore[no-untyped-def]
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("rgb8", "bgr8"):
        row_width = width * 3
        image = raw.reshape(height, step)[:, :row_width].reshape(height, width, 3)
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        return image.copy()

    if encoding in ("rgba8", "bgra8"):
        row_width = width * 4
        image = raw.reshape(height, step)[:, :row_width].reshape(height, width, 4)
        if encoding == "bgra8":
            image = image[:, :, [2, 1, 0, 3]]
        return image[:, :, :3].copy()

    if encoding == "mono8":
        image = raw.reshape(height, step)[:, :width].reshape(height, width)
        return np.repeat(image[:, :, None], 3, axis=2).copy()

    raise ValueError(f"Unsupported ROS image encoding for RGB conversion: {msg.encoding}")


class ROS2RGBCamera:
    """Small ROS2 RGB camera reader that avoids cv_bridge/NumPy ABI issues."""

    def __init__(self, config: ROS2RGBCameraConfig) -> None:
        self.config = config
        self.frame_index = 0
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ros_node: _ROS2RGBSubscriber | None = None
        if config.mode == "ros2":
            self._init_ros2()

    def _init_ros2(self) -> None:
        if rclpy is None:
            raise RuntimeError("ROS2 RGB camera mode requires rclpy and sensor_msgs.")
        if not rclpy.ok():
            rclpy.init(args=None)
        self.ros_node = _ROS2RGBSubscriber(self.config)

    def capture_frame(self) -> RGBFrame:
        if self.config.mode != "ros2":
            return self._capture_mock()
        return self._capture_ros2()

    def _capture_mock(self) -> RGBFrame:
        self.frame_index += 1
        rgb = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        rgb_path = self.output_dir / f"openpi_rgb_{self.frame_index:04d}.png"
        Image.fromarray(rgb).save(rgb_path)
        return RGBFrame(str(rgb_path), self.config.width, self.config.height)

    def _capture_ros2(self) -> RGBFrame:
        if self.ros_node is None:
            raise RuntimeError("ROS2 RGB camera node is not initialized.")

        deadline = time.monotonic() + self.config.capture_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            if self.ros_node.latest_rgb is not None:
                break

        if self.ros_node.latest_rgb is None:
            raise TimeoutError(f"Timed out waiting for RGB image on {self.config.color_topic}")

        self.frame_index += 1
        rgb = self.ros_node.latest_rgb
        rgb_path = self.output_dir / f"openpi_rgb_{self.frame_index:04d}.png"
        Image.fromarray(rgb).save(rgb_path)

        fx = fy = cx = cy = None
        if self.ros_node.latest_camera_info is not None:
            intrinsics = self.ros_node.latest_camera_info.k
            fx = float(intrinsics[0])
            fy = float(intrinsics[4])
            cx = float(intrinsics[2])
            cy = float(intrinsics[5])

        return RGBFrame(
            rgb_path_hint=str(rgb_path),
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
