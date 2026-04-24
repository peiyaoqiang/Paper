from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

from common.types import CameraFrame

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


@dataclass
class RealSenseConfig:
    width: int
    height: int
    mode: str = "mock"
    color_topic: str = "/camera/camera/color/image_raw"
    aligned_depth_topic: str = "/camera/camera/aligned_depth_to_color/image_raw"
    camera_info_topic: str = "/camera/camera/color/camera_info"
    capture_timeout_s: float = 3.0
    output_dir: str = "analysis/captures"
    ros_node_name: str = "paper_realsense_driver"


class _RealSenseROSSubscriber(Node):
    def __init__(self, config: RealSenseConfig) -> None:
        super().__init__(config.ros_node_name)
        self.latest_rgb: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_camera_info: CameraInfo | None = None

        self.create_subscription(ROSImage, config.color_topic, self._on_color, 10)
        self.create_subscription(ROSImage, config.aligned_depth_topic, self._on_depth, 10)
        self.create_subscription(CameraInfo, config.camera_info_topic, self._on_camera_info, 10)

    def _on_color(self, msg: ROSImage) -> None:
        self.latest_rgb = ros_image_to_rgb(msg)

    def _on_depth(self, msg: ROSImage) -> None:
        self.latest_depth = ros_image_to_depth(msg)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg


def ros_image_to_rgb(msg: ROSImage) -> np.ndarray:
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

    if encoding in ("mono8", "8uc1"):
        image = raw.reshape(height, step)[:, :width].reshape(height, width)
        return np.repeat(image[:, :, None], 3, axis=2).copy()

    raise ValueError(f"Unsupported ROS color image encoding: {msg.encoding}")


def ros_image_to_depth(msg: ROSImage) -> np.ndarray:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if encoding in ("16uc1", "mono16"):
        depth = _ros_image_to_2d_array(msg, np.uint16, height, width, step)
    elif encoding == "32fc1":
        depth = _ros_image_to_2d_array(msg, np.float32, height, width, step)
    elif encoding in ("8uc1", "mono8"):
        depth = _ros_image_to_2d_array(msg, np.uint8, height, width, step)
    else:
        raise ValueError(f"Unsupported ROS depth image encoding: {msg.encoding}")

    return depth.copy()


def _ros_image_to_2d_array(
    msg: ROSImage,
    dtype: type[np.generic],
    height: int,
    width: int,
    step: int,
) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    row_items = step // itemsize
    image = np.frombuffer(msg.data, dtype=dtype).reshape(height, row_items)[:, :width]
    msg_big_endian = bool(msg.is_bigendian)
    host_big_endian = sys.byteorder == "big"
    if msg_big_endian != host_big_endian and itemsize > 1:
        image = image.byteswap()
    return image


class RealSenseDriver:
    """
    Minimal RealSense driver interface.

    Supports:

    - `mock`: synthetic file hints only
    - `ros2`: subscribe to RealSense ROS2 image topics and save the latest RGB/depth locally
    """

    def __init__(self, config: RealSenseConfig) -> None:
        self.config = config
        self.frame_index = 0
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ros_node: _RealSenseROSSubscriber | None = None

        if self.config.mode == "ros2":
            self._init_ros2()

    def _init_ros2(self) -> None:
        if rclpy is None:
            raise RuntimeError(
                "ROS2 RealSense mode requires `rclpy` and `sensor_msgs` to be installed."
            )
        if not rclpy.ok():
            rclpy.init(args=None)
        self.ros_node = _RealSenseROSSubscriber(self.config)

    def capture_frame(self) -> CameraFrame:
        if self.config.mode == "ros2":
            return self._capture_frame_ros2()
        return self._capture_frame_mock()

    def _capture_frame_mock(self) -> CameraFrame:
        self.frame_index += 1
        return CameraFrame(
            rgb_path_hint=f"mock_rgb_frame_{self.frame_index:04d}.png",
            depth_path_hint=f"mock_depth_frame_{self.frame_index:04d}.npy",
            width=self.config.width,
            height=self.config.height,
        )

    def _capture_frame_ros2(self) -> CameraFrame:
        if self.ros_node is None:
            raise RuntimeError("ROS2 RealSense node is not initialized.")

        deadline = time.monotonic() + self.config.capture_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            if (
                self.ros_node.latest_rgb is not None
                and self.ros_node.latest_depth is not None
                and self.ros_node.latest_camera_info is not None
            ):
                break

        if self.ros_node.latest_rgb is None:
            raise TimeoutError(f"Timed out waiting for RGB image on {self.config.color_topic}")
        if self.ros_node.latest_depth is None:
            raise TimeoutError(
                f"Timed out waiting for aligned depth image on {self.config.aligned_depth_topic}"
            )
        if self.ros_node.latest_camera_info is None:
            raise TimeoutError(
                f"Timed out waiting for camera info on {self.config.camera_info_topic}"
            )

        self.frame_index += 1
        rgb = self.ros_node.latest_rgb
        depth = self.ros_node.latest_depth

        rgb_path = self.output_dir / f"realsense_rgb_{self.frame_index:04d}.png"
        depth_path = self.output_dir / f"realsense_depth_{self.frame_index:04d}.npy"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth)

        fx = fy = cx = cy = None
        if self.ros_node.latest_camera_info is not None:
            intrinsics = self.ros_node.latest_camera_info.k
            fx = float(intrinsics[0])
            fy = float(intrinsics[4])
            cx = float(intrinsics[2])
            cy = float(intrinsics[5])

        return CameraFrame(
            rgb_path_hint=str(rgb_path),
            depth_path_hint=str(depth_path),
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
