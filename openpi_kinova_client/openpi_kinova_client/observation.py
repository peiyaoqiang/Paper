from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from .config import ObservationConfig
from .types import RobotState


logger = logging.getLogger(__name__)


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    return image.astype(np.uint8, copy=False)


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    image = convert_to_uint8(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
    if image.shape[:2] == (height, width):
        return image

    pil_image = Image.fromarray(image)
    cur_width, cur_height = pil_image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_width = max(1, int(cur_width / ratio))
    resized_height = max(1, int(cur_height / ratio))
    resized = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    padded = Image.new("RGB", (width, height), 0)
    pad_x = max(0, (width - resized_width) // 2)
    pad_y = max(0, (height - resized_height) // 2)
    padded.paste(resized, (pad_x, pad_y))
    return np.asarray(padded, dtype=np.uint8)


def make_dummy_image(size: int, value: int = 0) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


class CameraSource:
    def __init__(self, config: ObservationConfig) -> None:
        self.config = config
        self._cv2 = None
        self._capture = None

    def __enter__(self) -> "CameraSource":
        if self.config.dummy_images:
            return self
        try:
            import cv2  # type: ignore[import-not-found]

            self._cv2 = cv2
            self._capture = cv2.VideoCapture(self.config.camera_index)
            if not self._capture.isOpened():
                logger.warning("Camera %s is not available; using dummy images", self.config.camera_index)
                self._capture = None
        except Exception as exc:  # pragma: no cover - depends on local camera stack
            logger.warning("OpenCV camera init failed: %s; using dummy images", exc)
            self._capture = None
        return self

    def read_rgb(self) -> np.ndarray:
        if self._capture is None or self._cv2 is None:
            return make_dummy_image(self.config.image_size)
        ok, frame_bgr = self._capture.read()
        if not ok:
            logger.warning("Camera read failed; using dummy image")
            return make_dummy_image(self.config.image_size)
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        return resize_with_pad(frame_rgb, self.config.image_size, self.config.image_size)

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def make_droid_observation(
    *,
    exterior_image: np.ndarray,
    wrist_image: np.ndarray | None,
    robot_state: RobotState | None,
    prompt: str,
    image_size: int = 224,
) -> dict[str, Any]:
    """Build the DROID-style observation expected by pi0_droid/pi05_droid configs."""

    exterior = resize_with_pad(exterior_image, image_size, image_size)
    wrist = resize_with_pad(wrist_image if wrist_image is not None else exterior_image, image_size, image_size)
    if robot_state is None:
        joint_position = np.zeros((7,), dtype=np.float32)
        gripper_position = np.zeros((1,), dtype=np.float32)
    else:
        joints = np.asarray(robot_state.joint_position, dtype=np.float32)
        if joints.size < 7:
            joints = np.pad(joints, (0, 7 - joints.size))
        joint_position = joints[:7]
        gripper_position = np.asarray([robot_state.gripper_position], dtype=np.float32)

    return {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }
