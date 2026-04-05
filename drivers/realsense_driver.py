from __future__ import annotations

from dataclasses import dataclass

from common.types import CameraFrame


@dataclass
class RealSenseConfig:
    width: int
    height: int


class RealSenseDriver:
    """
    Minimal RealSense driver interface.

    Replace `capture_frame` with pyrealsense2 integration when hardware is ready.
    """

    def __init__(self, config: RealSenseConfig) -> None:
        self.config = config
        self.frame_index = 0

    def capture_frame(self) -> CameraFrame:
        self.frame_index += 1
        return CameraFrame(
            rgb_path_hint=f"mock_rgb_frame_{self.frame_index:04d}.png",
            depth_path_hint=f"mock_depth_frame_{self.frame_index:04d}.npy",
            width=self.config.width,
            height=self.config.height,
        )
