from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

ImageArray = NDArray[np.uint8]
DepthArray = NDArray[np.uint16]


@dataclass(frozen=True)
class CameraInfo:
    width: int = 640
    height: int = 480
    fps: int = 30
    serial: str | None = None
    enable_depth: bool = False


class RealSenseCamera:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
        dry_run: bool = False,
        enable_depth: bool = False,
        frame_timeout_ms: int = 5000,
    ) -> None:
        self.info = CameraInfo(
            width=width,
            height=height,
            fps=fps,
            serial=serial,
            enable_depth=enable_depth,
        )
        self.dry_run = dry_run
        self.frame_timeout_ms = frame_timeout_ms
        self._pipeline: Any | None = None
        self._config: Any | None = None
        self._rs: Any | None = None
        self._started = False
        self._frame_index = 0
        self._last_depth: DepthArray | None = None

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            return
        if self.dry_run:
            self._started = True
            return

        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required when dry_run=False") from exc

        try:
            config = rs.config()
            if self.info.serial is not None:
                config.enable_device(self.info.serial)
            config.enable_stream(
                rs.stream.color,
                self.info.width,
                self.info.height,
                rs.format.rgb8,
                self.info.fps,
            )
            if self.info.enable_depth:
                config.enable_stream(
                    rs.stream.depth,
                    self.info.width,
                    self.info.height,
                    rs.format.z16,
                    self.info.fps,
                )
            pipeline = rs.pipeline()
            pipeline.start(config)
        except Exception as exc:
            serial_msg = f" serial={self.info.serial}" if self.info.serial else ""
            raise RuntimeError(
                "Failed to start RealSense camera"
                f"{serial_msg} at {self.info.width}x{self.info.height}@{self.info.fps}."
            ) from exc

        self._rs = rs
        self._config = config
        self._pipeline = pipeline
        self._started = True

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:
                raise RuntimeError("Failed to stop RealSense pipeline cleanly") from exc
        self._pipeline = None
        self._config = None
        self._rs = None
        self._started = False
        self._last_depth = None

    def connect(self) -> None:
        self.start()

    def disconnect(self) -> None:
        self.stop()

    def get_rgb(self) -> ImageArray:
        if self.dry_run:
            if not self._started:
                self.start()
            return self._make_dry_rgb()
        if self._pipeline is None or not self._started:
            raise RuntimeError("RealSenseCamera is not started. Call start() before get_rgb().")

        try:
            frames = self._pipeline.wait_for_frames(self.frame_timeout_ms)
        except Exception as exc:
            raise RuntimeError(
                "Timed out or failed while waiting for RealSense frames. "
                "Check USB connection, camera power, and stream configuration."
            ) from exc

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(
                "RealSense frame set did not contain an RGB frame. "
                "Check that the color stream is enabled and the camera is connected."
            )

        if self.info.enable_depth:
            depth_frame = frames.get_depth_frame()
            self._last_depth = (
                np.asanyarray(depth_frame.get_data()).astype(np.uint16)
                if depth_frame
                else None
            )

        rgb = np.asanyarray(color_frame.get_data())
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"Unexpected RealSense RGB frame shape: {rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        return np.ascontiguousarray(rgb)

    def get_depth(self) -> DepthArray:
        if self.dry_run:
            if not self._started:
                self.start()
            return self._make_dry_depth()
        if not self.info.enable_depth:
            raise RuntimeError("Depth stream is disabled. Construct RealSenseCamera(enable_depth=True).")
        self.get_rgb()
        if self._last_depth is None:
            raise RuntimeError("RealSense frame set did not contain a depth frame.")
        return np.ascontiguousarray(self._last_depth)

    def read_image(self) -> ImageArray:
        return self.get_rgb()

    def _make_dry_rgb(self) -> ImageArray:
        self._frame_index += 1
        height = self.info.height
        width = self.info.width
        x_gradient = np.linspace(20, 220, width, dtype=np.uint8)
        y_gradient = np.linspace(30, 180, height, dtype=np.uint8)[:, None]
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = x_gradient[None, :]
        image[:, :, 1] = y_gradient
        image[:, :, 2] = np.uint8((self._frame_index * 5) % 255)

        block_size = max(28, min(width, height) // 7)
        x0 = width // 2 - block_size // 2
        y0 = height // 2 - block_size // 2
        image[y0 : y0 + block_size, x0 : x0 + block_size] = np.array([255, 0, 0], dtype=np.uint8)

        pil_image = Image.fromarray(image, mode="RGB")
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        lines = [
            "DRY RUN RealSense D435i RGB",
            f"frame: {self._frame_index}",
            f"time:  {timestamp}",
            f"size:  {width}x{height}@{self.info.fps}",
        ]
        margin = 10
        line_height = 14
        box_height = margin * 2 + line_height * len(lines)
        draw.rectangle((0, 0, width, box_height), fill=(0, 0, 0))
        for index, line in enumerate(lines):
            draw.text((margin, margin + index * line_height), line, fill=(255, 255, 255), font=font)
        return np.asarray(pil_image, dtype=np.uint8)

    def _make_dry_depth(self) -> DepthArray:
        self._frame_index += 1
        height = self.info.height
        width = self.info.width
        base = np.linspace(300, 1200, width, dtype=np.uint16)[None, :]
        depth = np.repeat(base, height, axis=0)
        return np.ascontiguousarray(depth)


def _preview_with_pygame(camera: RealSenseCamera) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required for preview display") from exc

    pygame.init()
    screen = pygame.display.set_mode((camera.info.width, camera.info.height))
    pygame.display.set_caption("RealSense RGB preview")
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in {pygame.K_ESCAPE, pygame.K_q}:
                running = False

        rgb = camera.get_rgb()
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(camera.info.fps)
    pygame.quit()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Test Intel RealSense D435i RGB camera.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-depth", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args(argv)

    with RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        dry_run=args.dry_run,
        enable_depth=args.enable_depth,
    ) as camera:
        if args.no_preview:
            while True:
                rgb = camera.get_rgb()
                print(f"rgb shape={rgb.shape} dtype={rgb.dtype} time={time.time():.3f}")
                time.sleep(1.0 / float(args.fps))
        else:
            _preview_with_pygame(camera)


if __name__ == "__main__":
    main()
