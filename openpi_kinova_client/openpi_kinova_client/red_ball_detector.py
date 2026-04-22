from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RedBallDetection:
    center_xy: tuple[int, int]
    radius_px: float
    area_px: int
    image_size: tuple[int, int]


def detect_red_ball(rgb: np.ndarray) -> RedBallDetection | None:
    """Detect the red sponge ball with a conservative RGB color mask."""

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {rgb.shape}")

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    # Works for the attached image where the ball is dark red under weak light.
    mask = (r > 45) & (r > g + 22) & (r > b + 22) & (r > 1.35 * g) & (r > 1.35 * b)

    ys, xs = np.nonzero(mask)
    if xs.size < 80:
        return None

    center_x = int(round(float(xs.mean())))
    center_y = int(round(float(ys.mean())))
    area = int(xs.size)
    radius = float(np.sqrt(area / np.pi))
    height, width = rgb.shape[:2]
    return RedBallDetection(
        center_xy=(center_x, center_y),
        radius_px=radius,
        area_px=area,
        image_size=(width, height),
    )
