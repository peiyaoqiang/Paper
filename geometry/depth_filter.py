from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class DepthSample:
    pixel_xy: Tuple[int, int]
    depth_m: float
    valid: bool


class DepthFilter:
    """
    Minimal depth helper that samples the saved aligned depth map around the target pixel.
    """

    def sample_target_depth(self, pixel_xy: Tuple[int, int], depth_path_hint: str) -> DepthSample:
        depth_path = Path(depth_path_hint)
        if not depth_path.is_file():
            return DepthSample(pixel_xy=pixel_xy, depth_m=0.18, valid=False)

        depth = np.load(depth_path)
        if depth.ndim != 2:
            return DepthSample(pixel_xy=pixel_xy, depth_m=0.18, valid=False)

        u, v = pixel_xy
        u = int(np.clip(u, 0, depth.shape[1] - 1))
        v = int(np.clip(v, 0, depth.shape[0] - 1))

        half_window = 2
        y0 = max(0, v - half_window)
        y1 = min(depth.shape[0], v + half_window + 1)
        x0 = max(0, u - half_window)
        x1 = min(depth.shape[1], u + half_window + 1)
        patch = depth[y0:y1, x0:x1]

        patch = patch.astype(np.float32)
        valid_patch = patch[np.isfinite(patch) & (patch > 0)]
        if valid_patch.size == 0:
            return DepthSample(pixel_xy=(u, v), depth_m=0.18, valid=False)

        depth_value = float(np.median(valid_patch))

        # RealSense aligned depth is commonly stored as uint16 in millimeters.
        if depth.dtype.kind in ("u", "i") and depth_value > 10.0:
            depth_value *= 0.001

        return DepthSample(pixel_xy=(u, v), depth_m=depth_value, valid=True)
