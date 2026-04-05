from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DepthSample:
    pixel_xy: Tuple[int, int]
    depth_m: float
    valid: bool


class DepthFilter:
    """
    Placeholder depth helper.

    Replace with actual depth map lookup, denoising, ROI cropping, and validity checks.
    """

    def sample_target_depth(self, pixel_xy: Tuple[int, int]) -> DepthSample:
        return DepthSample(pixel_xy=pixel_xy, depth_m=0.18, valid=True)
