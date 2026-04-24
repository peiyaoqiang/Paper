from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


def identity_pose_xyzw() -> FloatArray:
    return np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def normalize_quaternion_xyzw(quaternion: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quaternion / norm).astype(np.float32)
