from __future__ import annotations

import base64
import io
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.float32]


@dataclass
class KinovaBCRemoteClient:
    """HTTP client for the Kinova BC baseline service.

    The GPU service contract is:
      POST /act {"image_b64": "...", "state": [14 floats], "prompt": "..."}
      -> {"action": [dx, dy, dz, droll, dpitch, dyaw, gripper], ...}
    """

    base_url: str
    timeout_s: float = 10.0
    jpeg_quality: int = 90
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/act"):
            self.base_url = self.base_url[: -len("/act")]
        self._dry_step = 0

    def healthz(self) -> dict[str, Any]:
        if self.dry_run:
            return {"ok": True, "dry_run": True}
        return self._get_json("/healthz")

    def metadata(self) -> dict[str, Any]:
        if self.dry_run:
            return {"dry_run": True, "action_dim": 7, "state_dim": 14}
        return self._get_json("/metadata")

    def reset(self) -> dict[str, Any]:
        if self.dry_run:
            self._dry_step = 0
            return {"ok": True, "dry_run": True}
        return self._post_json("/reset", {})

    def act(
        self,
        image_rgb: ImageArray,
        state_14: FloatArray,
        prompt: str = "pick up the red ball",
    ) -> dict[str, Any]:
        if self.dry_run:
            action = self._dry_run_action()
            return {
                "action": action.tolist(),
                "raw_action": action.tolist(),
                "gripper_value": float(action[-1]),
                "gripper_label": _gripper_label(float(action[-1])),
                "gripper_probs": [0.0, 1.0, 0.0],
                "prompt": prompt,
                "timing_ms": {"total": 0.0},
            }

        state = np.asarray(state_14, dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected state shape (14,), got {state.shape}")
        payload = {
            "image_b64": self._encode_rgb_image_base64(image_rgb),
            "state": state.tolist(),
            "prompt": prompt,
        }
        response = self._post_json("/act", payload)
        action = self.parse_action(response)
        response["action"] = action.astype(float).tolist()
        return response

    @staticmethod
    def parse_action(response: Any) -> FloatArray:
        if not isinstance(response, dict) or "action" not in response:
            raise RuntimeError("BC policy response must be a JSON object containing key 'action'")
        action = np.asarray(response["action"], dtype=np.float32)
        if action.shape != (7,):
            raise RuntimeError(f"BC policy action must have shape (7,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise RuntimeError(f"BC policy action contains NaN or Inf: {action.tolist()}")
        return action.astype(np.float32)

    def _get_json(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            headers={"Accept": "application/json"},
            method="GET",
        )
        return self._open_json(request)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        return self._open_json(request)

    def _open_json(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"BC policy server HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach BC policy server {self.base_url}: {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"BC policy server request timed out after {self.timeout_s}s") from exc

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"BC policy server returned non-JSON response: {body[:200]}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(f"BC policy server returned non-object JSON: {type(decoded).__name__}")
        return decoded

    def _encode_rgb_image_base64(self, image_rgb: ImageArray) -> str:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image [H, W, 3], got {image_rgb.shape}")
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(image_rgb, mode="RGB").save(buffer, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _dry_run_action(self) -> FloatArray:
        self._dry_step += 1
        phase = self._dry_step / 12.0
        return np.array(
            [
                0.001 * math.sin(phase),
                0.001 * math.cos(phase),
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0 if self._dry_step < 20 else 0.0 if self._dry_step < 40 else 1.0,
            ],
            dtype=np.float32,
        )


def _gripper_label(value: float) -> str:
    if value < -0.5:
        return "open"
    if value > 0.5:
        return "close"
    return "hold"
