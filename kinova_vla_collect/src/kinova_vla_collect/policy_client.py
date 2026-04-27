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
class PolicyClient:
    server_url: str
    timeout_s: float = 10.0
    dry_run: bool = False

    def __post_init__(self) -> None:
        self._dry_step = 0

    def predict(
        self,
        wrist_rgb: ImageArray,
        robot_state: FloatArray,
        task_prompt: str,
    ) -> FloatArray:
        actions = self.predict_chunk(wrist_rgb=wrist_rgb, robot_state=robot_state, task_prompt=task_prompt)
        if actions.shape[0] < 1:
            raise RuntimeError("Policy returned an empty action chunk")
        return actions[0]

    def predict_chunk(
        self,
        wrist_rgb: ImageArray,
        robot_state: FloatArray,
        task_prompt: str,
    ) -> FloatArray:
        if self.dry_run:
            return self._dry_run_actions()
        payload = {
            "observation": {
                "images": {
                    "wrist": self._encode_rgb_jpeg_base64(wrist_rgb),
                },
                "state": np.asarray(robot_state, dtype=np.float32).tolist(),
            },
            "task": task_prompt,
            "action_definition": "[dx, dy, dz, droll, dpitch, dyaw, gripper]",
        }
        response = self._post_json(payload)
        return self._parse_actions(response)

    def _post_json(self, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self.server_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Policy server HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach policy server {self.server_url}: {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Policy server request timed out after {self.timeout_s}s") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Policy server returned non-JSON response: {body[:200]}") from exc

    @staticmethod
    def _parse_actions(response: Any) -> FloatArray:
        candidate: Any
        if isinstance(response, dict):
            if "actions" in response:
                candidate = response["actions"]
            elif "action" in response:
                candidate = response["action"]
            elif "predicted_actions" in response:
                candidate = response["predicted_actions"]
            else:
                raise RuntimeError(
                    "Policy response JSON must contain one of: action, actions, predicted_actions"
                )
        else:
            candidate = response

        actions = np.asarray(candidate, dtype=np.float32)
        if actions.shape == (7,):
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise RuntimeError(f"Policy action must have shape (7,) or (T, 7), got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise RuntimeError("Policy action contains NaN or Inf")
        return actions.astype(np.float32)

    @staticmethod
    def _encode_rgb_jpeg_base64(image: ImageArray) -> str:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected wrist RGB image [H, W, 3], got {image.shape}")
        buffer = io.BytesIO()
        Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB").save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _dry_run_actions(self) -> FloatArray:
        self._dry_step += 1
        phase = self._dry_step / 10.0
        dx = 0.001 * math.sin(phase)
        dy = 0.001 * math.cos(phase)
        dz = 0.0
        gripper = 0.0
        return np.array([[dx, dy, dz, 0.0, 0.0, 0.0, gripper]], dtype=np.float32)
