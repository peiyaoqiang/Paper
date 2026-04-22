from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request

from common.types import Observation, PolicyAction


@dataclass
class OpenVLAConfig:
    model_name: str = "openvla-mock"
    mode: str = "mock"
    remote_url: str = "http://127.0.0.1:8000/predict"
    remote_timeout_s: float = 10.0
    unnorm_key: str = "libero_spatial"
    image_input_key: str = "wrist_image"


class OpenVLAWrapper:
    """
    Mock wrapper that exposes the interface we want for the real system.

    Replace `predict_action` with actual model loading and inference.
    """

    def __init__(self, config: OpenVLAConfig) -> None:
        self.config = config

    def predict_action(self, observation: Observation) -> PolicyAction:
        if self.config.mode == "remote_api":
            return self._predict_action_remote(observation)
        return self._predict_action_mock(observation)

    def check_remote_health(self) -> tuple[bool, str]:
        if self.config.mode != "remote_api":
            return (True, "OpenVLA is not using remote_api mode.")
        if not self.config.remote_url:
            return (False, "OpenVLA remote_api mode requires `remote_url` to be set.")

        health_url = self.config.remote_url
        if health_url.endswith("/predict"):
            health_url = health_url[: -len("/predict")] + "/health"

        http_request = request.Request(health_url, method="GET")
        try:
            with request.urlopen(http_request, timeout=min(self.config.remote_timeout_s, 5.0)) as response:
                body = response.read().decode("utf-8", errors="replace")
        except error.URLError as exc:
            return (False, f"OpenVLA health check failed for {health_url}: {exc}")

        return (True, body or f"OpenVLA health endpoint reachable: {health_url}")

    def _predict_action_mock(self, observation: Observation) -> PolicyAction:
        instruction = observation.instruction.lower()
        is_pick = "pick" in instruction or "grasp" in instruction
        target_pixel = (observation.frame.width // 2, observation.frame.height // 2)
        delta_z = -0.03 if is_pick else 0.0
        notes = f"Mock {self.config.model_name} action for instruction: {observation.instruction}"
        return PolicyAction(
            delta_xyz_m=(0.00, 0.00, delta_z),
            delta_yaw_deg=0.0,
            gripper_command="open" if is_pick else "hold",
            confidence=0.68,
            target_pixel=target_pixel,
            notes=notes,
        )

    def _predict_action_remote(self, observation: Observation) -> PolicyAction:
        if not self.config.remote_url:
            raise ValueError("OpenVLA remote_api mode requires `remote_url` to be set.")

        payload = self._build_remote_payload(observation)
        response = self._post_json(self.config.remote_url, payload)
        return self._policy_action_from_remote_response(response, observation)

    def _build_remote_payload(self, observation: Observation) -> dict[str, Any]:
        return {
            "instruction": observation.instruction,
            "unnorm_key": self.config.unnorm_key,
            "image_input_key": self.config.image_input_key,
            "frame": {
                "rgb_path_hint": observation.frame.rgb_path_hint,
                "depth_path_hint": observation.frame.depth_path_hint,
                "width": observation.frame.width,
                "height": observation.frame.height,
                "rgb_b64": self._maybe_base64_file(observation.frame.rgb_path_hint),
            },
            "robot_state": {
                "joint_positions": observation.robot_state.joint_positions,
                "ee_position_m": observation.robot_state.ee_position_m,
                "ee_yaw_deg": observation.robot_state.ee_yaw_deg,
                "gripper_opening_m": observation.robot_state.gripper_opening_m,
            },
        }

    def _post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.config.remote_timeout_s) as response:
                body = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(
                f"OpenVLA remote request failed: {exc}. "
                f"Expected service at {url}. Check the remote server and local port-forward."
            ) from exc

        parsed = json.loads(body)
        if not isinstance(parsed, Mapping):
            raise RuntimeError("OpenVLA remote response must be a JSON object.")
        return parsed

    def _policy_action_from_remote_response(
        self,
        response: Mapping[str, Any],
        observation: Observation,
    ) -> PolicyAction:
        if "action" in response:
            action = response["action"]
            if not isinstance(action, Sequence) or len(action) < 7:
                raise RuntimeError("OpenVLA remote `action` must contain at least 7 elements.")
            delta_xyz_m = tuple(float(v) for v in action[:3])
            delta_yaw_deg = float(action[5])
            gripper_command = "close" if float(action[6]) > 0.5 else "open"
        else:
            delta_xyz_raw = response.get("delta_xyz_m", (0.0, 0.0, 0.0))
            if not isinstance(delta_xyz_raw, Sequence) or len(delta_xyz_raw) != 3:
                raise RuntimeError("OpenVLA remote `delta_xyz_m` must contain exactly 3 elements.")
            delta_xyz_m = tuple(float(v) for v in delta_xyz_raw)
            delta_yaw_deg = float(response.get("delta_yaw_deg", 0.0))
            gripper_command = str(response.get("gripper_command", "hold"))

        confidence = float(response.get("confidence", 0.75))
        target_pixel = self._parse_target_pixel(response.get("target_pixel"), observation)
        notes = str(response.get("notes", f"Remote {self.config.model_name} response"))
        metadata = self._extract_remote_metadata(response)

        return PolicyAction(
            delta_xyz_m=delta_xyz_m,
            delta_yaw_deg=delta_yaw_deg,
            gripper_command=gripper_command,
            confidence=confidence,
            target_pixel=target_pixel,
            notes=notes,
            metadata=metadata,
        )

    def _parse_target_pixel(
        self,
        raw_target_pixel: Any,
        observation: Observation,
    ) -> tuple[int, int] | None:
        if raw_target_pixel is None:
            return (observation.frame.width // 2, observation.frame.height // 2)
        if not isinstance(raw_target_pixel, Sequence) or len(raw_target_pixel) != 2:
            raise RuntimeError("OpenVLA remote `target_pixel` must contain exactly 2 elements.")
        return (int(raw_target_pixel[0]), int(raw_target_pixel[1]))

    def _maybe_base64_file(self, path_hint: str) -> str | None:
        path = Path(path_hint)
        if not path.is_file():
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _extract_remote_metadata(self, response: Mapping[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in (
            "preprocess_ms",
            "infer_ms",
            "total_ms",
            "model_name",
            "image_size",
            "server_timestamp_utc",
        ):
            if key in response:
                metadata[key] = response[key]
        return metadata
