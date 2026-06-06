from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
ButtonDict = dict[str, bool]


@dataclass(frozen=True)
class SpaceMouseSigns:
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    droll: float = 1.0
    dpitch: float = 1.0
    dyaw: float = 1.0


@dataclass(frozen=True)
class SpaceMouseButtons:
    enable: int = 0
    stop: int = 1
    gripper_toggle: int = 0


@dataclass(frozen=True)
class SpaceMouseMapping:
    signs: SpaceMouseSigns = field(default_factory=SpaceMouseSigns)
    buttons: SpaceMouseButtons = field(default_factory=SpaceMouseButtons)


class SpaceMouseController:
    def __init__(
        self,
        device: str | None = None,
        device_index: int = 0,
        device_path: str | None = None,
        deadzone: float = 0.08,
        max_delta_m: float = 0.005,
        max_delta_rad: float = 0.034906585,
        require_enable_button: bool = False,
        mapping: SpaceMouseMapping | None = None,
        debug: bool = False,
        calibrate_on_connect: bool = True,
        calibration_duration_s: float = 0.6,
    ) -> None:
        self.device = device
        self.device_index = device_index
        self.device_path = device_path
        self.deadzone = deadzone
        self.max_delta_m = max_delta_m
        self.max_delta_rad = max_delta_rad
        self.require_enable_button = require_enable_button
        self.mapping = mapping or SpaceMouseMapping()
        self.debug = debug
        self.calibrate_on_connect = calibrate_on_connect
        self.calibration_duration_s = calibration_duration_s

        self._pyspacemouse: Any | None = None
        self._spacemouse: Any | None = None
        self._gripper_target = -1.0
        self._last_button_values: list[int] = []
        self._last_buttons = self._buttons()
        self._read_failure_reported = False
        self._axis_bias = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }

    @property
    def input_available(self) -> bool:
        return self._spacemouse is not None

    def connect(self) -> None:
        try:
            import pyspacemouse
        except ImportError as exc:
            raise RuntimeError(
                "pyspacemouse is required. Install it with `pip install pyspacemouse`."
            ) from exc

        self._pyspacemouse = pyspacemouse
        try:
            if self.device_path:
                self._spacemouse = pyspacemouse.open_by_path(
                    self.device_path,
                    nonblocking=True,
                )
            else:
                self._spacemouse = self._open_first_available_device(pyspacemouse)
        except Exception as exc:
            hid_paths = self._find_3dconnexion_hid_paths()
            chmod_hint = (
                "sudo chmod a+rw " + " ".join(hid_paths)
                if hid_paths
                else "sudo chmod a+rw /dev/hidraw*"
            )
            raise RuntimeError(
                "Failed to open SpaceMouse. If the device is connected, check that "
                "`/dev/hidraw*` is visible and readable by this user. On Linux this "
                "usually needs a udev rule or running outside a container without "
                "the HID device passed through.\n"
                f"Detected 3Dconnexion HID paths: {hid_paths or 'none'}\n"
                f"Temporary fix: `{chmod_hint}`"
            ) from exc

        self._last_button_values = self._read_button_values(self._spacemouse.read())
        if self.calibrate_on_connect:
            self._calibrate_zero()
        description = (
            self._spacemouse.describe_connection()
            if hasattr(self._spacemouse, "describe_connection")
            else "SpaceMouse connected"
        )
        print(f"\033[36m已连接 SpaceMouse: {description}\033[0m")

    def _open_first_available_device(self, pyspacemouse: Any) -> Any:
        errors: list[str] = []
        candidate_devices: list[str | None]
        if self.device is not None:
            candidate_devices = [self.device]
        else:
            try:
                candidate_devices = list(dict.fromkeys(pyspacemouse.get_connected_devices()))
            except Exception:
                candidate_devices = [None]
            if not candidate_devices:
                candidate_devices = [None]

        # pyspacemouse may expose the same physical receiver through several
        # HID interfaces. Try several indices before giving up.
        for device_name in candidate_devices:
            for index in range(max(self.device_index, 0), max(self.device_index, 0) + 8):
                try:
                    return pyspacemouse.open(
                        device=device_name,
                        device_index=index,
                        nonblocking=True,
                    )
                except Exception as exc:
                    errors.append(f"device={device_name!r} index={index}: {exc}")

        raise RuntimeError("Failed to open any SpaceMouse HID interface: " + "; ".join(errors[-8:]))

    @staticmethod
    def _find_3dconnexion_hid_paths() -> list[str]:
        try:
            from easyhid import Enumeration
        except Exception:
            return []

        paths: list[str] = []
        try:
            devices = Enumeration().find()
        except Exception:
            return []

        for device in devices:
            vendor_id = getattr(device, "vendor_id", None)
            manufacturer = str(getattr(device, "manufacturer_string", "") or "")
            product = str(getattr(device, "product_string", "") or "")
            if vendor_id != 0x256F and "3Dconnexion" not in manufacturer + product:
                continue
            path = str(getattr(device, "path", "") or "")
            if not path:
                continue
            if Path(path).name.startswith("hidraw") and path not in paths:
                paths.append(path)
        return paths

    def disconnect(self) -> None:
        if self._spacemouse is not None:
            self._spacemouse.close()
        self._spacemouse = None

    def reset_gripper_target(self, value: float = -1.0) -> None:
        """Reset persistent gripper target.

        Args:
            value:
                -1.0 = desired open state
                +1.0 = desired close state
        """
        self._gripper_target = 1.0 if float(value) > 0.0 else -1.0

    def read(self) -> tuple[FloatArray, ButtonDict]:
        if self._spacemouse is None:
            raise RuntimeError("SpaceMouseController is not connected")

        try:
            state = self._spacemouse.read()
            self._read_failure_reported = False
        except Exception as exc:
            if not self._read_failure_reported:
                print(f"\033[33mSpaceMouse 读取失败，当前动作置零：{exc}\033[0m")
                self._read_failure_reported = True
            action = np.zeros((7,), dtype=np.float32)
            action[-1] = self._gripper_target
            return action, self._buttons()
        current_buttons = self._read_button_values(state)
        buttons = self._buttons(
            enable=self._button_down(current_buttons, self.mapping.buttons.enable),
            stop=self._button_down(current_buttons, self.mapping.buttons.stop),
        )

        gripper_button = self.mapping.buttons.gripper_toggle
        if self._button_pressed(current_buttons, gripper_button):
            self._gripper_target = 1.0 if self._gripper_target < 0.0 else -1.0

        self._last_button_values = current_buttons
        action = self._state_to_action(state, enabled=buttons["enable"])

        if self.debug:
            raw = {
                "x": round(float(state.x), 4),
                "y": round(float(state.y), 4),
                "z": round(float(state.z), 4),
                "roll": round(float(state.roll), 4),
                "pitch": round(float(state.pitch), 4),
                "yaw": round(float(state.yaw), 4),
                "buttons": current_buttons,
            }
            print(f"spacemouse={raw} action={[round(float(v), 5) for v in action]}")

        return action, buttons

    def read_action(self) -> tuple[FloatArray, ButtonDict]:
        return self.read()

    def _state_to_action(self, state: Any, *, enabled: bool) -> FloatArray:
        if self.require_enable_button and not enabled:
            action = np.zeros((7,), dtype=np.float32)
            action[-1] = self._gripper_target
            return action

        signs = self.mapping.signs
        return np.array(
            [
                signs.dx * self._apply_deadzone(self._axis_value(state, "x")) * self.max_delta_m,
                signs.dy * self._apply_deadzone(self._axis_value(state, "y")) * self.max_delta_m,
                signs.dz * self._apply_deadzone(self._axis_value(state, "z")) * self.max_delta_m,
                signs.droll * self._apply_deadzone(self._axis_value(state, "roll")) * self.max_delta_rad,
                signs.dpitch * self._apply_deadzone(self._axis_value(state, "pitch")) * self.max_delta_rad,
                signs.dyaw * self._apply_deadzone(self._axis_value(state, "yaw")) * self.max_delta_rad,
                self._gripper_target,
            ],
            dtype=np.float32,
        )

    def _calibrate_zero(self) -> None:
        if self._spacemouse is None:
            return
        samples: list[dict[str, float]] = []
        deadline = time.monotonic() + max(0.0, self.calibration_duration_s)
        print("\033[36m正在校准 SpaceMouse 零点，请松手不要碰旋钮...\033[0m")
        while time.monotonic() < deadline:
            try:
                state = self._spacemouse.read()
            except Exception:
                time.sleep(0.01)
                continue
            samples.append(
                {
                    "x": float(state.x),
                    "y": float(state.y),
                    "z": float(state.z),
                    "roll": float(state.roll),
                    "pitch": float(state.pitch),
                    "yaw": float(state.yaw),
                }
            )
            time.sleep(0.01)
        if not samples:
            return
        for axis in self._axis_bias:
            self._axis_bias[axis] = float(np.median([sample[axis] for sample in samples]))
        print(
            "\033[36mSpaceMouse 零点偏置: "
            + ", ".join(f"{axis}={value:+.4f}" for axis, value in self._axis_bias.items())
            + "\033[0m"
        )

    def _axis_value(self, state: Any, axis: str) -> float:
        return float(getattr(state, axis)) - self._axis_bias[axis]

    def _apply_deadzone(self, value: float) -> float:
        clipped = float(np.clip(value, -1.0, 1.0))
        magnitude = abs(clipped)
        if magnitude < self.deadzone:
            return 0.0
        if self.deadzone >= 1.0:
            return float(np.sign(clipped))
        scaled = (magnitude - self.deadzone) / (1.0 - self.deadzone)
        return float(np.sign(clipped) * np.clip(scaled, 0.0, 1.0))

    def _button_down(self, buttons: list[int], index: int) -> bool:
        if index < 0 or index >= len(buttons):
            return False
        return bool(buttons[index])

    def _button_pressed(self, buttons: list[int], index: int) -> bool:
        if index < 0 or index >= len(buttons):
            return False
        previous = self._last_button_values[index] if index < len(self._last_button_values) else 0
        return bool(buttons[index]) and not bool(previous)

    @staticmethod
    def _read_button_values(state: Any) -> list[int]:
        return [int(value) for value in getattr(state, "buttons", [])]

    @staticmethod
    def _buttons(enable: bool = False, stop: bool = False) -> ButtonDict:
        return {
            "enable": enable,
            "start": enable,
            "success": False,
            "abort": False,
            "stop": stop,
        }
