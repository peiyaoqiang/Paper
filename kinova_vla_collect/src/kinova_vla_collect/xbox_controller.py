from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
ButtonDict = dict[str, bool]
DryRunMode = Literal["keyboard", "random", "scripted"]
GripperActionMode = Literal["hold_on_release", "persistent_target"]


@dataclass(frozen=True)
class AxisMapping:
    left_x: int = 0
    left_y: int = 1
    right_x: int = 3
    right_y: int = 4
    lt: int = 2
    rt: int = 5


@dataclass(frozen=True)
class ActionSigns:
    dx: float = -1.0
    dy: float = -1.0
    dz: float = -1.0
    droll: float = 1.0
    dpitch: float = 1.0
    dyaw: float = 1.0


@dataclass(frozen=True)
class ButtonMapping:
    success: int = 0
    abort: int = 1
    roll_negative: int = 4
    roll_positive: int = 5
    stop: int = 6
    start: int = 7


@dataclass(frozen=True)
class XboxMapping:
    axes: AxisMapping = field(default_factory=AxisMapping)
    signs: ActionSigns = field(default_factory=ActionSigns)
    buttons: ButtonMapping = field(default_factory=ButtonMapping)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "XboxMapping":
        if data is None:
            return cls()

        axes = data.get("axes", {})
        signs = data.get("signs", {})
        buttons = data.get("buttons", {})

        if not isinstance(axes, dict):
            raise ValueError("xbox mapping axes must be a mapping")
        if not isinstance(signs, dict):
            raise ValueError("xbox mapping signs must be a mapping")
        if not isinstance(buttons, dict):
            raise ValueError("xbox mapping buttons must be a mapping")

        return cls(
            axes=AxisMapping(
                left_x=int(axes.get("left_x", AxisMapping.left_x)),
                left_y=int(axes.get("left_y", AxisMapping.left_y)),
                right_x=int(axes.get("right_x", AxisMapping.right_x)),
                right_y=int(axes.get("right_y", AxisMapping.right_y)),
                lt=int(axes.get("lt", AxisMapping.lt)),
                rt=int(axes.get("rt", AxisMapping.rt)),
            ),
            signs=ActionSigns(
                dx=float(signs.get("dx", ActionSigns.dx)),
                dy=float(signs.get("dy", ActionSigns.dy)),
                dz=float(signs.get("dz", ActionSigns.dz)),
                droll=float(signs.get("droll", ActionSigns.droll)),
                dpitch=float(signs.get("dpitch", ActionSigns.dpitch)),
                dyaw=float(signs.get("dyaw", ActionSigns.dyaw)),
            ),
            buttons=ButtonMapping(
                success=int(buttons.get("success", ButtonMapping.success)),
                abort=int(buttons.get("abort", ButtonMapping.abort)),
                roll_negative=int(buttons.get("roll_negative", ButtonMapping.roll_negative)),
                roll_positive=int(buttons.get("roll_positive", ButtonMapping.roll_positive)),
                stop=int(buttons.get("stop", ButtonMapping.stop)),
                start=int(buttons.get("start", ButtonMapping.start)),
            ),
        )


class XboxController:
    def __init__(
        self,
        device_index: int = 0,
        deadzone: float = 0.12,
        max_delta_m: float = 0.005,
        max_delta_rad: float = 0.034906585,
        action_dim: int = 7,
        dry_run: bool = False,
        mapping: XboxMapping | dict[str, Any] | None = None,
        debug: bool = False,
        dry_run_mode: DryRunMode = "keyboard",
        gripper_action_mode: GripperActionMode = "persistent_target",
        random_seed: int | None = None,
    ) -> None:
        self.device_index = device_index
        self.deadzone = deadzone
        self.max_delta_m = max_delta_m
        self.max_delta_rad = max_delta_rad
        if action_dim not in {4, 7}:
            raise ValueError(f"action_dim must be 4 or 7, got {action_dim}")
        self.action_dim = action_dim
        self.dry_run = dry_run
        self.mapping = mapping if isinstance(mapping, XboxMapping) else XboxMapping.from_dict(mapping)
        self.debug = debug
        self.dry_run_mode = dry_run_mode
        if gripper_action_mode not in {"hold_on_release", "persistent_target"}:
            raise ValueError(
                "gripper_action_mode must be 'hold_on_release' or 'persistent_target', "
                f"got {gripper_action_mode!r}"
            )
        self.gripper_action_mode = gripper_action_mode

        self._pygame: Any | None = None
        self._joystick: Any | None = None
        self._joystick_instance_id: int | None = None
        self._last_reconnect_attempt = 0.0
        self._disconnected_reported = False
        self._dry_counter = 0
        self._rng = random.Random(random_seed)

        self._trigger_negative_seen: dict[str, bool] = {"lt": False, "rt": False}
        self._latched_buttons: ButtonDict = self._buttons()

        # Gripper target used by persistent_target mode.
        # -1.0 = desired open state, +1.0 = desired close state.
        self._gripper_target: float = -1.0

    @property
    def input_available(self) -> bool:
        return bool(self.dry_run or self._joystick is not None)

    def connect(self) -> None:
        if self.dry_run and self.dry_run_mode in {"random", "scripted"}:
            return

        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required for XboxController") from exc

        pygame.init()

        if self.dry_run and self.dry_run_mode == "keyboard":
            try:
                pygame.display.set_mode((420, 160))
                pygame.display.set_caption("kinova_vla_collect dry-run controller")
            except Exception as exc:
                print(f"Warning: pygame keyboard window failed ({exc}); falling back to random dry-run.")
                self.dry_run_mode = "random"

            self._pygame = pygame
            return

        if self.dry_run:
            self._pygame = pygame
            return

        self._pygame = pygame
        self._connect_joystick_or_raise()

    def disconnect(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()

        self._pygame = None
        self._joystick = None
        self._joystick_instance_id = None

    def reset_gripper_target(self, value: float = -1.0) -> None:
        """Reset persistent gripper target.

        Args:
            value:
                -1.0 = desired open state
                +1.0 = desired close state
        """
        self._gripper_target = 1.0 if float(value) > 0.0 else -1.0

    def read(self) -> tuple[FloatArray, ButtonDict]:
        if self.dry_run:
            return self._read_dry_run()

        if self._pygame is None:
            raise RuntimeError("XboxController is not connected")

        self._poll_events()
        if self._joystick is None:
            self._try_reconnect_joystick()
        if self._joystick is None:
            return self._zero_action(), self._buttons()

        axes = self._read_axes()
        buttons = self._read_buttons()

        if self.debug:
            self.print_debug()

        return self._axes_to_action(axes), buttons

    def read_action(self) -> tuple[FloatArray, ButtonDict]:
        return self.read()

    def print_debug(self) -> None:
        if self._pygame is None:
            print("pygame is not initialized")
            return

        if self.dry_run:
            print("dry_run=True; no joystick axis/button table available")
            return

        if self._joystick is None:
            print("No joystick connected")
            return

        try:
            axis_values = [
                round(float(self._joystick.get_axis(index)), 4)
                for index in range(int(self._joystick.get_numaxes()))
            ]
            button_values = [
                int(self._joystick.get_button(index))
                for index in range(int(self._joystick.get_numbuttons()))
            ]
            hat_values = [
                self._joystick.get_hat(index)
                for index in range(int(self._joystick.get_numhats()))
            ]
        except Exception as exc:
            self._mark_joystick_disconnected(f"debug read failed: {exc}")
            return

        print(f"axes={axis_values} buttons={button_values} hats={hat_values}")

    def _read_axes(self) -> dict[str, float]:
        if self._joystick is None:
            raise RuntimeError("XboxController is not connected")

        axes = self.mapping.axes

        return {
            "left_x": self._apply_deadzone(self._safe_axis(axes.left_x)),
            "left_y": self._apply_deadzone(self._safe_axis(axes.left_y)),
            "right_x": self._apply_deadzone(self._safe_axis(axes.right_x)),
            "right_y": self._apply_deadzone(self._safe_axis(axes.right_y)),
            "lt": self._normalize_trigger("lt", self._safe_axis(axes.lt)),
            "rt": self._normalize_trigger("rt", self._safe_axis(axes.rt)),
            "roll": self._read_roll_axis(),
            "pitch": self._read_pitch_axis(),
        }

    def _read_buttons(self) -> ButtonDict:
        if self._joystick is None:
            raise RuntimeError("XboxController is not connected")

        buttons = self.mapping.buttons

        current_buttons = {
            "start": self._safe_button(buttons.start) or self._latched_buttons["start"],
            "success": self._safe_button(buttons.success) or self._latched_buttons["success"],
            "abort": self._safe_button(buttons.abort) or self._latched_buttons["abort"],
            "stop": self._safe_button(buttons.stop) or self._latched_buttons["stop"],
        }

        self._latched_buttons = self._buttons()
        return current_buttons

    def _poll_events(self) -> None:
        if self._pygame is None:
            return

        for event in self._pygame.event.get():
            if event.type == self._pygame.JOYBUTTONDOWN:
                self._latch_button(int(event.button))
            elif event.type == getattr(self._pygame, "JOYDEVICEREMOVED", object()):
                instance_id = getattr(event, "instance_id", None)
                if self._joystick_instance_id is None or instance_id == self._joystick_instance_id:
                    self._mark_joystick_disconnected(f"设备移除 instance_id={instance_id}")
            elif event.type == getattr(self._pygame, "JOYDEVICEADDED", object()):
                if self._joystick is None:
                    self._try_reconnect_joystick(force=True)
            elif event.type == self._pygame.QUIT:
                self._latched_buttons["stop"] = True

    def _latch_button(self, button_index: int) -> None:
        mapping = self.mapping.buttons

        if button_index == mapping.start:
            self._latched_buttons["start"] = True
        elif button_index == mapping.success:
            self._latched_buttons["success"] = True
        elif button_index == mapping.abort:
            self._latched_buttons["abort"] = True
        elif button_index == mapping.stop:
            self._latched_buttons["stop"] = True

    def _axes_to_action(self, axes: dict[str, float]) -> FloatArray:
        lt_pressed = axes["lt"] > 0.2
        rt_pressed = axes["rt"] > 0.2

        gripper_action = self._gripper_axis_action(lt_pressed=lt_pressed, rt_pressed=rt_pressed)

        signs = self.mapping.signs
        translation = [
            signs.dx * axes["left_y"] * self.max_delta_m,
            signs.dy * axes["left_x"] * self.max_delta_m,
            signs.dz * axes["right_y"] * self.max_delta_m,
        ]
        if self.action_dim == 4:
            return np.array([*translation, gripper_action], dtype=np.float32)

        return np.array(
            [
                *translation,
                signs.droll * axes["roll"] * self.max_delta_rad,
                signs.dpitch * axes["pitch"] * self.max_delta_rad,
                signs.dyaw * axes["right_x"] * self.max_delta_rad,
                gripper_action,
            ],
            dtype=np.float32,
        )

    def _gripper_axis_action(self, lt_pressed: bool, rt_pressed: bool) -> float:
        if rt_pressed and not lt_pressed:
            if self.gripper_action_mode == "persistent_target":
                self._gripper_target = 1.0
            return 1.0
        if lt_pressed and not rt_pressed:
            if self.gripper_action_mode == "persistent_target":
                self._gripper_target = -1.0
            return -1.0
        if self.gripper_action_mode == "persistent_target":
            return self._gripper_target
        return 0.0

    def _read_dry_run(self) -> tuple[FloatArray, ButtonDict]:
        if self.dry_run_mode == "random":
            return self._read_random_dry_run()

        if self.dry_run_mode == "scripted":
            return self._read_scripted_dry_run()

        return self._read_keyboard_dry_run()

    def _read_keyboard_dry_run(self) -> tuple[FloatArray, ButtonDict]:
        if self._pygame is None:
            self.connect()

        if self._pygame is None:
            return self._read_random_dry_run()

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                action = self._zero_action()
                action[-1] = self._gripper_target
                return action, self._buttons(stop=True)

        keys = self._pygame.key.get_pressed()

        dx = float(keys[self._pygame.K_w] - keys[self._pygame.K_s]) * self.max_delta_m
        dy = float(keys[self._pygame.K_d] - keys[self._pygame.K_a]) * self.max_delta_m
        dz = float(keys[self._pygame.K_r] - keys[self._pygame.K_f]) * self.max_delta_m
        droll = float(keys[self._pygame.K_o] - keys[self._pygame.K_u]) * self.max_delta_rad
        dpitch = float(keys[self._pygame.K_i] - keys[self._pygame.K_k]) * self.max_delta_rad
        dyaw = float(keys[self._pygame.K_l] - keys[self._pygame.K_j]) * self.max_delta_rad

        gripper_action = self._gripper_axis_action(
            lt_pressed=bool(keys[self._pygame.K_q]),
            rt_pressed=bool(keys[self._pygame.K_e]),
        )

        if self.action_dim == 4:
            action = np.array([dx, dy, dz, gripper_action], dtype=np.float32)
        else:
            action = np.array([dx, dy, dz, droll, dpitch, dyaw, gripper_action], dtype=np.float32)

        buttons = self._buttons(
            start=bool(keys[self._pygame.K_RETURN]),
            success=bool(keys[self._pygame.K_SPACE]),
            abort=bool(keys[self._pygame.K_b]),
            stop=bool(keys[self._pygame.K_ESCAPE]),
        )

        return action, buttons

    def _read_random_dry_run(self) -> tuple[FloatArray, ButtonDict]:
        scale = self.max_delta_m * 0.35
        if self._rng.random() < 0.05:
            self._gripper_target = self._rng.choice([-1.0, 1.0])

        if self.action_dim == 4:
            action = np.array(
                [
                    self._rng.uniform(-scale, scale),
                    self._rng.uniform(-scale, scale),
                    self._rng.uniform(-scale, scale),
                    self._gripper_target,
                ],
                dtype=np.float32,
            )
        else:
            action = np.array(
                [
                    self._rng.uniform(-scale, scale),
                    self._rng.uniform(-scale, scale),
                    self._rng.uniform(-scale, scale),
                    self._rng.uniform(-self.max_delta_rad * 0.25, self.max_delta_rad * 0.25),
                    self._rng.uniform(-self.max_delta_rad * 0.25, self.max_delta_rad * 0.25),
                    self._rng.uniform(-self.max_delta_rad * 0.25, self.max_delta_rad * 0.25),
                    self._gripper_target,
                ],
                dtype=np.float32,
            )

        return action, self._buttons()

    def _read_scripted_dry_run(self) -> tuple[FloatArray, ButtonDict]:
        self._dry_counter += 1

        action = self._zero_action()
        action[-1] = self._gripper_target

        if 2 <= self._dry_counter <= 20:
            action[0] = self.max_delta_m * 0.25
        if self._dry_counter == 10:
            self._gripper_target = 1.0
            action[-1] = self._gripper_target

        return action, self._buttons(
            start=self._dry_counter == 1,
            success=self._dry_counter == 21,
            stop=self._dry_counter >= 22,
        )

    def _safe_axis(self, axis_index: int) -> float:
        if self._joystick is None:
            return 0.0

        try:
            if axis_index < 0 or axis_index >= int(self._joystick.get_numaxes()):
                return 0.0
            return float(self._joystick.get_axis(axis_index))
        except Exception as exc:
            self._mark_joystick_disconnected(f"axis read failed: {exc}")
            return 0.0

    def _safe_button(self, button_index: int) -> bool:
        if self._joystick is None:
            return False

        try:
            if button_index < 0 or button_index >= int(self._joystick.get_numbuttons()):
                return False
            return bool(self._joystick.get_button(button_index))
        except Exception as exc:
            self._mark_joystick_disconnected(f"button read failed: {exc}")
            return False

    def _apply_deadzone(self, value: float) -> float:
        clipped = float(np.clip(value, -1.0, 1.0))

        if abs(clipped) < self.deadzone:
            return 0.0

        return clipped

    def _normalize_trigger(self, name: str, raw_value: float) -> float:
        if raw_value < -0.05:
            self._trigger_negative_seen[name] = True

        if self._trigger_negative_seen[name]:
            normalized = (raw_value + 1.0) * 0.5
        else:
            normalized = raw_value

        return float(np.clip(normalized, 0.0, 1.0))

    def _read_roll_axis(self) -> float:
        buttons = self.mapping.buttons
        return float(self._safe_button(buttons.roll_positive)) - float(self._safe_button(buttons.roll_negative))

    def _read_pitch_axis(self) -> float:
        if self._joystick is None:
            return 0.0
        try:
            hat_x, hat_y = self._joystick.get_hat(0)
            del hat_x
            return float(hat_y)
        except Exception as exc:
            self._mark_joystick_disconnected(f"hat read failed: {exc}")
            return 0.0

    def _connect_joystick_or_raise(self) -> None:
        if self._pygame is None:
            raise RuntimeError("pygame is not initialized")
        self._pygame.joystick.quit()
        self._pygame.joystick.init()
        joystick_count = self._pygame.joystick.get_count()
        if joystick_count <= self.device_index:
            raise RuntimeError(
                f"No Xbox controller found at index {self.device_index}; "
                f"pygame sees {joystick_count} joystick(s)"
            )
        joystick = self._pygame.joystick.Joystick(self.device_index)
        joystick.init()
        self._joystick = joystick
        try:
            self._joystick_instance_id = int(joystick.get_instance_id())
        except Exception:
            self._joystick_instance_id = None
        self._disconnected_reported = False
        print(
            "\033[36m"
            "已连接手柄 "
            f"{self.device_index}: {joystick.get_name()} "
            f"轴数量={joystick.get_numaxes()} 按钮数量={joystick.get_numbuttons()}"
            "\033[0m"
        )

    def _try_reconnect_joystick(self, force: bool = False) -> None:
        if self._pygame is None:
            return
        now = time.monotonic()
        if not force and now - self._last_reconnect_attempt < 1.0:
            return
        self._last_reconnect_attempt = now
        try:
            self._connect_joystick_or_raise()
            print("\033[32m手柄已自动重连。\033[0m")
        except Exception as exc:
            if not self._disconnected_reported:
                print(f"\033[33m等待手柄重连：{exc}\033[0m")
                self._disconnected_reported = True

    def _mark_joystick_disconnected(self, reason: str) -> None:
        if not self._disconnected_reported:
            print(f"\033[33m手柄连接失效，动作置零并尝试自动重连：{reason}\033[0m")
            self._disconnected_reported = True
        self._joystick = None
        self._joystick_instance_id = None

    def _zero_action(self) -> FloatArray:
        action = np.zeros((self.action_dim,), dtype=np.float32)
        action[-1] = self._gripper_target
        return action

    @staticmethod
    def _buttons(
        start: bool = False,
        success: bool = False,
        abort: bool = False,
        stop: bool = False,
    ) -> ButtonDict:
        return {
            "start": start,
            "success": success,
            "abort": abort,
            "stop": stop,
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Test Xbox controller action mapping.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--max-delta-m", type=float, default=0.005)
    parser.add_argument("--max-delta-rad", type=float, default=0.034906585)
    parser.add_argument("--action-dim", type=int, choices=[4, 7], default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-mode",
        choices=["keyboard", "random", "scripted"],
        default="keyboard",
        help="Dry-run input source. Keyboard: WASD/RF move, Q/E gripper, Enter/Space/B/Esc buttons.",
    )
    parser.add_argument("--debug", action="store_true", help="Print raw axis/button values.")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument(
        "--gripper-action-mode",
        choices=["hold_on_release", "persistent_target"],
        default="persistent_target",
    )

    args = parser.parse_args(argv)

    controller = XboxController(
        device_index=args.device_index,
        deadzone=args.deadzone,
        max_delta_m=args.max_delta_m,
        max_delta_rad=args.max_delta_rad,
        action_dim=args.action_dim,
        dry_run=args.dry_run,
        debug=args.debug,
        dry_run_mode=args.dry_run_mode,
        gripper_action_mode=args.gripper_action_mode,
    )

    controller.connect()
    period_s = 1.0 / args.hz

    print("Press Ctrl+C to exit.")
    print("Dry-run keyboard: WASD/RF, Q/E, Enter/Space/B/Esc.")

    try:
        while True:
            action, buttons = controller.read()
            print(f"action={action.tolist()} buttons={buttons}")

            if buttons["stop"]:
                break

            time.sleep(period_s)

    except KeyboardInterrupt:
        pass

    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
