from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from drivers.ctag_gripper import CTAGGripper
from drivers.kinova_driver import KinovaDriver


@dataclass
class GripperConfig:
    open_width_m: float
    close_width_m: float
    mode: str = "state_only"
    ctag_serial_port: str = "/dev/ttyUSB0"
    ctag_baudrate: int = 115200
    ctag_device_id: int = 1
    ctag_timeout_s: float = 0.2
    ctag_open_pos_mm: float = 850.0
    ctag_close_pos_mm: float = 0.0
    ctag_max_stroke_mm: float = 850.0
    ctag_speed: int = 80
    ctag_close_torque: int = 80
    ctag_open_torque: int = 40
    ctag_acc_dec: int = 80
    ctag_parity: str = "N"
    ctag_stopbits: int = 1
    ctag_enable_rs485_mode: bool = False
    ctag_accept_pos_reached_as_success: bool = False
    ctag_rs485_rts_level_for_tx: bool = True
    ctag_rs485_rts_level_for_rx: bool = False
    ctag_rs485_delay_before_tx: float = 0.0
    ctag_rs485_delay_before_rx: float = 0.0
    open_timeout_s: float = 3.0
    close_timeout_s: float = 5.0


class GripperDriver:
    """Wrapper around either the state-only gripper or the real CTAG Modbus gripper."""

    def __init__(self, robot: KinovaDriver, config: GripperConfig) -> None:
        self.robot = robot
        self.config = config
        self.logger = logging.getLogger("paper.gripper")
        self.backend: Optional[CTAGGripper] = None

        if self.config.mode == "ctag":
            self.backend = CTAGGripper(
                logger=self.logger,
                serial_port=self.config.ctag_serial_port,
                baudrate=self.config.ctag_baudrate,
                device_id=self.config.ctag_device_id,
                timeout=self.config.ctag_timeout_s,
                open_pos_mm=self.config.ctag_open_pos_mm,
                close_pos_mm=self.config.ctag_close_pos_mm,
                max_stroke_mm=self.config.ctag_max_stroke_mm,
                speed=self.config.ctag_speed,
                close_torque=self.config.ctag_close_torque,
                open_torque=self.config.ctag_open_torque,
                acc_dec=self.config.ctag_acc_dec,
                parity=self.config.ctag_parity,
                stopbits=self.config.ctag_stopbits,
                enable_rs485_mode=self.config.ctag_enable_rs485_mode,
                accept_pos_reached_as_success=self.config.ctag_accept_pos_reached_as_success,
                rs485_rts_level_for_tx=self.config.ctag_rs485_rts_level_for_tx,
                rs485_rts_level_for_rx=self.config.ctag_rs485_rts_level_for_rx,
                rs485_delay_before_tx=self.config.ctag_rs485_delay_before_tx,
                rs485_delay_before_rx=self.config.ctag_rs485_delay_before_rx,
            )
            if not self.backend.connect():
                raise RuntimeError(
                    f"Failed to connect CTAG gripper on {self.config.ctag_serial_port}."
                )

    def open(self) -> bool:
        if self.backend is not None and not self.backend.open(timeout_s=self.config.open_timeout_s):
            return False
        self.robot.set_gripper_opening(self.config.open_width_m)
        return True

    def close(self) -> bool:
        if self.backend is not None and not self.backend.close(timeout_s=self.config.close_timeout_s):
            return False
        self.robot.set_gripper_opening(self.config.close_width_m)
        return True

    def shutdown(self) -> None:
        if self.backend is not None:
            self.backend.shutdown()
