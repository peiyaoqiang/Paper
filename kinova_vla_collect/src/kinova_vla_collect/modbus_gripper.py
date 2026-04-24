from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal


GripperName = Literal["open", "hold", "close"]


@dataclass(frozen=True)
class GripperCommand:
    value: float
    name: GripperName


class ModbusGripper:
    """
    Modbus gripper wrapper.

    Command convention:
    - `-1.0`: open
    - `0.0`: hold
    - `+1.0`: close

    The default register map is the CTAG map used by the previous project code.
    """

    REG_ENABLE: int = 0x0100
    REG_POS_H: int = 0x0102
    REG_SPEED: int = 0x0104
    REG_TORQUE: int = 0x0105
    REG_ACC: int = 0x0106
    REG_DEC: int = 0x0107
    REG_TRIGGER: int = 0x0108
    REG_TORQUE_REACHED: int = 0x0601
    REG_POS_REACHED: int = 0x0602
    REG_READY: int = 0x0604
    REG_ALARM: int = 0x0612

    def __init__(
        self,
        host: str,
        port: int,
        unit_id: int,
        dry_run: bool = False,
        mode: str = "tcp",
        serial_port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout_s: float = 4.0,
        open_pos_mm: float = 0.0,
        close_pos_mm: float = 120.0,
        max_stroke_mm: float = 120.0,
        speed: int = 30,
        close_torque: int = 10,
        open_torque: int = 100,
        acc_dec: int = 2000,
        parity: str = "N",
        stopbits: int = 1,
        enable_rs485_mode: bool = False,
        accept_pos_reached_as_success: bool = True,
        open_timeout_s: float = 4.0,
        close_timeout_s: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.dry_run = dry_run
        self.mode = mode
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.open_pos_mm = open_pos_mm
        self.close_pos_mm = close_pos_mm
        self.max_stroke_mm = max_stroke_mm
        self.speed = speed
        self.close_torque = close_torque
        self.open_torque = open_torque
        self.acc_dec = acc_dec
        self.parity = parity
        self.stopbits = stopbits
        self.enable_rs485_mode = enable_rs485_mode
        self.accept_pos_reached_as_success = accept_pos_reached_as_success
        self.open_timeout_s = open_timeout_s
        self.close_timeout_s = close_timeout_s
        self._client: Any | None = None
        self._modbus_exception: type[BaseException] = Exception
        self._connected = False
        self._position = 0.0
        self._current_command = 0.0
        self._last_sent_command: float | None = None

    def __enter__(self) -> "ModbusGripper":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def connect(self) -> None:
        if self._connected:
            return
        if self.dry_run:
            self._connected = True
            return
        try:
            from pymodbus.client import ModbusSerialClient
            from pymodbus.client import ModbusTcpClient
            from pymodbus.exceptions import ModbusException
        except ImportError as exc:
            raise RuntimeError("pymodbus is required when dry_run=False") from exc

        try:
            mode = self.mode.lower()
            if mode in {"rtu", "serial", "ctag", "ctag_rtu", "modbus_rtu"}:
                client = ModbusSerialClient(
                    port=self.serial_port,
                    baudrate=self.baudrate,
                    parity=self.parity,
                    stopbits=self.stopbits,
                    bytesize=8,
                    timeout=self.timeout_s,
                )
                target = self.serial_port
            elif mode in {"tcp", "modbus_tcp"}:
                client = ModbusTcpClient(host=self.host, port=self.port, timeout=self.timeout_s)
                target = f"{self.host}:{self.port}"
            else:
                raise ValueError(f"Unsupported gripper mode: {self.mode}")
            if not client.connect():
                raise RuntimeError("client.connect() returned False")
        except Exception as exc:
            raise RuntimeError(f"Failed to connect Modbus gripper in mode={self.mode}") from exc

        self._client = client
        self._modbus_exception = ModbusException
        self._connected = True
        if self.mode.lower() in {"rtu", "serial", "ctag", "ctag_rtu", "modbus_rtu"}:
            self._write_register(self.REG_ENABLE, 1, f"enable CTAG gripper on {target}")

    def close(self) -> None:
        if self._client is not None:
            try:
                if self._connected and not self.dry_run and self.mode.lower() in {"rtu", "serial", "ctag", "ctag_rtu", "modbus_rtu"}:
                    self._write_register(self.REG_ENABLE, 0, "disable CTAG gripper")
                self._client.close()
            except Exception as exc:
                raise RuntimeError("Failed to close Modbus gripper client") from exc
        self._client = None
        self._connected = False

    def disconnect(self) -> None:
        self.close()

    def open(self) -> None:
        self.open_gripper()

    def close_gripper(self) -> None:
        self._set_command(GripperCommand(value=1.0, name="close"))

    def open_gripper(self) -> None:
        self._set_command(GripperCommand(value=-1.0, name="open"))

    def hold(self) -> None:
        self._set_command(GripperCommand(value=0.0, name="hold"))

    def apply_action(self, gripper_action: float) -> GripperCommand:
        if gripper_action > 0.5:
            self.close_gripper()
        elif gripper_action < -0.5:
            self.open_gripper()
        else:
            self.hold()
        return GripperCommand(value=self._current_command, name=self._command_name(self._current_command))

    def command(self, value: float) -> GripperCommand:
        return self.apply_action(value)

    def get_position(self) -> float:
        self._ensure_connected()
        if self.dry_run:
            self._advance_dry_position()
            return self._position
        if self.mode.lower() in {"rtu", "serial", "ctag", "ctag_rtu", "modbus_rtu"}:
            return self._position

        return self._position

    def get_current_command(self) -> float:
        return self._current_command

    def _set_command(self, command: GripperCommand) -> None:
        self._ensure_connected()
        if self._last_sent_command == command.value:
            self._current_command = command.value
            if self.dry_run:
                self._advance_dry_position()
            return
        self._current_command = command.value
        self._last_sent_command = command.value
        if self.dry_run:
            self._advance_dry_position()
            return

        if command.name == "open":
            self._set_ctag_jaw_param(self.open_pos_mm, self.open_torque)
            self._trigger_ctag_move()
            self._position = 0.0
        elif command.name == "close":
            self._set_ctag_jaw_param(self.close_pos_mm, self.close_torque)
            self._trigger_ctag_move()
            self._position = 1.0
        else:
            self._write_register(self.REG_TRIGGER, 0, "hold gripper")

    def _advance_dry_position(self) -> None:
        step = 0.15
        if self._current_command > 0.5:
            self._position = min(1.0, self._position + step)
        elif self._current_command < -0.5:
            self._position = max(0.0, self._position - step)

    def _write_register(self, address: int, value: int, action_name: str) -> None:
        if self._client is None:
            raise RuntimeError("Modbus gripper client is not initialized")
        try:
            result = self._call_modbus(
                self._client.write_register,
                address=address,
                value=int(value),
            )
        except self._modbus_exception as exc:
            raise RuntimeError(
                f"Modbus error during {action_name}: address=0x{address:04X}, value={value}, unit_id={self.unit_id}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed during {action_name}: address=0x{address:04X}, value={value}, unit_id={self.unit_id}"
            ) from exc
        self._check_result(result, action_name)

    def _read_register(self, address: int, action_name: str) -> int:
        if self._client is None:
            raise RuntimeError("Modbus gripper client is not initialized")
        try:
            result = self._call_modbus(
                self._client.read_holding_registers,
                address=address,
                count=1,
            )
        except self._modbus_exception as exc:
            raise RuntimeError(
                f"Modbus error during {action_name}: address=0x{address:04X}, unit_id={self.unit_id}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed during {action_name}: address=0x{address:04X}, unit_id={self.unit_id}"
            ) from exc
        self._check_result(result, action_name)
        registers = getattr(result, "registers", None)
        if not registers:
            raise RuntimeError(f"Modbus {action_name} returned no registers at address=0x{address:04X}")
        return int(registers[0])

    def _call_modbus(self, method: Any, **kwargs: Any) -> Any:
        for unit_keyword in ("device_id", "slave", "unit"):
            try:
                return method(**kwargs, **{unit_keyword: self.unit_id})
            except TypeError:
                continue
        return method(**kwargs)

    @staticmethod
    def _check_result(result: Any, action_name: str) -> None:
        if result is None:
            raise RuntimeError(f"Modbus {action_name} failed: no response")
        if hasattr(result, "isError") and result.isError():
            raise RuntimeError(f"Modbus {action_name} failed: {result}")

    def _set_ctag_jaw_param(self, pos_mm: float, torque: int) -> None:
        pos_value = self._mm_to_pos_value(pos_mm)
        pos_high = (pos_value >> 16) & 0xFFFF
        pos_low = pos_value & 0xFFFF
        self._write_registers(self.REG_POS_H, [pos_high, pos_low], "write CTAG position")
        self._write_register(self.REG_SPEED, self.speed, "write CTAG speed")
        self._write_register(self.REG_TORQUE, torque, "write CTAG torque")
        self._write_register(self.REG_ACC, self.acc_dec, "write CTAG acceleration")
        self._write_register(self.REG_DEC, self.acc_dec, "write CTAG deceleration")

    def _trigger_ctag_move(self) -> None:
        self._write_register(self.REG_TRIGGER, 1, "trigger CTAG move")
        time.sleep(0.05)
        self._write_register(self.REG_TRIGGER, 0, "reset CTAG trigger")

    def _wait_until_ready(self, timeout_s: float, action_name: str) -> None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while time.monotonic() < deadline:
            ready = self._read_register(self.REG_READY, f"read ready status after {action_name}")
            if ready == 1:
                return
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for CTAG gripper ready after {action_name}")

    def _wait_until_close_done(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while time.monotonic() < deadline:
            torque_reached = self._read_register(self.REG_TORQUE_REACHED, "read torque reached status")
            pos_reached = self._read_register(self.REG_POS_REACHED, "read position reached status")
            ready = self._read_register(self.REG_READY, "read ready status after close")
            if torque_reached == 1:
                return
            if pos_reached == 1:
                if self.accept_pos_reached_as_success:
                    return
                raise RuntimeError("CTAG gripper fully closed but did not report torque reached")
            if ready == 1:
                return
            time.sleep(0.05)
        alarm = self._read_register(self.REG_ALARM, "read alarm after close timeout")
        raise TimeoutError(f"Timed out waiting for CTAG close; alarm=0x{alarm:04X}")

    def _write_registers(self, address: int, values: list[int], action_name: str) -> None:
        if self._client is None:
            raise RuntimeError("Modbus gripper client is not initialized")
        try:
            result = self._call_modbus(
                self._client.write_registers,
                address=address,
                values=[int(value) for value in values],
            )
        except self._modbus_exception as exc:
            raise RuntimeError(
                f"Modbus error during {action_name}: address=0x{address:04X}, values={values}, unit_id={self.unit_id}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed during {action_name}: address=0x{address:04X}, values={values}, unit_id={self.unit_id}"
            ) from exc
        self._check_result(result, action_name)

    def _mm_to_pos_value(self, pos_mm: float) -> int:
        clipped_mm = max(0.0, min(self.max_stroke_mm, pos_mm))
        return int(round(clipped_mm * 100.0))

    def _position_raw_to_normalized(self, raw_position: int) -> float:
        raw_open = self._mm_to_pos_value(self.open_pos_mm)
        raw_close = self._mm_to_pos_value(self.close_pos_mm)
        span = float(raw_close - raw_open)
        if abs(span) <= 1e-9:
            return 0.0
        normalized = (float(raw_position) - float(raw_open)) / span
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def _command_name(command: float) -> GripperName:
        if command > 0.5:
            return "close"
        if command < -0.5:
            return "open"
        return "hold"

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("ModbusGripper is not connected. Call connect() first.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Test Modbus gripper wrapper.")
    parser.add_argument("--mode", type=str, default="ctag_rtu")
    parser.add_argument("--host", type=str, default="192.168.1.20")
    parser.add_argument("--port", type=int, default=502)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--serial-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--dry-run", action="store_true", help="Simulate gripper without opening Modbus connection.")
    parser.add_argument("--wait", action="store_true", help="After each command, wait for CTAG ready/status for bench testing.")
    parser.add_argument("--sequence", type=str, default="open,hold,close,close,hold,open")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args(argv)

    action_map = {
        "open": -1.0,
        "hold": 0.0,
        "close": 1.0,
    }
    print(
        f"Starting gripper test mode={args.mode} dry_run={args.dry_run} "
        f"serial={args.serial_port} baudrate={args.baudrate} unit_id={args.unit_id}"
    )
    with ModbusGripper(
        host=args.host,
        port=args.port,
        unit_id=args.unit_id,
        dry_run=args.dry_run,
        mode=args.mode,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
    ) as gripper:
        for token in [item.strip().lower() for item in args.sequence.split(",") if item.strip()]:
            if token not in action_map:
                raise ValueError(f"Unknown sequence token: {token}. Use open,hold,close.")
            command = gripper.apply_action(action_map[token])
            if args.wait and not args.dry_run:
                if token == "open":
                    gripper._wait_until_ready(gripper.open_timeout_s, "open gripper")
                elif token == "close":
                    gripper._wait_until_close_done(gripper.close_timeout_s)
            position = gripper.get_position()
            print(
                f"requested={token:>5s} command={command.value:+.1f} "
                f"current={gripper.get_current_command():+.1f} position={position:.3f}"
            )
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
