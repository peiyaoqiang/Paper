from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.gripper_driver import GripperConfig, GripperDriver
from drivers.kinova_driver import KinovaConfig, KinovaDriver


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "default_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone CTAG gripper open/close test.")
    parser.add_argument(
        "--action",
        type=str,
        choices=("open", "close", "cycle"),
        default="cycle",
        help="Which gripper action to run.",
    )
    parser.add_argument(
        "--pause-s",
        type=float,
        default=1.0,
        help="Pause between open and close when action=cycle.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the requested action.",
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        default="",
        help="Optional CTAG serial port override such as /dev/ttyUSB0.",
    )
    return parser.parse_args()


def build_gripper(config: dict, serial_port_override: str = "") -> GripperDriver:
    serial_port = serial_port_override.strip() or config["gripper"].get("ctag_serial_port", "/dev/ttyUSB0")
    robot = KinovaDriver(
        KinovaConfig(
            max_translation_step_m=config["robot"]["max_translation_step_m"],
            max_rotation_step_deg=config["robot"]["max_rotation_step_deg"],
            mode="mock",
        )
    )
    return GripperDriver(
        robot,
        GripperConfig(
            open_width_m=config["gripper"]["open_width_m"],
            close_width_m=config["gripper"]["close_width_m"],
            mode=config["gripper"].get("mode", "state_only"),
            ctag_serial_port=serial_port,
            ctag_baudrate=config["gripper"].get("ctag_baudrate", 115200),
            ctag_device_id=config["gripper"].get("ctag_device_id", 1),
            ctag_timeout_s=config["gripper"].get("ctag_timeout_s", 1.0),
            ctag_open_pos_mm=config["gripper"].get("ctag_open_pos_mm", 0.0),
            ctag_close_pos_mm=config["gripper"].get("ctag_close_pos_mm", 120.0),
            ctag_max_stroke_mm=config["gripper"].get("ctag_max_stroke_mm", 120.0),
            ctag_speed=config["gripper"].get("ctag_speed", 30),
            ctag_close_torque=config["gripper"].get("ctag_close_torque", 10),
            ctag_open_torque=config["gripper"].get("ctag_open_torque", 100),
            ctag_acc_dec=config["gripper"].get("ctag_acc_dec", 2000),
            ctag_parity=config["gripper"].get("ctag_parity", "N"),
            ctag_stopbits=config["gripper"].get("ctag_stopbits", 1),
            ctag_enable_rs485_mode=config["gripper"].get("ctag_enable_rs485_mode", False),
            ctag_accept_pos_reached_as_success=config["gripper"].get(
                "ctag_accept_pos_reached_as_success", True
            ),
            ctag_rs485_rts_level_for_tx=config["gripper"].get("ctag_rs485_rts_level_for_tx", True),
            ctag_rs485_rts_level_for_rx=config["gripper"].get("ctag_rs485_rts_level_for_rx", False),
            ctag_rs485_delay_before_tx=config["gripper"].get("ctag_rs485_delay_before_tx", 0.0),
            ctag_rs485_delay_before_rx=config["gripper"].get("ctag_rs485_delay_before_rx", 0.0),
            open_timeout_s=config["gripper"].get("open_timeout_s", 3.0),
            close_timeout_s=config["gripper"].get("close_timeout_s", 5.0),
        ),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_config()

    print("Gripper mode:", config["gripper"].get("mode", "state_only"))
    print("Serial port:", args.serial_port.strip() or config["gripper"].get("ctag_serial_port", "/dev/ttyUSB0"))
    print("Action:", args.action)
    print("Repeat:", args.repeat)

    gripper = build_gripper(config, args.serial_port)
    try:
        for idx in range(args.repeat):
            print(f"Run {idx + 1}/{args.repeat}")
            if args.action == "open":
                success = gripper.open()
                print("Open success:", success)
            elif args.action == "close":
                success = gripper.close()
                print("Close success:", success)
            else:
                open_success = gripper.open()
                print("Open success:", open_success)
                time.sleep(max(args.pause_s, 0.0))
                close_success = gripper.close()
                print("Close success:", close_success)
    finally:
        gripper.shutdown()


if __name__ == "__main__":
    main()
