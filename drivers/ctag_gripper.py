import array
import errno
import os
import time


class CTAGGripper:
    REG_ENABLE = 0x0100
    REG_POS_H = 0x0102
    REG_SPEED = 0x0104
    REG_TORQUE = 0x0105
    REG_ACC = 0x0106
    REG_DEC = 0x0107
    REG_TRIGGER = 0x0108

    REG_TORQUE_REACHED = 0x0601
    REG_POS_REACHED = 0x0602
    REG_READY = 0x0604
    REG_ALARM = 0x0612
    TIOCGRS485 = 0x542E
    TIOCSRS485 = 0x542F
    ERRNO_ENOIOCTLCMD = getattr(errno, "ENOIOCTLCMD", None)

    def __init__(
        self,
        logger,
        serial_port: str,
        baudrate: int,
        device_id: int,
        timeout: float,
        open_pos_mm: float,
        close_pos_mm: float,
        max_stroke_mm: float,
        speed: int,
        close_torque: int,
        open_torque: int,
        acc_dec: int,
        parity: str = 'N',
        stopbits: int = 1,
        enable_rs485_mode: bool = False,
        accept_pos_reached_as_success: bool = False,
        rs485_rts_level_for_tx: bool = True,
        rs485_rts_level_for_rx: bool = False,
        rs485_delay_before_tx: float = 0.0,
        rs485_delay_before_rx: float = 0.0,
    ):
        try:
            from pymodbus.client import ModbusSerialClient as ModbusClient
            from pymodbus.exceptions import ModbusException
            import serial
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "缺少 pymodbus。请先执行: python3 -m pip install pymodbus"
            ) from exc

        self.ModbusClient = ModbusClient
        self.ModbusException = ModbusException
        self.serial_module = serial
        self.logger = logger

        self.serial_port = serial_port
        self.baudrate = baudrate
        self.device_id = device_id
        self.timeout = timeout
        self.parity = parity
        self.stopbits = stopbits

        self.open_pos_mm = open_pos_mm
        self.close_pos_mm = close_pos_mm
        self.max_stroke_mm = max_stroke_mm

        self.speed = speed
        self.close_torque = close_torque
        self.open_torque = open_torque
        self.acc_dec = acc_dec
        self.enable_rs485_mode = enable_rs485_mode
        self.accept_pos_reached_as_success = accept_pos_reached_as_success
        self.rs485_rts_level_for_tx = rs485_rts_level_for_tx
        self.rs485_rts_level_for_rx = rs485_rts_level_for_rx
        self.rs485_delay_before_tx = rs485_delay_before_tx
        self.rs485_delay_before_rx = rs485_delay_before_rx

        self.client = None

    def connect(self) -> bool:
        self.client = self.ModbusClient(
            port=self.serial_port,
            baudrate=self.baudrate,
            parity=self.parity,
            stopbits=self.stopbits,
            bytesize=8,
            timeout=self.timeout,
        )

        if not self.client.connect():
            self.logger.error(f"夹爪串口连接失败: {self.serial_port}")
            return False

        self.log_linux_serial_diagnostics()

        if not self.configure_rs485_mode():
            self.client.close()
            return False

        if not self.write_single_register(self.REG_ENABLE, 1, "执行器使能"):
            return False

        self.logger.info("CTAG夹爪已连接并使能")
        return True

    def log_linux_serial_diagnostics(self):
        driver_name, driver_path = self.get_linux_tty_driver()
        if driver_name is not None:
            self.logger.info(
                f"夹爪串口驱动: {driver_name} ({driver_path})"
            )
            if "ch341" in driver_name.lower():
                self.logger.warn(
                    "检测到 CH341/CH340 Linux 串口驱动。"
                    "这类适配器通常按普通 UART 使用，但 RS485 ioctl 支持往往有限。"
                )

        supported, detail = self.probe_linux_rs485_ioctl()
        if supported is True:
            self.logger.info(f"Linux RS485 ioctl 可用: {detail}")
        elif supported is False:
            self.logger.warn(f"Linux RS485 ioctl 不可用: {detail}")

    def get_linux_tty_driver(self):
        if os.name != "posix" or not self.serial_port.startswith("/dev/"):
            return None, None

        tty_name = os.path.basename(self.serial_port)
        driver_link = f"/sys/class/tty/{tty_name}/device/driver"
        if not os.path.exists(driver_link):
            return None, None

        real_path = os.path.realpath(driver_link)
        return os.path.basename(real_path), real_path

    def probe_linux_rs485_ioctl(self):
        if os.name != "posix":
            return None, "当前系统不是 POSIX/Linux"

        socket = getattr(self.client, "socket", None)
        if socket is None or not hasattr(socket, "fileno"):
            return None, "pymodbus 串口对象不存在或不支持 fileno()"

        try:
            import fcntl
        except ModuleNotFoundError:
            return None, "fcntl 不可用"

        try:
            fd = socket.fileno()
        except Exception as exc:
            return None, f"获取串口 fd 失败: {exc}"

        buffer = array.array("I", [0] * 8)
        try:
            fcntl.ioctl(fd, self.TIOCGRS485, buffer, True)
        except OSError as exc:
            unsupported_errnos = {
                errno.ENOTTY,
                errno.EINVAL,
                errno.EOPNOTSUPP,
                errno.ENOSYS,
            }
            if self.ERRNO_ENOIOCTLCMD is not None:
                unsupported_errnos.add(self.ERRNO_ENOIOCTLCMD)
            if exc.errno in unsupported_errnos:
                return False, f"{exc.strerror} (errno={exc.errno})"
            return False, f"{exc.strerror} (errno={exc.errno})"
        except Exception as exc:
            return False, str(exc)

        flags = buffer[0]
        return True, f"flags=0x{flags:08X}"

    def configure_rs485_mode(self) -> bool:
        if not self.enable_rs485_mode:
            return True

        socket = getattr(self.client, "socket", None)
        if socket is None:
            self.logger.warn(
                "启用 rs485_mode 失败: pymodbus 串口对象不存在，"
                "继续按普通串口方式通信"
            )
            return True

        if not hasattr(self.serial_module, "rs485"):
            self.logger.warn(
                "启用 rs485_mode 失败: 当前 pyserial 不支持 rs485 扩展，"
                "继续按普通串口方式通信"
            )
            return True

        supported, detail = self.probe_linux_rs485_ioctl()
        if supported is False:
            self.logger.warn(
                "跳过 rs485_mode: Linux 驱动不支持 RS485 ioctl，"
                f"pyserial 无法接管收发方向控制 ({detail})"
            )
            self.logger.warn("继续按普通串口方式通信")
            return True

        try:
            rs485_settings = self.serial_module.rs485.RS485Settings(
                rts_level_for_tx=self.rs485_rts_level_for_tx,
                rts_level_for_rx=self.rs485_rts_level_for_rx,
                delay_before_tx=self.rs485_delay_before_tx,
                delay_before_rx=self.rs485_delay_before_rx,
            )
            socket.rs485_mode = rs485_settings
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.logger.warn(
                f"启用 rs485_mode 失败: {exc}，继续按普通串口方式通信"
            )
            return True

        self.logger.info(
            "已启用 pyserial rs485_mode | "
            f"tx={self.rs485_rts_level_for_tx} | "
            f"rx={self.rs485_rts_level_for_rx} | "
            f"delay_before_tx={self.rs485_delay_before_tx:.4f}s | "
            f"delay_before_rx={self.rs485_delay_before_rx:.4f}s"
        )

        supported, detail = self.probe_linux_rs485_ioctl()
        if supported is True:
            self.logger.info(f"rs485_mode 生效后的内核状态: {detail}")
        return True

    def shutdown(self):
        if self.client is None:
            return

        try:
            self.write_single_register(self.REG_ENABLE, 0, "执行器失能")
            self.client.close()
        except Exception as exc:  # pragma: no cover - hardware cleanup
            self.logger.warn(f"关闭夹爪资源时出现异常: {exc}")

    def mm_to_pos_value(self, mm: float) -> int:
        mm = max(0.0, min(self.max_stroke_mm, mm))
        return int(round(mm * 100))

    def check_result(self, result, action_name: str) -> bool:
        if result is None:
            self.logger.error(f"{action_name}失败: 无返回结果")
            return False
        if hasattr(result, "isError") and result.isError():
            self.logger.error(f"{action_name}失败: {result}")
            return False
        return True

    def write_single_register(self, address: int, value: int, action_name: str) -> bool:
        try:
            result = self.client.write_register(
                address=address,
                value=value,
                device_id=self.device_id,
            )
            return self.check_result(result, action_name)
        except self.ModbusException as exc:
            self.logger.error(f"{action_name}失败: {exc}")
            return False

    def write_multi_registers(self, address: int, values: list[int], action_name: str) -> bool:
        try:
            result = self.client.write_registers(
                address=address,
                values=values,
                device_id=self.device_id,
            )
            return self.check_result(result, action_name)
        except self.ModbusException as exc:
            self.logger.error(f"{action_name}失败: {exc}")
            return False

    def read_single_register(self, address: int, action_name: str):
        try:
            result = self.client.read_holding_registers(
                address=address,
                count=1,
                device_id=self.device_id,
            )
            if not self.check_result(result, action_name):
                return None
            return result.registers[0]
        except self.ModbusException as exc:
            self.logger.error(f"{action_name}失败: {exc}")
            return None

    def set_jaw_param(self, pos_mm: float, torque: int) -> bool:
        pos_value = self.mm_to_pos_value(pos_mm)
        pos_high = (pos_value >> 16) & 0xFFFF
        pos_low = pos_value & 0xFFFF

        if not self.write_multi_registers(self.REG_POS_H, [pos_high, pos_low], "写入位置参数"):
            return False
        if not self.write_single_register(self.REG_SPEED, self.speed, "写入速度参数"):
            return False
        if not self.write_single_register(self.REG_TORQUE, torque, "写入力矩参数"):
            return False
        if not self.write_single_register(self.REG_ACC, self.acc_dec, "写入加速度参数"):
            return False
        if not self.write_single_register(self.REG_DEC, self.acc_dec, "写入减速度参数"):
            return False
        return True

    def trigger_jaw_move(self) -> bool:
        if not self.write_single_register(self.REG_TRIGGER, 1, "触发运动"):
            return False
        time.sleep(0.05)
        if not self.write_single_register(self.REG_TRIGGER, 0, "复位触发位"):
            return False
        return True

    def wait_until_ready(self, timeout_s: float) -> bool:
        start = time.time()
        while time.time() - start < timeout_s:
            ready = self.read_single_register(self.REG_READY, "读取执行器准备完成状态")
            if ready == 1:
                return True
            time.sleep(0.05)
        return False

    def open(self, timeout_s: float = 3.0) -> bool:
        self.logger.info("执行夹爪张开")
        if not self.set_jaw_param(self.open_pos_mm, self.open_torque):
            return False
        if not self.trigger_jaw_move():
            return False
        ready = self.wait_until_ready(timeout_s)
        if not ready:
            self.logger.warn("夹爪张开等待超时")
        return ready

    def close(self, timeout_s: float = 5.0) -> bool:
        self.logger.info("执行夹爪闭合")
        if not self.set_jaw_param(self.close_pos_mm, self.close_torque):
            return False
        if not self.trigger_jaw_move():
            return False

        start = time.time()
        while time.time() - start < timeout_s:
            torque_reached = self.read_single_register(self.REG_TORQUE_REACHED, "读取力矩到达状态")
            pos_reached = self.read_single_register(self.REG_POS_REACHED, "读取位置到达状态")
            ready = self.read_single_register(self.REG_READY, "读取执行器准备完成状态")

            if torque_reached == 1:
                self.logger.info("夹爪检测到受力，判定已抓住目标")
                return True
            if pos_reached == 1:
                if self.accept_pos_reached_as_success:
                    self.logger.warn(
                        "夹爪完全闭合但未检测到明显受力。"
                        "已启用软物体模式，按抓取成功处理。"
                    )
                    return True
                self.logger.warn("夹爪完全闭合但未检测到受力，可能没有抓到物体")
                return False
            if ready == 1:
                self.logger.info("夹爪动作完成")
                return True

            time.sleep(0.05)

        alarm = self.read_single_register(self.REG_ALARM, "读取报警信息")
        if alarm not in (None, 0):
            self.logger.warn(f"夹爪报警码: 0x{alarm:04X}")
        self.logger.warn("夹爪闭合等待超时")
        return False
