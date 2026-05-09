from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _serial_available, require_package

from .types import PaxiniSample

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None  # type: ignore[assignment]

FIN_PCL_UART_HEAD_LEN = 13
TX_RX_BUFFER_SIZE = 4096
CMD_TX_HEAD = b"\x55\xAA"
CMD_RX_HEAD = b"\xAA\x55"
READ_FLAG = 1 << 7

_NUMPY_DTYPES = {
    "uint8": np.dtype("<u1"),
    "int8": np.dtype("<i1"),
    "uint16": np.dtype("<u2"),
    "int16": np.dtype("<i2"),
    "uint32": np.dtype("<u4"),
    "int32": np.dtype("<i4"),
    "float32": np.dtype("<f4"),
}


class PaxiniProtocolError(RuntimeError):
    pass


def lrc_cal(data: bytes | bytearray | memoryview) -> int:
    """Calculate the two's-complement LRC used by the Paxini UART protocol."""
    return (-sum(data)) & 0xFF


def _u16_le(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=False)


def _u32_le(value: int) -> bytes:
    return int(value).to_bytes(4, "little", signed=False)


def build_paxini_frame(
    *,
    device_id: int,
    func_code: int,
    addr: int,
    data_len: int,
    payload: bytes = b"",
    read: bool,
) -> bytes:
    """Build a Paxini master request frame.

    The MCU example sends read requests with header ``55 AA`` and function bit 7 set.
    The length field follows the C implementation: ``total_frame_len - 5``.
    """
    func_byte = (func_code | READ_FLAG) if read else (func_code & ~READ_FLAG)
    total = FIN_PCL_UART_HEAD_LEN + len(payload) + 1
    if total > TX_RX_BUFFER_SIZE:
        raise ValueError(f"Paxini frame too large: {total} bytes.")

    frame = bytearray()
    frame += CMD_TX_HEAD
    frame += _u16_le(total - 5)
    frame.append(device_id & 0xFF)
    frame.append(0x00)
    frame.append(func_byte & 0xFF)
    frame += _u32_le(addr)
    frame += _u16_le(data_len)
    frame += payload
    frame.append(lrc_cal(frame))
    return bytes(frame)


class PaxiniReader:
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def read_latest(self) -> PaxiniSample:
        raise NotImplementedError


@dataclass
class MockPaxiniReader(PaxiniReader):
    num_taxels: int
    phase: float = 0.0
    connected: bool = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_latest(self) -> PaxiniSample:
        if not self.connected:
            return PaxiniSample(taxels=np.zeros(self.num_taxels, dtype=np.float32), timestamp_s=time.time())

        self.phase += 0.07
        idx = np.arange(self.num_taxels, dtype=np.float32)
        center = (np.sin(self.phase) * 0.5 + 0.5) * max(self.num_taxels - 1, 1)
        taxels = np.exp(-0.5 * ((idx - center) / max(self.num_taxels / 12.0, 1.0)) ** 2).astype(np.float32)
        return PaxiniSample(
            taxels=taxels,
            fx=float(0.01 * np.sin(self.phase)),
            fy=float(0.01 * np.cos(self.phase)),
            fz=float(taxels.max()),
            timestamp_s=time.time(),
        )


class SerialPaxiniReader(PaxiniReader):
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        *,
        num_taxels: int = 100,
        device_id: int = 0x01,
        read_func_code: int = 0x7B,
        read_addr: int = 0x040E,
        read_len: int | None = None,
        timeout_s: float = 0.5,
        taxel_dtype: str = "uint16",
        taxel_scale: float | None = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.num_taxels = num_taxels
        self.device_id = device_id
        self.read_func_code = read_func_code
        self.read_addr = read_addr
        self.timeout_s = timeout_s
        self.taxel_dtype = taxel_dtype
        self.taxel_scale = taxel_scale
        self.connected = False
        self._serial: serial.Serial | None = None

        if taxel_dtype not in _NUMPY_DTYPES:
            supported = sorted(_NUMPY_DTYPES)
            raise ValueError(f"Unsupported taxel_dtype={taxel_dtype!r}. Choose one of {supported}.")
        self._taxel_np_dtype = _NUMPY_DTYPES[taxel_dtype]
        self.read_len = read_len if read_len is not None else num_taxels * self._taxel_np_dtype.itemsize

    def connect(self) -> None:
        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=min(self.timeout_s, 0.05),
            write_timeout=self.timeout_s,
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        self.connected = True

    def disconnect(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None
        self.connected = False

    def read_register(self, func_code: int, addr: int, length: int) -> bytes:
        frame = build_paxini_frame(
            device_id=self.device_id,
            func_code=func_code,
            addr=addr,
            data_len=length,
            read=True,
        )
        response = self._exchange(frame)
        return self._parse_response(
            response,
            expected_func_code=func_code | READ_FLAG,
            expected_addr=addr,
            expected_data_len=length,
        )

    def write_register(self, func_code: int, addr: int, payload: bytes) -> None:
        frame = build_paxini_frame(
            device_id=self.device_id,
            func_code=func_code,
            addr=addr,
            data_len=len(payload),
            payload=payload,
            read=False,
        )
        response = self._exchange(frame)
        self._parse_response(
            response,
            expected_func_code=func_code & ~READ_FLAG,
            expected_addr=addr,
            expected_data_len=0,
        )

    def read_latest(self) -> PaxiniSample:
        if not self.connected:
            raise RuntimeError("SerialPaxiniReader is not connected.")

        payload = self.read_register(self.read_func_code, self.read_addr, self.read_len)
        return self._decode_sample(payload)

    def _exchange(self, request: bytes) -> bytes:
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("SerialPaxiniReader is not connected.")

        self._serial.reset_input_buffer()
        self._serial.write(request)
        self._serial.flush()
        return self._read_response_frame()

    def _read_response_frame(self) -> bytes:
        if self._serial is None:
            raise RuntimeError("SerialPaxiniReader is not connected.")

        deadline = time.monotonic() + self.timeout_s
        buffer = bytearray()

        while time.monotonic() < deadline:
            read_size = max(1, min(256, getattr(self._serial, "in_waiting", 0) or 1))
            chunk = self._serial.read(read_size)
            if not chunk:
                continue

            buffer.extend(chunk)
            head_idx = buffer.find(CMD_RX_HEAD)
            if head_idx < 0:
                del buffer[:-1]
                continue
            if head_idx > 0:
                del buffer[:head_idx]

            if len(buffer) < 4:
                continue

            body_len = int.from_bytes(buffer[2:4], "little", signed=False)
            frame_len = body_len + 5
            if frame_len < FIN_PCL_UART_HEAD_LEN + 1 or frame_len > TX_RX_BUFFER_SIZE:
                raise PaxiniProtocolError(f"Invalid Paxini response length field: {body_len}.")

            if len(buffer) < frame_len:
                continue

            frame = bytes(buffer[:frame_len])
            if lrc_cal(frame[:-1]) != frame[-1]:
                raise PaxiniProtocolError("Paxini response LRC check failed.")

            return frame

        raise TimeoutError(f"Timed out waiting for Paxini response on {self.port}.")

    def _parse_response(
        self,
        frame: bytes,
        *,
        expected_func_code: int,
        expected_addr: int,
        expected_data_len: int,
    ) -> bytes:
        if len(frame) < FIN_PCL_UART_HEAD_LEN + 2:
            raise PaxiniProtocolError(f"Paxini response too short: {len(frame)} bytes.")
        if not frame.startswith(CMD_RX_HEAD):
            raise PaxiniProtocolError(f"Unexpected Paxini response header: {frame[:2]!r}.")
        if frame[4] != (self.device_id & 0xFF):
            raise PaxiniProtocolError(f"Unexpected Paxini device id: {frame[4]:#04x}.")
        if frame[6] != (expected_func_code & 0xFF):
            raise PaxiniProtocolError(f"Unexpected Paxini function code: {frame[6]:#04x}.")

        response_addr = int.from_bytes(frame[7:11], "little", signed=False)
        if response_addr != expected_addr:
            raise PaxiniProtocolError(
                f"Unexpected Paxini response address: {response_addr:#010x}, expected {expected_addr:#010x}."
            )

        status = frame[FIN_PCL_UART_HEAD_LEN]
        if status != 0:
            raise PaxiniProtocolError(f"Paxini returned non-zero status: {status}.")

        data = frame[FIN_PCL_UART_HEAD_LEN + 1 : -1]
        if expected_data_len and len(data) < expected_data_len:
            raise PaxiniProtocolError(
                f"Paxini payload too short: expected {expected_data_len}, got {len(data)}."
            )

        response_data_len = int.from_bytes(frame[11:13], "little", signed=False)
        if expected_data_len and response_data_len not in (expected_data_len, len(data)):
            raise PaxiniProtocolError(
                f"Unexpected Paxini data length: {response_data_len}, expected {expected_data_len}."
            )

        return data[:expected_data_len] if expected_data_len else b""

    def _decode_sample(self, payload: bytes) -> PaxiniSample:
        taxel_bytes = self.num_taxels * self._taxel_np_dtype.itemsize
        if len(payload) < taxel_bytes:
            raise PaxiniProtocolError(
                f"Paxini taxel payload too short: expected at least {taxel_bytes}, got {len(payload)}."
            )

        raw_taxels = np.frombuffer(payload[:taxel_bytes], dtype=self._taxel_np_dtype, count=self.num_taxels)
        taxels = raw_taxels.astype(np.float32)
        if self.taxel_scale is not None:
            taxels *= float(self.taxel_scale)

        fx = fy = fz = tx = ty = tz = 0.0
        rest = payload[taxel_bytes:]
        if len(rest) >= 24:
            fx, fy, fz, tx, ty, tz = np.frombuffer(rest[:24], dtype=np.dtype("<f4"), count=6).astype(float)
        elif len(rest) >= 12:
            fx, fy, fz, tx, ty, tz = np.frombuffer(rest[:12], dtype=np.dtype("<i2"), count=6).astype(float)
        else:
            fz = float(taxels.max(initial=0.0))

        return PaxiniSample(
            taxels=taxels,
            fx=float(fx),
            fy=float(fy),
            fz=float(fz),
            tx=float(tx),
            ty=float(ty),
            tz=float(tz),
            timestamp_s=time.time(),
        )
