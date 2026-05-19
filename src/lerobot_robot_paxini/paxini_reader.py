from __future__ import annotations

import csv
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from lerobot.utils.import_utils import require_package

from .types import PaxiniSample

REQ_HEAD = b"\x55\xAA"
RESP_HEAD_AUTO_RETURN = b"\xAA\x56"
RESERVED = b"\x00"

FUNC_WRITE = 0x10
AUTO_RETURN_REG = 0x0017

MAX_FRAME_BYTES = 8192

TaxelValueMode = Literal["x", "y", "z", "magnitude"]


class PaxiniProtocolError(RuntimeError):
    pass


def lrc_cal(data: bytes | bytearray | memoryview) -> int:
    """Calculate the two's-complement LRC used by the GEN3 serial protocol."""
    return (-sum(data)) & 0xFF


def build_gen3_request_frame(
    *,
    func_code: int,
    reg_addr: int,
    data_len: int,
    payload: bytes = b"",
) -> bytes:
    frame = bytearray()
    frame += REQ_HEAD
    frame += RESERVED
    frame.append(func_code & 0xFF)
    frame += int(reg_addr).to_bytes(2, "little", signed=False)
    frame += int(data_len).to_bytes(2, "little", signed=False)
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
        return PaxiniSample(taxels=taxels, fz=float(taxels.max(initial=0.0)), timestamp_s=time.time())


class CsvPaxiniReader(PaxiniReader):
    """Replay PXSR DataLogging CSV rows as tactile samples.

    PXSR logs distributed force columns such as ``3-0-NxN-Z[0]``.  For ring+pinky
    experiments this normally gives two 77-taxel arrays, concatenated in CSV
    column order into a 154-value tactile vector.
    """

    _NXN_COL_RE = re.compile(r"^(?P<sensor>.+)-NxN-(?P<axis>[XYZ])\[(?P<idx>\d+)\]$")

    def __init__(
        self,
        csv_path: str | Path,
        *,
        num_taxels: int,
        taxel_value_mode: TaxelValueMode = "z",
        taxel_scale: float = 1.0,
        loop: bool = True,
    ):
        if num_taxels <= 0:
            raise ValueError(f"num_taxels must be positive, got {num_taxels}.")
        if taxel_value_mode not in ("x", "y", "z", "magnitude"):
            raise ValueError(
                "taxel_value_mode must be one of 'x', 'y', 'z', or 'magnitude', "
                f"got {taxel_value_mode!r}."
            )

        self.csv_path = Path(csv_path)
        self.num_taxels = num_taxels
        self.taxel_value_mode = taxel_value_mode
        self.taxel_scale = float(taxel_scale)
        self.loop = loop
        self.connected = False
        self._rows: list[dict[str, str]] = []
        self._columns: list[str] = []
        self._selected_columns: list[str] = []
        self._triplet_columns: list[tuple[str, str, str]] = []
        self._cursor = 0

    def connect(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"PXSR CSV not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            self._columns = list(reader.fieldnames or [])
            self._rows = [row for row in reader]

        if not self._rows:
            raise ValueError(f"PXSR CSV has no data rows: {self.csv_path}")

        self._select_columns()
        self._cursor = 0
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_latest(self) -> PaxiniSample:
        if not self.connected:
            raise RuntimeError("CsvPaxiniReader is not connected.")
        if self._cursor >= len(self._rows):
            if not self.loop:
                self._cursor = len(self._rows) - 1
            else:
                self._cursor = 0

        row = self._rows[self._cursor]
        self._cursor += 1

        if self.taxel_value_mode == "magnitude":
            values = []
            for x_col, y_col, z_col in self._triplet_columns:
                x = self._parse_float(row.get(x_col))
                y = self._parse_float(row.get(y_col))
                z = self._parse_float(row.get(z_col))
                values.append(float(np.sqrt(x * x + y * y + z * z)))
            values_np = np.asarray(values, dtype=np.float32)
        else:
            values_np = np.asarray(
                [self._parse_float(row.get(col)) for col in self._selected_columns],
                dtype=np.float32,
            )

        taxels = np.zeros(self.num_taxels, dtype=np.float32)
        count = min(self.num_taxels, values_np.shape[0])
        taxels[:count] = values_np[:count] * self.taxel_scale
        return PaxiniSample(taxels=taxels, fz=float(taxels.max(initial=0.0)), timestamp_s=time.time())

    def _select_columns(self) -> None:
        parsed: list[tuple[int, str, str, int, str]] = []
        column_order = {col: order for order, col in enumerate(self._columns)}
        for col in self._columns:
            match = self._NXN_COL_RE.match(col)
            if not match:
                continue
            parsed.append(
                (
                    column_order[col],
                    match.group("sensor"),
                    match.group("axis").lower(),
                    int(match.group("idx")),
                    col,
                )
            )

        if not parsed:
            numeric_cols = [col for col in self._columns if col.lower() != "timestamp"]
            if len(numeric_cols) < self.num_taxels:
                raise ValueError(
                    f"Could not find PXSR NxN columns in {self.csv_path}; "
                    f"only {len(numeric_cols)} generic numeric columns are available."
                )
            self._selected_columns = numeric_cols[: self.num_taxels]
            return

        if self.taxel_value_mode == "magnitude":
            by_point: dict[tuple[str, int], dict[str, str]] = {}
            first_order: dict[tuple[str, int], int] = {}
            for order, sensor, axis, idx, col in parsed:
                key = (sensor, idx)
                by_point.setdefault(key, {})[axis] = col
                first_order[key] = min(first_order.get(key, order), order)

            triplets = []
            for key, axes in by_point.items():
                if {"x", "y", "z"} <= axes.keys():
                    triplets.append((first_order[key], axes["x"], axes["y"], axes["z"]))
            triplets.sort(key=lambda item: item[0])
            self._triplet_columns = [(x, y, z) for _, x, y, z in triplets[: self.num_taxels]]
            if not self._triplet_columns:
                raise ValueError(f"No complete NxN X/Y/Z triplets found in {self.csv_path}.")
            return

        axis = self.taxel_value_mode
        selected = [(order, col) for order, _, parsed_axis, _, col in parsed if parsed_axis == axis]
        selected.sort(key=lambda item: item[0])
        self._selected_columns = [col for _, col in selected[: self.num_taxels]]
        if not self._selected_columns:
            raise ValueError(f"No NxN {axis.upper()} columns found in {self.csv_path}.")

    @staticmethod
    def _parse_float(value: str | None) -> float:
        if value is None or value == "":
            return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0


class SerialPaxiniReader(PaxiniReader):
    """Read PXSR GEN3 auto-return tactile frames from a serial port.

    The verified adapter protocol is:
    - host request header: 55 AA
    - enable auto-return: write 0x01 to register 0x0017
    - stream response header: AA 56
    - valid data is emitted as 3-byte force groups, usually X/Y/Z.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        *,
        num_taxels: int = 158,
        timeout_s: float = 0.5,
        taxel_value_mode: TaxelValueMode = "z",
        taxel_scale: float = 0.1,
        skip_bytes: int = 0,
        async_read: bool = True,
    ):
        if num_taxels <= 0:
            raise ValueError(f"num_taxels must be positive, got {num_taxels}.")
        if skip_bytes < 0:
            raise ValueError(f"skip_bytes must be non-negative, got {skip_bytes}.")
        if taxel_value_mode not in ("x", "y", "z", "magnitude"):
            raise ValueError(
                "taxel_value_mode must be one of 'x', 'y', 'z', or 'magnitude', "
                f"got {taxel_value_mode!r}."
            )

        self.port = port
        self.baudrate = baudrate
        self.num_taxels = num_taxels
        self.timeout_s = timeout_s
        self.taxel_value_mode = taxel_value_mode
        self.taxel_scale = float(taxel_scale)
        self.skip_bytes = skip_bytes
        self.async_read = async_read
        self.connected = False
        self._serial = None
        self._last_sample: PaxiniSample | None = None
        self._last_error: Exception | None = None
        self._sample_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._read_thread: threading.Thread | None = None

    def connect(self) -> None:
        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        import serial

        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=min(self.timeout_s, 0.05),
            write_timeout=self.timeout_s,
            inter_byte_timeout=0.001,
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        with self._sample_lock:
            self._last_sample = None
            self._last_error = None
        self._set_auto_return(False)
        self._serial.reset_input_buffer()
        self._set_auto_return(True)
        self.connected = True
        if self.async_read:
            self._start_background_reader()

    def disconnect(self) -> None:
        self._stop_background_reader()
        if self._serial is not None and self._serial.is_open:
            try:
                self._set_auto_return(False)
                time.sleep(0.05)
            finally:
                self._serial.close()
        self._serial = None
        self.connected = False

    def read_latest(self) -> PaxiniSample:
        if not self.connected:
            raise RuntimeError("SerialPaxiniReader is not connected.")

        if self.async_read:
            return self._read_latest_from_background()

        return self._read_and_decode_once()

    def _read_and_decode_once(self) -> PaxiniSample:
        try:
            frame = self._read_auto_return_frame()
        except TimeoutError:
            with self._sample_lock:
                if self._last_sample is not None:
                    return self._last_sample
            raise

        data = self._parse_auto_return_frame(frame)
        sample = self._decode_taxels(data)
        with self._sample_lock:
            self._last_sample = sample
            self._last_error = None
        return sample

    def _read_latest_from_background(self) -> PaxiniSample:
        deadline = time.monotonic() + self.timeout_s
        while True:
            with self._sample_lock:
                if self._last_sample is not None:
                    return self._last_sample
                last_error = self._last_error

            if time.monotonic() >= deadline:
                if last_error is not None:
                    raise RuntimeError(f"PXSR background reader has no sample yet: {last_error}") from last_error
                raise TimeoutError(f"Timed out waiting for first PXSR sample on {self.port}.")
            time.sleep(0.002)

    def _start_background_reader(self) -> None:
        self._stop_event.clear()
        self._read_thread = threading.Thread(
            target=self._background_read_loop,
            name=f"paxini-reader-{self.port}",
            daemon=True,
        )
        self._read_thread.start()

    def _stop_background_reader(self) -> None:
        self._stop_event.set()
        if self._read_thread is not None:
            self._read_thread.join(timeout=self.timeout_s + 0.2)
            self._read_thread = None

    def _background_read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._read_auto_return_frame()
                data = self._parse_auto_return_frame(frame)
                sample = self._decode_taxels(data)
                with self._sample_lock:
                    self._last_sample = sample
                    self._last_error = None
            except TimeoutError:
                continue
            except Exception as exc:
                with self._sample_lock:
                    self._last_error = exc
                if not self._stop_event.wait(0.01):
                    continue

    def _set_auto_return(self, enabled: bool) -> None:
        if self._serial is None or not self._serial.is_open:
            return

        payload = b"\x01" if enabled else b"\x00"
        frame = build_gen3_request_frame(
            func_code=FUNC_WRITE,
            reg_addr=AUTO_RETURN_REG,
            data_len=len(payload),
            payload=payload,
        )
        self._serial.write(frame)
        self._serial.flush()
        time.sleep(0.05)

    def _read_auto_return_frame(self) -> bytes:
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("SerialPaxiniReader is not connected.")

        deadline = time.monotonic() + self.timeout_s
        buffer = bytearray()

        while time.monotonic() < deadline:
            read_size = max(1, min(512, getattr(self._serial, "in_waiting", 0) or 1))
            chunk = self._serial.read(read_size)
            if not chunk:
                continue

            buffer.extend(chunk)
            while True:
                head_idx = buffer.find(RESP_HEAD_AUTO_RETURN)
                if head_idx < 0:
                    del buffer[:-1]
                    break
                if head_idx > 0:
                    del buffer[:head_idx]

                if len(buffer) < 6:
                    break

                valid_frame_len = int.from_bytes(buffer[3:5], "little", signed=False)
                frame_len = valid_frame_len + 6
                if frame_len < 7 or frame_len > MAX_FRAME_BYTES:
                    del buffer[0]
                    continue
                if len(buffer) < frame_len:
                    break

                frame = bytes(buffer[:frame_len])
                if lrc_cal(frame[:-1]) != frame[-1]:
                    del buffer[0]
                    continue

                return frame

        raise TimeoutError(f"Timed out waiting for PXSR auto-return data on {self.port}.")

    @staticmethod
    def _parse_auto_return_frame(frame: bytes) -> bytes:
        if len(frame) < 7:
            raise PaxiniProtocolError(f"Auto-return frame too short: {len(frame)} bytes.")
        if frame[:2] != RESP_HEAD_AUTO_RETURN:
            raise PaxiniProtocolError(f"Unexpected auto-return header: {frame[:2]!r}.")
        if lrc_cal(frame[:-1]) != frame[-1]:
            raise PaxiniProtocolError("Auto-return LRC check failed.")

        valid_frame_len = int.from_bytes(frame[3:5], "little", signed=False)
        valid_data_len = valid_frame_len - 1
        error_code = frame[5]
        if error_code != 0:
            raise PaxiniProtocolError(f"PXSR auto-return error code: 0x{error_code:02X}.")

        data_end = 6 + valid_data_len
        if data_end > len(frame) - 1:
            raise PaxiniProtocolError(
                f"Incomplete auto-return data: expected {valid_data_len} bytes, "
                f"got {max(len(frame) - 7, 0)} bytes."
            )
        return frame[6:data_end]

    def _decode_taxels(self, payload: bytes) -> PaxiniSample:
        if self.skip_bytes:
            payload = payload[self.skip_bytes :]

        group_count = len(payload) // 3
        taxels = np.zeros(self.num_taxels, dtype=np.float32)
        if group_count == 0:
            return PaxiniSample(taxels=taxels, timestamp_s=time.time())

        raw = np.frombuffer(payload[: group_count * 3], dtype=np.uint8).reshape(group_count, 3)
        x = raw[:, 0].astype(np.int16)
        y = raw[:, 1].astype(np.int16)
        x = np.where(x <= 127, x, x - 256).astype(np.float32)
        y = np.where(y <= 127, y, y - 256).astype(np.float32)
        z = raw[:, 2].astype(np.float32)

        if self.taxel_value_mode == "x":
            values = np.abs(x)
        elif self.taxel_value_mode == "y":
            values = np.abs(y)
        elif self.taxel_value_mode == "magnitude":
            values = np.sqrt(x * x + y * y + z * z)
        else:
            values = z

        values = values.astype(np.float32, copy=False) * self.taxel_scale
        count = min(self.num_taxels, values.shape[0])
        taxels[:count] = values[:count]

        return PaxiniSample(taxels=taxels, fz=float(taxels.max(initial=0.0)), timestamp_s=time.time())
