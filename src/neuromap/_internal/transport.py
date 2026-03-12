"""Transport abstraction layer for Neuromap board communication.

Provides :class:`Transport` ABC, :class:`SerialTransport` for real hardware,
:class:`TcpTransport` for WiFi connections to ESP32 boards,
:class:`MockTransport` + :class:`MockFirmware` for testing without a board.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np
import torch

from neuromap._internal.packing import unpack_weights_nibble
from neuromap.protocol import (
    Cmd,
    ErrorCode,
    Packet,
    PacketCodec,
)

# ---------------------------------------------------------------------------
# Abstract Transport
# ---------------------------------------------------------------------------


class Transport(ABC):
    """Abstract byte-stream transport."""

    @abstractmethod
    def open(self) -> None:
        """Open the transport connection."""

    @abstractmethod
    def close(self) -> None:
        """Close the transport connection."""

    @abstractmethod
    def write(self, data: bytes) -> None:
        """Write bytes to the transport."""

    @abstractmethod
    def read(self, size: int, timeout_ms: int = 1000) -> bytes:
        """Read up to *size* bytes with a timeout.

        Args:
            size: Maximum number of bytes to read.
            timeout_ms: Timeout in milliseconds.

        Returns:
            Bytes read (may be fewer than *size*).
        """

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the transport is currently open."""


# ---------------------------------------------------------------------------
# Serial Transport (real hardware)
# ---------------------------------------------------------------------------


class SerialTransport(Transport):
    """USB CDC serial transport via pyserial.

    Args:
        port: Serial port path (e.g. ``/dev/ttyACM0``).
        baudrate: Baud rate (default 115200, ignored by CDC but set for compat).
    """

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self._port = port
        self._baudrate = baudrate
        self._serial: Any = None  # serial.Serial instance

    def open(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for hardware board communication. "
                "Install it with: pip install pyserial"
            ) from exc
        self._serial = serial.Serial(self._port, self._baudrate, timeout=0.1)

    def close(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            self._serial = None

    def write(self, data: bytes) -> None:
        if self._serial is None:
            raise RuntimeError("Transport not open.")
        self._serial.write(data)

    def read(self, size: int, timeout_ms: int = 1000) -> bytes:
        if self._serial is None:
            raise RuntimeError("Transport not open.")
        self._serial.timeout = timeout_ms / 1000.0
        return bytes(self._serial.read(size))

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open


# ---------------------------------------------------------------------------
# TCP Transport (WiFi connection to ESP32)
# ---------------------------------------------------------------------------


class TcpTransport(Transport):
    """TCP socket transport for WiFi connections to Neuromap boards.

    Connects to the ESP32's NMP TCP server (default port 4840).
    Carries the same binary NMP protocol as USB — no HTTP wrapping.

    Args:
        host: Hostname or IP address (e.g. ``"neuromap-A1B2.local"``
            or ``"192.168.1.42"``).
        port: TCP port number (default 4840).
    """

    NMP_DEFAULT_PORT = 4840

    def __init__(self, host: str, port: int = NMP_DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None

    def open(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5.0)
        try:
            self._sock.connect((self._host, self._port))
        except OSError as exc:
            self._sock.close()
            self._sock = None
            raise ConnectionError(f"Cannot connect to {self._host}:{self._port}: {exc}") from exc

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def write(self, data: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("Transport not open.")
        self._sock.sendall(data)

    def read(self, size: int, timeout_ms: int = 1000) -> bytes:
        if self._sock is None:
            raise RuntimeError("Transport not open.")
        self._sock.settimeout(timeout_ms / 1000.0)
        try:
            return self._sock.recv(size)
        except socket.timeout:
            return b""

    @property
    def is_open(self) -> bool:
        return self._sock is not None


# ---------------------------------------------------------------------------
# Mock Firmware
# ---------------------------------------------------------------------------


class MockFirmware:
    """Simulates MCU firmware using DynamicSNN model.

    Produces numerically identical results to :meth:`Network.infer` so
    developers can build applications without the physical board.

    Args:
        chip: The chip specification to simulate.
    """

    def __init__(self, chip: Any) -> None:
        from neuromap._internal.dynamic_snn import DynamicSNN

        self._chip = chip
        self._model = DynamicSNN(
            layers=list(chip.layers),
            neuron_kwargs=chip.neuron_params.to_dict(),
            use_decoder=True,
        )
        self._model.eval()
        self._programmed = False
        self._inferring = False
        self._monitoring = False
        self._state: dict[str, Any] | None = None
        # Stored weight data for verify
        self._weight_data: dict[int, bytes] = {}

    def process_packet(self, packet: Packet) -> list[Packet]:
        """Process an incoming packet and return response packets.

        Args:
            packet: Incoming packet from the host.

        Returns:
            List of response packets.
        """
        cmd = packet.cmd
        seq = packet.seq

        if cmd == Cmd.PING:
            # PONG: board_id(4) | hw_rev(1) | fw_major(1) | fw_minor(1) |
            #        fw_patch(1) | proto_ver(1) | chip_type(1) | status(1)
            status = 0x00 if not self._inferring else 0x01
            payload = struct.pack(
                "<IBBBBBBB",
                0x00000001,  # board_id
                0x01,  # hw_rev
                1,
                0,
                0,  # fw version 1.0.0
                0x01,  # protocol_version
                0x01,  # chip_type (NEUROSOC_V1)
                status,
            )
            return [Packet(cmd=Cmd.PONG, seq=seq, payload=payload)]

        if cmd == Cmd.GET_INFO:
            payload = struct.pack(
                "<IBBBBBBB",
                0x00000001,
                0x01,
                1,
                0,
                0,
                0x01,
                0x01,
                0x00 if not self._inferring else 0x01,
            )
            return [Packet(cmd=Cmd.INFO_RESP, seq=seq, payload=payload)]

        if cmd == Cmd.WRITE_WEIGHTS:
            layer_idx, bits, rows, cols = struct.unpack_from("<BBHH", packet.payload)
            packed_data = packet.payload[6:]
            self._weight_data[layer_idx] = packed_data
            # Apply weights to model
            weights = unpack_weights_nibble(packed_data, rows, cols, bits)
            layer = self._model.snn_layers[layer_idx]
            with torch.no_grad():
                layer.fc.weight.copy_(torch.from_numpy(weights.astype(np.float32)))
            return [Packet(cmd=Cmd.WRITE_WEIGHTS_ACK, seq=seq)]

        if cmd == Cmd.WRITE_NEURON_PARAMS:
            return [Packet(cmd=Cmd.WRITE_NEURON_PARAMS_ACK, seq=seq)]

        if cmd == Cmd.WRITE_CONFIG:
            return [Packet(cmd=Cmd.WRITE_CONFIG_ACK, seq=seq)]

        if cmd == Cmd.VERIFY_WEIGHTS:
            layer_idx = packet.payload[0] if packet.payload else 0
            data = self._weight_data.get(layer_idx, b"")
            return [Packet(cmd=Cmd.VERIFY_WEIGHTS_RESP, seq=seq, payload=data)]

        if cmd == Cmd.DEPLOY_COMPLETE:
            self._programmed = True
            return [Packet(cmd=Cmd.DEPLOY_COMPLETE_ACK, seq=seq)]

        if cmd == Cmd.INFER_START:
            if not self._programmed:
                return [self._error(seq, ErrorCode.ERR_NOT_PROGRAMMED)]
            self._inferring = True
            self._state = None
            return [Packet(cmd=Cmd.INFER_START_ACK, seq=seq)]

        if cmd == Cmd.INFER_STOP:
            self._inferring = False
            self._state = None
            return [Packet(cmd=Cmd.INFER_STOP_ACK, seq=seq)]

        if cmd == Cmd.INJECT_SPIKES:
            if not self._inferring:
                return [self._error(seq, ErrorCode.ERR_NOT_PROGRAMMED)]
            _timestep = packet.payload[0]
            input_size = struct.unpack_from("<H", packet.payload, 1)[0]
            spike_bytes = packet.payload[3:]
            # Unpack spike bit vector
            spikes = np.zeros(input_size, dtype=np.float32)
            for i in range(input_size):
                byte_idx = i // 8
                bit_idx = i % 8
                if byte_idx < len(spike_bytes) and (spike_bytes[byte_idx] >> bit_idx) & 1:
                    spikes[i] = 1.0

            x = torch.from_numpy(spikes).unsqueeze(0).unsqueeze(0)  # (1,1,input)
            with torch.no_grad():
                result = self._model(x, state=self._state, return_state=True)
                output, self._state = result

            out_np = output.squeeze(0).numpy()
            # Pack output as float32 array
            out_payload = out_np.astype(np.float32).tobytes()
            return [Packet(cmd=Cmd.OUTPUT_SPIKES, seq=seq, payload=out_payload)]

        if cmd == Cmd.INJECT_BATCH:
            if not self._inferring:
                return [self._error(seq, ErrorCode.ERR_NOT_PROGRAMMED)]
            # Payload: timesteps(2 LE) | input_size(2 LE) | data
            timesteps, input_size = struct.unpack_from("<HH", packet.payload)
            data_offset = 4
            # Each timestep: input_size bits packed into ceil(input_size/8) bytes
            bytes_per_step = (input_size + 7) // 8
            all_outputs = []
            for t in range(timesteps):
                step_start = data_offset + t * bytes_per_step
                spike_bytes = packet.payload[step_start : step_start + bytes_per_step]
                spikes = np.zeros(input_size, dtype=np.float32)
                for i in range(input_size):
                    byte_idx = i // 8
                    bit_idx = i % 8
                    if byte_idx < len(spike_bytes) and (spike_bytes[byte_idx] >> bit_idx) & 1:
                        spikes[i] = 1.0
                all_outputs.append(spikes)

            x = torch.from_numpy(np.array(all_outputs)).unsqueeze(0)  # (1, T, input)
            with torch.no_grad():
                output = self._model(x)
            out_np = output.squeeze(0).numpy().astype(np.float32)
            return [Packet(cmd=Cmd.OUTPUT_BATCH, seq=seq, payload=out_np.tobytes())]

        if cmd == Cmd.MONITOR_START:
            self._monitoring = True
            return [Packet(cmd=Cmd.MONITOR_START_ACK, seq=seq)]

        if cmd == Cmd.MONITOR_STOP:
            self._monitoring = False
            return [Packet(cmd=Cmd.MONITOR_STOP_ACK, seq=seq)]

        if cmd == Cmd.RESET:
            self._programmed = False
            self._inferring = False
            self._monitoring = False
            self._state = None
            self._weight_data.clear()
            return [Packet(cmd=Cmd.RESET_ACK, seq=seq)]

        if cmd == Cmd.SET_LED:
            return []  # LEDs are no-ops in mock

        return [self._error(seq, ErrorCode.ERR_UNKNOWN_CMD)]

    @staticmethod
    def _error(seq: int, code: ErrorCode, message: str = "") -> Packet:
        msg_bytes = message.encode("utf-8")[:256]
        payload = struct.pack("<HH", int(code), seq) + msg_bytes
        return Packet(cmd=Cmd.ERROR, seq=seq, payload=payload)


# ---------------------------------------------------------------------------
# Mock Transport
# ---------------------------------------------------------------------------


class MockTransport(Transport):
    """In-memory transport backed by :class:`MockFirmware`.

    Args:
        firmware: The mock firmware that processes packets.
    """

    def __init__(self, firmware: MockFirmware) -> None:
        self._firmware = firmware
        self._codec = PacketCodec()
        self._rx_buffer = deque[bytes]()
        self._open = False
        self._lock = threading.Lock()

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def write(self, data: bytes) -> None:
        if not self._open:
            raise RuntimeError("Transport not open.")
        # Decode packet(s) from written data
        packets = self._codec.feed(data)
        for pkt in packets:
            responses = self._firmware.process_packet(pkt)
            for resp in responses:
                with self._lock:
                    self._rx_buffer.append(resp.encode())

    def read(self, size: int, timeout_ms: int = 1000) -> bytes:
        if not self._open:
            raise RuntimeError("Transport not open.")
        deadline = time.monotonic() + timeout_ms / 1000.0
        result = bytearray()
        while len(result) < size and time.monotonic() < deadline:
            with self._lock:
                if self._rx_buffer:
                    chunk = self._rx_buffer.popleft()
                    result.extend(chunk)
                    continue
            time.sleep(0.001)
        return bytes(result[:size])

    @property
    def is_open(self) -> bool:
        return self._open
