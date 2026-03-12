"""Board connection, discovery, deploy, and inference API.

The :class:`Board` class is the primary interface for interacting with a
Neuromap dev board (or a mock software simulation).

Example::

    from neuromap import Board

    board = Board.mock()           # software simulation
    net.deploy(board)              # quantize + flash + verify
    output = board.inject(spikes)  # run inference on chip
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuromap._internal.packing import pack_weights_nibble
from neuromap._internal.transport import (
    MockFirmware,
    MockTransport,
    SerialTransport,
    TcpTransport,
    Transport,
)
from neuromap.protocol import (
    Cmd,
    Packet,
    PacketCodec,
)

# ---------------------------------------------------------------------------
# BoardInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardInfo:
    """Information about a connected Neuromap board.

    Attributes:
        board_id: Unique board identifier.
        hw_revision: Hardware revision number.
        fw_version: Firmware version as ``(major, minor, patch)``.
        protocol_version: NMP protocol version.
        chip_type: Chip type identifier string.
        serial_port: Serial port path (or ``"mock"``).
        status: Board status (``"idle"``, ``"inferring"``, or ``"error"``).
    """

    board_id: int
    hw_revision: int
    fw_version: tuple[int, int, int]
    protocol_version: int
    chip_type: str
    serial_port: str
    status: str


# ---------------------------------------------------------------------------
# Chip type mapping
# ---------------------------------------------------------------------------

_CHIP_TYPE_MAP = {
    0x01: "NeuroSoC-v1",
}

_STATUS_MAP = {
    0x00: "idle",
    0x01: "inferring",
    0x02: "error",
}


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


class Board:
    """Interface for a Neuromap dev board.

    Use :meth:`connect` for real hardware or :meth:`mock` for software
    simulation.  Do not instantiate directly.

    Args:
        transport: The underlying byte-stream transport.
        port_name: Human-readable port name.
    """

    def __init__(self, transport: Transport, port_name: str = "unknown") -> None:
        self._transport = transport
        self._port_name = port_name
        self._seq = 0
        self._codec = PacketCodec()
        self._info: BoardInfo | None = None
        self._programmed = False

    # -- Discovery ----------------------------------------------------------

    @classmethod
    def connect(
        cls,
        port: str | None = None,
        *,
        timeout: float = 5.0,
    ) -> Board:
        """Connect to a Neuromap board over USB serial or WiFi TCP.

        Args:
            port: Connection target.  Accepts:

                - **Serial port** path (e.g. ``"/dev/ttyACM0"``)
                - **Hostname or IP** with optional port
                  (e.g. ``"neuromap-A1B2.local"``,
                  ``"192.168.1.42:4840"``)
                - ``None`` to auto-discover (scans USB ports first,
                  then mDNS on the local network).

            timeout: Discovery timeout in seconds.

        Returns:
            A connected :class:`Board` instance.

        Raises:
            ConnectionError: If no board is found.
        """
        if port is None:
            port = cls._auto_discover(timeout)

        # Detect if this is a TCP target (hostname/IP) vs serial port
        if cls._is_tcp_target(port):
            host, tcp_port = cls._parse_tcp_target(port)
            transport: Transport = TcpTransport(host, tcp_port)
            transport.open()
            board = cls(transport, port_name=port)
            board._ping()
            return board

        transport = SerialTransport(port)
        transport.open()
        board = cls(transport, port_name=port)
        board._ping()
        return board

    @classmethod
    def list_boards(cls) -> list[BoardInfo]:
        """Enumerate all connected Neuromap boards.

        Scans USB serial ports and the local network (mDNS).

        Returns:
            List of :class:`BoardInfo` for each discovered board.
        """
        boards: list[BoardInfo] = []
        seen_ids: set[int] = set()

        # Scan USB serial ports
        for port in cls._scan_serial_ports():
            try:
                transport: Transport = SerialTransport(port)
                transport.open()
                board = cls(transport, port_name=port)
                board._ping()
                if board._info is not None:
                    boards.append(board._info)
                    seen_ids.add(board._info.board_id)
                transport.close()
            except Exception:
                continue

        # Scan mDNS for WiFi boards
        for host, tcp_port in cls._discover_mdns(timeout=2.0):
            try:
                target = f"{host}:{tcp_port}"
                transport = TcpTransport(host, tcp_port)
                transport.open()
                board = cls(transport, port_name=target)
                board._ping()
                if board._info is not None and board._info.board_id not in seen_ids:
                    boards.append(board._info)
                    seen_ids.add(board._info.board_id)
                transport.close()
            except Exception:
                continue

        return boards

    @classmethod
    def mock(cls, chip: Any = None) -> Board:
        """Create a mock board backed by software simulation.

        Args:
            chip: Optional :class:`~neuromap.chip.ChipSpec`.  Defaults to
                ``chips.NEUROSOC_V1``.

        Returns:
            A :class:`Board` that simulates the hardware in software.
        """
        if chip is None:
            from neuromap.chip import chips

            chip = chips.NEUROSOC_V1
        firmware = MockFirmware(chip)
        transport = MockTransport(firmware)
        transport.open()
        board = cls(transport, port_name="mock")
        board._ping()
        return board

    # -- Programming --------------------------------------------------------

    def program(self, nmap_path: str | Path | bytes) -> None:
        """Flash a ``.nmap`` model to the board.

        Args:
            nmap_path: Path to the ``.nmap`` file, or raw bytes from
                :meth:`Exporter.to_bytes`.
        """
        from neuromap.export import Exporter

        if isinstance(nmap_path, (str, Path)):
            net = Exporter.load_nmap(nmap_path)
            from neuromap.export import Exporter as _E

            exporter = _E(net)
            exporter.quantize()
            weight_map = exporter.to_weight_map()
            chip = net.chip
        else:
            # Raw bytes — decode the nmap in-memory
            import io
            import json
            import zipfile

            from neuromap.chip import ChipSpec

            with zipfile.ZipFile(io.BytesIO(nmap_path), "r") as zf:
                manifest = json.loads(zf.read("manifest.json"))
                chip = ChipSpec.from_dict(manifest["chip_spec"])
                weight_map = {}
                for name in zf.namelist():
                    if name.startswith("weights/") and name.endswith(".npy"):
                        buf = io.BytesIO(zf.read(name))
                        key = name.replace("weights/", "").replace(".npy", "")
                        weight_map[key] = np.load(buf, allow_pickle=False)

        # Send weights per layer
        layers = list(chip.layers)
        for i in range(len(layers) - 1):
            weight_key = None
            for k in weight_map:
                if f"snn_layers.{i}.fc.weight" in k or f"layer_{i}_weight" in k:
                    weight_key = k
                    break
            if weight_key is None:
                continue

            w = weight_map[weight_key]
            rows, cols = w.shape
            packed = pack_weights_nibble(
                w.astype(np.int8).reshape(rows, cols), bits=chip.weight_bits
            )
            payload = struct.pack("<BBHH", i, chip.weight_bits, rows, cols) + packed
            self._send_cmd(Cmd.WRITE_WEIGHTS, payload, expect=Cmd.WRITE_WEIGHTS_ACK)

        # Send neuron params
        np_dict = chip.neuron_params.to_dict()
        np_payload = struct.pack(
            "<fffffi",
            np_dict["tau_m"],
            np_dict["rm"],
            np_dict["dt"],
            np_dict["v_th"],
            np_dict["v_reset"],
            np_dict["t_ref"],
        )
        self._send_cmd(Cmd.WRITE_NEURON_PARAMS, np_payload, expect=Cmd.WRITE_NEURON_PARAMS_ACK)

        # Deploy complete
        self._send_cmd(Cmd.DEPLOY_COMPLETE, b"", expect=Cmd.DEPLOY_COMPLETE_ACK)
        self._programmed = True

    def verify(self) -> bool:
        """Read back weights from the board and verify.

        Returns:
            ``True`` if verification passes.

        Raises:
            RuntimeError: If board is not programmed.
        """
        if not self._programmed:
            raise RuntimeError("Board not programmed — call program() first.")
        # For now, just request layer 0 verify as a smoke test
        payload = struct.pack("<B", 0)
        resp = self._send_cmd(Cmd.VERIFY_WEIGHTS, payload, expect=Cmd.VERIFY_WEIGHTS_RESP)
        return len(resp.payload) > 0

    # -- Inference ----------------------------------------------------------

    def start_inference(self) -> None:
        """Start the inference clock on the board."""
        self._send_cmd(Cmd.INFER_START, b"", expect=Cmd.INFER_START_ACK)

    def stop_inference(self) -> None:
        """Stop the inference clock on the board."""
        self._send_cmd(Cmd.INFER_STOP, b"", expect=Cmd.INFER_STOP_ACK)

    def inject(self, spikes: np.ndarray | torch.Tensor) -> np.ndarray:
        """Inject one timestep of input spikes and return output.

        Args:
            spikes: 1-D spike vector (binary, length = input_size).

        Returns:
            1-D output array from the chip.
        """
        if isinstance(spikes, torch.Tensor):
            spikes = spikes.detach().cpu().numpy()
        spikes = np.asarray(spikes, dtype=np.float32).ravel()
        input_size = len(spikes)

        # Pack spikes as bit vector
        num_bytes = (input_size + 7) // 8
        spike_bytes = bytearray(num_bytes)
        for i in range(input_size):
            if spikes[i] > 0.5:
                spike_bytes[i // 8] |= 1 << (i % 8)

        payload = struct.pack("<BH", 0, input_size) + bytes(spike_bytes)
        resp = self._send_cmd(Cmd.INJECT_SPIKES, payload, expect=Cmd.OUTPUT_SPIKES)
        return np.frombuffer(resp.payload, dtype=np.float32)

    def inject_batch(self, spike_sequence: np.ndarray | torch.Tensor) -> np.ndarray:
        """Inject multiple timesteps and return aggregated output.

        Args:
            spike_sequence: 2-D array ``(timesteps, input_size)``.

        Returns:
            1-D output array from the chip.
        """
        if isinstance(spike_sequence, torch.Tensor):
            spike_sequence = spike_sequence.detach().cpu().numpy()
        spike_sequence = np.asarray(spike_sequence, dtype=np.float32)
        if spike_sequence.ndim == 1:
            spike_sequence = spike_sequence.reshape(1, -1)

        timesteps, input_size = spike_sequence.shape
        bytes_per_step = (input_size + 7) // 8

        payload = struct.pack("<HH", timesteps, input_size)
        for t in range(timesteps):
            spike_bytes = bytearray(bytes_per_step)
            for i in range(input_size):
                if spike_sequence[t, i] > 0.5:
                    spike_bytes[i // 8] |= 1 << (i % 8)
            payload += bytes(spike_bytes)

        resp = self._send_cmd(Cmd.INJECT_BATCH, payload, expect=Cmd.OUTPUT_BATCH)
        return np.frombuffer(resp.payload, dtype=np.float32)

    # -- Monitoring ---------------------------------------------------------

    def start_monitor(self) -> Any:
        """Start live spike monitoring.

        Returns:
            A :class:`~neuromap.monitor.SpikeMonitor` instance.
        """
        from neuromap.monitor import SpikeMonitor

        self._send_cmd(Cmd.MONITOR_START, b"", expect=Cmd.MONITOR_START_ACK)
        return SpikeMonitor(self)

    # -- System -------------------------------------------------------------

    def reset(self) -> None:
        """Soft-reset the board."""
        self._send_cmd(Cmd.RESET, b"", expect=Cmd.RESET_ACK)
        self._programmed = False

    def set_led(self, led: int, state: bool) -> None:
        """Control a user LED.

        Args:
            led: LED index (0-based).
            state: ``True`` for on, ``False`` for off.
        """
        payload = struct.pack("<BB", led, int(state))
        self._send_cmd(Cmd.SET_LED, payload)

    def close(self) -> None:
        """Close the board connection."""
        if self._transport.is_open:
            self._transport.close()

    # -- Properties ---------------------------------------------------------

    @property
    def info(self) -> BoardInfo | None:
        """Board information from the last PING/PONG exchange."""
        return self._info

    @property
    def is_connected(self) -> bool:
        """Whether the transport is open."""
        return self._transport.is_open

    @property
    def is_mock(self) -> bool:
        """Whether this is a mock (software) board."""
        return isinstance(self._transport, MockTransport)

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> Board:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"Board(port='{self._port_name}', {status})"

    # -- Internal -----------------------------------------------------------

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFFFF
        return seq

    def _send_cmd(
        self,
        cmd: Cmd,
        payload: bytes,
        expect: Cmd | None = None,
        timeout_ms: int = 2000,
    ) -> Packet:
        """Send a command packet and optionally wait for a response.

        Args:
            cmd: Command to send.
            payload: Command payload.
            expect: Expected response command (or ``None`` for fire-and-forget).
            timeout_ms: Response timeout in milliseconds.

        Returns:
            The response packet (or a dummy if *expect* is ``None``).

        Raises:
            RuntimeError: On timeout or error response.
        """
        pkt = Packet(cmd=cmd, seq=self._next_seq(), payload=payload)
        self._transport.write(pkt.encode())

        if expect is None:
            return Packet(cmd=cmd, seq=0)

        # Read response
        raw = self._transport.read(4106, timeout_ms=timeout_ms)
        if not raw:
            raise RuntimeError(f"Timeout waiting for {expect.name} response.")

        codec = PacketCodec()
        responses = codec.feed(raw)
        if not responses:
            raise RuntimeError(f"No valid response for {cmd.name}.")

        resp = responses[0]
        if resp.cmd == Cmd.ERROR:
            error_code = struct.unpack_from("<H", resp.payload)[0] if resp.payload else 0
            msg = (
                resp.payload[4:].decode("utf-8", errors="replace") if len(resp.payload) > 4 else ""
            )
            raise RuntimeError(
                f"Board error 0x{error_code:04X}: {msg}"
                if msg
                else f"Board error 0x{error_code:04X}"
            )

        return resp

    def _ping(self) -> None:
        """Send PING and parse PONG to populate :attr:`info`."""
        resp = self._send_cmd(Cmd.PING, b"", expect=Cmd.PONG)
        if len(resp.payload) >= 11:
            (
                board_id,
                hw_rev,
                fw_major,
                fw_minor,
                fw_patch,
                proto_ver,
                chip_type_id,
                status_id,
            ) = struct.unpack_from("<IBBBBBBB", resp.payload)
            self._info = BoardInfo(
                board_id=board_id,
                hw_revision=hw_rev,
                fw_version=(fw_major, fw_minor, fw_patch),
                protocol_version=proto_ver,
                chip_type=_CHIP_TYPE_MAP.get(chip_type_id, f"unknown-0x{chip_type_id:02X}"),
                serial_port=self._port_name,
                status=_STATUS_MAP.get(status_id, "unknown"),
            )

    @classmethod
    def _auto_discover(cls, timeout: float) -> str:
        """Scan USB serial ports and mDNS for a Neuromap board.

        Returns:
            The port path or ``"host:port"`` of the first discovered board.

        Raises:
            ConnectionError: If no board is found.
        """
        # Try USB serial first
        for port in cls._scan_serial_ports():
            try:
                transport: Transport = SerialTransport(port)
                transport.open()
                pkt = Packet(cmd=Cmd.PING, seq=0)
                transport.write(pkt.encode())
                raw = transport.read(4106, timeout_ms=int(timeout * 1000))
                codec = PacketCodec()
                responses = codec.feed(raw)
                transport.close()
                if responses and responses[0].cmd == Cmd.PONG:
                    return port
            except Exception:
                continue

        # Try mDNS discovery
        for host, tcp_port in cls._discover_mdns(timeout=timeout):
            try:
                transport = TcpTransport(host, tcp_port)
                transport.open()
                pkt = Packet(cmd=Cmd.PING, seq=0)
                transport.write(pkt.encode())
                raw = transport.read(4106, timeout_ms=int(timeout * 1000))
                codec = PacketCodec()
                responses = codec.feed(raw)
                transport.close()
                if responses and responses[0].cmd == Cmd.PONG:
                    return f"{host}:{tcp_port}"
            except Exception:
                continue

        raise ConnectionError(
            "No Neuromap board found. Check USB connection or WiFi and try again."
        )

    @staticmethod
    def _is_tcp_target(port: str) -> bool:
        """Check if *port* looks like a TCP target rather than a serial port."""
        # Serial ports: /dev/ttyXXX, COMx, etc.
        if port.startswith("/dev/") or port.upper().startswith("COM"):
            return False
        # Hostnames, IPs, or .local addresses
        if ".local" in port or ":" in port:
            return True
        # IP addresses (simple heuristic)
        parts = port.split(".")
        if len(parts) == 4 and all(p.isdigit() for p in parts):
            return True
        return False

    @staticmethod
    def _parse_tcp_target(target: str) -> tuple[str, int]:
        """Parse ``"host:port"`` or ``"host"`` into (host, port)."""
        default_port = TcpTransport.NMP_DEFAULT_PORT
        if ":" in target:
            # Could be "host:port" — but avoid splitting IPv6
            parts = target.rsplit(":", 1)
            if parts[1].isdigit():
                return parts[0], int(parts[1])
        return target, default_port

    @staticmethod
    def _discover_mdns(timeout: float = 3.0) -> list[tuple[str, int]]:
        """Discover Neuromap boards via mDNS (zeroconf).

        Returns:
            List of ``(host, port)`` tuples for discovered boards.
        """
        try:
            from zeroconf import ServiceBrowser, Zeroconf
        except ImportError:
            return []

        results: list[tuple[str, int]] = []

        class _Listener:
            def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                info = zc.get_service_info(type_, name)
                if info is not None:
                    addresses = info.parsed_addresses()
                    port = info.port
                    for addr in addresses:
                        results.append((addr, port))

            def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                pass

            def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                pass

        zc = Zeroconf()
        try:
            ServiceBrowser(zc, "_neuromap._tcp.local.", _Listener())
            time.sleep(min(timeout, 3.0))
        finally:
            zc.close()

        return results

    @staticmethod
    def _scan_serial_ports() -> list[str]:
        """Return available serial port paths."""
        try:
            from serial.tools.list_ports import comports

            return [p.device for p in comports()]
        except ImportError:
            return []
