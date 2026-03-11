"""Neuromap Protocol (NMP) — packet codec, CRC, and command enums.

Binary packet protocol for communication between the host Python SDK and
the MCU companion on a Neuromap dev board over USB CDC/ACM.

Packet format::

    SYNC(1) | VERSION(1) | CMD(1) | FLAGS(1) | SEQ(2 LE) | LEN(2 LE) | PAYLOAD(N) | CRC16(2 LE)

10 bytes overhead.  Max payload: 4096 bytes.  Max packet: 4106 bytes.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNC_BYTE = 0xAA
PROTOCOL_VERSION = 0x01
MAX_PAYLOAD = 4096
HEADER_SIZE = 8  # SYNC + VERSION + CMD + FLAGS + SEQ(2) + LEN(2)
CRC_SIZE = 2
MIN_PACKET_SIZE = HEADER_SIZE + CRC_SIZE  # 10 bytes, zero-length payload

# Flag bits
FLAG_ACK_REQ = 0x01
FLAG_STREAM = 0x02


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class Cmd(enum.IntEnum):
    """NMP command IDs."""

    # -- Discovery --
    PING = 0x01
    PONG = 0x02
    GET_INFO = 0x03
    INFO_RESP = 0x04

    # -- Programming --
    WRITE_WEIGHTS = 0x10
    WRITE_WEIGHTS_ACK = 0x11
    WRITE_NEURON_PARAMS = 0x12
    WRITE_NEURON_PARAMS_ACK = 0x13
    WRITE_CONFIG = 0x14
    WRITE_CONFIG_ACK = 0x15
    VERIFY_WEIGHTS = 0x16
    VERIFY_WEIGHTS_RESP = 0x17
    DEPLOY_COMPLETE = 0x18
    DEPLOY_COMPLETE_ACK = 0x19

    # -- Inference --
    INFER_START = 0x20
    INFER_START_ACK = 0x21
    INFER_STOP = 0x22
    INFER_STOP_ACK = 0x23
    INJECT_SPIKES = 0x24
    OUTPUT_SPIKES = 0x25
    INJECT_BATCH = 0x26
    OUTPUT_BATCH = 0x27

    # -- Monitoring --
    MONITOR_START = 0x30
    MONITOR_START_ACK = 0x31
    MONITOR_STOP = 0x32
    MONITOR_STOP_ACK = 0x33
    SPIKE_EVENT = 0x34
    STATE_SNAPSHOT = 0x35

    # -- System --
    RESET = 0x40
    RESET_ACK = 0x41
    SET_LED = 0x42
    ERROR = 0x43
    BOOTLOADER = 0x44
    BOOTLOADER_ACK = 0x45

    # -- Reserved --
    MULTI_BOARD_SYNC = 0xF0
    RAW_SPI = 0xF1
    RAW_SPI_RESP = 0xF2


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------


class ErrorCode(enum.IntEnum):
    """NMP error codes returned in ERROR packets."""

    ERR_UNKNOWN_CMD = 0x0001
    ERR_CRC_MISMATCH = 0x0002
    ERR_PAYLOAD_TOO_LARGE = 0x0003
    ERR_INVALID_LAYER = 0x0004
    ERR_INVALID_PARAM = 0x0005
    ERR_NOT_PROGRAMMED = 0x0006
    ERR_ALREADY_RUNNING = 0x0007
    ERR_SPI_TIMEOUT = 0x0008
    ERR_CHIP_NOT_DETECTED = 0x0009
    ERR_PROTOCOL_VERSION = 0x000A


# ---------------------------------------------------------------------------
# CRC-16/CCITT
# ---------------------------------------------------------------------------


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """Compute CRC-16/CCITT (XModem variant) over *data*.

    Args:
        data: Input bytes.
        init: Initial CRC value (default 0xFFFF).

    Returns:
        16-bit CRC value.
    """
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# Packet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Packet:
    """A single NMP packet.

    Args:
        cmd: Command ID.
        seq: Sequence number (uint16).
        flags: Flags byte.
        payload: Command-specific payload bytes.
    """

    cmd: Cmd
    seq: int = 0
    flags: int = 0
    payload: bytes = b""

    def encode(self) -> bytes:
        """Serialise this packet to wire bytes.

        Returns:
            Complete packet bytes including header, payload, and CRC.

        Raises:
            ValueError: If payload exceeds :data:`MAX_PAYLOAD`.
        """
        if len(self.payload) > MAX_PAYLOAD:
            raise ValueError(
                f"Payload length {len(self.payload)} exceeds max {MAX_PAYLOAD}."
            )
        header = struct.pack(
            "<BBBBHH",
            SYNC_BYTE,
            PROTOCOL_VERSION,
            int(self.cmd),
            self.flags,
            self.seq & 0xFFFF,
            len(self.payload),
        )
        body = header + self.payload
        crc = crc16_ccitt(body)
        return body + struct.pack("<H", crc)

    @classmethod
    def decode(cls, data: bytes) -> Packet:
        """Decode a complete packet from wire bytes.

        Args:
            data: Raw bytes (must include header + payload + CRC).

        Returns:
            Decoded :class:`Packet`.

        Raises:
            ValueError: On sync mismatch, truncation, or CRC failure.
        """
        if len(data) < MIN_PACKET_SIZE:
            raise ValueError(
                f"Packet too short: {len(data)} < {MIN_PACKET_SIZE} bytes."
            )
        sync, version, cmd, flags, seq, payload_len = struct.unpack_from(
            "<BBBBHH", data
        )
        if sync != SYNC_BYTE:
            raise ValueError(f"Bad sync byte: 0x{sync:02X} (expected 0xAA).")
        expected_len = HEADER_SIZE + payload_len + CRC_SIZE
        if len(data) < expected_len:
            raise ValueError(
                f"Packet truncated: have {len(data)}, need {expected_len}."
            )
        payload = data[HEADER_SIZE : HEADER_SIZE + payload_len]
        body = data[: HEADER_SIZE + payload_len]
        (crc_received,) = struct.unpack_from(
            "<H", data, HEADER_SIZE + payload_len
        )
        crc_computed = crc16_ccitt(body)
        if crc_received != crc_computed:
            raise ValueError(
                f"CRC mismatch: received 0x{crc_received:04X}, "
                f"computed 0x{crc_computed:04X}."
            )
        return cls(cmd=Cmd(cmd), seq=seq, flags=flags, payload=bytes(payload))


# ---------------------------------------------------------------------------
# PacketCodec — stateful stream decoder
# ---------------------------------------------------------------------------


class PacketCodec:
    """Stateful stream decoder: feed raw bytes, get complete packets.

    Buffers incoming data and extracts complete packets as they become
    available.  Handles partial reads gracefully.

    Example::

        codec = PacketCodec()
        packets = codec.feed(chunk1)
        packets += codec.feed(chunk2)
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[Packet]:
        """Append *data* to the internal buffer and return decoded packets.

        Args:
            data: Raw bytes from the transport layer.

        Returns:
            List of successfully decoded packets (may be empty).
        """
        self._buf.extend(data)
        packets: list[Packet] = []

        while True:
            # Find sync byte
            try:
                idx = self._buf.index(SYNC_BYTE)
            except ValueError:
                self._buf.clear()
                break
            if idx > 0:
                del self._buf[:idx]

            # Need at least a header to read payload length
            if len(self._buf) < HEADER_SIZE:
                break

            payload_len = struct.unpack_from("<H", self._buf, 6)[0]
            total = HEADER_SIZE + payload_len + CRC_SIZE

            if len(self._buf) < total:
                break  # wait for more data

            raw = bytes(self._buf[:total])
            del self._buf[:total]

            try:
                packets.append(Packet.decode(raw))
            except ValueError:
                # Bad packet — skip the sync byte and keep scanning
                continue

        return packets

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._buf.clear()
