"""Tests for NMP protocol: packet encoding/decoding, CRC, stream codec."""

import pytest
from neuromap.protocol import (
    CRC_SIZE,
    HEADER_SIZE,
    MAX_PAYLOAD,
    MIN_PACKET_SIZE,
    Cmd,
    ErrorCode,
    Packet,
    PacketCodec,
    crc16_ccitt,
)


class TestCRC16:
    """CRC-16/CCITT tests."""

    def test_empty(self) -> None:
        """Test empty data."""
        assert crc16_ccitt(b"") == 0xFFFF

    def test_known_value(self) -> None:
        """Test known value."""
        # "123456789" should produce 0x29B1 with init=0xFFFF (XModem variant)
        assert crc16_ccitt(b"123456789") == 0x29B1

    def test_deterministic(self) -> None:
        """Test deterministic."""
        data = b"neuromap"
        assert crc16_ccitt(data) == crc16_ccitt(data)


class TestPacket:
    """Packet encode/decode tests."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip."""
        pkt = Packet(cmd=Cmd.PING, seq=42, flags=0, payload=b"")
        raw = pkt.encode()
        decoded = Packet.decode(raw)
        assert decoded.cmd == Cmd.PING
        assert decoded.seq == 42
        assert decoded.payload == b""

    def test_encode_decode_with_payload(self) -> None:
        """Test encode/decode with payload."""
        payload = b"\x01\x02\x03\x04"
        pkt = Packet(cmd=Cmd.WRITE_WEIGHTS, seq=1, payload=payload)
        raw = pkt.encode()
        decoded = Packet.decode(raw)
        assert decoded.cmd == Cmd.WRITE_WEIGHTS
        assert decoded.payload == payload

    def test_packet_size(self) -> None:
        """Test packet size."""
        pkt = Packet(cmd=Cmd.PING, seq=0, payload=b"")
        raw = pkt.encode()
        assert len(raw) == MIN_PACKET_SIZE  # 10 bytes

    def test_payload_size(self) -> None:
        """Test payload size."""
        payload = b"x" * 100
        pkt = Packet(cmd=Cmd.PONG, seq=0, payload=payload)
        raw = pkt.encode()
        assert len(raw) == HEADER_SIZE + 100 + CRC_SIZE

    def test_max_payload(self) -> None:
        """Test max payload."""
        pkt = Packet(cmd=Cmd.PING, seq=0, payload=b"x" * MAX_PAYLOAD)
        raw = pkt.encode()
        decoded = Packet.decode(raw)
        assert len(decoded.payload) == MAX_PAYLOAD

    def test_exceeds_max_payload(self) -> None:
        """Test exceeds max payload."""
        with pytest.raises(ValueError, match="exceeds max"):
            Packet(cmd=Cmd.PING, seq=0, payload=b"x" * (MAX_PAYLOAD + 1)).encode()

    def test_bad_sync_byte(self) -> None:
        """Test bad sync byte."""
        raw = b"\x00" + b"\x00" * (MIN_PACKET_SIZE - 1)
        with pytest.raises(ValueError, match="Bad sync"):
            Packet.decode(raw)

    def test_truncated_packet(self) -> None:
        """Test truncated packet."""
        with pytest.raises(ValueError, match="too short"):
            Packet.decode(b"\xaa\x01")

    def test_crc_corruption(self) -> None:
        """Test CRC corruption."""
        pkt = Packet(cmd=Cmd.PING, seq=0)
        raw = bytearray(pkt.encode())
        raw[-1] ^= 0xFF  # corrupt CRC
        with pytest.raises(ValueError, match="CRC mismatch"):
            Packet.decode(bytes(raw))

    def test_seq_wraps(self) -> None:
        """Test seq wraps."""
        pkt = Packet(cmd=Cmd.PING, seq=0xFFFF)
        raw = pkt.encode()
        decoded = Packet.decode(raw)
        assert decoded.seq == 0xFFFF

    def test_all_commands_valid(self) -> None:
        """Every Cmd enum value can be encoded/decoded."""
        for cmd in Cmd:
            pkt = Packet(cmd=cmd, seq=0)
            raw = pkt.encode()
            decoded = Packet.decode(raw)
            assert decoded.cmd == cmd


class TestPacketCodec:
    """Stateful stream decoder tests."""

    def test_single_packet(self) -> None:
        """Test single packet."""
        codec = PacketCodec()
        pkt = Packet(cmd=Cmd.PING, seq=1)
        packets = codec.feed(pkt.encode())
        assert len(packets) == 1
        assert packets[0].cmd == Cmd.PING

    def test_two_packets_concatenated(self) -> None:
        """Test two packets concatenated."""
        codec = PacketCodec()
        p1 = Packet(cmd=Cmd.PING, seq=1).encode()
        p2 = Packet(cmd=Cmd.PONG, seq=2, payload=b"\x00" * 11).encode()
        packets = codec.feed(p1 + p2)
        assert len(packets) == 2
        assert packets[0].cmd == Cmd.PING
        assert packets[1].cmd == Cmd.PONG

    def test_partial_feed(self) -> None:
        """Test partial feed."""
        codec = PacketCodec()
        raw = Packet(cmd=Cmd.PING, seq=1).encode()
        # Feed first half, then second half
        packets = codec.feed(raw[:5])
        assert len(packets) == 0
        packets = codec.feed(raw[5:])
        assert len(packets) == 1
        assert packets[0].cmd == Cmd.PING

    def test_garbage_before_sync(self) -> None:
        """Test garbage before sync."""
        codec = PacketCodec()
        garbage = b"\x00\x01\x02"
        raw = Packet(cmd=Cmd.PING, seq=1).encode()
        packets = codec.feed(garbage + raw)
        assert len(packets) == 1

    def test_reset_clears_buffer(self) -> None:
        """Test reset clears buffer."""
        codec = PacketCodec()
        raw = Packet(cmd=Cmd.PING, seq=1).encode()
        codec.feed(raw[:5])  # partial
        codec.reset()
        packets = codec.feed(raw)  # full fresh packet
        assert len(packets) == 1


class TestErrorCode:
    """Error code enum sanity checks."""

    def test_values_unique(self) -> None:
        """Test values unique."""
        values = [e.value for e in ErrorCode]
        assert len(values) == len(set(values))
