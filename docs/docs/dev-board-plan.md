# Neuromap Dev Board - SDK Extension & Hardware Requirements Plan

> **Status**: Draft - ASIC taped out, awaiting fabrication
> **Last updated**: 2026-03-10

## Context

The NeuroSoC-v1 ASIC is taped out and awaiting fabrication. This document plans:
1. The **dev board** hardware that will host the ASIC for developers
2. The **SDK extensions** to deploy trained models and interact with the board
3. The **firmware protocol** between the host Python SDK and the MCU companion

The goal: "democratize neuromorphic computing like Arduino did for electronics" - plug-and-play USB board with a one-liner deploy experience, powerful enough for researchers.

### Key Decisions

- **Host link**: USB (CDC virtual serial port - driverless)
- **Board arch**: MCU companion (not decided which - design MCU-agnostic)
- **ASIC interface**: SPI registers, abstracted behind MCU firmware
- **Training**: Host-only (PyTorch on PC), chip for inference only
- **I/O**: Wide variety for maximum flexibility
- **DX**: One-liner deploy default (`net.deploy(board)`), explicit pipeline for power users
- **Multi-board**: Architecture supports it, v1 implements single-board only
- **CLI**: Ship `neuromap flash`, `neuromap monitor`, `neuromap info` alongside Python API
- **Monitoring**: Live spike visualization is essential for v1
- **Firmware scope**: Plan both SDK + firmware USB protocol

---

## 1. Dev Board Hardware Requirements

### 1.1 Core Components

| Component | Spec | Notes |
|-----------|------|-------|
| **NeuroSoC-v1 ASIC** | 48 LIF neurons, (16,16,16), 4-bit weights, SPI slave | 250mV supply, 300kHz max freq |
| **Companion MCU** | ARM Cortex-M4F+ or RISC-V class | Needs: SPI master ≥10MHz, USB 2.0 device, 256KB+ Flash, 64KB+ RAM |
| **USB-C connector** | Data + power (primary host link) | |
| **3.3V LDO** | For MCU | |
| **250mV regulator** | For ASIC supply | Low-noise, tight tolerance |
| **Crystal/oscillator** | For MCU USB clock | |

### 1.2 I/O & Expansion

| Interface | Pins | Purpose |
|-----------|------|---------|
| **GPIO** | 4x (3.3V logic) | Sensors, actuators, LEDs, buttons |
| **ADC** | 2x channels (12-bit) | Analog sensor input |
| **I2S / audio codec pads** | 1x | Microphone/speaker for audio use cases (prosthesis) |
| **SPI pass-through** | 1x | External peripherals, daisy-chaining |
| **I2C bus** | 1x | IMU, environmental sensors |
| **3.3V + GND** | On headers | Power for external modules |
| **PMOD / expansion header** | 2x 10-pin (2.54mm) | Arduino-shield-compatible spacing |

### 1.3 On-Board Extras

- **IMU** (e.g. LSM6DS3) on I2C - optional populate (DNP for cost-reduced variant)
- **Reset button + user button** (wired to MCU GPIO)
- **Power LED** (green), **Activity LED** (blue, MCU-controlled), **Spike LED** (amber)
- **SWD/JTAG debug header** (1.27mm Tag-Connect) for firmware development

### 1.4 Form Factor

- PCB: ~50mm × 35mm (smaller than Arduino Nano 33 BLE)
- 4× M2 mounting holes
- Castellated edges optional for module integration
- Power budget: ~100mA typical from USB (well within USB 2.0 500mA)

### 1.5 Block Diagram

```
Host PC                    Dev Board
+--------+    USB-C     +------------------------------------------+
|        |<============>| USB-C --> MCU (Cortex-M4F / RISC-V)      |
| Python |   CDC/ACM    |              |    |    |                 |
|  SDK   |              |         SPI Master  I2C  GPIO            |
|        |              |              |      |    |               |
+--------+              |         NeuroSoC   IMU  Expansion        |
                        |          ASIC       (opt) Headers        |
                        |              |                           |
                        |         250mV LDO <-- 3.3V LDO <-- 5V    |
                        +------------------------------------------+
```

---

## 2. Firmware Protocol (NMP - Neuromap Protocol)

### 2.1 Transport

- USB CDC/ACM (virtual serial port) - driverless on Windows 10+, macOS, Linux
- Binary packet protocol over the byte stream

### 2.2 Packet Format

```
Offset  Size  Field         Description
------  ----  -----         -----------
0       1     SYNC          0xAA (magic byte)
1       1     VERSION       Protocol version (0x01)
2       1     CMD           Command ID
3       1     FLAGS         [0]=ACK_REQ, [1]=STREAM, [2..7]=reserved
4       2     SEQ           Sequence number (uint16 LE)
6       2     PAYLOAD_LEN   Payload length (uint16 LE, max 4096)
8       N     PAYLOAD       Command-specific data
8+N     2     CRC16         CRC-16/CCITT over bytes [0..8+N-1]
```

10 bytes overhead. Max packet: 4106 bytes.

### 2.3 Command Table

```
CMD   Name                  Dir          Description
---   ----                  ---          -----------
- Discovery -
0x01  PING                  Host→Dev     Board identification request
0x02  PONG                  Dev→Host     Board ID, HW rev, FW version, chip type, status
0x03  GET_INFO              Host→Dev     Detailed board info query
0x04  INFO_RESP             Dev→Host     Full board info payload

- Programming -
0x10  WRITE_WEIGHTS         Host→Dev     Write weight matrix for one layer
0x11  WRITE_WEIGHTS_ACK     Dev→Host     Acknowledge
0x12  WRITE_NEURON_PARAMS   Host→Dev     Write LIF parameters (per-layer or global)
0x13  WRITE_NEURON_PARAMS_ACK Dev→Host   Acknowledge
0x14  WRITE_CONFIG          Host→Dev     Inference config (clock rate, etc.)
0x15  WRITE_CONFIG_ACK      Dev→Host     Acknowledge
0x16  VERIFY_WEIGHTS        Host→Dev     Read back weights for verification
0x17  VERIFY_WEIGHTS_RESP   Dev→Host     Weight readback data
0x18  DEPLOY_COMPLETE       Host→Dev     All programming done
0x19  DEPLOY_COMPLETE_ACK   Dev→Host     Chip ready for inference

- Inference -
0x20  INFER_START           Host→Dev     Start inference clock
0x21  INFER_START_ACK       Dev→Host     Running
0x22  INFER_STOP            Host→Dev     Stop inference clock
0x23  INFER_STOP_ACK        Dev→Host     Stopped
0x24  INJECT_SPIKES         Host→Dev     Inject one timestep of input spikes
0x25  OUTPUT_SPIKES         Dev→Host     Output spike vector
0x26  INJECT_BATCH          Host→Dev     Inject multiple timesteps
0x27  OUTPUT_BATCH          Dev→Host     Batch output

- Monitoring -
0x30  MONITOR_START         Host→Dev     Begin live spike streaming
0x31  MONITOR_START_ACK     Dev→Host     Streaming active
0x32  MONITOR_STOP          Host→Dev     Stop streaming
0x33  MONITOR_STOP_ACK      Dev→Host     Stopped
0x34  SPIKE_EVENT           Dev→Host     Single spike event (layer, neuron, timestamp)
0x35  STATE_SNAPSHOT        Dev→Host     Full membrane state readback

- System -
0x40  RESET                 Host→Dev     Soft-reset board
0x41  RESET_ACK             Dev→Host     Reset done
0x42  SET_LED               Host→Dev     Control user LEDs
0x43  ERROR                 Dev→Host     Error response (replaces any expected ACK)
0x44  BOOTLOADER            Host→Dev     Enter DFU mode
0x45  BOOTLOADER_ACK        Dev→Host     Entering bootloader

- Reserved -
0xF0  MULTI_BOARD_SYNC      Host→Dev     Future multi-board clock sync
0xF1  RAW_SPI               Host→Dev     SPI pass-through (power-user escape hatch)
0xF2  RAW_SPI_RESP          Dev→Host     SPI response
```

### 2.4 Key Payload Formats

**PONG** (11 bytes):

```
board_id(4) | hw_rev(1) | fw_ver_major(1) | fw_ver_minor(1) | fw_ver_patch(1)
| protocol_ver(1) | chip_type(1) | status(1)
```

**WRITE_WEIGHTS** (6 + N bytes):

```
layer_idx(1) | bits(1) | rows(2 LE) | cols(2 LE) | packed_data(N)
```

4-bit packing: 2 signed weights per byte (high nibble, low nibble), row-major.
For NEUROSOC_V1 layer 0 (16×16): 128 bytes of packed data.

**INJECT_SPIKES** (3 + N bytes):

```
timestep_idx(1) | input_size(2 LE) | spike_vector(N bits, packed into bytes)
```

**SPIKE_EVENT** (7 bytes):

```
timestamp_us(4 LE) | layer_idx(1) | neuron_idx(1) | event_type(1)
```

**ERROR** (4 + N bytes):

```
error_code(2 LE) | orig_seq(2 LE) | message(UTF-8, up to 256 bytes)
```

### 2.5 Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0x0001 | ERR_UNKNOWN_CMD | Unrecognized command ID |
| 0x0002 | ERR_CRC_MISMATCH | Packet CRC verification failed |
| 0x0003 | ERR_PAYLOAD_TOO_LARGE | Payload exceeds max size |
| 0x0004 | ERR_INVALID_LAYER | Layer index out of range |
| 0x0005 | ERR_INVALID_PARAM | Invalid parameter value |
| 0x0006 | ERR_NOT_PROGRAMMED | Inference attempted before deploy |
| 0x0007 | ERR_ALREADY_RUNNING | Inference already active |
| 0x0008 | ERR_SPI_TIMEOUT | MCU could not reach ASIC over SPI |
| 0x0009 | ERR_CHIP_NOT_DETECTED | ASIC not responding |
| 0x000A | ERR_PROTOCOL_VERSION | Protocol version mismatch |

---

## 3. SDK Architecture Changes

### 3.1 New File Structure

```
src/neuromap/
    __init__.py              # Add: Board, BoardInfo
    chip.py                  # Unchanged
    network.py               # Add: deploy() method
    trainer.py               # Unchanged
    export.py                # Add: to_bytes(), nmap-v2 manifest with hw/ dir
    board.py                 # NEW - Board connection, discovery, deploy, inference
    protocol.py              # NEW - Packet encode/decode, CRC, command enums
    monitor.py               # NEW - SpikeMonitor, raster plot, firing rates
    cli.py                   # NEW - CLI entry points
    _internal/
        transport.py         # NEW - Transport ABC, SerialTransport, MockTransport, MockFirmware
        packing.py           # NEW - Weight nibble packing/unpacking
        (existing files unchanged)
```

### 3.2 `protocol.py` - Packet Codec

```python
SYNC_BYTE = 0xAA
PROTOCOL_VERSION = 0x01
MAX_PAYLOAD = 4096


class Cmd(enum.IntEnum):
    PING = 0x01
    PONG = 0x02
    # ... all command IDs


@dataclass(frozen=True)
class Packet:
    cmd: Cmd
    seq: int
    flags: int
    payload: bytes

    def encode(self) -> bytes: ...

    @classmethod
    def decode(cls, data: bytes) -> Packet: ...


class PacketCodec:
    """Stateful stream decoder - feed raw bytes, get complete packets."""

    def feed(self, data: bytes) -> list[Packet]: ...
```

### 3.3 `_internal/transport.py` - Abstraction Layer

```python
class Transport(ABC):
    """Abstract byte-stream transport."""

    def open(self) -> None: ...
    def close(self) -> None: ...
    def write(self, data: bytes) -> None: ...
    def read(self, size: int, timeout_ms: int = 1000) -> bytes: ...
    @property
    def is_open(self) -> bool: ...


class SerialTransport(Transport):
    """USB CDC serial via pyserial."""

    def __init__(self, port: str, baudrate: int = 115200): ...


class MockTransport(Transport):
    """In-memory transport for testing without hardware."""


class MockFirmware:
    """Simulates MCU firmware using DynamicSNN model.

    Board.mock() produces numerically identical results to Network.infer(),
    so developers can build apps without the physical board.
    """

    def __init__(self, chip: ChipSpec): ...
    def process_packet(self, packet: Packet) -> list[Packet]: ...
```

### 3.4 `_internal/packing.py` - Weight Packing

```python
def pack_weights_nibble(weights: np.ndarray, bits: int = 4) -> bytes:
    """Pack int8 weight matrix into nibble-packed bytes for ASIC SRAM."""


def unpack_weights_nibble(data: bytes, rows: int, cols: int, bits: int = 4) -> np.ndarray:
    """Unpack nibble bytes back to int8 matrix."""
```

### 3.5 `board.py` - Board API

```python
@dataclass(frozen=True)
class BoardInfo:
    board_id: int
    hw_revision: int
    fw_version: tuple[int, int, int]
    protocol_version: int
    chip_type: str
    chip_spec: ChipSpec
    serial_port: str
    status: str  # "idle" | "inferring" | "error"


class Board:
    # -- Discovery --
    @classmethod
    def connect(cls, port: str | None = None, *, timeout: float = 5.0) -> Board:
        """Auto-discover USB board (scan serial ports, send PING)."""

    @classmethod
    def list_boards(cls) -> list[BoardInfo]:
        """Enumerate all connected boards."""

    @classmethod
    def mock(cls, chip: ChipSpec | None = None) -> Board:
        """Create mock board (software simulation, no hardware needed)."""

    # -- Programming --
    def program(self, nmap_path: str | Path) -> None:
        """Flash .nmap to board (weights + neuron params + config)."""

    def verify(self) -> bool:
        """Read back weights and verify against local copy."""

    # -- Inference --
    def start_inference(self) -> None: ...
    def stop_inference(self) -> None: ...

    def inject(self, spikes: np.ndarray | torch.Tensor) -> np.ndarray:
        """Inject one timestep, return output spikes."""

    def inject_batch(self, spike_sequence: np.ndarray | torch.Tensor) -> np.ndarray:
        """Inject multiple timesteps, return all outputs."""

    # -- Monitoring --
    def start_monitor(self) -> SpikeMonitor: ...

    # -- System --
    def reset(self) -> None: ...
    def set_led(self, led: int, state: bool) -> None: ...
    def close(self) -> None: ...

    # -- Context manager --
    def __enter__(self) -> Board: ...
    def __exit__(self, *exc) -> None: ...
```

### 3.6 `network.py` - Add `deploy()` Method

```python
def deploy(self, board: Board, *, verify: bool = True) -> None:
    """One-liner deploy: quantize → export → program board.

    This is the 'Arduino upload' experience:
        board = Board.connect()
        net.deploy(board)
    """
    # 1. Create Exporter, quantize to chip's weight_bits
    # 2. Export to in-memory .nmap bytes (via Exporter.to_bytes())
    # 3. Call board.program() with the bytes
    # 4. Optionally verify weights on-chip
```

### 3.7 `export.py` - `.nmap` v2 & `to_bytes()`

Add `to_bytes()` for in-memory export (used by `deploy()`). Evolve manifest to v2:

```json
{
    "format": "nmap-v2",
    "chip_spec": { "..." : "..." },
    "quantized": true,
    "hw_layout": {
        "weight_packing": "nibble_signed_4bit",
        "byte_order": "little",
        "layers": [
            { "index": 0, "file": "hw/layer_0.bin", "rows": 16, "cols": 16, "size_bytes": 128 }
        ],
        "total_weight_bytes": 256,
        "neuron_params_file": "hw/neuron_params.bin"
    },
    "neuron_init": { "v_mem_reset": true, "refractory_clear": true },
    "deploy_checksum": "sha256:..."
}
```

ZIP gains `hw/` directory with pre-packed binaries alongside existing `weights/` (backward compatible - `load_nmap()` still uses `weights/`).

### 3.8 `monitor.py` - Live Spike Monitoring

```python
@dataclass
class SpikeEvent:
    timestamp_us: int
    layer: int
    neuron: int
    event_type: str  # "spike" | "membrane_cross"

class SpikeMonitor:
    """Live spike monitoring with callbacks and visualization."""

    def on_spike(self, callback: Callable[[SpikeEvent], None]) -> None: ...
    def collect(self, duration: float) -> list[SpikeEvent]: ...
    def raster_plot(self, ...) -> Figure: ...
    def firing_rate(self, ...) -> dict[int, float]: ...
    def stop(self) -> None: ...

    # Context manager
    def __enter__(self) -> SpikeMonitor: ...
    def __exit__(self, *exc) -> None: ...
```

Background thread reads SPIKE_EVENT packets, dispatches to callbacks or internal buffer.

### 3.9 `cli.py` - CLI Commands

Registered via `[project.scripts] neuromap = "neuromap.cli:main"` in pyproject.toml.

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `neuromap info` | Show board info | `--port`, `--json` |
| `neuromap flash <file.nmap>` | Deploy model to board | `--port`, `--no-verify`, `--verbose` |
| `neuromap monitor` | Live spike monitoring | `--duration`, `--output`, `--raster`, `--layer`, `--json` |
| `neuromap reset` | Soft-reset board | `--port` |
| `neuromap test` | Board self-test | `--port` |
| `neuromap inject` | Inject spikes from CLI | `--input`, `--file`, `--output` |

### 3.10 Updated `__init__.py`

```python
from neuromap.board import Board, BoardInfo
# Add to __all__: "Board", "BoardInfo"
```

### 3.11 Updated `pyproject.toml`

```toml
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "pyserial>=3.5",        # NEW
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "scipy"]
monitor = ["matplotlib>=3.5"]  # NEW - for raster plots

[project.scripts]
neuromap = "neuromap.cli:main"  # NEW
```

---

## 4. End-to-End User Workflows

### Beginner - "Arduino Experience"

```python
from neuromap import Network, Trainer, Board, chips

net = Network(chips.NEUROSOC_V1)
trainer = Trainer(net, epochs=20)
trainer.fit(train_loader, val_loader)

board = Board.connect()  # auto-discover USB
net.deploy(board)  # quantize + flash + verify

output = board.inject(input_spikes)
```

### Power User - Explicit Pipeline

```python
from neuromap import Network, Exporter, Board

net = Network.load("trained_model.pt")
exporter = Exporter(net)
exporter.quantize(bits=4)
exporter.save("model.nmap")

board = Board.connect(port="/dev/ttyACM0")
board.program("model.nmap")
board.verify()
board.start_inference()

with board.start_monitor() as mon:
    mon.on_spike(lambda e: print(f"L{e.layer}:N{e.neuron}"))
    for chunk in sensor_stream:
        output = board.inject(chunk)
```

### CLI User

```bash
neuromap flash model.nmap
neuromap monitor --duration 10 --raster
neuromap info --json
```

### Development Without Hardware

```python
board = Board.mock()  # software simulation
net.deploy(board)  # works identically
output = board.inject(spikes)  # runs DynamicSNN internally
```

---

## 5. Implementation Phases

### Phase 1: Protocol & Transport (no hardware needed)

- `_internal/packing.py` - weight nibble packing/unpacking
- `protocol.py` - packet encode/decode, CRC-16/CCITT, Cmd enum
- `_internal/transport.py` - Transport ABC, MockTransport, MockFirmware
- Tests for all

### Phase 2: Board API & Mock Backend

- `board.py` - Board class with full API, working via MockTransport
- Add `deploy()` to `network.py`
- Update `export.py` with `to_bytes()` and nmap-v2 manifest
- Tests for Board + mock workflows

### Phase 3: Real Hardware Support

- `SerialTransport` implementation (pyserial)
- Auto-discovery (scan serial ports, PING)
- Integration testing with real board (when available)

### Phase 4: Monitoring & CLI

- `monitor.py` - SpikeMonitor, background thread, raster plots
- `cli.py` - all subcommands
- Update pyproject.toml `[project.scripts]`

### Phase 5: Polish

- Update `__init__.py` public API
- Update pyproject.toml dependencies
- Documentation (MkDocs pages for board, protocol, CLI)
- Update examples

---

## 6. Files to Create

| File | Purpose |
|------|---------|
| `src/neuromap/board.py` | Board connection, discovery, deploy, inference |
| `src/neuromap/protocol.py` | NMP packet codec, CRC, command enums |
| `src/neuromap/monitor.py` | SpikeMonitor, raster plots, firing rates |
| `src/neuromap/cli.py` | CLI entry points |
| `src/neuromap/_internal/transport.py` | Transport ABC, Serial, Mock, MockFirmware |
| `src/neuromap/_internal/packing.py` | Weight nibble packing |
| `tests/test_protocol.py` | Packet encode/decode, CRC |
| `tests/test_packing.py` | Weight packing round-trips |
| `tests/test_board.py` | Board API with mock backend |
| `tests/test_monitor.py` | Spike monitoring |
| `tests/test_cli.py` | CLI commands |
| `docs/docs/board.md` | Board usage guide |
| `docs/docs/protocol.md` | Firmware protocol reference |
| `docs/docs/api/board.md` | Board API reference |

## 7. Files to Modify

| File | Change |
|------|--------|
| `src/neuromap/__init__.py` | Add Board, BoardInfo to public API |
| `src/neuromap/network.py` | Add `deploy()` method |
| `src/neuromap/export.py` | Add `to_bytes()`, nmap-v2 manifest, `hw/` directory |
| `pyproject.toml` | Add pyserial dep, monitor extra, `[project.scripts]` |
| `docs/mkdocs.yml` | Add board/protocol/CLI nav entries |

---

## 8. Verification Criteria

1. `pip install -e ".[dev]"` installs with pyserial
2. `python -c "from neuromap import Board; b = Board.mock(); print(b.info)"` - mock board works
3. Mock end-to-end: `net.deploy(Board.mock())` → `board.inject()` returns spikes
4. `pytest tests/` - all new + existing tests pass
5. `neuromap info` (with mock) shows board info
6. `neuromap flash model.nmap` (with mock) succeeds
7. `neuromap monitor --duration 2` (with mock) captures events

---

## 9. Design Rationale

- **USB CDC** - appears as serial port on all OSes without drivers; pyserial is mature; serial monitors can debug raw protocol
- **Binary protocol** - essential for future larger chips; CRC for integrity; 10-byte overhead is minimal
- **MockFirmware uses DynamicSNN** - numerically identical to Network.infer(), full CI/CD testing without hardware
- **CRC-16 over CRC-32** - sufficient for packets <4KB, saves 2 bytes; upgradeable in future protocol version
- **pyserial as core dep** - lightweight (~100KB), no native compilation, cross-platform
- **matplotlib as optional dep** - only needed for raster plots, not for core board operation
