# Neuromap SDK

Python SDK for programming Neuromap neuromorphic chips.

## Installation

```bash
pip install neuromap
```

For development (using Poetry):

```bash
cd neuro-sim
poetry install --with dev
```

## Quick Start

```python
from neuromap import Network, Trainer, Exporter, chips

# Build a network matching the NeuroSoC-v1 chip
net = Network(chips.NEUROSOC_V1)
print(net.summary())

# Train
trainer = Trainer(net, lr=1e-3, epochs=20)
history = trainer.fit(train_loader, val_loader)

# Export for deployment
Exporter(net).quantize().save("model.nmap")
```

## Simplified API (snnTorch)

The SDK provides convenience functions backed by [snnTorch](https://snntorch.readthedocs.io/) for a streamlined workflow:

```python
import neuromap as nm

# Encode raw data to spike trains (wraps snntorch.spikegen)
spike_data = nm.encode_sensor_data(raw_images, num_steps=100)

# Build network from the fixed chip topology
model = nm.Network(nm.chips.NEUROSOC_V1)

# Configure surrogate gradients for training
nm.compile_model(model, surrogate="atan")

# Train, then quantize & export to hardware
nm.quantize_and_export_to_spi(model, path="model.nmap")
```

Available surrogate gradients: `"atan"` (default), `"fast_sigmoid"`, `"straight_through"`, `"spike_rate_escape"`.

Available encoding methods: `"rate"` (default), `"latency"`, `"delta"`.

## Running Tests

```bash
poetry run pytest tests/ -v
```

Or via Nx from the repo root:

```bash
npx nx run neuro-sim:test
```

## Linting & Formatting

```bash
poetry run ruff check .
poetry run ruff format .
```

---

## API Reference

### Top-level convenience (`import neuromap as nm`)

| Function | Signature | Description |
|---|---|---|
| `encode_sensor_data` | `(data, num_steps=100, method="rate", **kwargs) -> Tensor` | Encode continuous data into spike trains (`"rate"`, `"latency"`, `"delta"`) |
| `compile_model` | `(network, surrogate="atan") -> None` | Inject surrogate gradient into all LIF layers for training |
| `quantize_and_export_to_spi` | `(network, bit_resolution=4, path=None) -> bytes\|None` | Quantize weights and export to `.nmap`; returns bytes if no path given |

---

### `Network`

Build and run spiking neural networks mapped to chip topology.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(chip, *, use_decoder=True, membrane_readout=False)` | Create network from a `ChipSpec` |
| `from_topology` *(cls)* | `(layers, *, weight_bits=4, neuron_params=None, name="custom", ...)` | Build from explicit layer list, e.g. `[16, 32, 4]` |
| `infer` | `(x, *, state=None, return_state=False) -> Tensor\|tuple` | Rate-coded forward pass |
| `infer_sequence` | `(x) -> Tensor` | Per-frame sequence inference |
| `infer_stream` | `(x_chunk, state=None) -> (Tensor, state)` | Stateful streaming inference |
| `deploy` | `(board, *, verify=True) -> None` | Quantize, export, and flash to board in one call |
| `save` | `(path) -> None` | Save checkpoint to `.pt` file |
| `load` *(cls)* | `(path, *, device="cpu") -> Network` | Load from `.pt` checkpoint |
| `summary` | `() -> str` | Human-readable architecture summary |
| `to` | `(device) -> Network` | Move model to device |
| `chip` *(prop)* | `-> ChipSpec` | The chip spec this network targets |
| `model` *(prop)* | `-> DynamicSNN` | Underlying PyTorch module |
| `layers` *(prop)* | `-> list[int]` | Layer sizes |
| `input_size` / `output_size` *(prop)* | `-> int` | Input / output feature counts |
| `device` *(prop)* | `-> torch.device` | Current device |

---

### `Trainer`

High-level training driver with LR scheduling and early stopping.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(network, *, lr=3e-4, epochs=40, device="cpu", optimizer="adam", weight_decay=1e-6, scheduler="plateau", early_stop_patience=8, grad_clip_norm=1.0, loss_fn=None, use_sequence_forward=False)` | Configure trainer |
| `fit` | `(train_loader, val_loader=None, *, verbose=True) -> TrainHistory` | Train and return epoch history |
| `evaluate` | `(dataloader) -> dict[str, float]` | Evaluate loss/accuracy on a loader |

#### `TrainHistory`

| Method | Description |
|---|---|
| `append(row)` | Append a dict of epoch metrics |
| `to_list()` | Return list of epoch metric dicts |

---

### `Board`

Interface to a physical or mock Neuromap board.

| Method | Signature | Description |
|---|---|---|
| `connect` *(cls)* | `(port=None, *, timeout=5.0) -> Board` | Auto-discover and connect to a board |
| `list_boards` *(cls)* | `() -> list[BoardInfo]` | List all connected boards |
| `mock` *(cls)* | `(chip=None) -> Board` | Create a software-simulated board |
| `program` | `(nmap_path) -> None` | Flash a `.nmap` model to the board |
| `verify` | `() -> bool` | Read back and verify flashed weights |
| `start_inference` | `() -> None` | Start the inference clock |
| `stop_inference` | `() -> None` | Stop the inference clock |
| `inject` | `(spikes) -> np.ndarray` | Inject one timestep of input spikes, return output spikes |
| `inject_batch` | `(spike_sequence) -> np.ndarray` | Inject multiple timesteps, return aggregated output |
| `start_monitor` | `() -> SpikeMonitor` | Begin live spike monitoring |
| `reset` | `() -> None` | Soft-reset the board |
| `set_led` | `(led, state) -> None` | Control a user LED |
| `close` | `() -> None` | Close the connection |
| `info` *(prop)* | `-> BoardInfo\|None` | Board info from last handshake |
| `is_connected` *(prop)* | `-> bool` | Whether transport is open |
| `is_mock` *(prop)* | `-> bool` | Whether this is a mock board |

`Board` supports context manager (`with Board.connect() as b:`).

#### `BoardInfo` *(frozen dataclass)*

Fields: `board_id`, `hw_revision`, `fw_version`, `protocol_version`, `chip_type`, `serial_port`, `status`.

---

### `Exporter`

Export trained networks to `.nmap` hardware bundles.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(network)` | Wrap a `Network` for export |
| `quantize` | `(bits=None) -> Exporter` | Apply post-training quantization (returns self for chaining) |
| `to_weight_map` | `() -> dict[str, np.ndarray]` | Extract quantized weight arrays |
| `save` | `(path) -> None` | Save `.nmap` bundle to disk |
| `to_bytes` | `() -> bytes` | Export as in-memory bytes |
| `load_nmap` *(cls)* | `(path, *, device="cpu") -> Network` | Load `.nmap` and reconstruct a `Network` |

---

### `SpikeMonitor`

Live spike monitoring returned by `Board.start_monitor()`.

| Method | Signature | Description |
|---|---|---|
| `on_spike` | `(callback) -> None` | Register callback `fn(SpikeEvent) -> None` |
| `collect` | `(duration) -> list[SpikeEvent]` | Block and collect events for `duration` seconds |
| `raster_plot` | `(events=None, *, layer=None, title=...) -> Figure` | Generate raster plot (matplotlib) |
| `firing_rate` | `(events=None) -> dict[int, float]` | Per-layer firing rates (Hz) |
| `stop` | `() -> None` | Stop monitoring |

`SpikeMonitor` supports context manager. Each `SpikeEvent` has: `timestamp_us`, `layer`, `neuron`, `event_type`.

---

### `chips` namespace

Predefined chip profiles.

| | Description |
|---|---|
| `chips.NEUROSOC_V1` | NeuroSoC v1 — 48 LIF neurons, 512 4-bit synapses |
| `chips.list()` | Return names of all predefined profiles |
| `chips.get(name)` | Look up a profile by name |

---

### `ChipSpec` *(frozen dataclass)*

Describes chip hardware topology and neuron parameters.

| | Description |
|---|---|
| `num_layers`, `total_neurons`, `total_synapses` *(props)* | Topology counts |
| `input_size`, `output_size` *(props)* | I/O sizes |
| `to_dict()` / `from_dict(data)` | Serialize / deserialize |

---

### `NeuronParams` *(frozen dataclass)*

LIF neuron parameters used to configure a `ChipSpec`.

| | Description |
|---|---|
| `beta` *(prop)* | Membrane decay rate for snnTorch |
| `to_snntorch_kwargs()` | Return kwargs dict for `NeuromapLIF` |
| `to_dict()` / `from_dict(data)` | Serialize / deserialize |

---

### CLI (`neuromap` command)

```
neuromap info    [--port PORT]          Show board info
neuromap flash   MODEL.nmap [--port]   Deploy model to board
neuromap monitor [--port] [--duration] Live spike monitoring
neuromap inject  SPIKES.npy [--port]   Inject spikes from file
neuromap reset   [--port]              Soft-reset board
neuromap test    [--port]              Board self-test
```

---

## Internal / Advanced API

These are not part of the stable public API but can be useful for advanced use cases, custom transports, or protocol-level work.

### `_internal.lif.NeuromapLIF` — raw LIF layer (nn.Module)

```python
from neuromap._internal.lif import NeuromapLIF
layer = NeuromapLIF(in_features=16, out_features=32, beta=0.9, threshold=1.0,
                    t_ref=2, output_mem=False)
spikes, state = layer(x_seq, state=None, return_state=True)
# also: NeuromapLIF.from_neuron_params(in, out, params)
# also: layer.init_state(batch_size, device=device)
```

### `_internal.dynamic_snn.DynamicSNN` — raw multi-layer SNN (nn.Module)

```python
from neuromap._internal.dynamic_snn import DynamicSNN
snn = DynamicSNN([16, 32, 4], neuron_kwargs=None, use_decoder=True)
out = snn(x)                        # rate-coded
out = snn.forward_sequence(x)       # per-frame
out, state = snn(x, return_state=True)
```

### `_internal.encoding` — raw spike encoders

```python
from neuromap._internal.encoding import encode_rate, encode_latency, encode_delta
spikes = encode_rate(data, num_steps=100)
spikes = encode_latency(data, num_steps=100)
spikes = encode_delta(data)
```

### `_internal.quantization` — weight quantization

```python
from neuromap._internal.quantization import quantize_model_weights, quantize_weights_4bit
model = quantize_model_weights(model, bits=4)
model = quantize_weights_4bit(model)   # convenience alias
```

### `_internal.packing` — nibble packing

```python
from neuromap._internal.packing import pack_weights_nibble, unpack_weights_nibble
raw = pack_weights_nibble(weights_int8, bits=4)
w   = unpack_weights_nibble(raw, rows=R, cols=C, bits=4)
```

### `_internal.transport` — custom transports

```python
from neuromap._internal.transport import SerialTransport, TcpTransport, MockTransport, MockFirmware
t = SerialTransport("/dev/ttyUSB0", baudrate=115200)
t = TcpTransport("192.168.1.10", port=4840)
fw = MockFirmware(chip=chips.NEUROSOC_V1)
t  = MockTransport(fw)
```
All transports implement `open()`, `close()`, `write(data)`, `read(size, timeout_ms)`, `is_open`.

### `_internal.spike` — surrogate gradients

```python
from neuromap._internal.spike import get_surrogate
grad_fn = get_surrogate("atan")   # "fast_sigmoid", "straight_through", "spike_rate_escape"
```

### `_internal.audio` — audio preprocessing

```python
from neuromap._internal.audio import load_audio_mono, frame_audio
audio, sr = load_audio_mono("clip.wav", target_sample_rate=16000)
frames     = frame_audio(audio, frame_size=256, hop_size=128)
```

### `protocol` — wire protocol primitives

```python
from neuromap.protocol import Packet, PacketCodec, Cmd, ErrorCode, crc16_ccitt
pkt   = Packet(cmd=Cmd.PING, seq=0, flags=0, payload=b"")
raw   = pkt.encode()
pkt2  = Packet.decode(raw)
codec = PacketCodec()
pkts  = codec.feed(stream_bytes)   # returns list[Packet] as frames arrive
```
