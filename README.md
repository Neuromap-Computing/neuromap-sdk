# Neuromap SDK

Python SDK for programming Neuromap neuromorphic chips. Build, train, quantize
and export spiking neural networks that fit real silicon — the NeuroSoC-v1
targets 48 LIF neurons, 512 4-bit synapses and 16 input pins.

Built on [PyTorch](https://pytorch.org/) and
[snnTorch](https://snntorch.readthedocs.io/). Requires Python 3.10+.

- Full documentation: **[docs.neuromap.ca](https://docs.neuromap.ca/)**
- Homepage: [neuromap.ca](https://neuromap.ca)

## Installation

```bash
pip install neuromap
```

For development (using Poetry):

```bash
cd neuro-sim
poetry install --with dev
```

A worked end-to-end pipeline lives in
[`examples/prosthesis/`](examples/prosthesis/) (EMG → spike encoding → training →
export).

## Quick Start

```python
from neuromap import Network, Trainer, Exporter, chips

# Build a network matching the NeuroSoC-v1 chip
net = Network(chips.NEUROSOC_V1)
print(net.summary())

# Train
trainer = Trainer(net, lr=1e-3, epochs=20)
history = trainer.fit(train_loader, val_loader)

# Export a hardware bundle
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

# Train, then quantize and export to a .nmap bundle
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

## Linting and Formatting

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

### `Exporter`

Export trained networks to `.nmap` bundles.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(network)` | Wrap a `Network` for export |
| `quantize` | `(bits=None) -> Exporter` | Apply post-training quantization (returns self for chaining) |
| `to_weight_map` | `() -> dict[str, np.ndarray]` | Extract quantized weight arrays |
| `save` | `(path) -> None` | Save `.nmap` bundle to disk |
| `to_bytes` | `() -> bytes` | Export the `.nmap` bundle as in-memory bytes |
| `load_nmap` *(cls)* | `(path, *, device="cpu") -> Network` | Load `.nmap` and reconstruct a `Network` |

A `.nmap` file is a ZIP archive with a `manifest.json` (chip spec, topology, quantization metadata) and per-layer weight arrays under `weights/`.

---

### `chips` namespace

Predefined chip profiles.

| | Description |
|---|---|
| `chips.NEUROSOC_V1` | NeuroSoC v1: 48 LIF neurons, 512 4-bit synapses |
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

## Internal / Advanced API

These are not part of the stable public API but can be useful for advanced use cases.

### `_internal.lif.NeuromapLIF`: raw LIF layer (nn.Module)

```python
from neuromap._internal.lif import NeuromapLIF

layer = NeuromapLIF(
    in_features=16, out_features=32, beta=0.9, threshold=1.0, t_ref=2, output_mem=False
)
spikes, state = layer(x_seq, state=None, return_state=True)
# also: NeuromapLIF.from_neuron_params(in, out, params)
# also: layer.init_state(batch_size, device=device)
```

### `_internal.dynamic_snn.DynamicSNN`: raw multi-layer SNN (nn.Module)

```python
from neuromap._internal.dynamic_snn import DynamicSNN

snn = DynamicSNN([16, 32, 4], neuron_kwargs=None, use_decoder=True)
out = snn(x)  # rate-coded
out = snn.forward_sequence(x)  # per-frame
out, state = snn(x, return_state=True)
```

### `_internal.encoding`: raw spike encoders

```python
from neuromap._internal.encoding import encode_rate, encode_latency, encode_delta

spikes = encode_rate(data, num_steps=100)
spikes = encode_latency(data, num_steps=100)
spikes = encode_delta(data)
```

### `_internal.quantization`: weight quantization

```python
from neuromap._internal.quantization import quantize_model_weights, quantize_weights_4bit

model = quantize_model_weights(model, bits=4)
model = quantize_weights_4bit(model)  # convenience alias
```

### `_internal.spike`: surrogate gradients

```python
from neuromap._internal.spike import get_surrogate

grad_fn = get_surrogate("atan")  # "fast_sigmoid", "straight_through", "spike_rate_escape"
```

### `_internal.audio`: audio preprocessing

```python
from neuromap._internal.audio import load_audio_mono, frame_audio

audio, sr = load_audio_mono("clip.wav", target_sample_rate=16000)
frames = frame_audio(audio, frame_size=256, hop_size=128)
```
