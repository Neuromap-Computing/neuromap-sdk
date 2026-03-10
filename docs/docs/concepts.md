# Concepts

The Neuromap SDK follows a four-stage pipeline:

## 1. Chip specification

A `ChipSpec` describes the physical chip: layer topology, weight
bit-width, neuron parameters, and optional metadata (frequency, voltage,
energy per spike).

```python
from neuromap import ChipSpec, NeuronParams

chip = ChipSpec(
    name="custom-chip",
    layers=(32, 16, 8),
    weight_bits=4,
    neuron_params=NeuronParams(tau_m=10.0, v_th=1.0),
)
```

Predefined profiles are available via the `chips` namespace:

```python
from neuromap import chips
chip = chips.NEUROSOC_V1
```

## 2. Network

A `Network` wraps a `DynamicSNN` PyTorch model with the chip spec. It
provides inference, streaming, persistence, and introspection methods.

The network's internal model is a stack of LIF (leaky integrate-and-fire)
layers with surrogate gradient training support.

## 3. Trainer

The `Trainer` handles the training loop with:

- **LR warmup** — linear warmup over the first N epochs.
- **LR scheduling** — ReduceLROnPlateau or cosine annealing.
- **Early stopping** — stops when validation loss stops improving.
- **Gradient clipping** — prevents exploding gradients in deep SNNs.
- **Best checkpoint** — automatically restores the best weights.

## 4. Exporter

The `Exporter` quantizes float32 weights to signed fixed-point (matching
the chip's DAC bit-width) and packages them into a `.nmap` archive.

The archive contains:

- `manifest.json` — chip spec and quantization metadata.
- `weights/*.npy` — per-layer integer weight matrices.
