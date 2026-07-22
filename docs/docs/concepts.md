# Concepts

The Neuromap SDK is organised around a four-stage pipeline. Each stage
is a small, focused object, and the output of one feeds the next:

```
ChipSpec  ->  Network  ->  Trainer  ->  Exporter
(hardware)   (model)      (training)   (.nmap bundle)
```

## Spiking neural networks

A spiking neural network communicates with binary events (spikes)
rather than continuous activations. Neuromap models are stacks of
leaky integrate-and-fire (LIF) neurons: each neuron accumulates
incoming current into a membrane potential that leaks over time, and
fires a spike when the potential crosses a threshold, after which it
resets and stays quiet for a short refractory period.

Because the spike function is not differentiable, the SDK trains with
surrogate gradients: a smooth stand-in used only on the backward pass.
This lets you train with ordinary PyTorch autograd while keeping the
forward pass faithful to the hardware.

Inputs are shaped `(batch, time_steps, input_size)`. The time dimension
is where the spiking dynamics unfold, so continuous data must first be
encoded into spike trains (see [Encoding](#encoding) below).

## 1. Chip specification

A `ChipSpec` describes the physical chip: layer topology, weight
bit-width, the default LIF neuron parameters, and optional metadata such
as maximum frequency, supply voltage, and energy per spike.

```python
from neuromap import ChipSpec, NeuronParams

chip = ChipSpec(
    name="custom-chip",
    layers=(32, 16, 8),
    weight_bits=4,
    neuron_params=NeuronParams(tau_m=10.0, v_th=1.0),
)
```

Predefined profiles are available through the `chips` namespace:

```python
from neuromap import chips

chip = chips.NEUROSOC_V1
print(chip.total_neurons)    # 48
print(chip.total_synapses)   # 512
```

`ChipSpec` is a frozen dataclass, so a spec is immutable once created.
Derived quantities (`num_layers`, `total_neurons`, `total_synapses`,
`input_size`, `output_size`) are exposed as properties. See the
[chip API](api/chip.md) for the full reference.

## 2. Network

A `Network` wraps a PyTorch `DynamicSNN` model together with the chip
spec used to build it. It is the central object in the SDK and provides
inference, streaming, persistence, and introspection.

```python
from neuromap import Network, chips

net = Network(chips.NEUROSOC_V1)
```

The underlying model is a stack of LIF layers sized to match the chip
topology. Two options shape the readout:

- `use_decoder` appends a trainable linear layer on the output, useful
  for continuous regression.
- `membrane_readout` makes the final layer emit continuous membrane
  potentials instead of binary spikes.

## Encoding

Real sensor data is continuous, but an SNN consumes spikes. The
`encode_sensor_data` helper converts tensors into spike trains using one
of three strategies:

- **rate** turns each value into a Poisson-like spike frequency over the
  time window. This is the default and the most common choice.
- **latency** encodes each value as the time of a single spike, so
  larger values fire earlier.
- **delta** emits a spike whenever the signal changes by more than a
  threshold, which suits already time-varying inputs.

```python
import neuromap as nm

spikes = nm.encode_sensor_data(features, num_steps=100, method="rate")
```

## 3. Trainer

The `Trainer` runs the training loop so you do not have to write one.
It bundles the pieces that make SNN training stable:

- **LR warmup** linearly ramps the learning rate over the first epochs.
- **LR scheduling** with ReduceLROnPlateau (default) or cosine
  annealing.
- **Early stopping** halts training when the validation loss stops
  improving.
- **Gradient clipping** keeps gradients bounded in deep SNNs.
- **Best checkpoint** restores the best-scoring weights when training
  ends.

```python
from neuromap import Trainer

trainer = Trainer(net, lr=1e-3, epochs=20)
history = trainer.fit(train_loader, val_loader)
```

The default loss is mean squared error; pass `loss_fn` to override it.
See the [trainer API](api/trainer.md) for every option.

## 4. Exporter

The `Exporter` quantizes float32 weights to signed fixed-point matching
the chip's DAC bit-width, then packages them into a `.nmap` archive:

```python
from neuromap import Exporter

Exporter(net).quantize().save("model.nmap")
```

A `.nmap` archive is a ZIP file containing:

- `manifest.json` with the chip spec, topology, and quantization
  metadata.
- `weights/*.npy` with per-layer integer weight and bias arrays.

Reload a bundle back into a `Network` with `Exporter.load_nmap(path)`.
See the [export API](api/export.md) for details.
