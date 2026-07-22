# Quickstart

This page takes you through the full Neuromap workflow: install, build a
network, run inference, train, and export a hardware bundle.

## Install

```bash
pip install neuromap
```

For development against a checkout of the repository, use an editable
install with the test dependencies:

```bash
cd neuro-sim
pip install -e ".[dev]"
```

## Build a network

Every network starts from a **chip specification** that describes the
hardware topology. Use a predefined profile from the `chips` namespace:

```python
from neuromap import Network, chips

net = Network(chips.NEUROSOC_V1)
print(net.summary())
```

`summary()` prints the topology, weight bit-width, neuron and synapse
counts, parameter count, and (when the chip advertises it) the energy
per spike.

For simulation-only experiments that are not tied to a physical chip,
build a network from an explicit topology:

```python
net = Network.from_topology([400, 128, 10], name="my-sim")
```

Two constructor flags change the output behaviour:

- `use_decoder=True` (the default) appends a trainable linear decoder on
  the output layer, which helps continuous regression tasks.
- `membrane_readout=True` makes the last layer emit continuous membrane
  potentials instead of binary spikes. Use it for regression tasks such
  as denoising.

## Run inference

Inputs are spike tensors shaped `(batch, time_steps, input_size)`:

```python
import torch

x = torch.rand(1, 50, 16)   # (batch, time_steps, input_size)
y = net.infer(x)            # (batch, output_size)
```

To turn raw continuous data into spikes, use `encode_sensor_data`:

```python
import neuromap as nm

spikes = nm.encode_sensor_data(raw_features, num_steps=100, method="rate")
y = net.infer(spikes)
```

For real-time use, process the signal chunk by chunk and carry the
neuron state forward between calls:

```python
y1, state = net.infer_stream(chunk_1)
y2, state = net.infer_stream(chunk_2, state=state)
```

## Train

`Trainer` wraps a full training loop. Each batch from your `DataLoader`
must be an `(x, y)` tuple, or an object exposing `.x` and `.y`:

```python
from neuromap import Trainer

trainer = Trainer(net, lr=1e-3, epochs=20, device="cpu")
history = trainer.fit(train_loader, val_loader)
```

The trainer applies linear LR warmup, an LR scheduler (plateau by
default), gradient clipping, and early stopping. When training ends it
restores the best-scoring weights automatically. Inspect the run with
`history.to_list()`, and score a held-out loader with
`trainer.evaluate(test_loader)`.

Before training you can swap in a different surrogate gradient:

```python
import neuromap as nm

nm.compile_model(net, surrogate="atan")   # or "fast_sigmoid", etc.
```

## Export

Quantize the trained weights to the chip's bit-width and write a `.nmap`
bundle:

```python
from neuromap import Exporter

Exporter(net).quantize().save("model.nmap")
```

A `.nmap` file is a ZIP archive containing a JSON manifest and per-layer
weight arrays. Reload it later with `Exporter.load_nmap("model.nmap")`.
See the [Exporter API](api/export.md) for the archive layout.
