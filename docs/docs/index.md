# Neuromap SDK

**Program neuromorphic chips with Python.**

Neuromap is a Python SDK for building, training, quantizing, and
exporting spiking neural networks (SNNs) for Neuromap neuromorphic
hardware. It gives you a PyTorch-based workflow that goes from an
untrained model to a portable `.nmap` hardware bundle.

## Why Neuromap

- **Chip-aware networks.** Define networks that match your target
  chip's topology, weight bit-width, and neuron parameters, so what you
  train is what you export.
- **Batteries-included training.** A single `Trainer` class handles LR
  warmup, scheduling, gradient clipping, early stopping, and best-model
  checkpointing.
- **Hardware export.** Quantize weights to the chip's DAC bit-width and
  package everything into a portable `.nmap` archive.
- **Streaming inference.** Stateful chunk-by-chunk processing for
  real-time applications.

## Install

```bash
pip install neuromap
```

## The pipeline in one screen

```python
from neuromap import Network, Trainer, Exporter, chips

# 1. Build a network matching the NeuroSoC-v1 chip
net = Network(chips.NEUROSOC_V1)
print(net.summary())

# 2. Train it (bring your own DataLoader)
trainer = Trainer(net, lr=1e-3, epochs=20)
history = trainer.fit(train_loader, val_loader)

# 3. Export a hardware bundle
Exporter(net).quantize().save("model.nmap")
```

## Where to go next

- [Quickstart](quickstart.md) walks through the same pipeline step by
  step, with runnable snippets.
- [Concepts](concepts.md) explains the four-stage model behind the SDK:
  chip spec, network, trainer, and exporter.
- [API Reference](api/network.md) is the full class-by-class reference,
  generated from the source docstrings.
