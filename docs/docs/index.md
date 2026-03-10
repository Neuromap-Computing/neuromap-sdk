# Neuromap SDK

**Program neuromorphic chips with Python.**

Neuromap is a Python SDK for building, training, quantizing, and
exporting spiking neural networks (SNNs) that run on Neuromap
neuromorphic hardware.

## Features

- **Chip-aware networks** — define networks that match your target
  chip's topology and neuron parameters.
- **Built-in training loop** — gradient clipping, LR warmup, early
  stopping, and checkpointing out of the box.
- **Hardware export** — quantize weights to match DAC bit-widths and
  package into `.nmap` archives.
- **Streaming inference** — stateful chunk-by-chunk processing for
  real-time applications.

## Quick install

```bash
pip install neuromap
```

## Minimal example

```python
from neuromap import Network, Trainer, Exporter, chips

# Build a network matching the NeuroSoC-v1 chip
net = Network(chips.NEUROSOC_V1)

# Train (assuming you have a DataLoader)
trainer = Trainer(net, lr=1e-3, epochs=20)
history = trainer.fit(train_loader, val_loader)

# Export for deployment
Exporter(net).quantize().save("model.nmap")
```
