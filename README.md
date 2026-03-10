# Neuromap SDK

Python SDK for programming Neuromap neuromorphic chips.

## Installation

```bash
pip install neuromap
```

For development:

```bash
pip install -e ".[dev]"
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

## Running Tests

```bash
pytest tests/
```
