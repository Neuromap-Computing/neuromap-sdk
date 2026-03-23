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
