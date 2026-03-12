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
