# Quickstart

## Installation

```bash
pip install neuromap
```

For development (editable install with test dependencies):

```bash
cd neuro-sim
pip install -e ".[dev]"
```

## Build a network

Every network starts from a **chip specification** that describes the
hardware topology: 

```python
from neuromap import Network, chips

# Use a predefined chip profile
net = Network(chips.NEUROSOC_V1)
print(net.summary())
```

Or define a custom topology for simulation:

```python
net = Network.from_topology([400, 128, 10], name="my-sim")
```

## Run inference

```python
import torch

x = torch.rand(1, 50, 16)  # (batch, time_steps, input_size)
y = net.infer(x)            # (batch, output_size)
```

For streaming (stateful chunk-by-chunk):

```python
y1, state = net.infer_stream(chunk_1)
y2, state = net.infer_stream(chunk_2, state=state)
```

## Train

```python
from neuromap import Trainer

trainer = Trainer(net, lr=1e-3, epochs=20, device="cpu")
history = trainer.fit(train_loader, val_loader)
```

## Export

```python
from neuromap import Exporter

Exporter(net).quantize().save("model.nmap")
```

The `.nmap` file is a ZIP archive containing a JSON manifest and
per-layer weight arrays, ready for chip deployment.
