import sys
from pathlib import Path

# Ensure neuro-sim is on the path regardless of working directory
_NEURO_SIM = str(Path(__file__).resolve().parents[3])
if _NEURO_SIM not in sys.path:
    sys.path.insert(0, _NEURO_SIM)

from neuromap_sim.sdk import Network, Trainer, Exporter, chips
import torch
from torch.utils.data import DataLoader, TensorDataset

# 1. Pick your chip
chip = chips.NEUROSOC_V1
print('=== Chip ===')
print(f'  {chip.name}: layers={list(chip.layers)}, {chip.weight_bits}-bit, {chip.total_neurons} neurons, {chip.total_synapses} synapses')
print()

# 2. Build a network
net = Network(chip, use_decoder=False)
print('=== Network ===')
print(net.summary())
print()

# 3. Dummy spike data
torch.manual_seed(42)
x_train = (torch.rand(100, 20, 16) > 0.7).float()
y_train = x_train.sum(dim=1)
x_val = (torch.rand(30, 20, 16) > 0.7).float()
y_val = x_val.sum(dim=1)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16)

# 4. Train
print('=== Training ===')
trainer = Trainer(net, lr=1e-3, epochs=10, device='cpu')
history = trainer.fit(train_loader, val_loader)
print()

# 5. Inference
sample = (torch.rand(1, 20, 16) > 0.7).float()
output = net.infer(sample)
print(f'=== Inference ===')
print(f'  Input:  {sample.shape}  (1 sample, 20 timesteps, 16 inputs)')
print(f'  Output: {output.shape}  (1 sample, 16 outputs)')
print()

# 6. Quantize + export
print('=== Export ===')
exporter = Exporter(net)
exporter.quantize(bits=4)
weights = exporter.to_weight_map()
for name, arr in weights.items():
    print(f'  {name}: shape={arr.shape}, range=[{arr.min()}, {arr.max()}]')
exporter.save('/tmp/neurosoc_v1_model.nmap')
print('  Saved neurosoc_v1_model.nmap')
print()

# 7. Reload
print('=== Reload ===')
net2 = Exporter.load_nmap('/tmp/neurosoc_v1_model.nmap')
print(net2.summary())