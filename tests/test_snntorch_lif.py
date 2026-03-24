"""Tests for the snnTorch-backed NeuromapLIF layer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from neuromap import Network, chips
from neuromap._internal.lif import NeuromapLIF
from neuromap.chip import NeuronParams


class TestNeuromapLIF:
    def test_forward_shape(self) -> None:
        layer = NeuromapLIF(16, 8)
        x = torch.rand(2, 10, 16)
        out = layer(x)
        assert out.shape == (2, 10, 8)

    def test_forward_returns_binary_spikes(self) -> None:
        layer = NeuromapLIF(16, 8)
        x = torch.rand(2, 10, 16)
        out = layer(x)
        unique = torch.unique(out)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_state_carry(self) -> None:
        layer = NeuromapLIF(8, 4)
        x1 = torch.rand(1, 5, 8)
        x2 = torch.rand(1, 3, 8)
        _, state1 = layer(x1, return_state=True)
        assert state1["step"] == 5
        _, state2 = layer(x2, state=state1, return_state=True)
        assert state2["step"] == 8

    def test_refractory_period(self) -> None:
        layer = NeuromapLIF(4, 4, beta=0.5, threshold=0.1, t_ref=100)
        # High input to force spike at t=0, then refractory should suppress
        x = torch.ones(1, 10, 4) * 10.0
        out = layer(x)
        # After the first spike, refractory should block subsequent spikes
        spike_counts = out.sum(dim=1).squeeze()
        # With t_ref=100, only the first timestep can spike
        assert spike_counts.max().item() <= 2.0

    def test_init_state(self) -> None:
        layer = NeuromapLIF(8, 4, v_reset=0.5, t_ref=3)
        state = layer.init_state(2, device=torch.device("cpu"))
        assert state["mem"].shape == (2, 4)
        assert state["step"] == 0
        assert (state["mem"] == 0.5).all()
        assert (state["last_spike_time"] == -3.0).all()

    def test_from_neuron_params(self) -> None:
        params = NeuronParams(tau_m=10.0, rm=1.0, dt=1.0, v_th=1.0, v_reset=0.0, t_ref=2)
        layer = NeuromapLIF.from_neuron_params(16, 8, params)
        assert layer.lif.beta == 0.9
        assert layer.lif.threshold == 1.0
        assert layer.t_ref == 2

    def test_from_neuron_params_custom(self) -> None:
        params = NeuronParams(tau_m=20.0, rm=1.0, dt=2.0, v_th=0.5, v_reset=0.1, t_ref=5)
        layer = NeuromapLIF.from_neuron_params(8, 4, params)
        assert abs(layer.lif.beta - 0.9) < 1e-6  # 1 - 2/20 = 0.9
        assert layer.lif.threshold == 0.5
        assert layer.v_reset == 0.1
        assert layer.t_ref == 5

    def test_output_mem_continuous(self) -> None:
        layer = NeuromapLIF(16, 8, output_mem=True)
        x = torch.rand(2, 10, 16)
        out = layer(x)
        assert out.shape == (2, 10, 8)
        # Membrane potentials should be continuous, not binary
        unique = torch.unique(out)
        assert len(unique) > 2

    def test_output_mem_vs_spikes_shape(self) -> None:
        layer_spk = NeuromapLIF(8, 4, output_mem=False)
        layer_mem = NeuromapLIF(8, 4, output_mem=True)
        # Share weights
        layer_mem.load_state_dict(layer_spk.state_dict())
        x = torch.rand(1, 5, 8)
        out_spk = layer_spk(x)
        out_mem = layer_mem(x)
        assert out_spk.shape == out_mem.shape


class TestMembraneReadoutNetwork:
    def test_network_membrane_readout(self) -> None:
        from neuromap import Network, chips

        net = Network(chips.NEUROSOC_V1, use_decoder=True, membrane_readout=True)
        x = torch.rand(2, 10, 16)
        y = net.infer_sequence(x)
        assert y.shape == (2, 10, 16)
        # Output should be continuous (not just 0/1 mapped through decoder)
        assert len(torch.unique(y)) > 2

    def test_save_load_membrane_readout(self) -> None:
        net = Network(chips.NEUROSOC_V1, membrane_readout=True)
        x = torch.rand(1, 5, 16)
        y_original = net.infer(x)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            net.save(path)
            net2 = Network.load(path)
        y_loaded = net2.infer(x)
        assert torch.allclose(y_original, y_loaded, atol=1e-6)
        assert net2._membrane_readout is True
