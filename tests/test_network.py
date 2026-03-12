"""Tests for neuromap.network (Network)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from neuromap import Network, chips


class TestNetwork:
    def test_from_chip(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        assert net.input_size == 16
        assert net.output_size == 16
        assert net.layers == [16, 16, 16]

    def test_from_topology(self) -> None:
        net = Network.from_topology([32, 16, 8])
        assert net.input_size == 32
        assert net.output_size == 8
        assert net.chip.name == "custom"

    def test_infer(self) -> None:
        net = Network.from_topology([32, 16, 8], use_decoder=False)
        x = torch.rand(2, 5, 32)
        y = net.infer(x)
        assert y.shape == (2, 8)

    def test_infer_sequence(self) -> None:
        net = Network.from_topology([32, 16, 8], use_decoder=False)
        x = torch.rand(2, 5, 32)
        y = net.infer_sequence(x)
        assert y.shape == (2, 5, 8)

    def test_infer_stream(self) -> None:
        net = Network.from_topology([32, 16, 8], use_decoder=False)
        x1 = torch.rand(1, 3, 32)
        x2 = torch.rand(1, 4, 32)
        y1, state = net.infer_stream(x1)
        y2, state2 = net.infer_stream(x2, state=state)
        assert y1.shape == (1, 8)
        assert y2.shape == (1, 8)
        assert int(state2["layer_0"]["step"]) == 7

    def test_summary(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        s = net.summary()
        assert "NeuroSoC-v1" in s
        assert "16 -> 16 -> 16" in s

    def test_save_load_roundtrip(self) -> None:
        net = Network.from_topology([32, 16, 8])
        x = torch.rand(1, 5, 32)
        y_original = net.infer(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_net.pt"
            net.save(path)
            net2 = Network.load(path)

        y_loaded = net2.infer(x)
        assert torch.allclose(y_original, y_loaded, atol=1e-6)
        assert net2.chip.name == "custom"
        assert net2.layers == [32, 16, 8]

    def test_repr(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        r = repr(net)
        assert "NeuroSoC-v1" in r
