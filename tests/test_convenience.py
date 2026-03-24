"""Tests for convenience functions (compile_model, quantize_and_export_to_spi)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from neuromap import Network, chips, compile_model, quantize_and_export_to_spi
from neuromap._internal.lif import NeuromapLIF


class TestCompileModel:
    def test_compile_changes_surrogate(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        compile_model(net, surrogate="fast_sigmoid")
        for module in net.model.modules():
            if isinstance(module, NeuromapLIF):
                # Check the spike_grad was replaced
                assert module.lif.spike_grad is not None

    def test_compile_default_atan(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        compile_model(net)  # default is "atan"


class TestQuantizeAndExportToSpi:
    def test_export_to_bytes(self) -> None:
        net = Network(chips.NEUROSOC_V1, use_decoder=False)
        data = quantize_and_export_to_spi(net, bit_resolution=4)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_export_to_file(self) -> None:
        net = Network(chips.NEUROSOC_V1, use_decoder=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.nmap"
            result = quantize_and_export_to_spi(net, path=path)
            assert result is None
            assert path.exists()
            assert path.stat().st_size > 0

    def test_roundtrip_via_convenience(self) -> None:
        from neuromap import Exporter

        net = Network(chips.NEUROSOC_V1, use_decoder=False)
        x = torch.rand(1, 10, 16)
        _ = net.infer(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.nmap"
            quantize_and_export_to_spi(net, path=path)
            net2 = Exporter.load_nmap(path)
            assert net2.chip.name == "NeuroSoC-v1"
