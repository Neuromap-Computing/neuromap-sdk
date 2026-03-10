"""Tests for neuromap.export (Exporter)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from neuromap import Exporter, Network


class TestExporter:
    def test_quantize(self) -> None:
        net = Network.from_topology([16, 8, 4])
        exporter = Exporter(net)
        exporter.quantize(bits=4)
        wmap = exporter.to_weight_map()
        weight_keys = [k for k in wmap if "weight" in k]
        assert len(weight_keys) > 0
        for k in weight_keys:
            assert wmap[k].dtype.kind == "i"
            assert int(wmap[k].max()) <= 7
            assert int(wmap[k].min()) >= -7

    def test_save_load_nmap_roundtrip(self) -> None:
        net = Network.from_topology([16, 8, 4], use_decoder=False)
        x = torch.rand(1, 5, 16)
        y_original = net.infer(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            nmap_path = Path(tmpdir) / "test.nmap"
            exporter = Exporter(net)
            exporter.save(nmap_path)
            net2 = Exporter.load_nmap(nmap_path)

        y_loaded = net2.infer(x)
        assert torch.allclose(y_original, y_loaded, atol=1e-5)

    def test_save_load_quantized_nmap(self) -> None:
        net = Network.from_topology([16, 8, 4])

        with tempfile.TemporaryDirectory() as tmpdir:
            nmap_path = Path(tmpdir) / "quantized.nmap"
            exporter = Exporter(net)
            exporter.quantize(bits=4)
            exporter.save(nmap_path)
            net2 = Exporter.load_nmap(nmap_path)
            assert net2.chip.name == "custom"

    def test_weight_map_without_quantize(self) -> None:
        net = Network.from_topology([16, 8, 4])
        exporter = Exporter(net)
        wmap = exporter.to_weight_map()
        weight_keys = [k for k in wmap if "weight" in k]
        assert len(weight_keys) > 0
        for k in weight_keys:
            assert wmap[k].dtype.kind == "f"

    def test_invalid_network_type(self) -> None:
        with pytest.raises(TypeError, match="Network"):
            Exporter("not_a_network")  # type: ignore[arg-type]
