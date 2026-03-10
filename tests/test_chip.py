"""Tests for neuromap.chip (ChipSpec, NeuronParams, chips)."""

from __future__ import annotations

import pytest

from neuromap import ChipSpec, NeuronParams, chips


class TestChipSpec:
    def test_neurosoc_v1_exists(self) -> None:
        chip = chips.NEUROSOC_V1
        assert chip.name == "NeuroSoC-v1"
        assert chip.layers == (16, 16, 16)
        assert chip.weight_bits == 4
        assert chip.total_neurons == 48
        assert chip.total_synapses == 16 * 16 + 16 * 16

    def test_custom_chip(self) -> None:
        chip = ChipSpec(name="test", layers=(64, 32, 16), weight_bits=8)
        assert chip.num_layers == 3
        assert chip.input_size == 64
        assert chip.output_size == 16
        assert chip.total_synapses == 64 * 32 + 32 * 16

    def test_invalid_layers(self) -> None:
        with pytest.raises(ValueError, match="at least 2 layers"):
            ChipSpec(name="bad", layers=(16,))

    def test_invalid_weight_bits(self) -> None:
        with pytest.raises(ValueError, match="weight_bits"):
            ChipSpec(name="bad", layers=(8, 8), weight_bits=0)

    def test_round_trip_dict(self) -> None:
        chip = chips.NEUROSOC_V1
        restored = ChipSpec.from_dict(chip.to_dict())
        assert restored.name == chip.name
        assert restored.layers == chip.layers
        assert restored.weight_bits == chip.weight_bits

    def test_chips_list(self) -> None:
        names = chips.list()
        assert "NEUROSOC_V1" in names

    def test_chips_get(self) -> None:
        chip = chips.get("NeuroSoC-v1")
        assert chip is chips.NEUROSOC_V1

    def test_chips_get_unknown(self) -> None:
        with pytest.raises(KeyError):
            chips.get("nonexistent-chip")


class TestNeuronParams:
    def test_defaults(self) -> None:
        np_obj = NeuronParams()
        assert np_obj.tau_m == 10.0
        assert np_obj.v_th == 1.0

    def test_round_trip(self) -> None:
        np_obj = NeuronParams(tau_m=5.0, v_th=0.5)
        restored = NeuronParams.from_dict(np_obj.to_dict())
        assert restored.tau_m == 5.0
        assert restored.v_th == 0.5
