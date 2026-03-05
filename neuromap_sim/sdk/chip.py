"""Chip hardware profile definitions for Neuromap neuromorphic processors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NeuronParams:
    """Default LIF neuron parameters matching the hardware defaults."""

    tau_m: float = 10.0
    rm: float = 1.0
    dt: float = 1.0
    v_th: float = 1.0
    v_reset: float = 0.0
    t_ref: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_m": self.tau_m,
            "rm": self.rm,
            "dt": self.dt,
            "v_th": self.v_th,
            "v_reset": self.v_reset,
            "t_ref": self.t_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeuronParams:
        return cls(
            tau_m=float(data.get("tau_m", 10.0)),
            rm=float(data.get("rm", 1.0)),
            dt=float(data.get("dt", 1.0)),
            v_th=float(data.get("v_th", 1.0)),
            v_reset=float(data.get("v_reset", 0.0)),
            t_ref=int(data.get("t_ref", 2)),
        )


@dataclass(frozen=True)
class ChipSpec:
    """Hardware specification of a Neuromap neuromorphic chip.

    Parameters
    ----------
    name:
        Human-readable chip identifier (e.g. ``"NeuroSoC-v1"``).
    layers:
        Number of neurons per layer.  For example ``[16, 16, 16]`` describes
        the 3-layer topology of the first NeuroSoC ASIC.
    weight_bits:
        Bit-width of the synaptic weight DACs (e.g. 4 for the v1 chip).
    neuron_params:
        Default LIF neuron parameters to use when building a network for
        this chip.
    max_frequency_khz:
        Maximum neuron spiking frequency in kHz (optional metadata).
    supply_voltage_mv:
        Operating supply voltage in millivolts (optional metadata).
    energy_per_spike_fj:
        Energy per spike in femtojoules (optional metadata).
    """

    name: str
    layers: tuple[int, ...] = (16, 16, 16)
    weight_bits: int = 4
    neuron_params: NeuronParams = field(default_factory=NeuronParams)
    max_frequency_khz: float | None = None
    supply_voltage_mv: int | None = None
    energy_per_spike_fj: float | None = None

    def __post_init__(self) -> None:
        if len(self.layers) < 2:
            raise ValueError("ChipSpec requires at least 2 layers (input + output).")
        if self.weight_bits < 1:
            raise ValueError("weight_bits must be >= 1.")
        for i, size in enumerate(self.layers):
            if size < 1:
                raise ValueError(f"Layer {i} size must be >= 1, got {size}.")

    # -- convenience properties ---------------------------------------------------

    @property
    def num_layers(self) -> int:
        """Number of layers in the network topology."""
        return len(self.layers)

    @property
    def total_neurons(self) -> int:
        """Total neuron count across all layers."""
        return sum(self.layers)

    @property
    def total_synapses(self) -> int:
        """Total synapse count (fully-connected between consecutive layers)."""
        return sum(a * b for a, b in zip(self.layers[:-1], self.layers[1:]))

    @property
    def input_size(self) -> int:
        return self.layers[0]

    @property
    def output_size(self) -> int:
        return self.layers[-1]

    # -- serialization ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "layers": list(self.layers),
            "weight_bits": self.weight_bits,
            "neuron_params": self.neuron_params.to_dict(),
        }
        if self.max_frequency_khz is not None:
            data["max_frequency_khz"] = self.max_frequency_khz
        if self.supply_voltage_mv is not None:
            data["supply_voltage_mv"] = self.supply_voltage_mv
        if self.energy_per_spike_fj is not None:
            data["energy_per_spike_fj"] = self.energy_per_spike_fj
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChipSpec:
        neuron_params = data.get("neuron_params", {})
        if isinstance(neuron_params, dict):
            neuron_params = NeuronParams.from_dict(neuron_params)
        return cls(
            name=data["name"],
            layers=tuple(data["layers"]),
            weight_bits=int(data.get("weight_bits", 4)),
            neuron_params=neuron_params,
            max_frequency_khz=data.get("max_frequency_khz"),
            supply_voltage_mv=data.get("supply_voltage_mv"),
            energy_per_spike_fj=data.get("energy_per_spike_fj"),
        )


# ---------------------------------------------------------------------------
# Predefined chip profiles
# ---------------------------------------------------------------------------

class chips:  # noqa: N801  – lowercase class name used as a namespace
    """Registry of predefined chip profiles."""

    NEUROSOC_V1 = ChipSpec(
        name="NeuroSoC-v1",
        layers=(16, 16, 16),
        weight_bits=4,
        neuron_params=NeuronParams(
            tau_m=10.0,
            rm=1.0,
            dt=1.0,
            v_th=1.0,
            v_reset=0.0,
            t_ref=2,
        ),
        max_frequency_khz=300.0,
        supply_voltage_mv=250,
        energy_per_spike_fj=1.61,
    )
    """NeuroSoC v1  48 LIF neurons, 512 4-bit synapses, 28 nm CMOS ASIC."""

    @classmethod
    def list(cls) -> list[str]:
        """Return names of all predefined chip profiles."""
        return [
            name
            for name, val in vars(cls).items()
            if isinstance(val, ChipSpec)
        ]

    @classmethod
    def get(cls, name: str) -> ChipSpec:
        """Look up a predefined chip profile by name (case-insensitive)."""
        for _attr_name, val in vars(cls).items():
            if isinstance(val, ChipSpec) and val.name.lower() == name.lower():
                return val
        available = cls.list()
        raise KeyError(f"Unknown chip '{name}'. Available: {available}")

