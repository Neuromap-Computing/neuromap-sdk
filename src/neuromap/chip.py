"""Chip hardware profile definitions for Neuromap neuromorphic processors.

This module defines :class:`ChipSpec` — the specification of a physical
neuromorphic chip — and :class:`NeuronParams` — the default LIF neuron
parameters baked into the hardware.  The :class:`chips` namespace
provides pre-defined profiles for every Neuromap chip revision.

Example::

    from neuromap import chips

    chip = chips.NEUROSOC_V1
    print(chip.total_neurons)   # 48
    print(chip.total_synapses)  # 512
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NeuronParams:
    """Default LIF neuron parameters matching the hardware defaults.

    Args:
        tau_m: Membrane time constant.
        rm: Membrane resistance.
        dt: Simulation time-step size.
        v_th: Firing threshold voltage.
        v_reset: Reset voltage after a spike.
        t_ref: Refractory period in time-steps.
    """

    tau_m: float = 10.0
    rm: float = 1.0
    dt: float = 1.0
    v_th: float = 1.0
    v_reset: float = 0.0
    t_ref: int = 2

    @property
    def beta(self) -> float:
        """Membrane decay rate for snnTorch: ``1 - dt / tau_m``."""
        return 1.0 - self.dt / self.tau_m

    def to_snntorch_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for :class:`NeuromapLIF`.

        Returns:
            Dictionary with ``beta``, ``threshold``, ``v_reset``, and
            ``t_ref`` keys.
        """
        return {
            "beta": self.beta,
            "threshold": self.v_th,
            "v_reset": self.v_reset,
            "t_ref": self.t_ref,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
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
        """Deserialise from a plain dictionary.

        Args:
            data: Dictionary with neuron parameter keys.

        Returns:
            A new :class:`NeuronParams` instance.
        """
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

    Args:
        name: Human-readable chip identifier (e.g. ``"NeuroSoC-v1"``).
        layers: Number of neurons per layer.  For example
            ``(16, 16, 16)`` describes the 3-layer topology of the first
            NeuroSoC ASIC.
        weight_bits: Bit-width of the synaptic weight DACs.
        neuron_params: Default LIF neuron parameters for this chip.
        max_frequency_khz: Maximum neuron spiking frequency in kHz.
        supply_voltage_mv: Operating supply voltage in millivolts.
        energy_per_spike_fj: Energy per spike in femtojoules.
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
        """Number of neurons in the first (input) layer."""
        return self.layers[0]

    @property
    def output_size(self) -> int:
        """Number of neurons in the last (output) layer."""
        return self.layers[-1]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the chip spec to a plain dictionary."""
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
        """Deserialise from a plain dictionary.

        Args:
            data: Dictionary with chip specification keys.

        Returns:
            A new :class:`ChipSpec` instance.
        """
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
    """NeuroSoC v1 — 48 LIF neurons, 512 4-bit synapses, 28 nm CMOS ASIC."""

    @classmethod
    def list(cls) -> list[str]:
        """Return names of all predefined chip profiles."""
        return [name for name, val in vars(cls).items() if isinstance(val, ChipSpec)]

    @classmethod
    def get(cls, name: str) -> ChipSpec:
        """Look up a predefined chip profile by name (case-insensitive).

        Args:
            name: The chip name to look up.

        Returns:
            The matching :class:`ChipSpec`.

        Raises:
            KeyError: If no chip with the given name is registered.
        """
        for _attr_name, val in vars(cls).items():
            if isinstance(val, ChipSpec) and val.name.lower() == name.lower():
                return val
        available = cls.list()
        raise KeyError(f"Unknown chip '{name}'. Available: {available}")
