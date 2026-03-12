"""High-level Network builder for Neuromap neuromorphic chips.

A :class:`Network` is the central object in the Neuromap SDK.  It wraps
a :class:`~neuromap._internal.dynamic_snn.DynamicSNN` model with the
:class:`~neuromap.chip.ChipSpec` that describes the target hardware,
providing convenience methods for inference, streaming, persistence and
introspection.

Example::

    from neuromap import Network, chips

    net = Network(chips.NEUROSOC_V1)
    print(net.summary())
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from neuromap._internal.dynamic_snn import DynamicSNN
from neuromap.chip import ChipSpec, NeuronParams


class Network:
    """Chip-aware SNN wrapper.

    A :class:`Network` owns a :class:`DynamicSNN` model and carries the
    :class:`ChipSpec` (or equivalent topology metadata) used to build it.

    Args:
        chip: The chip specification whose topology and neuron parameters
            will be used to construct the underlying SNN.
        use_decoder: Append a trainable linear decoder on the output
            layer.  Useful for continuous regression tasks.
    """

    def __init__(self, chip: ChipSpec, *, use_decoder: bool = True) -> None:
        self._chip = chip
        self._use_decoder = use_decoder
        self._model = self._build_model(chip, use_decoder=use_decoder)

    # -- alternative constructors -------------------------------------------------

    @classmethod
    def from_topology(
        cls,
        layers: list[int] | tuple[int, ...],
        *,
        weight_bits: int = 4,
        neuron_params: Mapping[str, Any] | None = None,
        name: str = "custom",
        use_decoder: bool = True,
    ) -> Network:
        """Build a :class:`Network` from an explicit layer topology.

        This is the escape-hatch for simulation-only experiments that are
        not constrained to a specific physical chip.

        Args:
            layers: Sequence of layer sizes (e.g. ``[32, 16, 8]``).
            weight_bits: Bit-width for quantization metadata.
            neuron_params: Optional dict of LIF neuron parameters.
            name: Name for the synthetic chip spec.
            use_decoder: Whether to include the linear decoder.

        Returns:
            A new :class:`Network` instance.
        """
        np_obj = (
            NeuronParams.from_dict(dict(neuron_params))
            if neuron_params is not None
            else NeuronParams()
        )
        chip = ChipSpec(
            name=name,
            layers=tuple(layers),
            weight_bits=weight_bits,
            neuron_params=np_obj,
        )
        return cls(chip, use_decoder=use_decoder)

    # -- properties ---------------------------------------------------------------

    @property
    def chip(self) -> ChipSpec:
        """The chip specification backing this network."""
        return self._chip

    @property
    def model(self) -> DynamicSNN:
        """The underlying PyTorch :class:`DynamicSNN` module."""
        return self._model

    @property
    def layers(self) -> list[int]:
        """Layer sizes as a list."""
        return list(self._chip.layers)

    @property
    def input_size(self) -> int:
        """Number of input features."""
        return self._chip.input_size

    @property
    def output_size(self) -> int:
        """Number of output features."""
        return self._chip.output_size

    @property
    def device(self) -> torch.device:
        """Device of the first model parameter (or cpu if empty)."""
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # -- inference helpers --------------------------------------------------------

    def to(self, device: str | torch.device) -> Network:
        """Move the model to *device* and return *self*.

        Args:
            device: Target device (e.g. ``"cpu"`` or ``"cuda"``).

        Returns:
            This :class:`Network` instance for chaining.
        """
        self._model.to(device)
        return self

    def infer(
        self,
        x: torch.Tensor,
        *,
        state: Mapping[str, Mapping[str, Any]] | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run rate-coded inference.

        Args:
            x: Input tensor ``(batch, time_steps, input_size)``.
            state: Optional carry-over state for streaming.
            return_state: If ``True``, also return the next state dict.

        Returns:
            Output tensor ``(batch, output_size)``, or a tuple
            ``(output, next_state)`` when *return_state* is set.
        """
        self._model.eval()
        with torch.no_grad():
            return self._model(x, state=state, return_state=return_state)

    def infer_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Run per-frame sequence inference.

        Args:
            x: Input tensor ``(batch, frames, input_size)``.

        Returns:
            Output tensor ``(batch, frames, output_size)``.
        """
        self._model.eval()
        with torch.no_grad():
            return self._model.forward_sequence(x)

    def infer_stream(
        self,
        x_chunk: torch.Tensor,
        state: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Stateful streaming inference — process one chunk at a time.

        Args:
            x_chunk: Input chunk ``(batch, time_steps, input_size)``.
            state: State from the previous call, or ``None`` for the
                first chunk.

        Returns:
            Tuple of ``(output, next_state)``.
        """
        result = self.infer(x_chunk, state=state, return_state=True)
        assert isinstance(result, tuple)
        return result

    # -- introspection ------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the network."""
        lines: list[str] = [
            f"Network '{self._chip.name}'",
            f"  Topology      : {' -> '.join(str(s) for s in self._chip.layers)}",
            f"  Weight bits   : {self._chip.weight_bits}",
            f"  Total neurons : {self._chip.total_neurons}",
            f"  Total synapses: {self._chip.total_synapses}",
            f"  Decoder       : {'yes' if self._use_decoder else 'no'}",
        ]
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        lines.append(f"  Parameters    : {total_params:,} (trainable: {trainable:,})")
        if self._chip.energy_per_spike_fj is not None:
            lines.append(f"  Energy/spike  : {self._chip.energy_per_spike_fj} fJ")
        return "\n".join(lines)

    # -- deploy -------------------------------------------------------------------

    def deploy(self, board: Any, *, verify: bool = True) -> None:
        """One-liner deploy: quantize, export, and program the board.

        This is the "Arduino upload" experience::

            board = Board.connect()
            net.deploy(board)

        Args:
            board: A :class:`~neuromap.board.Board` instance (real or mock).
            verify: Whether to verify weights on-chip after programming.

        Raises:
            RuntimeError: If verification fails.
        """
        from neuromap.export import Exporter

        exporter = Exporter(self)
        exporter.quantize()
        nmap_bytes = exporter.to_bytes()
        board.program(nmap_bytes)
        if verify:
            ok = board.verify()
            if not ok:
                raise RuntimeError("Weight verification failed after deploy.")

    # -- persistence --------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the network (model weights + chip spec) to a ``.pt`` file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "chip_spec": self._chip.to_dict(),
                "use_decoder": self._use_decoder,
                "model_state_dict": self._model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> Network:
        """Load a network previously saved with :meth:`save`.

        Args:
            path: Path to the ``.pt`` checkpoint.
            device: Device to load the model onto.

        Returns:
            A restored :class:`Network` instance.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        chip = ChipSpec.from_dict(checkpoint["chip_spec"])
        use_decoder = checkpoint.get("use_decoder", True)
        net = cls(chip, use_decoder=use_decoder)
        net._model.load_state_dict(checkpoint["model_state_dict"])
        net.to(device)
        return net

    # -- internal -----------------------------------------------------------------

    @staticmethod
    def _build_model(chip: ChipSpec, *, use_decoder: bool) -> DynamicSNN:
        neuron_kwargs = chip.neuron_params.to_dict()
        return DynamicSNN(
            layers=list(chip.layers),
            neuron_kwargs=neuron_kwargs,
            use_decoder=use_decoder,
        )

    def __repr__(self) -> str:
        topo = "->".join(str(s) for s in self._chip.layers)
        return f"Network(chip='{self._chip.name}', topology={topo}, bits={self._chip.weight_bits})"
