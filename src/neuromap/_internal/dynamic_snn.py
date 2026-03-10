"""N-layer dynamic SNN architecture.

Generalises the earlier fixed-two-layer model so that the SDK can build
networks matching any :class:`~neuromap.chip.ChipSpec` topology.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from neuromap._internal.lif import LIFLayer

DynamicModelState = dict[str, dict[str, torch.Tensor | int]]
"""Type alias for the per-model state dictionary (one entry per layer)."""


class DynamicSNN(nn.Module):
    """Fully-connected SNN with an arbitrary number of LIF layers.

    Args:
        layers: Sequence of layer sizes, e.g. ``[16, 16, 16]`` or
            ``[400, 128, 10]``.  Must contain at least 2 elements
            (input size + output size).
        neuron_kwargs: Optional keyword arguments forwarded to every
            :class:`~neuromap._internal.lif.LIFLayer` constructor
            (``tau_m``, ``v_th``, ``v_reset``, ``t_ref``, etc.).
        use_decoder: If ``True``, append a trainable linear decoder
            from the last layer to itself (useful for continuous
            regression outputs like denoising).
    """

    def __init__(
        self,
        layers: list[int] | tuple[int, ...],
        *,
        neuron_kwargs: Mapping[str, Any] | None = None,
        use_decoder: bool = True,
    ) -> None:
        super().__init__()
        if len(layers) < 2:
            raise ValueError("DynamicSNN requires at least 2 layer sizes (input + output).")
        self.layer_sizes = list(layers)
        self._neuron_kwargs = dict(neuron_kwargs or {})

        self.snn_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.snn_layers.append(
                LIFLayer(layers[i], layers[i + 1], **self._neuron_kwargs)
            )

        self.decoder: nn.Linear | None = None
        if use_decoder:
            self.decoder = nn.Linear(layers[-1], layers[-1])

    @property
    def input_size(self) -> int:
        """Number of input features."""
        return self.layer_sizes[0]

    @property
    def output_size(self) -> int:
        """Number of output features."""
        return self.layer_sizes[-1]

    @property
    def num_snn_layers(self) -> int:
        """Number of LIF layers in the network."""
        return len(self.snn_layers)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame continuous predictions ``(batch, frames, output_size)``.

        Useful for sequence-to-sequence tasks like denoising where a
        rate-coded collapse would lose temporal resolution.

        Args:
            x: Input tensor ``(batch, frames, input_size)``.

        Returns:
            Output tensor ``(batch, frames, output_size)``.
        """
        out = x
        for layer in self.snn_layers:
            out, _ = layer(out, return_state=True)
        if self.decoder is not None:
            out = self.decoder(out)
        return out

    def forward(
        self,
        x: torch.Tensor,
        state: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DynamicModelState]:
        """Rate-coded forward pass.

        Args:
            x: Input spike tensor ``(batch, time_steps, input_size)``.
            state: Optional dict mapping ``"layer_0"``, ``"layer_1"``, …
                to per-layer LIF states for stateful / streaming inference.
            return_state: If ``True``, return ``(rate_output, next_state)``
                instead of just ``rate_output``.

        Returns:
            Tensor ``(batch, output_size)`` with spike counts summed across
            the time axis, or a tuple ``(rate_output, next_state)`` when
            *return_state* is set.
        """
        next_states: DynamicModelState = {}
        out = x
        for i, layer in enumerate(self.snn_layers):
            layer_key = f"layer_{i}"
            layer_state = None if state is None else state.get(layer_key)
            out, lstate = layer(out, layer_state, return_state=True)
            next_states[layer_key] = lstate

        rate_output = out.sum(dim=1)
        if not return_state:
            return rate_output
        return rate_output, next_states
