"""N-layer dynamic SNN architecture generalising NeuromapSNN."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from neuromap_sim.core.lif import LIFLayer

DynamicModelState = dict[str, dict[str, torch.Tensor | int]]


class DynamicSNN(nn.Module):
    """Fully-connected SNN with an arbitrary number of LIF layers.

    This generalises :class:`NeuromapSNN` (which is fixed at 2 layers) so
    that the SDK can build networks matching any :class:`ChipSpec` topology.

    Parameters
    ----------
    layers:
        Sequence of layer sizes, e.g. ``[16, 16, 16]`` or ``[400, 128, 10]``.
        Must contain at least 2 elements (input size + output size).
    neuron_kwargs:
        Optional keyword arguments forwarded to every :class:`LIFLayer`
        constructor (``tau_m``, ``v_th``, ``v_reset``, ``t_ref``, etc.).
    use_decoder:
        If ``True``, append a trainable linear decoder from the last layer
        to itself (useful for continuous regression outputs like denoising).
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

    # -- properties ---------------------------------------------------------------

    @property
    def input_size(self) -> int:
        return self.layer_sizes[0]

    @property
    def output_size(self) -> int:
        return self.layer_sizes[-1]

    @property
    def num_snn_layers(self) -> int:
        return len(self.snn_layers)

    # -- forward methods ----------------------------------------------------------

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame continuous predictions (batch, frames, output_size).

        Useful for sequence-to-sequence tasks like denoising where a
        rate-coded collapse would lose temporal resolution.
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

        Parameters
        ----------
        x:
            Input spike tensor of shape ``(batch, time_steps, input_size)``.
        state:
            Optional dict mapping ``"layer_0"``, ``"layer_1"``, … to per-layer
            LIF states for stateful / streaming inference.
        return_state:
            If ``True``, return ``(rate_output, next_state)`` instead of just
            ``rate_output``.

        Returns
        -------
        rate_output:
            Tensor of shape ``(batch, output_size)`` — spike counts summed
            across the time axis.
        next_state (optional):
            Updated layer states for the next chunk.
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

