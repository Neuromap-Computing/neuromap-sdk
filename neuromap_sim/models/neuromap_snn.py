"""Neuromap SNN architecture definition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from neuromap_sim.core.lif import LIFLayer

ModelState = dict[str, dict[str, torch.Tensor | int]]


class NeuromapSNN(nn.Module):
    """Two-layer SNN with rate-coded output."""

    def __init__(
        self,
        *,
        input_size: int = 400,
        hidden_size: int = 128,
        output_size: int = 10,
        layer1_kwargs: Mapping[str, Any] | None = None,
        layer2_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer1 = LIFLayer(input_size, hidden_size, **dict(layer1_kwargs or {}))
        self.layer2 = LIFLayer(hidden_size, output_size, **dict(layer2_kwargs or {}))

    def forward(
        self,
        x: torch.Tensor,
        state: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelState]:
        layer1_state = None if state is None else state.get("layer1")
        layer2_state = None if state is None else state.get("layer2")
        spikes_hidden, next_layer1_state = self.layer1(x, layer1_state, return_state=True)
        spikes_out, next_layer2_state = self.layer2(spikes_hidden, layer2_state, return_state=True)
        rate_output = spikes_out.sum(dim=1)
        if not return_state:
            return rate_output
        return rate_output, {"layer1": next_layer1_state, "layer2": next_layer2_state}
