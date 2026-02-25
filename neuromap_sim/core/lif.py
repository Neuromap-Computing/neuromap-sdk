"""Leaky integrate-and-fire layer implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .spike import spike_function

LIFState = dict[str, torch.Tensor | int]


class LIFLayer(nn.Module):
    """A dense LIF layer with optional state carry-over across chunks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_m: float = 10.0,
        rm: float = 1.0,
        dt: float = 1.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        t_ref: int = 2,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.tau_m = tau_m
        self.rm = rm
        self.dt = dt
        self.v_th = v_th
        self.v_reset = v_reset
        self.t_ref = t_ref

    def init_state(self, batch_size: int, *, device: torch.device) -> LIFState:
        v_mem = torch.full((batch_size, self.fc.out_features), self.v_reset, device=device)
        last_spike_time = torch.full(
            (batch_size, self.fc.out_features), -float(self.t_ref), device=device
        )
        return {"v_mem": v_mem, "last_spike_time": last_spike_time, "step": 0}

    def _normalize_state(
        self, state: Mapping[str, Any], *, batch_size: int, device: torch.device
    ) -> LIFState:
        if not {"v_mem", "last_spike_time", "step"}.issubset(state):
            raise ValueError("State must contain 'v_mem', 'last_spike_time', and 'step'.")
        v_mem = state["v_mem"].to(device)
        last_spike_time = state["last_spike_time"].to(device)
        step = int(state["step"])
        if v_mem.shape[0] != batch_size:
            raise ValueError("State batch size does not match input batch size.")
        return {"v_mem": v_mem, "last_spike_time": last_spike_time, "step": step}

    def forward(
        self,
        x_seq: torch.Tensor,
        state: Mapping[str, Any] | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LIFState]:
        batch_size, time_steps, _ = x_seq.shape
        device = x_seq.device
        current_state = (
            self.init_state(batch_size, device=device)
            if state is None
            else self._normalize_state(state, batch_size=batch_size, device=device)
        )
        v_mem = current_state["v_mem"]
        last_spike_time = current_state["last_spike_time"]
        step = int(current_state["step"])
        threshold = torch.as_tensor(self.v_th, device=device, dtype=x_seq.dtype)
        out_spikes: list[torch.Tensor] = []

        for t in range(time_steps):
            absolute_time = float(step + t)
            i_syn = self.fc(x_seq[:, t, :])
            refractory_mask = (absolute_time - last_spike_time) >= self.t_ref
            dv = self.dt * (-(v_mem - self.v_reset) + self.rm * i_syn) / self.tau_m
            v_mem = v_mem + dv * refractory_mask.float()
            spike = spike_function(v_mem, threshold)
            time_tensor = torch.full_like(last_spike_time, absolute_time)
            last_spike_time = torch.where(spike > 0, time_tensor, last_spike_time)
            v_mem = v_mem * (1.0 - spike) + self.v_reset * spike
            out_spikes.append(spike)

        spikes = torch.stack(out_spikes, dim=1)
        if not return_state:
            return spikes
        next_state: LIFState = {
            "v_mem": v_mem.detach(),
            "last_spike_time": last_spike_time.detach(),
            "step": step + time_steps,
        }
        return spikes, next_state
