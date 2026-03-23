"""Leaky integrate-and-fire layer backed by snnTorch.

This module implements :class:`NeuromapLIF`, a dense LIF neuron layer
that wraps :class:`snntorch.Leaky` via composition, adding refractory
period support and streaming state carry.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import snntorch
import torch
import torch.nn as nn

from neuromap._internal.spike import get_surrogate

LIFState = dict[str, torch.Tensor | int]
"""Type alias for the per-layer neuron state dictionary."""


class NeuromapLIF(nn.Module):
    """A dense LIF layer using snnTorch with refractory period support.

    Each neuron integrates weighted input current through a leaky
    membrane potential (via :class:`snntorch.Leaky`) and fires a spike
    when the membrane exceeds a threshold.  A refractory period
    suppresses further firing for a configurable number of time-steps
    after each spike.

    Args:
        in_features: Number of input features (pre-synaptic neurons).
        out_features: Number of output neurons in this layer.
        beta: Membrane potential decay rate (``1 - dt/tau_m``).
        threshold: Firing threshold voltage.
        v_reset: Reset voltage after a spike.
        t_ref: Refractory period in time-steps.
        spike_grad: Optional snnTorch surrogate gradient function.
            Defaults to ``atan`` surrogate.
        output_mem: If ``True``, output the pre-spike membrane potential
            instead of binary spikes.  Useful as a continuous readout
            for the output layer in regression tasks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        v_reset: float = 0.0,
        t_ref: int = 2,
        spike_grad: Any | None = None,
        output_mem: bool = False,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.t_ref = t_ref
        self.v_reset = v_reset
        self.output_mem = output_mem

        if spike_grad is None:
            spike_grad = get_surrogate()

        self.lif = snntorch.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=False,
            reset_mechanism="zero",
        )

    @classmethod
    def from_neuron_params(
        cls,
        in_features: int,
        out_features: int,
        params: Any,
        spike_grad: Any | None = None,
    ) -> NeuromapLIF:
        """Create a :class:`NeuromapLIF` from a :class:`NeuronParams`.

        Args:
            in_features: Number of input features.
            out_features: Number of output neurons.
            params: A :class:`~neuromap.chip.NeuronParams` instance.
            spike_grad: Optional surrogate gradient function.

        Returns:
            A new :class:`NeuromapLIF` instance.
        """
        beta = 1.0 - params.dt / params.tau_m
        return cls(
            in_features=in_features,
            out_features=out_features,
            beta=beta,
            threshold=params.v_th,
            v_reset=params.v_reset,
            t_ref=params.t_ref,
            spike_grad=spike_grad,
        )

    def init_state(self, batch_size: int, *, device: torch.device) -> LIFState:
        """Create a zero-initialised neuron state.

        Args:
            batch_size: Number of samples in the batch.
            device: Torch device for the state tensors.

        Returns:
            A state dictionary with membrane voltages, last spike times
            and step counter.
        """
        mem = torch.full(
            (batch_size, self.fc.out_features), self.v_reset, device=device
        )
        last_spike_time = torch.full(
            (batch_size, self.fc.out_features), -float(self.t_ref), device=device
        )
        return {"mem": mem, "last_spike_time": last_spike_time, "step": 0}

    def _normalize_state(
        self, state: Mapping[str, Any], *, batch_size: int, device: torch.device
    ) -> LIFState:
        # Accept both old "v_mem" key and new "mem" key
        mem_key = "mem" if "mem" in state else "v_mem"
        if not {mem_key, "last_spike_time", "step"}.issubset(state):
            raise ValueError(
                "State must contain 'mem' (or 'v_mem'), 'last_spike_time', and 'step'."
            )
        mem = state[mem_key].to(device)
        last_spike_time = state["last_spike_time"].to(device)
        step = int(state["step"])
        if mem.shape[0] != batch_size:
            raise ValueError("State batch size does not match input batch size.")
        return {"mem": mem, "last_spike_time": last_spike_time, "step": step}

    def forward(
        self,
        x_seq: torch.Tensor,
        state: Mapping[str, Any] | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LIFState]:
        """Process an input spike sequence through the LIF layer.

        Args:
            x_seq: Input tensor of shape ``(batch, time_steps, in_features)``.
            state: Optional carry-over state from a previous chunk.
            return_state: If ``True``, also return the updated state.

        Returns:
            Output spike tensor ``(batch, time_steps, out_features)``, or a
            tuple ``(spikes, next_state)`` when *return_state* is set.
        """
        batch_size, time_steps, _ = x_seq.shape
        device = x_seq.device
        current_state = (
            self.init_state(batch_size, device=device)
            if state is None
            else self._normalize_state(state, batch_size=batch_size, device=device)
        )
        mem = current_state["mem"]
        last_spike_time = current_state["last_spike_time"]
        step = int(current_state["step"])
        out_spikes: list[torch.Tensor] = []

        for t in range(time_steps):
            absolute_time = float(step + t)
            i_syn = self.fc(x_seq[:, t, :])

            # Apply refractory mask
            refractory_mask = (absolute_time - last_spike_time) >= self.t_ref
            i_syn = i_syn * refractory_mask.float()

            # Pre-spike membrane potential (leaky integration before reset)
            pre_spike_mem = self.lif.beta * mem + i_syn

            # snnTorch leaky neuron step
            spike, mem = self.lif(i_syn, mem)

            # Track spike times for refractory period
            time_tensor = torch.full_like(last_spike_time, absolute_time)
            last_spike_time = torch.where(spike > 0, time_tensor, last_spike_time)

            out_spikes.append(pre_spike_mem if self.output_mem else spike)

        spikes = torch.stack(out_spikes, dim=1)
        if not return_state:
            return spikes
        next_state: LIFState = {
            "mem": mem.detach(),
            "last_spike_time": last_spike_time.detach(),
            "step": step + time_steps,
        }
        return spikes, next_state
