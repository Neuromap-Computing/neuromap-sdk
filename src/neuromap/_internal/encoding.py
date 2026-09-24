"""Spike encoding wrappers around snnTorch spikegen.

Provides thin wrappers for common spike encoding strategies used to
convert continuous sensor data into spike trains.
"""

from __future__ import annotations

import snntorch.spikegen as spikegen
import torch


def _scale_to_unit_interval(data: torch.Tensor) -> torch.Tensor:
    """Min-max scale to ``[0, 1]`` per feature when values lie outside that range.

    snnTorch ``rate`` / ``latency`` require inputs in ``[0, 1]``.  Data that is
    already in range (e.g. pre-normalised sensors) is returned unchanged.
    Leading dimensions are reduced so each column (last dim) is scaled using
    its own min/max - e.g. ``(time, features)`` normalises over time per channel.
    """
    if data.numel() == 0:
        return data
    if bool((data >= 0).all()) and bool((data <= 1).all()):
        return data
    lead = tuple(range(data.ndim - 1))
    dmin = data.amin(dim=lead, keepdim=True) if lead else data.min()
    dmax = data.amax(dim=lead, keepdim=True) if lead else data.max()
    span = (dmax - dmin).clamp(min=1e-6)
    return ((data - dmin) / span).clamp(0.0, 1.0)


def encode_rate(data: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Rate-code encoding via :func:`snntorch.spikegen.rate`.

    Args:
        data: Input tensor of shape ``(batch, features)`` with values
            in ``[0, 1]``.
        num_steps: Number of time steps to generate.

    Returns:
        Spike tensor of shape ``(batch, num_steps, features)``.
    """
    data = _scale_to_unit_interval(data)
    # spikegen.rate returns (num_steps, batch, features)
    spikes = spikegen.rate(data, num_steps=num_steps)
    # Transpose to (batch, num_steps, features)
    return spikes.permute(1, 0, 2)


def encode_latency(data: torch.Tensor, num_steps: int, **kwargs: object) -> torch.Tensor:
    """Latency encoding via :func:`snntorch.spikegen.latency`.

    Args:
        data: Input tensor of shape ``(batch, features)`` with values
            in ``[0, 1]``.
        num_steps: Number of time steps.
        **kwargs: Extra arguments forwarded to
            :func:`snntorch.spikegen.latency`.

    Returns:
        Spike tensor of shape ``(batch, num_steps, features)``.
    """
    data = _scale_to_unit_interval(data)
    spikes = spikegen.latency(data, num_steps=num_steps, normalize=True, **kwargs)
    return spikes.permute(1, 0, 2)


def encode_delta(data: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """Delta modulation encoding via :func:`snntorch.spikegen.delta`.

    Args:
        data: Input tensor of shape ``(batch, time_steps, features)``
            or ``(time_steps, features)``.
        **kwargs: Extra arguments forwarded to
            :func:`snntorch.spikegen.delta`.

    Returns:
        Spike tensor with the same shape as *data*.
    """
    if data.ndim == 3:
        # snnTorch delta treats dim-0 as the time axis.
        # Transpose (batch, time, features) -> (time, batch, features)
        # so differences are computed across time steps, not batches.
        spikes = spikegen.delta(data.permute(1, 0, 2), **kwargs)
        return spikes.permute(1, 0, 2)
    return spikegen.delta(data, **kwargs)
