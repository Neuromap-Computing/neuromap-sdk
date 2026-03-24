"""Spike encoding wrappers around snnTorch spikegen.

Provides thin wrappers for common spike encoding strategies used to
convert continuous sensor data into spike trains.
"""

from __future__ import annotations

import snntorch.spikegen as spikegen
import torch


def encode_rate(data: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Rate-code encoding via :func:`snntorch.spikegen.rate`.

    Args:
        data: Input tensor of shape ``(batch, features)`` with values
            in ``[0, 1]``.
        num_steps: Number of time steps to generate.

    Returns:
        Spike tensor of shape ``(batch, num_steps, features)``.
    """
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
    return spikegen.delta(data, **kwargs)
