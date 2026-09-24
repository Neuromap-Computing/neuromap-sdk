"""Convenience functions for the simplified Neuromap workflow.

These high-level helpers wrap the core SDK classes to provide a
streamlined developer experience for common tasks: encoding sensor
data, configuring surrogate gradients, and exporting to hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuromap._internal.encoding import encode_delta, encode_latency, encode_rate
from neuromap._internal.spike import get_surrogate


def encode_sensor_data(
    data: torch.Tensor,
    num_steps: int = 100,
    method: str = "rate",
    **kwargs: Any,
) -> torch.Tensor:
    """Encode continuous data into spike trains.

    Args:
        data: Input tensor.  For ``"rate"`` and ``"latency"`` methods
            the shape should be ``(batch, features)`` (or ``(time, features)``
            for a single trace).  Values should be in ``[0, 1]``; otherwise
            they are min-max scaled per feature over leading dimensions.
            For ``"delta"`` the shape should be
            ``(batch, time_steps, features)``.
        num_steps: Number of time steps (used by ``"rate"`` and
            ``"latency"``).
        method: Encoding strategy - ``"rate"``, ``"latency"``, or
            ``"delta"``.
        **kwargs: Extra arguments forwarded to the underlying encoder.

    Returns:
        Spike tensor.  For ``"rate"`` and ``"latency"`` the shape is
        ``(batch, num_steps, features)``.

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method == "rate":
        return encode_rate(data, num_steps=num_steps, **kwargs)
    if method == "latency":
        return encode_latency(data, num_steps=num_steps, **kwargs)
    if method == "delta":
        return encode_delta(data, **kwargs)
    raise ValueError(f"Unknown encoding method '{method}'. Use 'rate', 'latency', or 'delta'.")


def compile_model(network: Any, surrogate: str = "atan") -> None:
    """Inject an snnTorch surrogate gradient into all LIF layers.

    This configures the network's neuron layers for training by
    replacing their surrogate gradient functions.

    Args:
        network: A :class:`~neuromap.network.Network` instance.
        surrogate: Surrogate gradient name (see
            :func:`~neuromap._internal.spike.get_surrogate`).
    """
    from neuromap._internal.lif import NeuromapLIF

    spike_grad = get_surrogate(surrogate)
    for module in network.model.modules():
        if isinstance(module, NeuromapLIF):
            module.lif.spike_grad = spike_grad


def quantize_and_export_to_spi(
    network: Any,
    bit_resolution: int = 4,
    path: str | Path | None = None,
) -> bytes | None:
    """Quantize model weights and export to ``.nmap`` format.

    Args:
        network: A :class:`~neuromap.network.Network` instance.
        bit_resolution: Bit-width for weight quantization.
        path: If provided, save the ``.nmap`` file to this path and
            return ``None``.  Otherwise return the raw bytes.

    Returns:
        The ``.nmap`` bytes when *path* is ``None``, otherwise ``None``.
    """
    from neuromap.export import Exporter

    exporter = Exporter(network)
    exporter.quantize(bits=bit_resolution)
    if path is not None:
        exporter.save(path)
        return None
    return exporter.to_bytes()
