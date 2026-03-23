"""Surrogate gradient functions backed by snnTorch.

Provides a registry of surrogate gradient functions for backpropagation
through the discontinuous spike function in spiking neurons.
"""

from __future__ import annotations

from typing import Callable

import snntorch.surrogate as surrogate
import torch

_SURROGATES: dict[str, Callable[[], Callable[..., torch.Tensor]]] = {
    "atan": surrogate.atan,
    "fast_sigmoid": surrogate.fast_sigmoid,
    "straight_through": surrogate.straight_through_estimator,
    "spike_rate_escape": surrogate.spike_rate_escape,
}

DEFAULT_SURROGATE = "atan"


def get_surrogate(name: str | None = None) -> Callable[..., torch.Tensor]:
    """Return an snnTorch surrogate gradient function by name.

    Args:
        name: One of ``"atan"``, ``"fast_sigmoid"``,
            ``"straight_through"``, or ``"spike_rate_escape"``.
            Defaults to ``"atan"``.

    Returns:
        A callable suitable for the ``spike_grad`` parameter of
        snnTorch neuron models.

    Raises:
        ValueError: If *name* is not a recognised surrogate.
    """
    name = name or DEFAULT_SURROGATE
    factory = _SURROGATES.get(name)
    if factory is None:
        raise ValueError(f"Unknown surrogate '{name}'. Available: {list(_SURROGATES)}")
    return factory()


spike_function = get_surrogate(DEFAULT_SURROGATE)
"""Default surrogate gradient function (atan) for internal use."""
