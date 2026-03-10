"""Spike functions used by SNN layers.

Provides the surrogate gradient mechanism needed for backpropagation
through the discontinuous Heaviside step function in spiking neurons.
"""

from __future__ import annotations

import torch


class SurrogateHeaviside(torch.autograd.Function):
    """Heaviside step function with a smooth surrogate gradient.

    During the forward pass the function behaves as a hard threshold
    (outputs 0 or 1).  During the backward pass it substitutes a
    Gaussian-shaped surrogate gradient so that standard optimisers can
    train the upstream weights.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input_tensor: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Heaviside step.

        Args:
            ctx: Autograd context for saving tensors.
            input_tensor: Membrane potentials.
            threshold: Firing threshold.

        Returns:
            Binary spike tensor (1 where input >= threshold, else 0).
        """
        ctx.save_for_backward(input_tensor, threshold)
        return (input_tensor >= threshold).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """Compute the surrogate gradient.

        Args:
            ctx: Autograd context with saved tensors.
            grad_output: Upstream gradient.

        Returns:
            Tuple of (gradient w.r.t. input, ``None`` for threshold).
        """
        input_tensor, threshold = ctx.saved_tensors
        surrogate_grad = torch.exp(-((input_tensor - threshold) ** 2))
        return grad_output * surrogate_grad, None


spike_function = SurrogateHeaviside.apply
"""Convenience alias for :meth:`SurrogateHeaviside.apply`."""
