"""Spike functions used by SNN layers."""

from __future__ import annotations

import torch


class SurrogateHeaviside(torch.autograd.Function):
    """Heaviside step function with a smooth surrogate gradient."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input_tensor: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(input_tensor, threshold)
        return (input_tensor >= threshold).float()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        input_tensor, threshold = ctx.saved_tensors
        surrogate_grad = torch.exp(-((input_tensor - threshold) ** 2))
        return grad_output * surrogate_grad, None


spike_function = SurrogateHeaviside.apply
