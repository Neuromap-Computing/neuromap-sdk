"""Quantization helpers for SNN models.

Provides post-training weight quantization to signed fixed-point
representation, matching the bit-widths of Neuromap hardware DACs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_model_weights(model: nn.Module, *, bits: int = 4, eps: float = 1e-8) -> nn.Module:
    """Quantize all weight tensors of a model to signed fixed-point.

    Each weight tensor is independently scaled so that its maximum
    absolute value maps to ``2^(bits-1) - 1``, rounded to the nearest
    integer level, then scaled back to floating-point.

    Args:
        model: The PyTorch module whose weights will be quantized in-place.
        bits: Bit-width for quantization (must be >= 2).
        eps: Small constant to avoid division by zero.

    Returns:
        The same *model* instance (modified in-place).

    Raises:
        ValueError: If *bits* is less than 2.
    """
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            scale = torch.max(torch.abs(param)) / qmax
            if scale <= eps:
                continue
            quantized = torch.clamp(torch.round(param / scale), qmin, qmax)
            param.copy_(quantized * scale)
    return model


def quantize_weights_4bit(model: nn.Module) -> nn.Module:
    """Convenience wrapper that quantizes to 4-bit signed fixed-point.

    Args:
        model: The PyTorch module to quantize.

    Returns:
        The same *model* instance (modified in-place).
    """
    return quantize_model_weights(model, bits=4)
