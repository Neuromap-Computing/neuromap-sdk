"""Quantization helpers for SNN models."""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_model_weights(model: nn.Module, *, bits: int = 4, eps: float = 1e-8) -> nn.Module:
    """Quantize all weight tensors of a model to signed fixed-point."""
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
    """Compatibility helper matching the previous simulator API."""
    return quantize_model_weights(model, bits=4)
