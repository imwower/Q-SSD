"""
Quantization layers and utility modules for Q-SSD.

This file implements:
- BitLinear: a ternary/1.58-bit linear layer using BitNet b1.58-style quantization.
- RMSNorm: a lightweight normalization layer.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _RoundSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for rounding.

    Forward: round to nearest integer.
    Backward: passthrough gradient (identity).
    """

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:  # type: ignore[override]
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        return grad_output


def _round_ste(x: Tensor) -> Tensor:
    """Helper to apply STE rounding."""
    return _RoundSTE.apply(x)


def weight_quant(
    weight: Tensor,
    *,
    bitnet_scale: bool = True,
    eps: float = 1e-5,
) -> Tensor:
    """
    Quantize weights to ternary {-1, 0, +1} with BitNet b1.58 style scaling.

    Args:
        weight: Full-precision weight tensor.
        bitnet_scale: Whether to use the BitNet scaling factor (1.58).
        eps: Numerical stability term to avoid divide-by-zero.

    Returns:
        Quantized weight tensor with values in {-1, 0, +1}.
    """
    gamma = weight.abs().mean()
    q_b = 1.58 if bitnet_scale else 1.0
    scaled = weight / (gamma + eps) * q_b
    # STE rounding followed by clipping to keep ternary support.
    ternary = torch.clamp(_round_ste(scaled), -1.0, 1.0)
    return ternary


def activation_quant(
    x: Tensor,
    num_bits: int = 8,
    eps: float = 1e-6,
) -> Tensor:
    """
    Symmetric per-tensor activation quantization (simulated) to int8 range.

    Uses dynamic scaling based on the current tensor max to avoid overflow,
    then applies STE rounding and de-quantizes back to float.
    """
    if num_bits is None:
        return x

    max_int = 2 ** (num_bits - 1) - 1  # e.g., 127 for int8
    # Dynamic scale so that max(|x|) maps to max_int.
    scale = x.detach().abs().max()
    scale = torch.clamp(scale, min=eps) / max_int
    x_scaled = x / scale
    x_q = torch.clamp(_round_ste(x_scaled), -max_int, max_int)
    return x_q * scale


class BitLinear(nn.Module):
    """
    Linear layer with ternary/1.58-bit quantized weights and optional int8 activations.

    This layer simulates low-bitweight/activation math while keeping outputs in FP32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        activation_bits: Optional[int] = 8,
        bitnet_scale: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.bitnet_scale = bitnet_scale

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same initialization as nn.Linear for familiarity.
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Optional activation quantization.
        if self.activation_bits is not None:
            x = activation_quant(x, num_bits=self.activation_bits)

        # Ternary weight quantization.
        w_q = weight_quant(self.weight, bitnet_scale=self.bitnet_scale)
        return F.linear(x, w_q, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, activation_bits={self.activation_bits}, "
            f"bitnet_scale={self.bitnet_scale}"
        )


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Computes: y = x * weight / sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Mean over the last dimension as in standard RMSNorm.
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"


__all__ = [
    "BitLinear",
    "RMSNorm",
    "activation_quant",
    "weight_quant",
]
