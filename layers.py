"""
Core Q-SSD layers built on top of BitLinear/RMSNorm and Mamba SSM.

This module wires quantized projections into the mixer and channel blocks
while keeping the SSM core in higher precision (FP16/BF16/FP32).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from quantization import BitLinear, RMSNorm


class QuantizedStateSpaceMixer(nn.Module):
    """
    Temporal mixing block: BitLinear input projection + depthwise conv + SSM core + BitLinear output.
    """

    def __init__(
        self,
        d_model: int,
        ssm_cfg: Dict[str, int],
        *,
        activation_bits: Optional[int] = 8,
        bitnet_scale: bool = True,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        expand = int(ssm_cfg.get("expand", 2))
        d_conv = int(ssm_cfg.get("d_conv", 4))
        d_state = int(ssm_cfg.get("d_state", 16))

        inner_dim = d_model * expand

        self.norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.input_proj = BitLinear(
            d_model,
            inner_dim,
            activation_bits=activation_bits,
            bitnet_scale=bitnet_scale,
        )

        # Short depthwise convolution to capture local context before SSM.
        self.short_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=inner_dim,
        )

        # High-precision SSM core (keeps internal A, B, C in FP16/BF16/FP32).
        try:
            from mamba_ssm.modules.mamba_simple import Mamba

            # NOTE: Mamba internally uses nn.Linear; to fully quantize, replace those with BitLinear in the library itself.
            self.ssm_core = Mamba(
                d_model=inner_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=1,  # already expanded via BitLinear input
            )
        except ImportError:
            # Fallback: lightweight residual MLP as a placeholder when mamba_ssm is unavailable.
            # This keeps the interface intact for smoke tests; replace with real Mamba when installed.
            self.ssm_core = nn.Sequential(
                nn.LayerNorm(inner_dim),
                nn.Linear(inner_dim, inner_dim),
                nn.SiLU(),
                nn.Linear(inner_dim, inner_dim),
            )

        self.out_proj = BitLinear(
            inner_dim,
            d_model,
            activation_bits=activation_bits,
            bitnet_scale=bitnet_scale,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        residual = x
        x = self.norm(x)

        x = self.input_proj(x)

        # Conv1d expects (B, C, T), while incoming is (B, T, C).
        x = x.transpose(1, 2)
        x = self.short_conv(x)
        x = x.transpose(1, 2)

        # Keep SSM computation in higher precision.
        x = self.ssm_core(x)

        x = F.silu(x)
        x = self.out_proj(x)
        # Align sequence length in case convolution padding changed it.
        if x.shape[1] != residual.shape[1]:
            min_t = min(x.shape[1], residual.shape[1])
            x = x[:, :min_t]
            residual = residual[:, :min_t]
        return residual + x


class QuantizedChannelMixer(nn.Module):
    """
    Channel mixing (FFN) block using SwiGLU-style gating with BitLinear projections.
    """

    def __init__(
        self,
        d_model: int,
        *,
        activation_bits: Optional[int] = 8,
        bitnet_scale: bool = True,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        hidden = 4 * d_model  # expansion factor

        self.norm = RMSNorm(d_model, eps=rms_norm_eps)
        # Project to gate/value branches.
        self.gate_value_proj = BitLinear(
            d_model,
            hidden * 2,
            activation_bits=activation_bits,
            bitnet_scale=bitnet_scale,
        )
        self.out_proj = BitLinear(
            hidden,
            d_model,
            activation_bits=activation_bits,
            bitnet_scale=bitnet_scale,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        residual = x
        x = self.norm(x)

        gate_value = self.gate_value_proj(x)
        value, gate = gate_value.chunk(2, dim=-1)
        x = value * F.silu(gate)  # SwiGLU-style gating

        x = self.out_proj(x)
        return residual + x


__all__ = [
    "QuantizedStateSpaceMixer",
    "QuantizedChannelMixer",
]
