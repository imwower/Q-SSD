"""
Core Q-SSD layers built on top of BitLinear/RMSNorm and a pure PyTorch Mamba-style SSM.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from quantization import BitLinear, RMSNorm


class QuantizedMambaBlock(nn.Module):
    """Minimal Mamba/SSM block with quantized projections and high-precision state update.

    Args:
        d_model: Model dimension.
        d_state: State dimension per channel.
        activation_bits: Activation quantization bits for BitLinear.
        bitnet_scale: Whether to apply BitNet scaling in BitLinear.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        *,
        activation_bits: Optional[int] = 8,
        bitnet_scale: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Quantized projections for input-dependent B, C, and delta.
        self.x_proj = BitLinear(
            d_model,
            2 * d_model,
            activation_bits=activation_bits,
            bitnet_scale=bitnet_scale,
        )
        self.dt_proj = BitLinear(
            d_model,
            d_model,
            activation_bits=activation_bits,
            bitnet_scale=bitnet_scale,
        )

        # High-precision dynamics parameters (kept in FP32).
        self.A_log = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Runs the selective scan over a sequence (vectorized, no Python loop).

        Args:
            x: Tensor of shape (B, T, D).
            state: Optional state dict storing "h" for inference or streaming.
        Returns:
            Tensor of shape (B, T, D).
        """
        bsz, seq_len, dim = x.shape
        assert dim == self.d_model, "Input dim must match d_model."

        A = -F.softplus(self.A_log).to(dtype=torch.float32)  # ensure negative for stability
        if state is not None and "h" in state:
            h0 = state["h"].to(dtype=torch.float32, device=x.device)
        else:
            h0 = torch.zeros(bsz, dim, device=x.device, dtype=torch.float32)

        # Quantized projections (stay in compute dtype), then promote to FP32 for dynamics.
        dt = F.softplus(self.dt_proj(x)).to(dtype=torch.float32)  # (B, T, D)
        dt = dt.clamp(min=1e-3, max=1.0)
        x_proj = self.x_proj(x)
        b_t, c_t = x_proj.chunk(2, dim=-1)
        b_t = b_t.to(dtype=torch.float32).clamp(min=-5.0, max=5.0)
        c_t = c_t.to(dtype=torch.float32).clamp(min=-5.0, max=5.0)

        alpha = torch.exp((dt * A).clamp(min=-10.0, max=0.0))  # (B, T, D)

        # Prefix products for alpha to compute recurrence in closed form.
        prefix = torch.cumprod(alpha, dim=1)  # (B, T, D)
        prefix_shift = torch.cat(
            [torch.ones(bsz, 1, dim, device=x.device, dtype=torch.float32), prefix[:, :-1, :]],
            dim=1,
        )  # prod over j<k

        g = dt * b_t / prefix_shift
        cumsum = torch.cumsum(g, dim=1)
        h = prefix * (h0.unsqueeze(1) + cumsum)  # (B, T, D)
        y = h * torch.tanh(c_t)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = y.to(dtype=x.dtype)

        if state is not None:
            state["h"] = h[:, -1, :].detach()

        return y


class QuantizedStateSpaceMixer(nn.Module):
    """Temporal mixing block using quantized projections and pure PyTorch Mamba block."""

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

        # Depthwise causal convolution (maintain buffer for inference).
        self.short_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=inner_dim,
        )

        self.ssm_core = QuantizedMambaBlock(
            d_model=inner_dim,
            d_state=d_state,
            activation_bits=activation_bits,
            bitnet_scale=bitnet_scale,
        )

        self.out_proj = BitLinear(
            inner_dim,
            d_model,
            activation_bits=activation_bits,
            bitnet_scale=bitnet_scale,
        )

        self.d_conv = d_conv
        self.inner_dim = inner_dim

    def _causal_conv_step(
        self,
        x_proj: Tensor,
        layer_state: Dict[str, Tensor],
    ) -> Tensor:
        """Performs a single causal depthwise conv step using buffered inputs."""
        # x_proj: (B, 1, C)
        buffer = layer_state.get("conv_buffer")
        bsz = x_proj.size(0)
        if buffer is None:
            buffer = torch.zeros(
                bsz,
                self.inner_dim,
                self.d_conv,
                device=x_proj.device,
                dtype=x_proj.dtype,
            )

        # Shift and append new token
        x_proj_t = x_proj.transpose(1, 2)  # (B, C, 1)
        buffer = torch.cat([buffer[:, :, 1:], x_proj_t], dim=2)

        conv_out = F.conv1d(
            buffer,
            self.short_conv.weight,
            self.short_conv.bias,
            padding=0,
            groups=self.inner_dim,
        )
        conv_out = conv_out.transpose(1, 2)  # (B, 1, C)
        layer_state["conv_buffer"] = buffer.detach()
        return conv_out

    def forward(
        self,
        x: Tensor,
        layer_state: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Forward pass with optional streaming states.

        Args:
            x: Tensor of shape (B, T, d_model).
            layer_state: Optional dict holding "conv_buffer" and "ssm_state".
        """
        residual = x
        x = self.norm(x)

        x_proj = self.input_proj(x)

        if layer_state is None or x_proj.size(1) > 1:
            # Training / full sequence path.
            x_conv = self.short_conv(x_proj.transpose(1, 2)).transpose(1, 2)
            if x_conv.size(1) != x_proj.size(1):
                t = min(x_conv.size(1), x_proj.size(1))
                x_conv = x_conv[:, :t]
                x_proj = x_proj[:, :t]
        else:
            # Streaming single-step path.
            x_conv = self._causal_conv_step(x_proj, layer_state)

        ssm_state = None if layer_state is None else layer_state.setdefault("ssm_state", {})
        x_ssm = self.ssm_core(x_conv, ssm_state)
        x_ssm = F.silu(x_ssm)
        x_out = self.out_proj(x_ssm)

        return residual + x_out


class QuantizedChannelMixer(nn.Module):
    """Channel mixing (FFN) block using SwiGLU-style gating with BitLinear projections."""

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

    def forward(
        self,
        x: Tensor,
        layer_state: Optional[Dict[str, Tensor]] = None,  # kept for API consistency
    ) -> Tensor:
        residual = x
        x = self.norm(x)

        gate_value = self.gate_value_proj(x)
        value, gate = gate_value.chunk(2, dim=-1)
        x = value * F.silu(gate)  # SwiGLU-style gating

        x = self.out_proj(x)
        return residual + x


__all__ = [
    "QuantizedMambaBlock",
    "QuantizedStateSpaceMixer",
    "QuantizedChannelMixer",
]
