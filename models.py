"""
Q-SSD model assembly from quantized building blocks.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from config import QSSDConfig
from layers import QuantizedStateSpaceMixer, QuantizedChannelMixer
from quantization import BitLinear, RMSNorm


class QSSDBlock(nn.Module):
    """
    A single Q-SSD block: temporal mixer followed by channel mixer.
    Each sub-layer is pre-normed and uses residual connections internally.
    """

    def __init__(
        self,
        config: QSSDConfig,
        *,
        activation_bits: Optional[int] = 8,
    ) -> None:
        super().__init__()
        self.mixer = QuantizedStateSpaceMixer(
            d_model=config.d_model,
            ssm_cfg=config.ssm_cfg,
            activation_bits=activation_bits,
            bitnet_scale=config.bitnet_scale,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.channel_mixer = QuantizedChannelMixer(
            d_model=config.d_model,
            activation_bits=activation_bits,
            bitnet_scale=config.bitnet_scale,
            rms_norm_eps=config.rms_norm_eps,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.mixer(x)
        x = self.channel_mixer(x)
        return x


class QSSDModel(nn.Module):
    """
    Quantized State Space Dual (Q-SSD) language model.
    """

    def __init__(
        self,
        config: QSSDConfig,
        *,
        activation_bits: Optional[int] = 8,
    ) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            QSSDBlock(config, activation_bits=activation_bits)
            for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = BitLinear(
            config.d_model,
            config.vocab_size,
            activation_bits=activation_bits,
            bitnet_scale=config.bitnet_scale,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Embedding: normal init as in many LM setups.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        # BitLinear layers already initialized internally; ensure lm_head bias zeroed.
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)
        # RMSNorm weight is initialized to ones in its constructor.

    def forward(self, input_ids: Tensor) -> Tensor:  # type: ignore[override]
        """
        Args:
            input_ids: Tensor of shape (batch, seq_len) with token ids.
        Returns:
            logits: Tensor of shape (batch, seq_len, vocab_size).
        """
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


__all__ = [
    "QSSDBlock",
    "QSSDModel",
]
