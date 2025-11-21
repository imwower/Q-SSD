"""
Q-SSD model assembly from quantized building blocks with streaming/inference support.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from config import QSSDConfig
from layers import QuantizedStateSpaceMixer, QuantizedChannelMixer
from quantization import BitLinear, RMSNorm


class QSSDBlock(nn.Module):
    """A single Q-SSD block: temporal mixer followed by channel mixer."""

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

    def forward(
        self,
        x: Tensor,
        layer_state: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ) -> Tensor:  # type: ignore[override]
        mixer_state = None if layer_state is None else layer_state.setdefault("mixer", {})
        channel_state = None if layer_state is None else layer_state.setdefault("channel", {})

        x = self.mixer(x, mixer_state)
        x = self.channel_mixer(x, channel_state)
        return x


class QSSDModel(nn.Module):
    """Quantized State Space Dual (Q-SSD) language model with streaming inference."""

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
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        input_ids: Tensor,
        inference_params: Optional[Dict[str, Dict[str, Dict[str, Tensor]]]] = None,
    ) -> Tensor:  # type: ignore[override]
        """Full-sequence forward pass.

        Args:
            input_ids: Tensor of shape (B, T).
            inference_params: Optional streaming states; if provided, will be updated.
        """
        x = self.embed(input_ids)
        layer_states = None
        if inference_params is not None:
            layer_states = inference_params.setdefault(
                "layer_states", [dict() for _ in range(len(self.layers))]
            )

        for idx, layer in enumerate(self.layers):
            state = None if layer_states is None else layer_states[idx]
            x = layer(x, state)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def step(
        self,
        input_ids: Tensor,
        inference_params: Dict[str, Dict[str, Dict[str, Tensor]]],
    ) -> Tensor:
        """O(1) incremental decoding step.

        Args:
            input_ids: Tensor of shape (B, 1) containing the next token id(s).
            inference_params: Dict holding persistent layer states.
        Returns:
            Logits for the next token: Tensor of shape (B, vocab_size).
        """
        x = self.embed(input_ids)
        layer_states = inference_params.setdefault(
            "layer_states", [dict() for _ in range(len(self.layers))]
        )
        for idx, layer in enumerate(self.layers):
            x = layer(x, layer_states[idx])
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits[:, -1, :]


__all__ = [
    "QSSDBlock",
    "QSSDModel",
]
