"""
Configuration objects for the Q-SSD model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class QSSDConfig:
    """
    Lightweight configuration container for Q-SSD.

    Attributes
    ----------
    d_model : int
        Hidden size.
    n_layer : int
        Number of stacked Q-SSD blocks.
    vocab_size : int
        Vocabulary size for token-based tasks.
    ssm_cfg : dict
        Selective state space configuration. Defaults to
        {"d_state": 16, "d_conv": 4, "expand": 2}.
    rms_norm_eps : float
        Numerical stability epsilon for RMSNorm.
    bitnet_scale : bool
        Whether to enable BitNet (b1.58) scaling factor during quantization.
    """

    d_model: int = 512
    n_layer: int = 12
    vocab_size: int = 32000
    ssm_cfg: Dict[str, Any] = field(
        default_factory=lambda: {"d_state": 16, "d_conv": 4, "expand": 2}
    )
    rms_norm_eps: float = 1e-5
    bitnet_scale: bool = True

