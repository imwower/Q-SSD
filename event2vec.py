"""
Event2Vec: embed neuromorphic event tensors into dense vectors for Q-SSD.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Event2Vec(nn.Module):
    """
    Converts event voxel grids (B, T, C, H, W) into dense embeddings (B, T, d_model).

    Steps:
        1) Spatial embedding per time step with a lightweight Conv2d stack.
        2) Temporal convolution over the time dimension to capture short-term dynamics.
        3) Flatten + projection to d_model.
    """

    def __init__(
        self,
        resolution: Tuple[int, int],
        dim: int,
        in_channels: int = 2,
        hidden_channels: int = 32,
        temporal_kernel: int = 3,
    ) -> None:
        super().__init__()
        h, w = resolution
        self.resolution = resolution
        self.dim = dim

        # Spatial encoder: small Conv2d stack to compress HxW into a compact feature map.
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((h // 4, w // 4)),  # Downsample to reduce tokens.
        )

        # Temporal aggregation: Conv1d over time on flattened spatial features.
        reduced_h, reduced_w = h // 4, w // 4
        spatial_dim = hidden_channels * reduced_h * reduced_w
        self.temporal = nn.Conv1d(
            spatial_dim,
            spatial_dim,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
            groups=1,
        )

        # Final projection to model dimension.
        self.proj = nn.Linear(spatial_dim, dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """
        Args:
            x: Tensor of shape (B, T, C, H, W)
        Returns:
            embeddings: Tensor of shape (B, T, dim)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        x = self.spatial(x)  # (B*T, hidden, h', w')

        # Flatten spatial dims.
        x = x.flatten(2)  # (B*T, hidden, h'*w')
        x = x.transpose(1, 2).contiguous()  # (B*T, tokens, hidden)
        x = x.flatten(1)  # (B*T, spatial_dim)

        # Temporal conv expects (B, C, T). Reshape back to (B, spatial_dim, T).
        x = x.view(b, t, -1).transpose(1, 2)  # (B, spatial_dim, T)
        x = self.temporal(x)  # (B, spatial_dim, T)
        x = x.transpose(1, 2)  # (B, T, spatial_dim)

        # Project to model dimension.
        x = self.proj(x)  # (B, T, dim)
        return x


def _example() -> None:
    """
    Example usage: Convert DVS data (B, T, 2, 64, 64) to (B, T, d_model).
    """
    B, T, H, W = 2, 16, 64, 64
    d_model = 512
    dvs = torch.randn(B, T, 2, H, W)
    encoder = Event2Vec(resolution=(H, W), dim=d_model, in_channels=2)
    out = encoder(dvs)
    print("Input:", dvs.shape, "Output:", out.shape)


if __name__ == "__main__":  # pragma: no cover
    _example()


__all__ = ["Event2Vec"]
