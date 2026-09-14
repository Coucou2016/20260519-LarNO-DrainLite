from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .fno import normalize_drainage_features


class DrainageAdapter(nn.Module):
    """Residual drainage adapter for a frozen LarNO backbone.

    Input tensors use LarNO's convention ``B x C x H x W x T``. The adapter
    consumes the base water-depth prediction, static drainage rasters, current
    rainfall, and cumulative rainfall, and predicts a residual correction.
    """

    def __init__(
        self,
        drainage_channels: int = 8,
        hidden_channels: int = 32,
        nonpositive_residual: bool = True,
    ):
        super().__init__()
        self.drainage_channels = drainage_channels
        self.nonpositive_residual = nonpositive_residual
        in_channels = 1 + drainage_channels + 2
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, base_depth, inputs):
        """Return ``clamp(base_depth + residual, min=0)``.

        ``base_depth`` must be in the same units as the training labels.
        """
        adapter_input = self.build_features(base_depth, inputs)
        raw = self.net(adapter_input)
        residual = -F.softplus(raw) if self.nonpositive_residual else raw
        residual = residual.permute(0, 1, 3, 4, 2).contiguous()
        return torch.clamp(base_depth + residual, min=0.0)

    def build_features(self, base_depth, inputs):
        b, _, h, w, t = base_depth.shape
        drainage = inputs.get("drainage_features")
        if drainage is None:
            raise KeyError("DrainageAdapter requires inputs['drainage_features']")
        drainage = normalize_drainage_features(drainage).repeat(1, 1, 1, 1, t)

        rainfall = inputs["rainfall"][:, :, :, :, :t].float()
        cumsum = inputs["cumsum_rainfall"][:, :, :, :, :t].float()
        rainfall = torch.clamp(rainfall / 9.0, min=0.0, max=1.0)
        cumsum = torch.clamp(cumsum / 375.0, min=0.0, max=1.0)

        x = torch.cat([base_depth.float(), drainage, rainfall, cumsum], dim=1)
        return x.permute(0, 1, 4, 2, 3).contiguous()
