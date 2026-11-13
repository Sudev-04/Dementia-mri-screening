"""Model definitions. The default is deliberately small: the usable OASIS-1
cohort is about 200 people, and a network with tens of millions of weights
memorises that in a few epochs."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv3d(c_in, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c_out, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )


class SmallCNN3D(nn.Module):
    """Four conv stages, global average pooling, one logit.

    Instance norm instead of batch norm because batches are small (4-8 volumes)
    and it behaves identically in training and inference.
    """

    def __init__(self, widths: tuple[int, ...] = (16, 32, 64, 128), dropout: float = 0.3):
        super().__init__()
        stages, c_in = [], 1
        for i, w in enumerate(widths):
            stages.append(ConvBlock(c_in, w))
            if i < len(widths) - 1:
                stages.append(nn.MaxPool3d(2))
            c_in = w
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.head(self.dropout(x)).squeeze(1)


class _SqueezeLogit(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_model(arch: str = "small", dropout: float = 0.3) -> nn.Module:
    if arch == "small":
        return SmallCNN3D(dropout=dropout)
    if arch == "tiny":  # for smoke tests and the synthetic demo only
        return SmallCNN3D(widths=(8, 16, 32), dropout=dropout)
    if arch == "densenet121":
        from monai.networks.nets import DenseNet121
        return _SqueezeLogit(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1,
                                         dropout_prob=dropout))
    raise ValueError(f"Unknown architecture '{arch}'. Choose small, tiny or densenet121.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
