"""Squeeze-and-Excitation channel attention block (Hu et al., CVPR 2018)."""

import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Applies global average pooling to squeeze spatial dimensions,
    then a two-layer FC bottleneck to learn per-channel scaling factors.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        fc1 = nn.Linear(channels, mid)
        fc2 = nn.Linear(mid, channels)
        # Initialize near-identity: sigmoid outputs near 1.0 so the block
        # does not disrupt pretrained weights at the start of fine-tuning.
        nn.init.zeros_(fc2.weight)
        nn.init.constant_(fc2.bias, 5.0)  # sigmoid(5) ≈ 0.99
        self.excitation = nn.Sequential(fc1, nn.ReLU(inplace=True), fc2, nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1, 1)
        return x * s
