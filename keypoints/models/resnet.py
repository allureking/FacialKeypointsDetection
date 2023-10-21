"""PyTorch ResNet model for facial keypoint detection.

Architecture:
    Conv2d(1, 32) -> BN -> ReLU -> MaxPool
    ResBlock(32, 32, x2)   - first block, no downsampling
    ResBlock(32, 64, x2)   - stride-2 downsampling via 1x1 conv
    ResBlock(64, 128, x2)
    ResBlock(128, 256, x2)
    ResBlock(256, 512, x2)
    AdaptiveAvgPool(1, 1) -> Linear(512, 30)

Total: 12 residual blocks across 6 stages, processing 96x96
grayscale input down to a 512-dimensional feature vector.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with optional 1x1 convolution shortcut.

    When input and output channel counts differ (or stride > 1), a 1x1
    convolution aligns the shortcut dimensions so the skip connection
    can be added element-wise.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_1x1conv: bool = False,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if use_1x1conv
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.shortcut is not None:
            x = self.shortcut(x)
        return F.relu(y + x)


def _resnet_stage(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    first_stage: bool = False,
) -> nn.Sequential:
    """Build a sequence of residual blocks for one stage.

    The first block of non-initial stages uses stride=2 with a 1x1 conv
    shortcut to halve spatial dimensions and match channel counts.
    """
    blocks: List[ResidualBlock] = []
    for i in range(num_blocks):
        if i == 0 and not first_stage:
            blocks.append(
                ResidualBlock(in_channels, out_channels, use_1x1conv=True, stride=2)
            )
        else:
            blocks.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*blocks)


class ResNet(nn.Module):
    """Custom ResNet for 96x96 grayscale facial keypoint regression.

    Outputs 30 values (x, y coordinates for 15 keypoints).
    """

    def __init__(self, num_outputs: int = 30) -> None:
        super().__init__()
        # Initial convolution + pooling
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Residual stages
        self.stage1 = _resnet_stage(32, 32, 2, first_stage=True)
        self.stage2 = _resnet_stage(32, 64, 2)
        self.stage3 = _resnet_stage(64, 128, 2)
        self.stage4 = _resnet_stage(128, 256, 2)
        self.stage5 = _resnet_stage(256, 512, 2)
        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.fc(self.flatten(self.pool(x)))
        return x
