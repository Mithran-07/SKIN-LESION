"""
Deep-Narrow Branch for Structural Morphological Feature Extraction.

Deep architectures expand the receptive field exponentially through repeated
convolutions and pooling, allowing the network to evaluate macroscopic lesion
properties:
  - Border irregularity and asymmetry
  - Global pigment distribution across the entire neoplasm
  - Lesion-to-background contrast ratio
  - Macroscopic structural geometry distinguishing BCC/melanoma from benign nevi

Narrow constraint: By restricting channel width (base 64ch), the network is
forced to distill highly abstract, global semantic features rather than
storing redundant localized patterns — functioning as a low-pass filter.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models


class NarrowResidualBlock(nn.Module):
    """
    Narrow residual block with configurable depth (number of conv layers).

    Unlike standard ResNet blocks, the channel width stays constant to maintain
    the narrow constraint. Skip connections are added every 2 conv layers.

    Args:
        channels: Number of channels (kept constant — the 'narrow' constraint).
        depth: Number of Conv→BN→ReLU units in the block.
        downsample: If True, applies stride=2 on the first conv + 1x1 projection skip.
    """

    def __init__(self, channels: int, depth: int = 2, downsample: bool = False):
        super().__init__()
        out_channels = channels * 2 if downsample else channels
        stride = 2 if downsample else 1

        layers = []
        for i in range(depth):
            in_ch = channels if i == 0 else out_channels
            out_ch = out_channels
            s = stride if i == 0 else 1
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=s, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
            if i < depth - 1:
                layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection projection if spatial/channel dims change
        if downsample or channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.out_channels = out_channels
        # Expose last conv for Grad-CAM
        self.conv2 = self.conv_layers[-2]  # Last Conv2d (before final BN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv_layers(x)
        return self.relu(out + residual)


class DeepNarrowBranch(nn.Module):
    """
    Deep-Narrow Branch: expands receptive field via 4 stages of residual blocks
    while keeping channel width deliberately small.

    Architecture:
        Input (B, 3, 224, 224)
        → Stem: Conv(3→64, k=7, s=2) + BN + ReLU + MaxPool   → (B, 64, 56, 56)
        → Stage1: NarrowResBlock(64, depth=2)                  → (B, 64, 56, 56)
        → Stage2: NarrowResBlock(64→128, depth=2, ds=True)    → (B, 128, 28, 28)
        → Stage3: NarrowResBlock(128→128, depth=3)             → (B, 128, 28, 28)
        → Stage4: NarrowResBlock(128→128, depth=3, ds=True)   → (B, 128, 14, 14)
        → Bottleneck: Conv(128→256, k=1)                       → (B, 256, 14, 14)
        → AdaptiveAvgPool(1,1)                                 → (B, 256, 1, 1)
        → Flatten                                              → (B, 256)

    Args:
        base_channels: Starting channel width (default 64 — the 'narrow' constraint).
        bottleneck_channels: Final bottleneck output channels (default 256).
        num_blocks: Number of residual blocks per stage [2, 2, 3, 3].
        pretrained_init: If True, init stem from DenseNet121 conv0 weights.
        dropout_p: Dropout probability before output.
    """

    def __init__(
        self,
        base_channels: int = 64,
        bottleneck_channels: int = 256,
        num_blocks: List[int] = None,
        pretrained_init: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        num_blocks = num_blocks or [2, 2, 3, 3]
        assert len(num_blocks) == 4, "num_blocks must have exactly 4 elements."

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.stage1 = NarrowResidualBlock(base_channels, depth=num_blocks[0], downsample=False)
        self.stage2 = NarrowResidualBlock(base_channels, depth=num_blocks[1], downsample=True)
        ch2 = self.stage2.out_channels  # 128
        self.stage3 = NarrowResidualBlock(ch2, depth=num_blocks[2], downsample=False)
        self.stage4 = NarrowResidualBlock(ch2, depth=num_blocks[3], downsample=True)
        ch4 = self.stage4.out_channels

        # Bottleneck projection to fixed dim
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch4, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.output_dim = bottleneck_channels

        self._init_weights(pretrained_init)

    def _init_weights(self, pretrained_init: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained_init:
            try:
                # Transfer stem conv weights from DenseNet121
                densenet = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
                # DenseNet121 conv0: (64, 3, 7, 7) — exact match for our stem!
                self.stem[0].weight.data.copy_(densenet.features.conv0.weight.data)
                self.stem[1].weight.data.copy_(densenet.features.norm0.weight.data)
                self.stem[1].bias.data.copy_(densenet.features.norm0.bias.data)
                print("[DeepNarrowBranch] DenseNet121 stem weights transferred successfully.")
            except Exception as e:
                print(f"[DeepNarrowBranch] Pretrained stem init skipped: {e}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            feature_vector: (B, 256) — flattened structure vector
            last_feature_map: (B, 256, H, W) — raw feature map before pooling
                              (required by Grad-CAM hook)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bottleneck(x)
        last_feature_map = x  # (B, 256, H, W) — Grad-CAM target
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return x, last_feature_map
