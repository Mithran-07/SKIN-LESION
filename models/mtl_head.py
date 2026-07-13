"""
Multi-Task Learning Dual-Branch Network.

Extends DualBranchNet with a U-Net-style segmentation decoder that shares
the shallow-wide branch encoder as its backbone. The segmentation task acts
as a spatial regularizer, penalizing the shared encoder for attending to
background artifacts (rulers, hair, bubbles) by forcing it to precisely
delineate the lesion boundary.

MTL Benefits:
- Joint optimization improves feature quality for both tasks
- Segmentation head provides explicit geometric priors to the classifier
- Inference: both classification + lesion mask in a single forward pass
- Reduces total inference time vs. sequential pipeline
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dual_branch_net import DualBranchNet
from models.shallow_wide_branch import ShallowWideBranch, ConvBlock


class SegmentationDecoder(nn.Module):
    """
    U-Net-style decoder that upsamples from 1024-ch bottleneck to a binary mask.

    Uses TransposedConv2d for learnable upsampling (vs. bilinear which may
    lose spatial precision needed for boundary delineation).

    Args:
        in_channels: Number of channels from the encoder (1024).
    """

    def __init__(self, in_channels: int = 1024):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        # Final 1×1 conv to single-channel binary mask
        self.mask_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.mask_head(x)
        # Ensure output matches target spatial size
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)  # Binary mask probabilities (B, 1, H, W)


class MTLDualBranchNet(nn.Module):
    """
    Multi-Task Dual-Branch Network: simultaneous classification + segmentation.

    Architecture:
        Input → Shared (ShallowWideBranch + DeepNarrowBranch + Fusion)
                    ├── Classification Head → (B, num_classes) logits
                    └── Segmentation Decoder → (B, 1, H, W) binary mask

    Args:
        num_classes: Number of diagnostic classes.
        pretrained_init: Transfer stem weights from pretrained backbones.
    """

    def __init__(self, num_classes: int = 7, pretrained_init: bool = True):
        super().__init__()
        # Reuse the full DualBranchNet as the shared encoder + classifier
        self.dual_branch = DualBranchNet(
            num_classes=num_classes,
            pretrained_init=pretrained_init,
        )
        # Segmentation decoder takes the shallow branch's last feature map
        self.seg_decoder = SegmentationDecoder(
            in_channels=self.dual_branch.shallow_branch.output_dim  # 1024
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            logits: (B, num_classes)
            seg_mask: (B, 1, H, W) — predicted binary lesion mask
            texture_fmap: (B, 1024, H', W') — for Grad-CAM
            structure_fmap: (B, 256, H'', W'') — for Grad-CAM
        """
        target_size = (x.shape[2], x.shape[3])
        logits, texture_fmap, structure_fmap = self.dual_branch(x)
        seg_mask = self.seg_decoder(texture_fmap, target_size=target_size)
        return logits, seg_mask, texture_fmap, structure_fmap
