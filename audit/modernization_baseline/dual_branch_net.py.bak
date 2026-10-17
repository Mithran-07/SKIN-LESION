"""
Dual-Branch CNN: Full classifier combining shallow-wide and deep-narrow branches.

This is the core architectural innovation: by physically instantiating two
topologically distinct CNN branches, the network is forced to maintain
separate high-frequency (texture) and low-frequency (structure) representations
rather than collapsing both into a single abstract feature vector.

The dual forward returns both the classification logits AND the raw feature
maps from the last conv layer of each branch. These feature maps are
required by the Grad-CAM explainability pipeline.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn

from models.shallow_wide_branch import ShallowWideBranch
from models.deep_narrow_branch import DeepNarrowBranch
from models.fusion import AttentionFusionHead


class DualBranchNet(nn.Module):
    """
    Full Dual-Branch Dermoscopic Classifier.

    Args:
        num_classes: Number of diagnostic classes (7 for HAM10000, 8 for ISIC2019).
        shallow_channels: Channel config for shallow-wide branch [256, 512, 1024].
        deep_base_channels: Base channel width for deep-narrow branch (64).
        deep_bottleneck: Final bottleneck channels for deep-narrow branch (256).
        deep_num_blocks: Residual block counts per stage [2, 2, 3, 3].
        fusion_hidden_dim: Hidden dim in fusion MLP (512).
        fusion_output_dim: Output dim of fusion head (256).
        pretrained_init: Initialize branch stems from pretrained weights.
        dropout_cls: Dropout before the final classification layer.
    """

    def __init__(
        self,
        num_classes: int = 7,
        shallow_channels: list = None,
        deep_base_channels: int = 64,
        deep_bottleneck: int = 256,
        deep_num_blocks: list = None,
        fusion_hidden_dim: int = 512,
        fusion_output_dim: int = 256,
        pretrained_init: bool = True,
        dropout_cls: float = 0.3,
    ):
        super().__init__()
        shallow_channels = shallow_channels or [256, 512, 1024]
        deep_num_blocks = deep_num_blocks or [2, 2, 3, 3]

        # ── Branch instantiation ────────────────────────────────────────────
        self.shallow_branch = ShallowWideBranch(
            channels=shallow_channels,
            pretrained_init=pretrained_init,
        )
        self.deep_branch = DeepNarrowBranch(
            base_channels=deep_base_channels,
            bottleneck_channels=deep_bottleneck,
            num_blocks=deep_num_blocks,
            pretrained_init=pretrained_init,
        )

        # ── Attention-gated fusion ──────────────────────────────────────────
        self.fusion = AttentionFusionHead(
            texture_dim=self.shallow_branch.output_dim,
            structure_dim=self.deep_branch.output_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
        )

        # ── Classification head ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_cls),
            nn.Linear(fusion_output_dim, num_classes),
        )

        self.num_classes = num_classes
        self.fusion_output_dim = fusion_output_dim

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through both branches, fusion, and classifier.

        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            logits: (B, num_classes) — raw unnormalized class scores
            texture_fmap: (B, 1024, H', W') — shallow branch last conv feature map
            structure_fmap: (B, 256, H'', W'') — deep branch last conv feature map
                            Both feature maps are returned for Grad-CAM visualization.
        """
        texture_vec, texture_fmap = self.shallow_branch(x)
        structure_vec, structure_fmap = self.deep_branch(x)
        fused = self.fusion(texture_vec, structure_vec)
        logits = self.classifier(fused)
        return logits, texture_fmap, structure_fmap

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_branch(self, branch: str = "deep") -> None:
        """
        Freeze a branch for stage-wise fine-tuning.

        Args:
            branch: 'shallow', 'deep', or 'both'
        """
        if branch in ("shallow", "both"):
            for p in self.shallow_branch.parameters():
                p.requires_grad = False
        if branch in ("deep", "both"):
            for p in self.deep_branch.parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True


def build_model_from_config(cfg: dict) -> DualBranchNet:
    """
    Instantiate DualBranchNet from a parsed config.yaml dict.

    Args:
        cfg: Full config dict from config.yaml

    Returns:
        Configured DualBranchNet instance
    """
    model_cfg = cfg["model"]["dual_branch"]
    sw_cfg = model_cfg["shallow_wide"]
    dn_cfg = model_cfg["deep_narrow"]
    fu_cfg = model_cfg["fusion"]

    return DualBranchNet(
        num_classes=cfg["dataset"]["num_classes"],
        shallow_channels=sw_cfg["channels"],
        deep_base_channels=dn_cfg["base_channels"],
        deep_bottleneck=dn_cfg["bottleneck_channels"],
        deep_num_blocks=dn_cfg["num_blocks"],
        fusion_hidden_dim=fu_cfg["hidden_dim"],
        fusion_output_dim=fu_cfg["output_dim"],
        pretrained_init=sw_cfg.get("pretrained_init", True),
    )
