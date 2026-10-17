"""
Dual-Branch CNN: Full classifier combining shallow-wide and deep-narrow branches.

Supports explicit architecture versioning:
- "v1_matrix" (`dual_branch_v1_matrix`): 10,669,639 parameters (used by `dual_branch_seed123` and `dual_branch_seed999`).
- "v2_scalar" (`dual_branch_v2_scalar`): 9,031,241 parameters (used by `dual_branch_seed42`).
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union

import torch
import torch.nn as nn

from models.shallow_wide_branch import ShallowWideBranch
from models.deep_narrow_branch import DeepNarrowBranch
from models.fusion import AttentionFusionHead, FusionVersion, SUPPORTED_FUSION_VERSIONS


class DualBranchNet(nn.Module):
    """
    Full Dual-Branch Dermoscopic Classifier with Explicit Architecture Versioning.

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
        fusion_version: Explicit fusion architecture version ("v1_matrix" or "v2_scalar").
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
        fusion_version: FusionVersion = "v2_scalar",
    ):
        super().__init__()
        if fusion_version not in SUPPORTED_FUSION_VERSIONS:
            raise ValueError(
                f"Unsupported fusion_version '{fusion_version}'. "
                f"Expected one of {SUPPORTED_FUSION_VERSIONS}."
            )

        shallow_channels = shallow_channels or [256, 512, 1024]
        deep_num_blocks = deep_num_blocks or [2, 2, 3, 3]

        self.fusion_version: FusionVersion = fusion_version
        self.architecture_version: str = f"dual_branch_{fusion_version}"

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
            fusion_version=fusion_version,
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

    @staticmethod
    def inspect_checkpoint_version(state_dict: Dict[str, torch.Tensor]) -> str:
        """
        Inspect a state_dict's fusion keys to report its declared fusion version.
        Does NOT silently mutate the model; used to provide clear diagnostics.
        """
        keys = set(state_dict.keys())
        if "fusion.gate.0.weight" in keys:
            return "v1_matrix"
        if "fusion.gate_t.0.weight" in keys and "fusion.gate_s.0.weight" in keys:
            return "v2_scalar"
        return "unknown"

    def load_verified_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Strictly load a checkpoint state_dict (`strict=True`), raising a clear
        diagnostic error if the checkpoint's fusion version does not match
        `self.fusion_version`. Never uses `strict=False`.
        """
        ckpt_version = self.inspect_checkpoint_version(state_dict)
        if ckpt_version != "unknown" and ckpt_version != self.fusion_version:
            raise ValueError(
                f"Architecture version mismatch: DualBranchNet was instantiated with "
                f"fusion_version='{self.fusion_version}' ({self.get_num_params():,} params), "
                f"but the checkpoint state_dict uses fusion_version='{ckpt_version}'. "
                f"Instantiate DualBranchNet(fusion_version='{ckpt_version}') to load this checkpoint strictly."
            )
        self.load_state_dict(state_dict, strict=True)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        fusion_version: FusionVersion,
        device: Union[str, torch.device] = "cpu",
        num_classes: int = 7,
        pretrained_init: bool = False,
    ) -> "DualBranchNet":
        """
        Instantiate DualBranchNet with an explicit `fusion_version` and load
        a checkpoint with `strict=True` verification.
        """
        ckpt_obj = torch.load(str(checkpoint_path), map_location=device)
        if isinstance(ckpt_obj, dict):
            state_dict = ckpt_obj.get("model_state_dict") or ckpt_obj.get("state_dict") or ckpt_obj
        else:
            state_dict = ckpt_obj

        model = cls(
            num_classes=num_classes,
            pretrained_init=pretrained_init,
            fusion_version=fusion_version,
        )
        model.load_verified_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

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
    fusion_version = fu_cfg.get("fusion_version", "v2_scalar")

    return DualBranchNet(
        num_classes=cfg["dataset"]["num_classes"],
        shallow_channels=sw_cfg["channels"],
        deep_base_channels=dn_cfg["base_channels"],
        deep_bottleneck=dn_cfg["bottleneck_channels"],
        deep_num_blocks=dn_cfg["num_blocks"],
        fusion_hidden_dim=fu_cfg["hidden_dim"],
        fusion_output_dim=fu_cfg["output_dim"],
        pretrained_init=sw_cfg.get("pretrained_init", True),
        fusion_version=fusion_version,
    )
