"""
Multi-Task Loss combiner for joint segmentation + classification training.

Combines Focal Loss (classification) with Dice + BCE Loss (segmentation)
using configurable task-weighting coefficients λ_cls and λ_seg.

The MTL loss has two key effects:
1. The classification gradient is modulated by Focal Loss to handle imbalance.
2. The segmentation gradient forces the shared encoder to focus on lesion
   boundaries, acting as a spatial regularizer against background artifacts.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.focal_loss import FocalLoss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice loss is preferred over BCE alone for medical image segmentation
    because it directly optimizes the overlap metric (Dice score / F1)
    rather than pixel-level accuracy, making it robust to class imbalance
    between foreground (lesion) and background pixels.

    Args:
        smooth: Laplace smoothing factor to prevent division by zero.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, H, W), already sigmoid-activated.
            target: Binary ground-truth mask (B, 1, H, W), values in {0, 1}.

        Returns:
            Scalar Dice loss: 1 - Dice coefficient.
        """
        pred_flat = pred.view(pred.shape[0], -1)      # (B, H*W)
        target_flat = target.view(target.shape[0], -1)  # (B, H*W)

        intersection = (pred_flat * target_flat).sum(dim=1)  # (B,)
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class MTLLoss(nn.Module):
    """
    Multi-Task Loss: λ_cls × FocalLoss + λ_seg × (DiceLoss + BCELoss).

    Args:
        alpha: Per-class alpha tensor for Focal Loss.
        gamma: Focal Loss focusing parameter.
        cls_lambda: Weight for the classification loss component.
        seg_lambda: Weight for the segmentation loss component.
        label_smoothing: Label smoothing for classification.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        cls_lambda: float = 1.0,
        seg_lambda: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.cls_lambda = cls_lambda
        self.seg_lambda = seg_lambda

        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        seg_pred: Optional[torch.Tensor] = None,
        seg_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the combined multi-task loss.

        Args:
            logits: (B, num_classes) — classification logits
            labels: (B,) — integer class labels
            seg_pred: (B, 1, H, W) — predicted segmentation mask (sigmoid output)
            seg_target: (B, 1, H, W) — ground-truth binary mask

        Returns:
            total_loss: Weighted combined scalar loss
            loss_dict: Dict with individual loss components for logging
        """
        cls_loss = self.focal_loss(logits, labels)

        loss_dict = {"cls_loss": cls_loss.item()}

        if seg_pred is not None and seg_target is not None:
            dice = self.dice_loss(seg_pred, seg_target.float())
            bce = self.bce_loss(seg_pred, seg_target.float())
            seg_loss = dice + bce
            loss_dict["dice_loss"] = dice.item()
            loss_dict["bce_loss"] = bce.item()
            loss_dict["seg_loss"] = seg_loss.item()
            total_loss = self.cls_lambda * cls_loss + self.seg_lambda * seg_loss
        else:
            # Classification-only mode (no masks available)
            total_loss = cls_loss
            loss_dict["seg_loss"] = 0.0

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
