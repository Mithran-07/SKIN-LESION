"""
Focal Loss with per-class alpha weighting for imbalanced dermoscopic datasets.

Focal Loss addresses the fundamental failure of standard Cross-Entropy in
heavily imbalanced datasets like HAM10000 (NV class: 6705 samples vs.
DF class: 115 samples — a 58:1 imbalance ratio).

Mathematical formulation:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Where:
    p_t   = model's predicted probability for the ground-truth class
    α_t   = per-class balancing factor (inverse class frequency)
    γ     = focusing parameter (typically 2.0)
              → When p_t → 1: (1-p_t)^γ → 0, down-weights easy examples
              → When p_t → 0: (1-p_t)^γ → 1, maintains loss for hard examples

Effect on HAM10000 training:
    - The network cannot coast on NV (nevi) predictions
    - BCC, AKIEC, DF, VASC minority classes receive proportionally higher gradients
    - Prevents the 'majority class collapse' that plagues naive CE training
"""

from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    label_list: List[int],
    num_classes: int,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute per-class alpha weights from inverse class frequency.

    Args:
        label_list: Integer labels from the training set.
        num_classes: Total number of classes.
        device: Target device.
        normalize: If True, normalize weights to have mean = 1.0.

    Returns:
        Tensor of shape (num_classes,) containing per-class alpha weights.
    """
    labels = np.array(label_list)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)  # Avoid division by zero
    weights = 1.0 / counts
    if normalize:
        weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class dermoscopic classification.

    Supports both auto-computed alpha from inverse class frequency and
    manually specified alpha vectors.

    Args:
        alpha: Per-class weight tensor of shape (num_classes,).
               If None, uniform weights are used.
        gamma: Focusing parameter. Higher values focus more on hard examples.
               Typical values: 0.5 (mild), 2.0 (standard), 5.0 (aggressive).
        reduction: 'mean' | 'sum' | 'none'
        label_smoothing: Label smoothing factor (0.0–0.2). Reduces overconfidence.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            logits: Raw model outputs (B, num_classes) — unnormalized scores.
            targets: Ground-truth integer labels (B,).

        Returns:
            Scalar loss value.
        """
        num_classes = logits.shape[1]
        B = logits.shape[0]

        # ── Compute log-softmax and probabilities ───────────────────────────
        log_probs = F.log_softmax(logits, dim=1)   # (B, C)
        probs = torch.exp(log_probs)               # (B, C)

        # ── Label smoothing ─────────────────────────────────────────────────
        if self.label_smoothing > 0:
            # Soft targets: one-hot * (1 - ε) + ε / C
            with torch.no_grad():
                smooth_targets = torch.full_like(log_probs, self.label_smoothing / num_classes)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            # Cross-entropy with smooth targets
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)  # (B,)
            # p_t from hard labels for focal modulation
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        else:
            # Standard cross-entropy: -log(p_t)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
            ce_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # ── Focal modulation: (1 - p_t)^γ ──────────────────────────────────
        focal_weight = (1.0 - p_t) ** self.gamma  # (B,)

        # ── Per-class alpha weighting ───────────────────────────────────────
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (B,)
            focal_loss = alpha_t * focal_weight * ce_loss  # (B,)
        else:
            focal_loss = focal_weight * ce_loss  # (B,)

        # ── Reduction ───────────────────────────────────────────────────────
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

    def __repr__(self) -> str:
        return (
            f"FocalLoss(gamma={self.gamma}, reduction='{self.reduction}', "
            f"label_smoothing={self.label_smoothing}, "
            f"alpha={'auto' if self.alpha is not None else 'uniform'})"
        )
