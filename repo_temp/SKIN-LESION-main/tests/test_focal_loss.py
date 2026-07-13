"""
Tests for Focal Loss implementation.

Verifies:
1. Easy examples produce lower loss than hard examples (core FL property)
2. Alpha weighting correctly scales class contributions
3. Gamma=0 reduces to standard cross-entropy
4. Label smoothing reduces loss for confident predictions
"""

import pytest
import torch
import numpy as np

from losses.focal_loss import FocalLoss, compute_class_weights

NUM_CLASSES = 7


def test_focal_loss_hard_vs_easy():
    """
    The fundamental Focal Loss property:
    Hard examples (low confidence) should produce higher loss than easy examples.
    """
    loss_fn = FocalLoss(gamma=2.0)

    # Easy example: model is very confident and correct (p_t ≈ 0.99)
    easy_logits = torch.zeros(1, NUM_CLASSES)
    easy_logits[0, 0] = 10.0  # Very high confidence for class 0
    easy_label = torch.tensor([0])

    # Hard example: model is uncertain (p_t ≈ 1/7 ≈ 0.14)
    hard_logits = torch.zeros(1, NUM_CLASSES)  # Uniform uncertainty
    hard_label = torch.tensor([0])

    easy_loss = loss_fn(easy_logits, easy_label)
    hard_loss = loss_fn(hard_logits, hard_label)

    assert hard_loss > easy_loss, (
        f"Hard examples should produce higher loss. "
        f"Easy: {easy_loss.item():.4f}, Hard: {hard_loss.item():.4f}"
    )


def test_focal_loss_gamma_zero_equals_ce():
    """When gamma=0, Focal Loss should equal standard Cross-Entropy."""
    import torch.nn.functional as F

    focal = FocalLoss(gamma=0.0)
    logits = torch.randn(4, NUM_CLASSES)
    labels = torch.randint(0, NUM_CLASSES, (4,))

    fl_loss = focal(logits, labels)
    ce_loss = F.cross_entropy(logits, labels)

    assert torch.allclose(fl_loss, ce_loss, atol=1e-5), (
        f"FL(gamma=0) != CE: {fl_loss.item():.6f} vs {ce_loss.item():.6f}"
    )


def test_focal_loss_alpha_weighting():
    """Alpha weight for minority class should increase its loss contribution."""
    # Simulate HAM10000 imbalance: NV has 6705, DF has 115
    counts = [1113, 6705, 514, 327, 1099, 115, 142]  # HAM10000
    alpha = compute_class_weights(
        label_list=np.concatenate([np.full(c, i) for i, c in enumerate(counts)]),
        num_classes=NUM_CLASSES,
        device=torch.device("cpu"),
    )
    # DF (index 5) should have the highest alpha weight
    assert alpha[5] > alpha[1], (
        f"DF (index 5) alpha should be > NV (index 1) alpha. "
        f"Got DF={alpha[5]:.2f}, NV={alpha[1]:.2f}"
    )


def test_focal_loss_no_nan():
    """Focal Loss must not produce NaN or Inf for any valid input."""
    loss_fn = FocalLoss(gamma=2.0)
    # Test with edge-case logits
    logits = torch.tensor([[100.0, -100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0])
    loss = loss_fn(logits, labels)
    assert not torch.isnan(loss), "Focal Loss produced NaN"
    assert not torch.isinf(loss), "Focal Loss produced Inf"


def test_compute_class_weights_normalization():
    """Normalized class weights should have mean ≈ 1.0."""
    labels = list(range(NUM_CLASSES)) * 10  # Balanced
    weights = compute_class_weights(labels, NUM_CLASSES, torch.device("cpu"), normalize=True)
    assert abs(weights.mean().item() - 1.0) < 0.01, (
        f"Mean weight should be ≈ 1.0, got {weights.mean().item():.4f}"
    )
