"""
Imbalanced dataset sampler using inverse-frequency class weights.

Addresses the severe class imbalance in HAM10000 where NV accounts for
~67% of all samples while DF and VASC have fewer than 150 samples each.
"""

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class ImbalancedDatasetSampler(WeightedRandomSampler):
    """
    A WeightedRandomSampler that automatically computes per-sample weights
    based on inverse class frequency.

    For HAM10000 with counts [MEL:1113, NV:6705, BCC:514, AKIEC:327,
    BKL:1099, DF:115, VASC:142], this upsamples rare classes so each
    class appears approximately equally in each mini-batch.

    Args:
        dataset: A dataset with a `get_labels()` method returning int labels.
        num_samples: Number of samples to draw per epoch. Defaults to len(dataset).
        replacement: Sample with replacement (True) or without (False).
    """

    def __init__(
        self,
        dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        labels = np.array(dataset.get_labels())
        num_classes = len(np.unique(labels))

        # Compute per-class weight = 1 / class_frequency
        class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
        class_counts = np.where(class_counts == 0, 1, class_counts)  # Avoid divide-by-zero
        class_weights = 1.0 / class_counts

        # Assign per-sample weight
        sample_weights = torch.tensor(class_weights[labels], dtype=torch.float)

        n = num_samples if num_samples is not None else len(dataset)
        super().__init__(weights=sample_weights, num_samples=n, replacement=replacement)


def compute_class_weights_tensor(
    labels: List[int], num_classes: int, device: torch.device
) -> torch.Tensor:
    """
    Compute normalized inverse-frequency class weights as a 1D tensor.

    Used to initialize the alpha vector in FocalLoss. Weights are
    normalized so they sum to num_classes for stable gradient scaling.

    Args:
        labels: List of integer class labels from the training set.
        num_classes: Total number of classes.
        device: Target device for the returned tensor.

    Returns:
        Tensor of shape (num_classes,) with per-class alpha weights.
    """
    labels_arr = np.array(labels)
    counts = np.bincount(labels_arr, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    # Normalize so the mean weight is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)
