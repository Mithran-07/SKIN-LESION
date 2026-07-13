"""
Metrics tracker for dermoscopic classification evaluation.

Key metrics for imbalanced medical classification:
- Macro-AUC: AUC averaged equally across all classes (not biased by NV dominance)
- Macro-F1: F1 score averaged per class
- Balanced Accuracy: Mean per-class recall (handles imbalance)
- Per-Class Recall: Critical for clinical deployment — we must not miss BCC/AKIEC/SCC
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


class MetricTracker:
    """
    Accumulates predictions and targets over an epoch, then computes all metrics.

    Args:
        num_classes: Number of classes.
        class_names: Human-readable class names for reporting.
    """

    def __init__(self, num_classes: int = 7, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated predictions."""
        self._all_probs: List[np.ndarray] = []
        self._all_preds: List[int] = []
        self._all_targets: List[int] = []
        self._loss_sum: float = 0.0
        self._n_batches: int = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
    ) -> None:
        """
        Accumulate one batch of predictions.

        Args:
            logits: (B, num_classes) raw logits from the model.
            targets: (B,) ground truth integer labels.
            loss: Scalar loss value for this batch.
        """
        probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
        preds = probs.argmax(axis=1).tolist()
        self._all_probs.append(probs)
        self._all_preds.extend(preds)
        self._all_targets.extend(targets.cpu().numpy().tolist())
        self._loss_sum += loss
        self._n_batches += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics over the accumulated epoch.

        Returns:
            Dictionary of metric name → float value.
        """
        all_probs = np.vstack(self._all_probs)   # (N, C)
        all_preds = np.array(self._all_preds)    # (N,)
        all_targets = np.array(self._all_targets)  # (N,)

        metrics: Dict[str, float] = {}

        # ── Loss ────────────────────────────────────────────────────────────
        metrics["loss"] = self._loss_sum / max(self._n_batches, 1)

        # ── Macro AUC ───────────────────────────────────────────────────────
        try:
            metrics["auc_macro"] = roc_auc_score(
                all_targets, all_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["auc_macro"] = 0.0  # Occurs if a class has 0 samples in split

        # ── Macro F1 ────────────────────────────────────────────────────────
        metrics["f1_macro"] = f1_score(
            all_targets, all_preds, average="macro", zero_division=0
        )

        # ── Balanced Accuracy ───────────────────────────────────────────────
        metrics["balanced_accuracy"] = balanced_accuracy_score(all_targets, all_preds)

        # ── Per-Class Recall ────────────────────────────────────────────────
        recalls = recall_score(
            all_targets, all_preds, average=None, zero_division=0, labels=list(range(self.num_classes))
        )
        for i, (name, rec) in enumerate(zip(self.class_names, recalls)):
            metrics[f"recall_{name}"] = rec

        return metrics

    def classification_report(self) -> str:
        """Return a formatted sklearn classification report."""
        return classification_report(
            np.array(self._all_targets),
            np.array(self._all_preds),
            target_names=self.class_names,
            zero_division=0,
        )

    def confusion_matrix(self) -> np.ndarray:
        """Return confusion matrix of shape (num_classes, num_classes)."""
        return confusion_matrix(
            np.array(self._all_targets),
            np.array(self._all_preds),
            labels=list(range(self.num_classes)),
        )
