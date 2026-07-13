"""
Metrics Module — Phase 3
Computes all classification metrics per epoch.
"""

import numpy as np
import torch
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize


NUM_CLASSES = 7
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class MetricTracker:
    """
    Accumulates predictions + labels across batches,
    computes all metrics at epoch end.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, class_names: list = CLASS_NAMES):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self._all_preds  = []   # integer predictions
        self._all_probs  = []   # softmax probabilities (N, C)
        self._all_labels = []   # true integer labels
        self._total_loss = 0.0
        self._n_batches  = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float):
        """
        Args:
            logits: (B, C) raw model output
            labels: (B,) true integer labels
            loss: scalar loss for this batch
        """
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=1)
            preds = torch.argmax(probs, dim=1)

        self._all_preds.extend(preds.cpu().numpy().tolist())
        self._all_probs.extend(probs.cpu().numpy().tolist())
        self._all_labels.extend(labels.cpu().numpy().tolist())
        self._total_loss += loss
        self._n_batches  += 1

    def compute(self) -> dict:
        """Return dict with all metrics for the epoch."""
        y_true  = np.array(self._all_labels)
        y_pred  = np.array(self._all_preds)
        y_probs = np.array(self._all_probs)   # (N, C)

        # ── Basic
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        avg_loss = self._total_loss / max(self._n_batches, 1)

        # ── Macro metrics
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # ── ROC-AUC (requires probabilities)
        try:
            y_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
            roc_auc = roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr")
        except Exception:
            roc_auc = float("nan")

        # ── PR-AUC (macro)
        try:
            pr_auc_per_class = [
                average_precision_score(y_bin[:, c], y_probs[:, c])
                for c in range(self.num_classes)
            ]
            pr_auc = float(np.mean(pr_auc_per_class))
        except Exception:
            pr_auc = float("nan")

        # ── Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

        # ── Per-class report
        clf_report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0,
            output_dict=True
        )

        return {
            "loss":              avg_loss,
            "accuracy":          acc,
            "balanced_accuracy": bal_acc,
            "precision_macro":   precision,
            "recall_macro":      recall,
            "f1_macro":          f1,
            "roc_auc":           roc_auc,
            "pr_auc":            pr_auc,
            "confusion_matrix":  cm,
            "clf_report":        clf_report,
            "y_true":            y_true,
            "y_pred":            y_pred,
            "y_probs":           y_probs,
        }

    def has_nan(self) -> bool:
        """Check for NaN values in accumulated predictions."""
        if not self._all_probs:
            return False
        arr = np.array(self._all_probs)
        return bool(np.any(np.isnan(arr)) or np.any(np.isinf(arr)))


def format_metrics(metrics: dict) -> str:
    """Format key metrics as a one-line string for logging."""
    return (
        f"loss={metrics['loss']:.4f} | "
        f"acc={metrics['accuracy']:.4f} | "
        f"bal_acc={metrics['balanced_accuracy']:.4f} | "
        f"f1={metrics['f1_macro']:.4f} | "
        f"auc={metrics['roc_auc']:.4f}"
    )
