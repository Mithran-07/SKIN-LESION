"""
Split Conformal Prediction for dermoscopic classification.

Conformal Prediction (CP) provides mathematically guaranteed prediction sets
with user-specified coverage. Unlike softmax outputs which are often
overconfident, CP gives a set of candidate classes that contains the true
diagnosis with probability ≥ 1 - α.

Clinical interpretation:
    Set size = 1  → Confident single diagnosis (e.g., "BCC")
    Set size > 1  → Ambiguous lesion → refer to dermatologist or mandate biopsy
    Set size = C  → Model is uncertain about this lesion type

Mathematical guarantee:
    P(Y_true ∈ Ĉ(X)) ≥ 1 - α
    where α is the user-specified miscoverage rate (default 0.1 → 90% coverage)

Method: Split Conformal Prediction (Angelopoulos & Bates, 2021)
    1. Calibration: Compute non-conformity scores s_i = 1 - p̂(y_i | x_i)
       on a held-out calibration set.
    2. Threshold: q̂ = quantile((1-α)(1 + 1/n)) of the calibration scores.
    3. Prediction: Ĉ(x) = {y : 1 - p̂(y | x) ≤ q̂}

Reference:
    Angelopoulos, A.N. & Bates, S. (2021). A Gentle Introduction to
    Conformal Prediction and Distribution-Free Uncertainty Quantification.
"""

from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class SplitConformalPredictor:
    """
    Split Conformal Predictor with LAC (Least Ambiguous Classifier) non-conformity scores.

    Args:
        alpha: Miscoverage level (0 < α < 1). Default 0.1 → 90% coverage.
        class_names: Human-readable class names for output formatting.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        class_names: Optional[List[str]] = None,
    ):
        assert 0 < alpha < 1, "Alpha must be in (0, 1)"
        self.alpha = alpha
        self.class_names = class_names or [f"Class_{i}" for i in range(7)]
        self.q_hat: Optional[float] = None
        self._cal_scores: Optional[np.ndarray] = None
        self._is_calibrated: bool = False

    def calibrate(
        self,
        model: nn.Module,
        cal_loader: DataLoader,
        device: torch.device,
    ) -> float:
        """
        Compute the calibration quantile from a held-out calibration set.

        Args:
            model: Trained classifier (frozen weights).
            cal_loader: DataLoader for the calibration set.
            device: Device to run inference on.

        Returns:
            q_hat: The calibrated threshold value.
        """
        model.eval()
        all_scores = []

        with torch.no_grad():
            for batch in tqdm(cal_loader, desc="Calibrating CP"):
                if isinstance(batch, dict):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                else:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                logits, *_ = model(images)
                probs = torch.softmax(logits, dim=1)  # (B, C)

                # Non-conformity score: 1 - p̂(true class | x)
                true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # (B,)
                scores = (1.0 - true_probs).cpu().numpy()  # High score = harder example
                all_scores.extend(scores.tolist())

        all_scores = np.array(all_scores)
        self._cal_scores = all_scores
        n = len(all_scores)

        # Finite-sample corrected quantile level
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)  # Clip to valid range
        self.q_hat = float(np.quantile(all_scores, level))
        self._is_calibrated = True

        print(f"[Conformal] Calibrated on {n} samples | q_hat = {self.q_hat:.4f} (alpha={self.alpha})")
        return self.q_hat

    def predict(
        self,
        model: nn.Module,
        image: torch.Tensor,
        device: torch.device,
    ) -> Dict:
        """
        Generate a conformal prediction set for a single image.

        Args:
            model: Trained classifier (frozen).
            image: Input tensor (1, 3, H, W).
            device: Device.

        Returns:
            Dictionary with:
                - prediction_set: List of class names in the prediction set
                - prediction_set_indices: List of class indices
                - probabilities: Softmax probability array
                - pred_class: Most likely class (argmax)
                - confident: True if set size == 1
                - q_hat: Calibration threshold used
        """
        assert self._is_calibrated, "Call calibrate() before predict()."
        model.eval()

        with torch.no_grad():
            image = image.to(device)
            logits, *_ = model(image)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C,)

        # Prediction set: all classes where non-conformity score ≤ q̂
        non_conformity = 1.0 - probs  # (C,)
        set_indices = [i for i, s in enumerate(non_conformity) if s <= self.q_hat]

        # If set is empty (shouldn't happen with valid q̂), include top-1
        if len(set_indices) == 0:
            set_indices = [int(probs.argmax())]

        return {
            "prediction_set": [self.class_names[i] for i in set_indices],
            "prediction_set_indices": set_indices,
            "probabilities": probs,
            "pred_class": int(probs.argmax()),
            "pred_class_name": self.class_names[int(probs.argmax())],
            "set_size": len(set_indices),
            "confident": len(set_indices) == 1,
            "q_hat": self.q_hat,
        }

    def coverage_report(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
    ) -> Dict:
        """
        Compute empirical coverage and average set size on the test set.

        Empirical coverage should be ≥ 1 - α (the mathematical guarantee).
        Average set size measures how informative the predictions are.

        Args:
            model, test_loader, device: Standard inference arguments.

        Returns:
            Dictionary with coverage, avg_set_size, set_size_distribution.
        """
        assert self._is_calibrated, "Call calibrate() before coverage_report()."
        model.eval()

        covered = []
        set_sizes = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Computing Coverage"):
                if isinstance(batch, dict):
                    images = batch["image"].to(device)
                    labels = batch["label"].cpu().numpy()
                else:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.cpu().numpy()
                logits, *_ = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, C)

                non_conformity = 1.0 - probs  # (B, C)
                for i in range(len(labels)):
                    set_idx = np.where(non_conformity[i] <= self.q_hat)[0].tolist()
                    if len(set_idx) == 0:
                        set_idx = [int(probs[i].argmax())]
                    covered.append(int(labels[i]) in set_idx)
                    set_sizes.append(len(set_idx))

        coverage = float(np.mean(covered))
        avg_set_size = float(np.mean(set_sizes))

        print(f"[Conformal] Coverage: {coverage:.4f} (target: {1-self.alpha:.2f})")
        print(f"[Conformal] Avg Set Size: {avg_set_size:.2f}")

        return {
            "empirical_coverage": coverage,
            "target_coverage": 1.0 - self.alpha,
            "avg_set_size": avg_set_size,
            "set_size_distribution": np.bincount(set_sizes).tolist(),
            "q_hat": self.q_hat,
        }
