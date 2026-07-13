"""
Monte Carlo Dropout for uncertainty estimation.

MC-Dropout approximates Bayesian inference by performing multiple stochastic
forward passes with dropout enabled at inference time. The variance across
N samples estimates the model's epistemic (knowledge) uncertainty.

Used as a COMPARISON BASELINE against Conformal Prediction.
CP is preferred for clinical deployment because:
- CP provides a mathematical coverage guarantee
- MC-Dropout lacks formal guarantees and is sensitive to dropout placement
- CP is computationally cheaper (single forward pass at inference)

However, MC-Dropout provides useful uncertainty decomposition:
- Aleatoric uncertainty: inherent data noise (variance in mean predictions)
- Epistemic uncertainty: model uncertainty (reducible with more data)
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference (override model.eval() behavior)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class MCDropoutPredictor:
    """
    Monte Carlo Dropout predictor for uncertainty quantification.

    Args:
        n_samples: Number of stochastic forward passes (default 50).
        class_names: Class name strings for output formatting.
    """

    def __init__(
        self,
        n_samples: int = 50,
        class_names: Optional[list] = None,
    ):
        self.n_samples = n_samples
        self.class_names = class_names or [f"Class_{i}" for i in range(7)]

    def predict(
        self,
        model: nn.Module,
        image: torch.Tensor,
        device: torch.device,
    ) -> Dict:
        """
        Run N stochastic forward passes and aggregate results.

        Args:
            model: Trained classifier with Dropout layers.
            image: Input tensor (1, 3, H, W).
            device: Device.

        Returns:
            Dictionary with mean predictions, variance, and entropy.
        """
        model.eval()            # Disable BN train mode
        enable_dropout(model)   # Re-enable Dropout for stochastic inference

        all_probs = []
        image = image.to(device)

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits, *_ = model(image)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                all_probs.append(probs)

        all_probs = np.stack(all_probs, axis=0)  # (N, C)

        mean_probs = all_probs.mean(axis=0)       # (C,)
        var_probs = all_probs.var(axis=0)         # (C,) — epistemic uncertainty
        pred_class = int(mean_probs.argmax())

        # Predictive entropy — total uncertainty
        epsilon = 1e-8
        entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon))

        # Mutual information — epistemic uncertainty
        sample_entropies = -np.sum(all_probs * np.log(all_probs + epsilon), axis=1)
        mutual_info = entropy - sample_entropies.mean()

        return {
            "mean_probabilities": mean_probs,
            "variance": var_probs,
            "pred_class": pred_class,
            "pred_class_name": self.class_names[pred_class],
            "predictive_entropy": float(entropy),
            "mutual_information": float(mutual_info),
            "n_samples": self.n_samples,
        }
