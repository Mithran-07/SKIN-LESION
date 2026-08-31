"""Multi-class ROC plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


def plot_roc_curves(fpr: Mapping[str, Sequence[float]], tpr: Mapping[str, Sequence[float]],
                    auc_scores: Mapping[str, float], save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in fpr:
        ax.plot(fpr[label], tpr[label], label=f"{label} (AUC={auc_scores.get(label, float('nan')):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OvR ROC Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig