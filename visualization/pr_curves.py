"""Precision-recall plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


def plot_pr_curves(recall: Mapping[str, Sequence[float]], precision: Mapping[str, Sequence[float]],
                   ap_scores: Mapping[str, float], save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in recall:
        ax.plot(recall[label], precision[label], label=f"{label} (AP={ap_scores.get(label, float('nan')):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig