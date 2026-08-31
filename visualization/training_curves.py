"""Training curve plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


def plot_training_curves(history: Mapping[str, Sequence[float]], save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in history.items():
        ax.plot(values, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig