"""Feature map grid plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_maps(feature_maps, max_channels: int = 16, save_path: str | None = None):
    fmap = np.asarray(feature_maps)
    channels = min(fmap.shape[0], max_channels)
    cols = min(4, channels)
    rows = int(np.ceil(channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for idx in range(rows * cols):
        ax = axes[idx]
        ax.axis("off")
        if idx < channels:
            ax.imshow(fmap[idx], cmap="magma")
            ax.set_title(f"Ch {idx}")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig