"""Architecture diagram utilities for publication figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_architecture_diagram(save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    boxes = [
        (0.05, 0.4, 0.18, 0.2, "Input"),
        (0.28, 0.62, 0.2, 0.2, "Shallow\nTexture Branch"),
        (0.28, 0.18, 0.2, 0.2, "Deep\nStructure Branch"),
        (0.55, 0.4, 0.15, 0.2, "Fusion"),
        (0.77, 0.4, 0.18, 0.2, "Classifier"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
    arrows = [
        ((0.23, 0.5), (0.28, 0.72)),
        ((0.23, 0.5), (0.28, 0.28)),
        ((0.48, 0.72), (0.55, 0.5)),
        ((0.48, 0.28), (0.55, 0.5)),
        ((0.70, 0.5), (0.77, 0.5)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2))
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig