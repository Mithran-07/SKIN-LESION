"""
Visualization utilities for Grad-CAM overlays and clinical reporting.

Produces publication-quality figures showing:
- Original dermoscopic image
- Texture branch Grad-CAM (highlighting micro-vascular patterns)
- Structure branch Grad-CAM (highlighting lesion boundary/asymmetry)
- Prediction probabilities bar chart

All visualizations are suitable for clinical reporting and model debugging.
"""

from typing import List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# Use non-interactive backend if running headlessly
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")


CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_COLORS = {
    "MEL": "#e74c3c",
    "NV": "#2ecc71",
    "BCC": "#e67e22",
    "AKIEC": "#9b59b6",
    "BKL": "#3498db",
    "DF": "#1abc9c",
    "VASC": "#f39c12",
}
MALIGNANT_CLASSES = {"MEL", "BCC", "AKIEC", "SCC"}


def _tensor_to_numpy(image_tensor) -> np.ndarray:
    """
    Convert a normalized image tensor to a displayable numpy array.

    Reverses ImageNet normalization and converts to uint8 RGB.
    """
    import torch
    if isinstance(image_tensor, torch.Tensor):
        img = image_tensor.detach().cpu().squeeze(0)  # (3, H, W)
        # Reverse ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        img = img.numpy() * std + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img.transpose(1, 2, 0)  # (H, W, 3)
    return image_tensor


def overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto an original dermoscopic image.

    Args:
        image: Original RGB image (H, W, 3), dtype uint8 or float [0,1].
        cam: Grad-CAM heatmap (H', W'), values in [0, 1].
        alpha: Heatmap opacity (0=invisible, 1=opaque). Default 0.4.
        colormap: OpenCV colormap for heatmap rendering. Default JET.

    Returns:
        Overlay image (H, W, 3) as uint8.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Resize CAM to match image dimensions
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # (H, W, 3) RGB
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def plot_dual_cam(
    image,
    texture_cam: np.ndarray,
    structure_cam: np.ndarray,
    probabilities: np.ndarray,
    pred_class: int,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a 4-panel clinical visualization figure.

    Panels:
        1. Original dermoscopic image
        2. Texture branch Grad-CAM overlay (shallow-wide)
        3. Structure branch Grad-CAM overlay (deep-narrow)
        4. Class probability bar chart with prediction highlighted

    Args:
        image: Original image tensor or numpy array (H, W, 3).
        texture_cam: Grad-CAM from shallow-wide branch (H', W').
        structure_cam: Grad-CAM from deep-narrow branch (H'', W'').
        probabilities: Softmax probability array (num_classes,).
        pred_class: Predicted class index.
        class_names: List of class name strings.
        save_path: If provided, saves figure to this path.
        title: Optional figure title.

    Returns:
        matplotlib Figure object.
    """
    class_names = class_names or CLASS_NAMES
    image_np = _tensor_to_numpy(image)

    texture_overlay = overlay_heatmap(image_np.copy(), texture_cam)
    structure_overlay = overlay_heatmap(image_np.copy(), structure_cam)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#1a1a2e")

    panels = [
        (image_np, "Original Image"),
        (texture_overlay, "Texture CAM\n(Shallow-Wide Branch)"),
        (structure_overlay, "Structure CAM\n(Deep-Narrow Branch)"),
    ]

    for ax, (img, label) in zip(axes[:3], panels):
        ax.imshow(img)
        ax.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#1a1a2e")

    # Probability bar chart
    ax = axes[3]
    ax.set_facecolor("#16213e")
    bar_colors = [
        "#e74c3c" if i == pred_class else
        ("#e67e22" if class_names[i] in MALIGNANT_CLASSES else "#3498db")
        for i in range(len(class_names))
    ]
    bars = ax.barh(
        class_names, probabilities, color=bar_colors, edgecolor="white", linewidth=0.5
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="white", fontsize=10)
    ax.set_title(
        f"Prediction: {class_names[pred_class]}",
        color="#e74c3c" if class_names[pred_class] in MALIGNANT_CLASSES else "#2ecc71",
        fontsize=12, fontweight="bold"
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    # Add probability labels
    for bar, prob in zip(bars, probabilities):
        ax.text(
            min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}", va="center", color="white", fontsize=8
        )

    if title:
        fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Figure saved: {save_path}")

    return fig
