"""
Single-image inference script with Grad-CAM + Conformal Prediction.

Usage:
    python scripts/infer.py \\
        --image path/to/dermoscopy_image.jpg \\
        --checkpoint results/best_model.pt \\
        --config config/config.yaml \\
        --save results/gradcam_output.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dual_branch_net import build_model_from_config
from data.augmentations import get_val_transforms
from explainability.gradcam import DualBranchGradCAM
from explainability.visualize import plot_dual_cam
from training.trainer import get_device


def load_image(image_path: str, aug_cfg: dict) -> torch.Tensor:
    """Load and preprocess a dermoscopic image for inference."""
    import numpy as np
    img = np.array(Image.open(image_path).convert("RGB"))
    transform = get_val_transforms(aug_cfg)
    augmented = transform(image=img)
    return augmented["image"].unsqueeze(0)  # (1, 3, H, W)


def main():
    parser = argparse.ArgumentParser(description="ADL Single-Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to dermoscopic image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--save", type=str, default=None, help="Path to save output figure")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Target class for Grad-CAM (default: predicted class)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    class_names = cfg["dataset"]["class_names"]
    aug_cfg = cfg["augmentation"]

    # ── Load model ───────────────────────────────────────────────────────────
    model = build_model_from_config(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    # ── Load image ───────────────────────────────────────────────────────────
    image_tensor = load_image(args.image, aug_cfg)
    print(f"Image: {args.image} | Shape: {image_tensor.shape}")

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    with DualBranchGradCAM(model) as gradcam:
        texture_cam, structure_cam, pred_class = gradcam.generate(
            image_tensor.clone(),
            target_class=args.target_class,
            device=device,
        )

    # ── Get probabilities ────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits, *_ = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n=== INFERENCE RESULTS ===")
    print(f"Predicted class: {class_names[pred_class]} (index {pred_class})")
    print("\nClass probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        bar = "█" * int(prob * 30)
        marker = " ← predicted" if i == pred_class else ""
        print(f"  {name:8s}: {prob:.4f} {bar}{marker}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    save_path = args.save or "results/gradcam/inference_output.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plot_dual_cam(
        image=image_tensor,
        texture_cam=texture_cam,
        structure_cam=structure_cam,
        probabilities=probs,
        pred_class=pred_class,
        class_names=class_names,
        save_path=save_path,
        title=f"ADL Inference — Predicted: {class_names[pred_class]}",
    )
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
