"""
Grad-CAM implementation for the Dual-Branch CNN.

Gradient-weighted Class Activation Mapping (Grad-CAM) provides clinical
interpretability by highlighting the specific image regions that were most
influential in the model's classification decision.

For the dual-branch architecture, TWO heatmaps are generated per image:
1. Texture CAM: From the last conv layer of the shallow-wide branch
   → Should highlight localized micro-vascular patterns, surface textures
2. Structure CAM: From the bottleneck of the deep-narrow branch
   → Should highlight global lesion boundary and asymmetry regions

Clinical validation:
   If the Grad-CAM highlights artifacts (rulers, hair, frame borders)
   rather than lesion tissue, the model has learned spurious correlations
   and must be retrained with improved augmentation or data curation.

Reference:
   Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from
   Deep Networks via Gradient-based Localization. ICCV 2017.
"""

from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMHook:
    """
    Registers forward and backward hooks on a target layer to capture
    activations and gradients for Grad-CAM computation.
    """

    def __init__(self, layer: nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_hook = layer.register_forward_hook(self._save_activations)
        self._bwd_hook = layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        """Deregister both hooks. Always call this after use to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute_cam(self) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap from captured activations and gradients.

        Steps:
            1. Global average pool the gradients → importance weights α_k
            2. Weighted sum of activation maps: CAM = Σ_k α_k * A_k
            3. ReLU to isolate positive contributions
            4. Normalize to [0, 1]

        Returns:
            Normalized heatmap as numpy array of shape (H, W), values in [0, 1].
        """
        assert self.gradients is not None, "No gradients captured. Run backward first."
        assert self.activations is not None, "No activations captured. Run forward first."

        # (1, C, H, W) → global avg pool → (1, C, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # α_k
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Keep only positive contributions
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


class DualBranchGradCAM:
    """
    Dual-branch Grad-CAM: generates separate heatmaps for texture and structure.

    Args:
        model: A DualBranchNet instance.
        shallow_target_layer: The nn.Module in the shallow branch to hook
                              (default: block3.conv — last conv of shallow branch).
        deep_target_layer: The nn.Module in the deep branch to hook
                           (default: stage4 last conv — deepest structure layer).

    Usage:
        gradcam = DualBranchGradCAM(model)
        texture_cam, structure_cam, pred_class = gradcam.generate(
            image_tensor, target_class=2  # BCC
        )
        gradcam.remove_hooks()  # Important: always cleanup
    """

    def __init__(
        self,
        model: nn.Module,
        shallow_target_layer: Optional[nn.Module] = None,
        deep_target_layer: Optional[nn.Module] = None,
    ):
        self.model = model
        self.model.eval()

        # Auto-detect target layers if not provided
        if shallow_target_layer is None:
            shallow_target_layer = model.shallow_branch.block3.conv
        if deep_target_layer is None:
            deep_target_layer = model.deep_branch.stage4.conv_layers[-2]  # Last Conv2d

        self.shallow_hook = GradCAMHook(shallow_target_layer)
        self.deep_hook = GradCAMHook(deep_target_layer)

    def generate(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate Grad-CAM heatmaps for both branches.

        Args:
            image: Input tensor of shape (1, 3, H, W) — single image, batch=1.
            target_class: Class index for gradient computation.
                          If None, uses the predicted (argmax) class.
            device: Device to run computation on.

        Returns:
            texture_cam: (H', W') heatmap from the shallow-wide branch
            structure_cam: (H'', W'') heatmap from the deep-narrow branch
            pred_class: The predicted class index
        """
        if device is not None:
            image = image.to(device)
            self.model = self.model.to(device)

        # ── Forward pass ────────────────────────────────────────────────────
        image.requires_grad_(True)
        logits, _, _ = self.model(image)
        pred_class = logits.argmax(dim=1).item()

        if target_class is None:
            target_class = pred_class

        # ── Backward pass for target class ──────────────────────────────────
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        # ── Compute CAMs ────────────────────────────────────────────────────
        texture_cam = self.shallow_hook.compute_cam()    # (H', W')
        structure_cam = self.deep_hook.compute_cam()     # (H'', W'')

        return texture_cam, structure_cam, pred_class

    def remove_hooks(self) -> None:
        """Deregister all hooks. MUST be called after generation to free memory."""
        self.shallow_hook.remove()
        self.deep_hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()
