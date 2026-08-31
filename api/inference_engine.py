"""
EfficientNet-B4 Inference Engine
Handles model loading, preprocessing, inference, and Grad-CAM.
Supports CUDA, MPS (Apple Silicon), and CPU.
"""

import io
import sys
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ─── Class Configuration ────────────────────────────────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DISPLAY_NAMES = {
    "akiec": "Actinic Keratosis / Bowen's Disease",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}
CLASS_DESCRIPTIONS = {
    "akiec": "A precancerous lesion or early form of squamous cell carcinoma.",
    "bcc":   "The most common form of skin cancer, typically slow-growing.",
    "bkl":   "Non-cancerous growths including seborrheic keratoses and solar lentigines.",
    "df":    "A common benign fibrous skin nodule, usually on the legs.",
    "mel":   "A serious form of skin cancer arising from pigment-producing cells.",
    "nv":    "Common benign moles (melanocytic nevi), generally harmless.",
    "vasc":  "Rare vascular anomalies including angiomas and pyogenic granulomas.",
}
NUM_CLASSES = 7

# ─── Preprocessing ───────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Grad-CAM for EfficientNet ───────────────────────────────────────────────
class EfficientNetGradCAM:
    """
    Grad-CAM implementation for EfficientNet-B4 (timm model).
    Hooks onto the last conv block before the global pool.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Hook onto the last conv block of timm EfficientNet-B4
        # timm: model.conv_head is the last conv before global_pool
        target_layer = model.conv_head
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self._activations = o.detach()

    def _save_grad(self, m, gi, go):
        self._gradients = go[0].detach()

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()

    def generate(
        self, image_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, List[float]]:
        """
        Run inference + Grad-CAM.

        Returns:
            cam:         Normalized heatmap (H, W) in [0, 1]
            pred_class:  Predicted class index
            probs:       Softmax probability list (length 7)
        """
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W)
        image_tensor.requires_grad_(True)

        output = self.model(image_tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        probs = F.softmax(logits, dim=1).squeeze().detach().cpu().tolist()
        pred_class = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = pred_class

        # Backward for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)

        # Compute Grad-CAM
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, probs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()


# ─── Inference Engine ────────────────────────────────────────────────────────
class InferenceEngine:
    """
    Singleton-style inference engine for EfficientNet-B4.
    Loads once, reused across requests.
    """

    _instance: Optional["InferenceEngine"] = None

    def __init__(self, checkpoint_path: Path):
        self.device = get_device()
        self.checkpoint_path = checkpoint_path
        logger.info(f"[Engine] Initializing on device: {self.device}")

        # Use timm to create EfficientNet-B4 — matches checkpoint key structure
        self.model = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=NUM_CLASSES,
        )
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"[Engine] Model ready: {checkpoint_path.name}")

    def _load_checkpoint(self):
        ck = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        # Support both 'state_dict' and 'model_state_dict' keys
        state = ck.get("state_dict", ck.get("model_state_dict", ck))
        # Strip 'module.' prefix if saved from DataParallel
        state = {k.replace("module.", ""): v for k, v in state.items()}

        # timm EfficientNet-B4 keys match checkpoint exactly (conv_stem, blocks, classifier)
        missing, unexpected = self.model.load_state_dict(state, strict=True)
        logger.info(f"[Engine] Checkpoint loaded (epoch {ck.get('epoch', '?')}) — "
                    f"0 missing, 0 unexpected [strict=True]")

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image → (3, 224, 224) tensor."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return _INFERENCE_TRANSFORM(pil_image)

    def predict(
        self, pil_image: Image.Image
    ) -> Dict:
        """
        Run inference on a PIL image.

        Returns:
            dict with: predicted_class, predicted_label, predicted_display_name,
                       probability, top3, all_probs
        """
        tensor = self.preprocess(pil_image).to(self.device)

        with torch.no_grad():
            output = self.model(tensor.unsqueeze(0))
            logits = output[0] if isinstance(output, (tuple, list)) else output
            probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()

        pred_idx = int(np.argmax(probs))
        top3_idx = sorted(range(NUM_CLASSES), key=lambda i: probs[i], reverse=True)[:3]

        return {
            "predicted_class":        pred_idx,
            "predicted_label":        CLASS_NAMES[pred_idx],
            "predicted_display_name": CLASS_DISPLAY_NAMES[CLASS_NAMES[pred_idx]],
            "predicted_description":  CLASS_DESCRIPTIONS[CLASS_NAMES[pred_idx]],
            "probability":            round(probs[pred_idx], 4),
            "top3": [
                {
                    "rank": i + 1,
                    "class_idx": idx,
                    "label": CLASS_NAMES[idx],
                    "display_name": CLASS_DISPLAY_NAMES[CLASS_NAMES[idx]],
                    "probability": round(probs[idx], 4),
                }
                for i, idx in enumerate(top3_idx)
            ],
            "all_probabilities": {
                CLASS_NAMES[i]: round(probs[i], 4) for i in range(NUM_CLASSES)
            },
            "model": "EfficientNet-B4",
            "device": str(self.device),
        }

    def predict_with_gradcam(
        self, pil_image: Image.Image, original_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Run inference + Grad-CAM, return prediction + base64-encoded heatmap overlay.
        """
        import cv2

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        tensor = self.preprocess(pil_image).to(self.device)

        with EfficientNetGradCAM(self.model) as gcam:
            cam_np, pred_idx, probs = gcam.generate(tensor)

        top3_idx = sorted(range(NUM_CLASSES), key=lambda i: probs[i], reverse=True)[:3]

        # ── Build overlay ────────────────────────────────────────────────────
        orig_np = np.array(pil_image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        cam_resized = cv2.resize(cam_np, (IMAGE_SIZE, IMAGE_SIZE))
        cam_uint8 = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap_rgb, 0.4, 0)

        def img_to_b64(arr: np.ndarray) -> str:
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        return {
            "predicted_class":        pred_idx,
            "predicted_label":        CLASS_NAMES[pred_idx],
            "predicted_display_name": CLASS_DISPLAY_NAMES[CLASS_NAMES[pred_idx]],
            "predicted_description":  CLASS_DESCRIPTIONS[CLASS_NAMES[pred_idx]],
            "probability":            round(probs[pred_idx], 4),
            "top3": [
                {
                    "rank": i + 1,
                    "class_idx": idx,
                    "label": CLASS_NAMES[idx],
                    "display_name": CLASS_DISPLAY_NAMES[CLASS_NAMES[idx]],
                    "probability": round(probs[idx], 4),
                }
                for i, idx in enumerate(top3_idx)
            ],
            "all_probabilities": {
                CLASS_NAMES[i]: round(probs[i], 4) for i in range(NUM_CLASSES)
            },
            "gradcam": {
                "heatmap_b64":  img_to_b64(heatmap_rgb),
                "overlay_b64":  img_to_b64(overlay),
                "original_b64": img_to_b64(orig_np),
            },
            "model": "EfficientNet-B4",
            "device": str(self.device),
        }
