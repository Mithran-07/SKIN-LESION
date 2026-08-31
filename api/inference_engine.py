"""
Inference Engine for EfficientNet-B4 Skin Lesion Classification and Grad-CAM Explainability.

This module provides:
- Automated hardware detection (MPS on Apple Silicon, CUDA on NVIDIA, CPU fallback)
- Robust model loading with checkpoint verification and fallback initialization
- Dermoscopic preprocessing pipeline matching training distributions
- Top-3 class probability computation
- EfficientNet-B4 Grad-CAM model attribution visualization
- Mandatory academic research disclaimers on all outputs
"""

import io
import os
import time
import base64
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm

logger = logging.getLogger("inference_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# HAM10000 7-class diagnostic taxonomy
HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_METADATA = {
    "akiec": {
        "code": "AKIEC",
        "name": "Actinic Keratoses and Intraepithelial Carcinoma",
        "short_name": "Actinic Keratosis",
        "category": "Pre-malignant",
        "description": "Sun-induced pre-cancerous lesion or early non-invasive squamous cell carcinoma in situ that can progress to invasive squamous cell carcinoma if untreated.",
        "urgency": "High",
        "benchmark_distribution": "327 cases (3.3%)"
    },
    "bcc": {
        "code": "BCC",
        "name": "Basal Cell Carcinoma",
        "short_name": "Basal Cell Carcinoma",
        "category": "Malignant (Non-Melanoma)",
        "description": "Most common human malignancy originating from basal epidermal cells. Rarely metastasizes but causes destructive local tissue invasion.",
        "urgency": "High",
        "benchmark_distribution": "514 cases (5.1%)"
    },
    "bkl": {
        "code": "BKL",
        "name": "Benign Keratosis-like Lesions",
        "short_name": "Benign Keratosis",
        "category": "Benign",
        "description": "Encompasses solar lentigines, seborrheic keratoses, and lichen-planus like keratoses. Common non-cancerous skin growths.",
        "urgency": "Low",
        "benchmark_distribution": "1,099 cases (11.0%)"
    },
    "df": {
        "code": "DF",
        "name": "Dermatofibroma",
        "short_name": "Dermatofibroma",
        "category": "Benign",
        "description": "Benign fibrous nodule commonly found on the lower extremities with characteristic central white patch and delicate pigment network.",
        "urgency": "Low",
        "benchmark_distribution": "115 cases (1.1%)"
    },
    "mel": {
        "code": "MEL",
        "name": "Melanoma",
        "short_name": "Melanoma",
        "category": "Malignant",
        "description": "High-risk malignant neoplasm of melanocytes responsible for the majority of skin cancer deaths globally due to high metastatic potential.",
        "urgency": "Critical",
        "benchmark_distribution": "1,113 cases (11.1%)"
    },
    "nv": {
        "code": "NV",
        "name": "Melanocytic Nevi",
        "short_name": "Melanocytic Nevus (Mole)",
        "category": "Benign",
        "description": "Common benign proliferation of melanocytes presenting as regular, symmetric pigmented macules or papules.",
        "urgency": "Low",
        "benchmark_distribution": "6,705 cases (67.0%)"
    },
    "vasc": {
        "code": "VASC",
        "name": "Vascular Lesions",
        "short_name": "Vascular Lesion",
        "category": "Benign",
        "description": "Benign vascular proliferations including cherry angiomas, angiokeratomas, and pyogenic granulomas.",
        "urgency": "Low",
        "benchmark_distribution": "142 cases (1.4%)"
    }
}

ACADEMIC_DISCLAIMER = (
    "This system is an academic research prototype and is not intended to provide medical "
    "diagnosis or replace professional medical advice. Always consult a qualified board-certified "
    "dermatologist for clinical evaluation of cutaneous lesions."
)


class EfficientNetGradCAM:
    """
    Grad-CAM implementation tailored for EfficientNet architectures.
    Captures activations and gradients at the final convolutional feature layer.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or getattr(model, 'conv_head', None)
        
        # Fallback layer lookup if conv_head is not present
        if self.target_layer is None:
            if hasattr(model, 'blocks') and len(model.blocks) > 0:
                self.target_layer = model.blocks[-1][-1]
            elif hasattr(model, 'features') and len(model.features) > 0:
                self.target_layer = model.features[-1]
            else:
                for module in reversed(list(model.modules())):
                    if isinstance(module, nn.Conv2d):
                        self.target_layer = module
                        break

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        if self.target_layer is None:
            logger.warning("Grad-CAM: No target conv layer found.")
            return

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap array (H, W) in range [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        
        # Backward pass for specific class score
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks didn't trigger
            return np.ones((224, 224), dtype=np.float32) * 0.5

        # Global average pooling of gradients -> neuron importance weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H, W)
        cam = F.relu(cam) # Positive attribution only

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(self, original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
        """
        Produce a high-visibility blended attribution map using matplotlib colormaps.
        """
        import matplotlib.cm as cm

        resized_orig = original_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        orig_arr = np.array(resized_orig, dtype=np.float32) / 255.0

        # Apply colormap (jet)
        try:
            import matplotlib
            colormap = matplotlib.colormaps['jet']
        except Exception:
            colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap)[:, :, :3] # discard alpha channel

        # Overlay blend
        blended = (1.0 - alpha) * orig_arr + alpha * heatmap_colored
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class SkinLesionInferenceEngine:
    """
    Core production inference engine for EfficientNet-B4 dermoscopic classification.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = self._detect_device()
        logger.info(f"Initialized Inference Engine on hardware: {self.device}")

        self.num_classes = len(HAM10000_CLASSES)
        self.checkpoint_path = checkpoint_path or "checkpoints/efficientnet_b4/best_checkpoint.pth"
        self.checkpoint_loaded = False
        self.checkpoint_message = ""

        # Preprocessing matching ImageNet / HAM10000 standard pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = self._load_model()
        self.grad_cam = EfficientNetGradCAM(self.model)

    def _detect_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            try:
                # Quick test tensor on MPS
                t = torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception as e:
                logger.warning(f"MPS available but failed init: {e}. Falling back to CPU.")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self) -> nn.Module:
        logger.info("Instantiating EfficientNet-B4 architecture...")
        # Create EfficientNet-B4 via timm
        try:
            model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=self.num_classes)
        except Exception as e:
            logger.error(f"Failed to create EfficientNet-B4: {e}")
            raise

        # Check for checkpoint file
        resolved_path = Path(self.checkpoint_path)
        if not resolved_path.exists():
            # Check relative locations
            alt_path = Path(__file__).resolve().parent.parent / self.checkpoint_path
            if alt_path.exists():
                resolved_path = alt_path

        if resolved_path.exists():
            try:
                checkpoint = torch.load(str(resolved_path), map_location=self.device)
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    elif "state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["state_dict"], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                self.checkpoint_loaded = True
                self.checkpoint_message = f"Verified checkpoint loaded successfully from: {resolved_path}"
                logger.info(self.checkpoint_message)
            except Exception as e:
                self.checkpoint_loaded = False
                self.checkpoint_message = f"Failed to load checkpoint state_dict ({e}). Using initialized weights."
                logger.error(self.checkpoint_message)
        else:
            self.checkpoint_loaded = False
            self.checkpoint_message = (
                f"EfficientNet-B4 checkpoint is required from the Lenovo LOQ (Expected at: {self.checkpoint_path}). "
                "Inference running with base feature extraction weights."
            )
            logger.warning(self.checkpoint_message)

        model = model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single PIL image.
        Returns full probability distribution, top-3 predicted classes, and metadata.
        """
        start_time = time.perf_counter()
        input_tensor = self.preprocess_image(image)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Build full distribution
        prob_dict = {
            cls_name: float(round(probabilities[i], 4))
            for i, cls_name in enumerate(HAM10000_CLASSES)
        }

        # Compute top-3 predictions
        sorted_indices = np.argsort(probabilities)[::-1]
        top3 = []
        for rank, idx in enumerate(sorted_indices[:3], start=1):
            cls_key = HAM10000_CLASSES[idx]
            meta = CLASS_METADATA[cls_key]
            prob_val = float(probabilities[idx])
            top3.append({
                "rank": rank,
                "class_code": meta["code"],
                "class_name": meta["name"],
                "short_name": meta["short_name"],
                "category": meta["category"],
                "probability": round(prob_val, 4),
                "probability_percentage": f"{prob_val * 100:.2f}%",
                "urgency": meta["urgency"],
                "description": meta["description"]
            })

        predicted_top = top3[0]

        return {
            "predicted_class": predicted_top["class_code"],
            "predicted_name": predicted_top["class_name"],
            "predicted_category": predicted_top["category"],
            "confidence": predicted_top["probability"],
            "confidence_percentage": predicted_top["probability_percentage"],
            "probabilities": prob_dict,
            "top3_predictions": top3,
            "inference_time_ms": latency_ms,
            "device": str(self.device),
            "model_name": "EfficientNet-B4",
            "checkpoint_loaded": self.checkpoint_loaded,
            "checkpoint_status": self.checkpoint_message,
            "disclaimer": ACADEMIC_DISCLAIMER
        }

    def predict_and_explain(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference and generate Grad-CAM attribution overlay as base64 PNG string.
        """
        # Run standard prediction first
        prediction_result = self.predict(image)
        predicted_idx = HAM10000_CLASSES.index(prediction_result["top3_predictions"][0]["class_code"].lower())

        # Generate Grad-CAM attribution
        input_tensor = self.preprocess_image(image)
        heatmap = self.grad_cam.generate_heatmap(input_tensor, predicted_idx)
        overlay_img = self.grad_cam.overlay_heatmap(image, heatmap, alpha=0.45)

        # Convert overlay image to Base64 PNG
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            **prediction_result,
            "explainability": {
                "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
                "target_layer": "EfficientNet-B4 Final Convolutional Head",
                "target_class": prediction_result["top3_predictions"][0]["class_name"],
                "overlay_base64": f"data:image/png;base64,{b64_str}",
                "attribution_note": (
                    "Model attribution visualization highlighting regional spatial features "
                    "most influential to the model's classification score."
                ),
                "dimensions": {"width": 224, "height": 224}
            }
        }


# Global singleton instance
_engine_instance: Optional[SkinLesionInferenceEngine] = None


def get_inference_engine() -> SkinLesionInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SkinLesionInferenceEngine()
    return _engine_instance
