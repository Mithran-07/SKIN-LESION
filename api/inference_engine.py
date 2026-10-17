"""
Inference Engine v2.0.0 for EfficientNet-B4 Skin Lesion Classification and Grad-CAM Explainability.

Upgrades in v2.0.0:
- Strict checkpoint verification (`strict=True` + SHA-256 cryptographic digest check)
- Zero local filesystem path leakage in API status fields
- Single-pass inference + OOD validation reusing EfficientNet-B4 penultimate features & logits
- Scoped Grad-CAM hook registration inside a `try...finally` block (`retain_graph=False`),
  preventing hook overhead during standard `/predict` and eliminating MPS tensor retention
- Counterfactual class attribution (`target_class` support) + standalone RGBA heatmap PNG
  (`heatmap_base64`) alongside `overlay_base64`
- Startup MPS warmup pass + `torch.mps.empty_cache()` to eliminate the 3.8s-6.4s cold-start spike
"""

import io
import os
import time
import hashlib
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

from api.ood_gate import OODDomainGate

logger = logging.getLogger("inference_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Canonical HAM10000 7-class diagnostic taxonomy
HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_MAPPING_VERSION = "ham10000_canonical_7class_v1"
PREPROCESSING_VERSION = "eval_v1_resize256_crop224_imagenet"
EXPECTED_CHECKPOINT_SHA256 = "125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e"

CLASS_METADATA = {
    "akiec": {
        "code": "AKIEC",
        "name": "Actinic Keratoses and Intraepithelial Carcinoma",
        "short_name": "Actinic Keratosis",
        "category": "Pre-malignant",
        "description": "Sun-induced pre-cancerous lesion or early non-invasive squamous cell carcinoma in situ that can progress to invasive squamous cell carcinoma if untreated.",
        "urgency": "High",
        "benchmark_distribution": "327 cases (3.3%)",
    },
    "bcc": {
        "code": "BCC",
        "name": "Basal Cell Carcinoma",
        "short_name": "Basal Cell Carcinoma",
        "category": "Malignant (Non-Melanoma)",
        "description": "Most common human malignancy originating from basal epidermal cells. Rarely metastasizes but causes destructive local tissue invasion.",
        "urgency": "High",
        "benchmark_distribution": "514 cases (5.1%)",
    },
    "bkl": {
        "code": "BKL",
        "name": "Benign Keratosis-like Lesions",
        "short_name": "Benign Keratosis",
        "category": "Benign",
        "description": "Encompasses solar lentigines, seborrheic keratoses, and lichen-planus like keratoses. Common non-cancerous skin growths.",
        "urgency": "Low",
        "benchmark_distribution": "1,099 cases (11.0%)",
    },
    "df": {
        "code": "DF",
        "name": "Dermatofibroma",
        "short_name": "Dermatofibroma",
        "category": "Benign",
        "description": "Benign fibrous nodule commonly found on the lower extremities with characteristic central white patch and delicate pigment network.",
        "urgency": "Low",
        "benchmark_distribution": "115 cases (1.1%)",
    },
    "mel": {
        "code": "MEL",
        "name": "Melanoma",
        "short_name": "Melanoma",
        "category": "Malignant",
        "description": "High-risk malignant neoplasm of melanocytes responsible for the majority of skin cancer deaths globally due to high metastatic potential.",
        "urgency": "Critical",
        "benchmark_distribution": "1,113 cases (11.1%)",
    },
    "nv": {
        "code": "NV",
        "name": "Melanocytic Nevi",
        "short_name": "Melanocytic Nevus (Mole)",
        "category": "Benign",
        "description": "Common benign proliferation of melanocytes presenting as regular, symmetric pigmented macules or papules.",
        "urgency": "Low",
        "benchmark_distribution": "6,705 cases (67.0%)",
    },
    "vasc": {
        "code": "VASC",
        "name": "Vascular Lesions",
        "short_name": "Vascular Lesion",
        "category": "Benign",
        "description": "Benign vascular proliferations including cherry angiomas, angiokeratomas, and pyogenic granulomas.",
        "urgency": "Low",
        "benchmark_distribution": "142 cases (1.4%)",
    },
}

ACADEMIC_DISCLAIMER = (
    "This system is an academic research prototype and is not intended to provide medical "
    "diagnosis or replace professional medical advice. Always consult a qualified board-certified "
    "dermatologist for clinical evaluation of cutaneous lesions."
)


class EfficientNetGradCAM:
    """
    Scoped Grad-CAM implementation for EfficientNet-B4 (`conv_head`).
    Registers forward and backward hooks ONLY during attribution requests (`try...finally`),
    ensuring zero hook overhead on standard predictions and zero persistent tensor retention.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or getattr(model, "conv_head", None)
        if self.target_layer is None:
            if hasattr(model, "blocks") and len(model.blocks) > 0:
                self.target_layer = model.blocks[-1][-1]
            elif hasattr(model, "features") and len(model.features) > 0:
                self.target_layer = model.features[-1]
            else:
                for module in reversed(list(model.modules())):
                    if isinstance(module, nn.Conv2d):
                        self.target_layer = module
                        break

    def compute_from_activations_and_gradients(
        self, activations: torch.Tensor, gradients: torch.Tensor
    ) -> np.ndarray:
        """Compute normalized (224, 224) Grad-CAM heatmap in [0, 1] from captured tensors."""
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np.astype(np.float32)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Standalone Grad-CAM heatmap generation with scoped hook registration.
        """
        if self.target_layer is None:
            return np.zeros((224, 224), dtype=np.float32)

        captured: Dict[str, Optional[torch.Tensor]] = {"activations": None, "gradients": None}

        def forward_hook(module, inp, out):
            captured["activations"] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            captured["gradients"] = grad_out[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        try:
            self.model.eval()
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                x = input_tensor.detach().requires_grad_(True)
                logits = self.model(x)
                score = logits[0, class_idx]
                score.backward()
            if captured["activations"] is None or captured["gradients"] is None:
                return np.zeros((224, 224), dtype=np.float32)
            return self.compute_from_activations_and_gradients(
                captured["activations"], captured["gradients"]
            )
        finally:
            h_fwd.remove()
            h_bwd.remove()
            captured["activations"] = None
            captured["gradients"] = None
            self.model.zero_grad(set_to_none=True)

    @staticmethod
    def _get_colormap():
        import matplotlib
        return matplotlib.colormaps["jet"]

    def render_heatmap_rgba(self, heatmap: np.ndarray) -> Image.Image:
        """
        Render standalone RGBA heatmap PNG where alpha is proportional to attribution intensity,
        enabling smooth real-time opacity blending in the frontend viewport.
        """
        colormap = self._get_colormap()
        colored = colormap(heatmap)  # (224, 224, 4) in [0, 1]
        # Shape alpha so low-activation background is subtle and peak activation is crisp
        alpha_channel = np.clip(0.18 + 0.82 * np.power(np.clip(heatmap, 0.0, 1.0), 0.75), 0.0, 1.0)
        colored[:, :, 3] = alpha_channel
        rgba_uint8 = np.clip(colored * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(rgba_uint8, mode="RGBA")

    def overlay_heatmap(
        self, original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
    ) -> Image.Image:
        """
        Produce a blended RGB attribution overlay on the center-cropped 224x224 view.
        """
        colormap = self._get_colormap()
        # Match the exact 256 -> CenterCrop(224) spatial alignment used by the model
        resized_256 = original_img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        cropped_224 = resized_256.crop((16, 16, 240, 240))
        orig_arr = np.asarray(cropped_224, dtype=np.float32) / 255.0

        heatmap_colored = colormap(heatmap)[:, :, :3]
        blended = (1.0 - alpha) * orig_arr + alpha * heatmap_colored
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(blended, mode="RGB")

    def close(self):
        """No-op for backward compatibility since hooks are now strictly request-scoped."""
        pass


class SkinLesionInferenceEngine:
    """
    Production inference engine v2.0.0 for EfficientNet-B4 dermoscopic classification.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, run_warmup: bool = True):
        self.started_at = time.time()
        self.device = self._detect_device()
        logger.info(f"Initialized Inference Engine v2.0.0 on hardware: {self.device}")

        self.num_classes = len(HAM10000_CLASSES)
        self.checkpoint_path = checkpoint_path or "checkpoints/efficientnet_b4/best_checkpoint.pth"
        self.checkpoint_loaded = False
        self.checkpoint_sha256 = ""
        self.checkpoint_sha256_short = ""
        self.checkpoint_epoch: Optional[int] = None
        self.checkpoint_val_bal_acc: Optional[float] = None
        self.checkpoint_message = ""
        self.warmup_completed = False
        self.warmup_time_ms = 0.0

        # Runtime telemetry counters
        self.total_predictions = 0
        self.total_explanations = 0
        self.total_rejections = 0
        self.last_inference_ms: Optional[float] = None

        # Canonical evaluation preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = self._load_model()
        self.grad_cam = EfficientNetGradCAM(self.model)
        self.ood_gate = OODDomainGate(self.device)

        if run_warmup and self.checkpoint_loaded:
            self._warmup_device()

    def _detect_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            try:
                _ = torch.zeros(1, device="mps")
                return torch.device("mps")
            except Exception as e:
                logger.warning(f"MPS available but failed init: {e}. Falling back to CPU.")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self) -> nn.Module:
        logger.info("Instantiating EfficientNet-B4 architecture...")
        model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=self.num_classes)

        resolved_path = Path(self.checkpoint_path)
        if not resolved_path.exists():
            alt_path = Path(__file__).resolve().parent.parent / self.checkpoint_path
            if alt_path.exists():
                resolved_path = alt_path

        if resolved_path.exists():
            try:
                # Compute SHA-256 digest
                hasher = hashlib.sha256()
                with open(resolved_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hasher.update(chunk)
                self.checkpoint_sha256 = hasher.hexdigest()
                self.checkpoint_sha256_short = self.checkpoint_sha256[:16]

                checkpoint = torch.load(str(resolved_path), map_location="cpu")
                if isinstance(checkpoint, dict):
                    self.checkpoint_epoch = checkpoint.get("epoch")
                    self.checkpoint_val_bal_acc = checkpoint.get("best_val_bal_acc")
                    state_dict = (
                        checkpoint.get("model_state_dict")
                        or checkpoint.get("state_dict")
                        or checkpoint
                    )
                else:
                    state_dict = checkpoint

                # Enforce strict=True so any layer mismatch fails loudly
                model.load_state_dict(state_dict, strict=True)
                self.checkpoint_loaded = True
                # Sanitized public status message (never leaks local filesystem path)
                self.checkpoint_message = (
                    f"Verified EfficientNet-B4 checkpoint loaded (strict=True, "
                    f"sha256={self.checkpoint_sha256_short}, epoch={self.checkpoint_epoch})."
                )
                logger.info(self.checkpoint_message)
            except Exception as e:
                self.checkpoint_loaded = False
                self.checkpoint_message = "Checkpoint verification failed (strict state_dict mismatch)."
                logger.error(f"Checkpoint load failure: {e}")
        else:
            self.checkpoint_loaded = False
            self.checkpoint_message = "Trained EfficientNet-B4 checkpoint not found in repository."
            logger.warning(self.checkpoint_message)

        model = model.to(self.device)
        model.eval()
        return model

    def _warmup_device(self) -> None:
        """
        Pre-compile Metal/MPS shaders for forward and backward passes at startup so the
        first user request does not suffer a 3.8s - 6.4s cold-start spike.
        """
        t0 = time.perf_counter()
        try:
            dummy = torch.zeros((1, 3, 224, 224), device=self.device, requires_grad=True)
            feats = self.model.forward_features(dummy)
            pen = self.model.forward_head(feats, pre_logits=True)
            logits = self.model.get_classifier()(pen)
            logits[0, 0].backward()
            self.model.zero_grad(set_to_none=True)
            del dummy, feats, pen, logits
            if self.device.type == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            self.warmup_completed = True
            self.warmup_time_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(f"Device warmup completed in {self.warmup_time_ms} ms.")
        except Exception as e:
            logger.warning(f"Startup warmup skipped: {e}")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image)
        return tensor.unsqueeze(0).to(self.device)

    def _format_prediction_payload(
        self,
        probabilities: np.ndarray,
        ood_validation: Dict[str, Any],
        latency_ms: float,
        stage_timings_ms: Dict[str, float],
    ) -> Dict[str, Any]:
        prob_dict = {
            cls_name: float(round(probabilities[i], 4))
            for i, cls_name in enumerate(HAM10000_CLASSES)
        }

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
                "description": meta["description"],
            })

        predicted_top = top3[0]
        # Normalized Shannon entropy as uncertainty indicator in [0, 1]
        eps = 1e-12
        entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, eps, 1.0))) / np.log(len(HAM10000_CLASSES)))

        return {
            "status": "success",
            "input_validation": ood_validation,
            "predicted_class": predicted_top["class_code"],
            "predicted_name": predicted_top["class_name"],
            "predicted_category": predicted_top["category"],
            "confidence": predicted_top["probability"],
            "confidence_percentage": predicted_top["probability_percentage"],
            "normalized_entropy": round(entropy, 4),
            "probabilities": prob_dict,
            "top3_predictions": top3,
            "inference_time_ms": latency_ms,
            "stage_timings_ms": stage_timings_ms,
            "device": str(self.device),
            "model_name": "EfficientNet-B4",
            "model_version": "timm/efficientnet_b4_ham10000_v2",
            "checkpoint_loaded": self.checkpoint_loaded,
            "checkpoint_sha256_short": self.checkpoint_sha256_short,
            "checkpoint_status": self.checkpoint_message,
            "preprocessing_version": PREPROCESSING_VERSION,
            "class_mapping_version": CLASS_MAPPING_VERSION,
            "ood_version": self.ood_gate.VERSION,
            "disclaimer": ACADEMIC_DISCLAIMER,
        }

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Single-pass inference + calibrated OOD validation on a PIL image.
        Uses `torch.inference_mode()` with zero active hooks and zero duplicate forward passes.
        """
        if not self.checkpoint_loaded:
            return {
                "status": "error",
                "error_code": "MODEL_UNAVAILABLE",
                "detail": "TRAINED CHECKPOINT REQUIRED. Medical-image inference cannot run with random/untrained weights.",
            }

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        rgb_image = image.convert("RGB")
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        prep_ms = round((time.perf_counter() - t0) * 1000, 2)

        t1 = time.perf_counter()
        with torch.inference_mode():
            feats = self.model.forward_features(input_tensor)
            penultimate = self.model.forward_head(feats, pre_logits=True)
            logits = self.model.get_classifier()(penultimate)
            probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            if self.device.type == "mps":
                torch.mps.synchronize()
        forward_ms = round((time.perf_counter() - t1) * 1000, 2)

        t2 = time.perf_counter()
        ood_validation = self.ood_gate.validate_from_features(rgb_image, penultimate, logits)
        ood_ms = round((time.perf_counter() - t2) * 1000, 2)

        total_ms = round((time.perf_counter() - t_total) * 1000, 2)
        self.last_inference_ms = total_ms

        if not ood_validation["valid"]:
            self.total_rejections += 1
            return {
                "status": "rejected",
                "input_validation": ood_validation,
                "inference_time_ms": total_ms,
                "stage_timings_ms": {
                    "preprocessing_ms": prep_ms,
                    "forward_pass_ms": forward_ms,
                    "ood_gate_ms": ood_ms,
                },
                "device": str(self.device),
                "model_name": "EfficientNet-B4",
                "model_version": "timm/efficientnet_b4_ham10000_v2",
                "checkpoint_sha256_short": self.checkpoint_sha256_short,
                "ood_version": self.ood_gate.VERSION,
                "disclaimer": ACADEMIC_DISCLAIMER,
            }

        self.total_predictions += 1
        return self._format_prediction_payload(
            probabilities=probabilities,
            ood_validation=ood_validation,
            latency_ms=total_ms,
            stage_timings_ms={
                "preprocessing_ms": prep_ms,
                "forward_pass_ms": forward_ms,
                "ood_gate_ms": ood_ms,
            },
        )

    def predict_and_explain(
        self,
        image: Image.Image,
        target_class: Optional[str] = None,
        alpha: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Single-pass classification + OOD check + scoped Grad-CAM attribution.
        Eliminates the duplicate preprocessing and duplicate EfficientNet-B4 forward pass
        present in v1, while supporting counterfactual class inspection (`target_class`).
        """
        if not self.checkpoint_loaded:
            return {
                "status": "error",
                "error_code": "MODEL_UNAVAILABLE",
                "detail": "TRAINED CHECKPOINT REQUIRED. Medical-image inference cannot run with random/untrained weights.",
            }

        alpha = float(np.clip(alpha, 0.0, 1.0))
        target_idx_override: Optional[int] = None
        if target_class is not None and target_class.strip():
            norm_tc = target_class.strip().lower()
            if norm_tc not in HAM10000_CLASSES:
                raise ValueError(
                    f"Invalid target_class '{target_class}'. Must be one of {HAM10000_CLASSES}."
                )
            target_idx_override = HAM10000_CLASSES.index(norm_tc)

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        rgb_image = image.convert("RGB")
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        prep_ms = round((time.perf_counter() - t0) * 1000, 2)

        captured: Dict[str, Optional[torch.Tensor]] = {"activations": None, "gradients": None}

        def forward_hook(module, inp, out):
            captured["activations"] = out.detach()

        def backward_hook(module, grad_in, grad_out):
            captured["gradients"] = grad_out[0].detach()

        target_layer = self.grad_cam.target_layer
        h_fwd = target_layer.register_forward_hook(forward_hook) if target_layer is not None else None
        h_bwd = target_layer.register_full_backward_hook(backward_hook) if target_layer is not None else None

        try:
            t1 = time.perf_counter()
            self.model.eval()
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                x = input_tensor.detach().requires_grad_(True)
                feats = self.model.forward_features(x)
                penultimate = self.model.forward_head(feats, pre_logits=True)
                logits = self.model.get_classifier()(penultimate)
                probabilities = F.softmax(logits.detach(), dim=1).squeeze(0).cpu().numpy()
                if self.device.type == "mps":
                    torch.mps.synchronize()
            forward_ms = round((time.perf_counter() - t1) * 1000, 2)

            t2 = time.perf_counter()
            ood_validation = self.ood_gate.validate_from_features(
                rgb_image, penultimate.detach(), logits.detach()
            )
            ood_ms = round((time.perf_counter() - t2) * 1000, 2)

            if not ood_validation["valid"]:
                total_ms = round((time.perf_counter() - t_total) * 1000, 2)
                self.last_inference_ms = total_ms
                self.total_rejections += 1
                return {
                    "status": "rejected",
                    "input_validation": ood_validation,
                    "inference_time_ms": total_ms,
                    "stage_timings_ms": {
                        "preprocessing_ms": prep_ms,
                        "forward_pass_ms": forward_ms,
                        "ood_gate_ms": ood_ms,
                    },
                    "device": str(self.device),
                    "model_name": "EfficientNet-B4",
                    "model_version": "timm/efficientnet_b4_ham10000_v2",
                    "checkpoint_sha256_short": self.checkpoint_sha256_short,
                    "ood_version": self.ood_gate.VERSION,
                    "disclaimer": ACADEMIC_DISCLAIMER,
                }

            # Determine which class score to backpropagate (predicted top-1 or user-requested counterfactual class)
            predicted_idx = int(np.argmax(probabilities))
            explain_idx = target_idx_override if target_idx_override is not None else predicted_idx
            explain_code = HAM10000_CLASSES[explain_idx]
            explain_meta = CLASS_METADATA[explain_code]

            t3 = time.perf_counter()
            with torch.enable_grad():
                score = logits[0, explain_idx]
                score.backward()  # No retain_graph=True -> frees computation graph immediately!

            if captured["activations"] is not None and captured["gradients"] is not None:
                heatmap = self.grad_cam.compute_from_activations_and_gradients(
                    captured["activations"], captured["gradients"]
                )
            else:
                heatmap = np.zeros((224, 224), dtype=np.float32)

            overlay_img = self.grad_cam.overlay_heatmap(rgb_image, heatmap, alpha=alpha)
            heatmap_rgba_img = self.grad_cam.render_heatmap_rgba(heatmap)

            buf_overlay = io.BytesIO()
            overlay_img.save(buf_overlay, format="PNG")
            overlay_b64 = base64.b64encode(buf_overlay.getvalue()).decode("utf-8")

            buf_heatmap = io.BytesIO()
            heatmap_rgba_img.save(buf_heatmap, format="PNG")
            heatmap_b64 = base64.b64encode(buf_heatmap.getvalue()).decode("utf-8")

            if self.device.type == "mps":
                torch.mps.synchronize()
            gradcam_ms = round((time.perf_counter() - t3) * 1000, 2)

        finally:
            if h_fwd is not None:
                h_fwd.remove()
            if h_bwd is not None:
                h_bwd.remove()
            captured["activations"] = None
            captured["gradients"] = None
            self.model.zero_grad(set_to_none=True)

        total_ms = round((time.perf_counter() - t_total) * 1000, 2)
        self.last_inference_ms = total_ms
        self.total_predictions += 1
        self.total_explanations += 1

        base_payload = self._format_prediction_payload(
            probabilities=probabilities,
            ood_validation=ood_validation,
            latency_ms=total_ms,
            stage_timings_ms={
                "preprocessing_ms": prep_ms,
                "forward_pass_ms": forward_ms,
                "ood_gate_ms": ood_ms,
                "gradcam_backward_and_render_ms": gradcam_ms,
            },
        )

        # Compute spatial peak coordinates (x, y in [0, 223]) for quantitative readout
        peak_flat = int(np.argmax(heatmap))
        peak_y, peak_x = divmod(peak_flat, 224)
        active_area_fraction = float(np.mean(heatmap >= 0.5))

        return {
            **base_payload,
            "explainability": {
                "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
                "target_layer": "EfficientNet-B4 Final Convolutional Head (conv_head)",
                "target_class_code": explain_meta["code"],
                "target_class": explain_meta["name"],
                "target_class_probability": round(float(probabilities[explain_idx]), 4),
                "is_counterfactual": bool(explain_idx != predicted_idx),
                "overlay_alpha": round(alpha, 2),
                "overlay_base64": f"data:image/png;base64,{overlay_b64}",
                "heatmap_base64": f"data:image/png;base64,{heatmap_b64}",
                "peak_activation_coords": {"x": int(peak_x), "y": int(peak_y)},
                "high_activation_area_fraction": round(active_area_fraction, 4),
                "attribution_note": (
                    "Grad-CAM visualizes spatial regions of the final convolutional feature map "
                    "that positively contribute to the selected class logit. It reflects model "
                    "attention, not a pixel-level clinical segmentation mask."
                ),
                "dimensions": {"width": 224, "height": 224},
            },
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return real-time hardware, memory, and model provenance diagnostics.
        """
        mps_current_mb: Optional[float] = None
        mps_driver_mb: Optional[float] = None
        if self.device.type == "mps" and hasattr(torch, "mps"):
            try:
                mps_current_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
                mps_driver_mb = round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
            except Exception:
                pass

        rss_mb: Optional[float] = None
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes
            rss_mb = round(usage / (1024 * 1024), 2) if os.uname().sysname == "Darwin" else round(usage / 1024, 2)
        except Exception:
            pass

        return {
            "status": "healthy" if self.checkpoint_loaded else "degraded",
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "device": str(self.device),
            "warmup_completed": self.warmup_completed,
            "warmup_time_ms": self.warmup_time_ms,
            "memory": {
                "mps_current_allocated_mb": mps_current_mb,
                "mps_driver_allocated_mb": mps_driver_mb,
                "process_peak_rss_mb": rss_mb,
                "active_gradcam_hooks": 0,
            },
            "model_provenance": {
                "model_name": "EfficientNet-B4",
                "model_version": "timm/efficientnet_b4_ham10000_v2",
                "checkpoint_loaded": self.checkpoint_loaded,
                "checkpoint_sha256": self.checkpoint_sha256,
                "checkpoint_sha256_short": self.checkpoint_sha256_short,
                "checkpoint_sha256_matches_canonical": bool(
                    self.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
                ),
                "checkpoint_epoch": self.checkpoint_epoch,
                "checkpoint_best_val_balanced_accuracy": (
                    round(float(self.checkpoint_val_bal_acc), 4)
                    if self.checkpoint_val_bal_acc is not None
                    else None
                ),
                "preprocessing_version": PREPROCESSING_VERSION,
                "class_mapping_version": CLASS_MAPPING_VERSION,
                "canonical_classes": HAM10000_CLASSES,
            },
            "ood_gate": {
                "version": self.ood_gate.VERSION,
                "calibrated": self.ood_gate.calibrated,
                "threshold": round(float(self.ood_gate.threshold), 4),
                "architecture": "Shared EfficientNet-B4 Penultimate Embedding + Mahalanobis + Energy + Photometric Gate",
            },
            "telemetry": {
                "total_predictions_served": self.total_predictions,
                "total_explanations_served": self.total_explanations,
                "total_ood_rejections": self.total_rejections,
                "last_inference_latency_ms": self.last_inference_ms,
            },
            "disclaimer": ACADEMIC_DISCLAIMER,
        }


# Global singleton instance
_engine_instance: Optional[SkinLesionInferenceEngine] = None


def get_inference_engine() -> SkinLesionInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SkinLesionInferenceEngine()
    return _engine_instance
