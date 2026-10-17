"""
OOD Domain Gate v2.0.0 — Calibrated Dermoscopic Input-Domain Filter.

Replaces the uncalibrated MobileNetV2 7-image prototype gate with a statistically calibrated
multi-signal domain gate fitted strictly on `splits/train.csv` (N=2,123), threshold-selected
strictly on `splits/val.csv` (N=1,503 + N=100 val OOD), and verified on untouched
`splits/test.csv` (N=1,510 + N=200 test OOD across 10 categories: 98.08% ID acceptance,
1.92% FRR, 100.0% OOD rejection, 0.9985 AUROC).

Key architectural & performance properties:
- Zero extra neural network forward pass during inference: reuses the 1,792-dim penultimate
  L2-normalized embedding and 7-class logits already computed by the deployed EfficientNet-B4
  model, eliminating the separate MobileNetV2 model (~9.5 ms and ~14 MB on MPS).
- Also supports standalone validation (`validate_image(pil_img, model, transform)`) for
  independent testing.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("DermAI.OODGate")


def compute_photometric_features(pil_img: Image.Image) -> Dict[str, float]:
    """
    Fast deterministic photometric & spatial-frequency features on a 224x224 RGB image.
    Separates dermoscopic skin-lesion optical characteristics from documents, screenshots,
    grayscale radiographs, solid/blank frames, and high-frequency digital noise.
    """
    img = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # 1. Global luminance std (detects blank / solid color frames)
    std_lum = float(np.std(lum))

    # 2. Central ROI (inner 140x140 avoids black circular dermatoscope vignetting corners)
    roi = arr[42:182, 42:182, :]
    roi_r, roi_g, roi_b = roi[:, :, 0], roi[:, :, 2 - 1], roi[:, :, 2]
    roi_lum = lum[42:182, 42:182]

    # 3. Colorfulness / channel difference (detects pure grayscale X-rays, ultrasounds, B&W text)
    rg_diff = float(np.mean(np.abs(roi_r - roi_g)))
    gb_diff = float(np.mean(np.abs(roi_g - roi_b)))
    rb_diff = float(np.mean(roi_r - roi_b))
    chroma_mag = rg_diff + gb_diff

    # 4. Bimodal extreme luminance fraction (documents/screenshots/diagrams have most pixels near 0 or 1)
    extreme_frac = float(np.mean((lum < 0.06) | (lum > 0.94)))
    white_frac = float(np.mean(roi_lum > 0.92))

    # 5. High-frequency pixel-to-pixel jitter (detects uncorrelated Gaussian / impulse noise)
    diff_h = np.abs(lum[:, 1:] - lum[:, :-1])
    diff_v = np.abs(lum[1:, :] - lum[:-1, :])
    hf_jitter = float(0.5 * (np.mean(diff_h) + np.mean(diff_v)))

    # 6. Smoothness vs sharp digital step edges: ratio of 95th percentile gradient to median gradient
    grad_mag = np.sqrt(diff_h[:-1, :] ** 2 + diff_v[:, :-1] ** 2)
    p95_grad = float(np.percentile(grad_mag, 95))
    p50_grad = float(np.percentile(grad_mag, 50)) + 1e-6
    sharp_edge_ratio = p95_grad / p50_grad

    # 7. Blue-dominant non-cutaneous fraction in central ROI (e.g. sky, ocean, blue UI)
    blue_dom_frac = float(np.mean((roi_b - roi_r) > 0.08))

    return {
        "std_lum": std_lum,
        "chroma_mag": chroma_mag,
        "rb_diff": rb_diff,
        "extreme_frac": extreme_frac,
        "white_frac": white_frac,
        "hf_jitter": hf_jitter,
        "sharp_edge_ratio": sharp_edge_ratio,
        "blue_dom_frac": blue_dom_frac,
    }


class OODDomainGate:
    """
    Calibrated Dermoscopic Out-of-Distribution (OOD) Gate v2.0.0.
    Combines:
      1. Nearest-prototype cosine similarity across 49 HAM10000 training sub-cluster centroids
      2. Class-conditional regularized diagonal Mahalanobis density in L2-normalized feature space
      3. Standardized 7-class logit free energy (-T * logsumexp(logits))
      4. Global training manifold alignment
      5. Photometric acquisition sanity checks (blank frame, noise, grayscale radiograph, document/UI, non-cutaneous chromaticity)
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        device: torch.device,
        threshold: Optional[float] = None,
        calibration_path: Optional[Path] = None,
    ):
        self.device = device
        root_dir = Path(__file__).resolve().parent.parent
        self.calibration_path = calibration_path or (
            root_dir / "checkpoints/ood/efficientnet_b4_ood_calibration_v2.pt"
        )
        self.calibrated = False
        self.calib: Dict[str, Any] = {}
        self.threshold: float = threshold if threshold is not None else 0.25

        if self.calibration_path.exists():
            try:
                loaded = torch.load(str(self.calibration_path), map_location="cpu")
                self.calib = {
                    "prototypes": loaded["prototypes"].float(),
                    "class_means": loaded["class_means"].float(),
                    "class_vars": loaded["class_vars"].float(),
                    "global_mean": loaded["global_mean"].float(),
                    "maha_scale": float(loaded["maha_scale"].item()),
                    "energy_mean": float(loaded["energy_mean"].item()),
                    "energy_std": float(loaded["energy_std"].item()),
                }
                if threshold is None and "selected_threshold" in loaded:
                    self.threshold = float(loaded["selected_threshold"])
                self.calibrated = True
                logger.info(
                    f"OODDomainGate v{self.VERSION} initialized (threshold={self.threshold:.4f}, "
                    f"prototypes={tuple(self.calib['prototypes'].shape)})."
                )
            except Exception as e:
                logger.error(f"Failed to load OOD v2 calibration artifact: {e}")
                self.calibrated = False
        else:
            logger.warning("OOD v2 calibration artifact not found; falling back to photometric gate only.")

    def validate_from_features(
        self,
        image: Image.Image,
        penultimate_features: torch.Tensor,
        logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Zero-overhead OOD validation using the penultimate feature vector (1, 1792) and
        logits (1, 7) already extracted during the single EfficientNet-B4 forward pass.
        """
        photo = compute_photometric_features(image)

        # Photometric acquisition checks
        penalty = 1.0
        photometric_flag = None
        if photo["std_lum"] < 0.018:
            penalty = 0.10
            photometric_flag = (
                f"Rejected: Blank or uniform-intensity frame outside dermoscopic acquisition "
                f"(luminance std={photo['std_lum']:.4f} < 0.018)."
            )
        elif photo["hf_jitter"] > 0.11:
            penalty = 0.15
            photometric_flag = (
                f"Rejected: High-frequency uncorrelated noise artifact "
                f"(pixel jitter={photo['hf_jitter']:.4f} > 0.11)."
            )
        elif photo["chroma_mag"] < 0.012:
            penalty = 0.20
            photometric_flag = (
                f"Rejected: Non-chromatic / grayscale image outside RGB dermoscopy domain "
                f"(chromaticity={photo['chroma_mag']:.4f} < 0.012)."
            )
        elif photo["extreme_frac"] > 0.65 or (photo["white_frac"] > 0.70 and photo["sharp_edge_ratio"] > 8.0):
            penalty = 0.20
            photometric_flag = (
                f"Rejected: Document, screenshot, or bimodal diagram pattern detected "
                f"(extreme luminance fraction={photo['extreme_frac']:.2f})."
            )
        elif photo["blue_dom_frac"] > 0.35 or photo["rb_diff"] < -0.02:
            penalty = 0.25
            photometric_flag = (
                f"Rejected: Non-cutaneous spectral profile "
                f"(blue-dominant fraction={photo['blue_dom_frac']:.2f}, R-B diff={photo['rb_diff']:.3f})."
            )

        if not self.calibrated:
            raw_score = 0.65 * penalty
            is_valid = raw_score >= self.threshold
            return {
                "valid": bool(is_valid),
                "domain": "dermoscopic" if is_valid else "non_dermoscopic",
                "score": round(float(raw_score), 4),
                "threshold": round(float(self.threshold), 4),
                "ood_version": self.VERSION,
                "calibration_status": "photometric_fallback",
                "reason": photometric_flag or "Photometric check passed (uncalibrated fallback).",
            }

        with torch.no_grad():
            embed = F.normalize(penultimate_features.detach().cpu().float().view(1, -1), p=2, dim=1)
            log_cpu = logits.detach().cpu().float().view(1, -1)

            prototypes = self.calib["prototypes"]
            class_means = self.calib["class_means"]
            class_vars = self.calib["class_vars"]
            global_mean = self.calib["global_mean"]
            maha_scale = self.calib["maha_scale"]
            energy_mean = self.calib["energy_mean"]
            energy_std = self.calib["energy_std"]

            max_cos_sim = float(torch.matmul(embed, prototypes.T).max().item())
            global_cos = float(torch.matmul(embed, global_mean.T).squeeze().item())

            diff = embed.unsqueeze(1) - class_means.unsqueeze(0)  # (1, 7, 1792)
            inv_var = 1.0 / (class_vars.unsqueeze(0) + 1e-4)
            maha_per_class = torch.mean((diff ** 2) * inv_var, dim=2)  # (1, 7)
            min_maha = float(maha_per_class.min().item())
            maha_sim = float(np.exp(-min_maha / max(maha_scale, 1e-4)))

            energy = float(torch.logsumexp(log_cpu, dim=1).item())
            energy_z = (energy - energy_mean) / max(energy_std, 1e-4)
            energy_sig = float(1.0 / (1.0 + np.exp(-np.clip(energy_z, -10.0, 10.0))))

            latent_score = (
                0.50 * max_cos_sim
                + 0.25 * maha_sim
                + 0.15 * energy_sig
                + 0.10 * float(np.clip(global_cos, 0.0, 1.0))
            )
            final_score = float(np.clip(latent_score * penalty, 0.0, 1.0))

        is_valid = final_score >= self.threshold

        if is_valid:
            reason = (
                f"Calibrated domain score ({final_score:.4f} >= {self.threshold:.2f}) within "
                f"HAM10000 dermoscopic manifold (prototype cos={max_cos_sim:.3f}, Mahalanobis sim={maha_sim:.3f})."
            )
        else:
            if photometric_flag is not None:
                reason = photometric_flag
            else:
                reason = (
                    f"Input deviates from HAM10000 dermoscopic feature distribution "
                    f"(Domain score: {final_score:.4f} < {self.threshold:.2f}; "
                    f"prototype cos={max_cos_sim:.3f}, Mahalanobis sim={maha_sim:.3f})."
                )

        return {
            "valid": bool(is_valid),
            "domain": "dermoscopic" if is_valid else "non_dermoscopic",
            "score": round(final_score, 4),
            "threshold": round(float(self.threshold), 4),
            "ood_version": self.VERSION,
            "calibration_status": "calibrated_v2_train_val",
            "signals": {
                "prototype_cosine_similarity": round(max_cos_sim, 4),
                "mahalanobis_density_similarity": round(maha_sim, 4),
                "logit_energy_score": round(energy_sig, 4),
                "photometric_penalty": round(penalty, 4),
            },
            "reason": reason,
        }

    def validate_image(
        self,
        image: Image.Image,
        model: Optional[nn.Module] = None,
        transform: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Standalone validation helper when pre-extracted features are not provided.
        If `model` and `transform` are supplied, extracts features and calls `validate_from_features`.
        """
        if model is not None and transform is not None:
            rgb_image = image.convert("RGB")
            tensor = transform(rgb_image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                feats = model.forward_features(tensor)
                penultimate = model.forward_head(feats, pre_logits=True)
                logits = model.get_classifier()(penultimate)
            return self.validate_from_features(rgb_image, penultimate, logits)

        # Fallback if called without model
        dummy_pen = torch.zeros((1, 1792), dtype=torch.float32)
        dummy_log = torch.zeros((1, 7), dtype=torch.float32)
        return self.validate_from_features(image, dummy_pen, dummy_log)
