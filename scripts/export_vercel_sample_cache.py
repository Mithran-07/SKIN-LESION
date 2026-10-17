"""
Export Pre-Verified Inference & Counterfactual Grad-CAM Cache for Vercel Cloud Deployment.

Executes the real `SkinLesionInferenceEngine` (EfficientNet-B4 checkpoint SHA-256
`125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e` + Calibrated
OODDomainGate v2.0.0) across all 7 held-out HAM10000 demo images in
`app/skin-lesion-app/public/samples/` and all 7 counterfactual target classes.

This guarantees that when the Next.js 14 application is deployed on Vercel
(without a local 127.0.0.1:8000 PyTorch/MPS process), the Next.js `/api/*`
routes serve 100% genuine, bit-exact model outputs, OOD v2 telemetry, and
counterfactual Grad-CAM heatmaps rather than failing or fabricating numbers.
"""

import hashlib
import json
from pathlib import Path
from PIL import Image

from api.inference_engine import get_inference_engine, HAM10000_CLASSES

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "app/skin-lesion-app/public/samples"
OUT_PATH = ROOT / "app/skin-lesion-app/src/lib/verified_samples_cache.json"


def main():
    engine = get_inference_engine()
    assert engine.checkpoint_loaded, "Trained EfficientNet-B4 checkpoint must be loaded."

    cache = {
        "cache_version": "2.0.0",
        "checkpoint_sha256": engine.checkpoint_sha256,
        "checkpoint_sha256_short": engine.checkpoint_sha256_short,
        "ood_version": engine.ood_gate.VERSION,
        "ood_threshold": round(float(engine.ood_gate.threshold), 4),
        "samples_by_filename": {},
        "samples_by_sha256": {},
    }

    sample_files = sorted(SAMPLES_DIR.glob("*_sample.jpg"))
    assert len(sample_files) == 7, f"Expected 7 sample images, found {len(sample_files)}"

    for img_path in sample_files:
        raw_bytes = img_path.read_bytes()
        file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        pil_img = Image.open(img_path)

        # Base prediction + top-1 Grad-CAM explanation
        base_result = engine.predict_and_explain(pil_img, target_class=None, alpha=0.45)
        base_result["filename"] = img_path.name
        base_result["execution_mode"] = "vercel_verified_checkpoint_cache"

        # Counterfactual Grad-CAM explanations for all 7 classes
        counterfactuals = {}
        for cls_code in HAM10000_CLASSES:
            cf_res = engine.predict_and_explain(pil_img, target_class=cls_code, alpha=0.45)
            counterfactuals[cls_code] = cf_res["explainability"]

        entry = {
            "filename": img_path.name,
            "sha256": file_sha256,
            "base_response": base_result,
            "counterfactual_explainability": counterfactuals,
        }
        cache["samples_by_filename"][img_path.name] = entry
        cache["samples_by_sha256"][file_sha256] = img_path.name
        print(
            f"[OK] {img_path.name} (sha256={file_sha256[:12]}...) -> "
            f"pred={base_result['predicted_class']} ({base_result['confidence_percentage']}), "
            f"ood_score={base_result['input_validation']['score']:.4f}"
        )

    OUT_PATH.write_text(json.dumps(cache, indent=2))
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
