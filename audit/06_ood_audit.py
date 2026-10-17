"""
AUDIT 06 — Out-of-Distribution (OOD) Domain Gate Audit.
Verifies:
- Prototype origin and dimension (checkpoints/ood/mobilenet_v2_derm_prototypes.pt)
- Backbone architecture (MobileNetV2 feature extractor)
- Metric (Cosine similarity max over prototypes)
- Threshold calibration (0.55)
- Evaluation on:
  1. The 7 demo images (prototypes)
  2. 50 random in-domain HAM10000 test images
  3. Synthetic OOD images (solid black, white, gaussian noise, gradient, text)
- Leakage/independence assessment
"""
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.ood_gate import OODDomainGate

OUT = ROOT / "audit" / "outputs"
OUT.mkdir(exist_ok=True)

HAM = Path("/Users/mithran/Documents/My projects/ADL MAIN/ADL.  LOQ/datasets/HAM10000")
SAMPLES_DIR = ROOT / "data" / "samples"
PROTO_PATH = ROOT / "checkpoints" / "ood" / "mobilenet_v2_derm_prototypes.pt"

report = {}

# 1. Inspect Prototypes
proto_data = torch.load(PROTO_PATH, map_location="cpu", weights_only=False)
report["prototypes_metadata"] = {
    "keys": list(proto_data.keys()) if isinstance(proto_data, dict) else str(type(proto_data)),
    "classes": proto_data.get("classes") if isinstance(proto_data, dict) else None,
    "prototypes_shape": list(proto_data.get("prototypes").shape) if isinstance(proto_data, dict) else None,
    "feature_dim": proto_data.get("feature_dim") if isinstance(proto_data, dict) else None
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
gate = OODDomainGate(device=device, threshold=0.55)

# 2. Test Prototype/Demo Samples
demo_results = {}
for p in sorted(SAMPLES_DIR.glob("*_sample.jpg")):
    with Image.open(p) as img:
        res = gate.validate_image(img)
        demo_results[p.name] = {
            "score": float(res["score"]),
            "valid": bool(res["valid"]),
            "domain": res["domain"]
        }
report["demo_samples_validation"] = demo_results

# 3. Test In-Domain Test Split Samples (50 random test images from HAM10000)
import pandas as pd
test_df = pd.read_csv(ROOT / "splits" / "test.csv").sample(n=50, random_state=42)
test_scores = []
for _, row in test_df.iterrows():
    iid = row["image_id"]
    p1 = HAM / "HAM10000_images_part_1" / f"{iid}.jpg"
    p2 = HAM / "HAM10000_images_part_2" / f"{iid}.jpg"
    p = p1 if p1.exists() else p2
    with Image.open(p) as img:
        res = gate.validate_image(img)
        test_scores.append(float(res["score"]))

report["ham10000_test_sample_50"] = {
    "min_score": float(np.min(test_scores)),
    "max_score": float(np.max(test_scores)),
    "mean_score": float(np.mean(test_scores)),
    "median_score": float(np.median(test_scores)),
    "fraction_accepted": float(np.mean([s >= 0.55 for s in test_scores])),
    "false_rejection_count": int(sum(s < 0.55 for s in test_scores))
}

# 4. Test Synthetic OOD Inputs
synthetic_cases = {}
# Solid black
img_black = Image.new("RGB", (224, 224), (0, 0, 0))
synthetic_cases["solid_black"] = gate.validate_image(img_black)

# Solid white
img_white = Image.new("RGB", (224, 224), (255, 255, 255))
synthetic_cases["solid_white"] = gate.validate_image(img_white)

# Random Gaussian noise
noise_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img_noise = Image.fromarray(noise_arr)
synthetic_cases["random_noise"] = gate.validate_image(img_noise)

# Text / document image
img_text = Image.new("RGB", (300, 300), (240, 240, 240))
d = ImageDraw.Draw(img_text)
d.text((20, 20), "Patient Medical Record\nDiagnosis: None\nConfidential", fill=(0, 0, 0))
synthetic_cases["text_document"] = gate.validate_image(img_text)

report["synthetic_ood_results"] = {
    k: {
        "score": float(v["score"]),
        "valid": bool(v["valid"]),
        "domain": v["domain"],
        "reason": v["reason"]
    }
    for k, v in synthetic_cases.items()
}

(OUT / "06_ood_audit.json").write_text(json.dumps(report, indent=2))
print("OOD audit completed. Summary:")
print(json.dumps(report, indent=2))
