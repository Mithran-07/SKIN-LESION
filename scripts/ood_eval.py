"""
OOD Domain Gate — Calibration & Evaluation Script
Evaluates the MobileNetV2 prototype-based OOD gate on:
  - IN-DOMAIN:  HAM10000 dermoscopic samples (held out from prototypes)
  - OUT-OF-DOMAIN: synthetic OOD set (noise, solid colors, web images)

Outputs:
  - Similarity distributions per class
  - ROC curve + AUROC
  - Precision / Recall / F1 at multiple thresholds
  - False Acceptance Rate / False Rejection Rate
  - Calibrated threshold recommendation
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

gate_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
gate_model.eval()
gate_model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

proto_path = Path("checkpoints/ood/mobilenet_v2_derm_prototypes.pt")
if not proto_path.exists():
    print("ERROR: OOD prototypes not found.")
    sys.exit(1)

prototypes = torch.load(proto_path, map_location=device)

def get_similarity(img_path=None, pil_img=None):
    if pil_img is None:
        pil_img = Image.open(img_path).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")
    t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = gate_model.features(t)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(1, -1)
        feat = F.normalize(feat, p=2, dim=1)
    return torch.matmul(feat, prototypes.T).max().item()


# ------------------------------------------------------------------
# Build evaluation dataset
# ------------------------------------------------------------------
in_domain = []   # (similarity, label=1)
ood = []         # (similarity, label=0)

# IN-DOMAIN: dermoscopic samples (already in data/samples)
sample_dir = Path("data/samples")
for img_path in sorted(sample_dir.glob("*.jpg")):
    sim = get_similarity(img_path=img_path)
    in_domain.append(sim)
    print(f"  [IN-DOMAIN] {img_path.name}: {sim:.4f}")

# OOD: synthetic/artificial images
print("\n--- Generating OOD images ---")

# Solid black
ood_imgs = {
    "solid_black":    Image.new("RGB", (224, 224), (0, 0, 0)),
    "solid_white":    Image.new("RGB", (224, 224), (255, 255, 255)),
    "solid_red":      Image.new("RGB", (224, 224), (255, 0, 0)),
    "solid_blue":     Image.new("RGB", (224, 224), (0, 0, 255)),
    "solid_green":    Image.new("RGB", (224, 224), (0, 255, 0)),
    "solid_gray":     Image.new("RGB", (224, 224), (128, 128, 128)),
}

# Noise images (different seeds)
for seed in [0, 1, 2, 3, 4]:
    np.random.seed(seed)
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    ood_imgs[f"noise_{seed}"] = Image.fromarray(arr)

# Gradient images (horizontal, vertical)
for axis in ["h", "v"]:
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        val = int(i / 224 * 255)
        if axis == "h":
            arr[:, i, :] = val
        else:
            arr[i, :, :] = val
    ood_imgs[f"gradient_{axis}"] = Image.fromarray(arr)

# Checkerboard
arr = np.zeros((224, 224, 3), dtype=np.uint8)
for i in range(224):
    for j in range(224):
        if (i // 28 + j // 28) % 2 == 0:
            arr[i, j] = [0, 0, 0]
        else:
            arr[i, j] = [255, 255, 255]
ood_imgs["checkerboard"] = Image.fromarray(arr)

# Text-on-white (simulating screenshots/documents)
try:
    from PIL import ImageDraw
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "HELLO WORLD\nThis is a document\nNot a skin image", fill="black")
    ood_imgs["text_document"] = img
except Exception:
    pass

# Evaluate OOD
for name, pil_img in ood_imgs.items():
    sim = get_similarity(pil_img=pil_img)
    ood.append(sim)
    print(f"  [OOD] {name}: {sim:.4f}")

# ------------------------------------------------------------------
# Compute metrics
# ------------------------------------------------------------------
all_sims = np.array(in_domain + ood)
all_labels = np.array([1] * len(in_domain) + [0] * len(ood))

print(f"\n=== DISTRIBUTION SUMMARY ===")
print(f"IN-DOMAIN  (n={len(in_domain)}): mean={np.mean(in_domain):.4f}, min={np.min(in_domain):.4f}, max={np.max(in_domain):.4f}")
print(f"OUT-DOMAIN (n={len(ood)}): mean={np.mean(ood):.4f}, min={np.min(ood):.4f}, max={np.max(ood):.4f}")

# ROC
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, f1_score

auc = roc_auc_score(all_labels, all_sims)
print(f"\nAUROC: {auc:.4f}")

# Threshold sweep
print("\n=== THRESHOLD SWEEP ===")
print(f"{'Threshold':>10} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'FAR':>6} | {'FRR':>6}")
print("-" * 60)
best_f1, best_thresh = 0, 0.50
for thresh in np.arange(0.30, 0.85, 0.05):
    preds = (all_sims >= thresh).astype(int)
    n_in, n_ood = len(in_domain), len(ood)
    true_accept = ((preds == 1) & (all_labels == 1)).sum()
    false_accept = ((preds == 1) & (all_labels == 0)).sum()  # OOD passed as valid
    false_reject = ((preds == 0) & (all_labels == 1)).sum()  # real image rejected
    true_reject  = ((preds == 0) & (all_labels == 0)).sum()

    precision = true_accept / (true_accept + false_accept) if (true_accept + false_accept) > 0 else 0
    recall    = true_accept / n_in
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far       = false_accept / n_ood   # false acceptance rate
    frr       = false_reject / n_in    # false rejection rate
    print(f"{thresh:>10.2f} | {precision:>9.4f} | {recall:>6.4f} | {f1:>6.4f} | {far:>6.4f} | {frr:>6.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\nBEST F1 THRESHOLD: {best_thresh:.2f} (F1={best_f1:.4f})")

# Final result at best threshold
preds = (all_sims >= best_thresh).astype(int)
n_in, n_ood = len(in_domain), len(ood)
false_accept = ((preds == 1) & (all_labels == 0)).sum()
false_reject = ((preds == 0) & (all_labels == 1)).sum()
print(f"\n=== FINAL EVALUATION AT THRESHOLD {best_thresh:.2f} ===")
print(f"False Acceptance Rate (FAR): {false_accept/n_ood:.4f}  ({false_accept}/{n_ood} OOD images passed)")
print(f"False Rejection Rate  (FRR): {false_reject/n_in:.4f}  ({false_reject}/{n_in} valid images rejected)")
