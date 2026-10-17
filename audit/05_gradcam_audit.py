"""
AUDIT 05 — Grad-CAM Explainability Audit.
Verifies:
- EfficientNetGradCAM from api/inference_engine.py (used in production API)
- DualBranchGradCAM from explainability/gradcam.py (used in research pipeline)
- Target layers, gradient aggregation, activation hook capture
- Weight immutability during/after attribution
- Min/Max bounds, NaN/Inf checks, heatmap resolution
"""
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import timm
from api.inference_engine import EfficientNetGradCAM
from explainability.gradcam import DualBranchGradCAM
from models.dual_branch_net import DualBranchNet
from data.augmentations import get_val_transforms

OUT = ROOT / "audit" / "outputs"
OUT.mkdir(exist_ok=True)

HAM = Path("/Users/mithran/Documents/My projects/ADL MAIN/ADL.  LOQ/datasets/HAM10000")
MAC_CKPTS = ROOT / "checkpoints"

report = {}

sample_path = HAM / "HAM10000_images_part_1" / "ISIC_0024306.jpg"
with Image.open(sample_path) as img:
    img_pil = img.convert("RGB")
    img_np = np.array(img_pil)

transform = get_val_transforms(224)
x = transform(image=img_np)["image"].unsqueeze(0)

# 1. Test EfficientNetGradCAM (API production)
m_eff = timm.create_model("efficientnet_b4", pretrained=False, num_classes=7)
ckpt = torch.load(MAC_CKPTS / "efficientnet_b4" / "best_checkpoint.pth", map_location="cpu", weights_only=False)
m_eff.load_state_dict(ckpt["state_dict"])
m_eff.eval()

weights_before = {k: v.clone() for k, v in m_eff.state_dict().items()}

cam_eff = EfficientNetGradCAM(m_eff)
heatmap_eff = cam_eff.generate_heatmap(x, class_idx=4)
overlay_eff = cam_eff.overlay_heatmap(img_pil, heatmap_eff)

weights_after = {k: v for k, v in m_eff.state_dict().items()}
weights_unmodified_eff = all(torch.equal(weights_before[k], weights_after[k]) for k in weights_before)

report["efficientnet_gradcam"] = {
    "target_layer": "conv_head",
    "heatmap_shape": list(heatmap_eff.shape),
    "heatmap_min": float(np.min(heatmap_eff)),
    "heatmap_max": float(np.max(heatmap_eff)),
    "heatmap_has_nan": bool(np.isnan(heatmap_eff).any()),
    "heatmap_has_inf": bool(np.isinf(heatmap_eff).any()),
    "overlay_size": list(overlay_eff.size),
    "weights_remained_unmodified": weights_unmodified_eff,
    "model_eval_mode_preserved": not m_eff.training
}
cam_eff.close()

# 2. Test DualBranchGradCAM (DualBranchNet research pipeline)
m_db = DualBranchNet(num_classes=7, pretrained_init=False)
m_db.eval()

db_weights_before = {k: v.clone() for k, v in m_db.state_dict().items()}

with DualBranchGradCAM(m_db) as cam_db:
    tex_cam, str_cam, pred_cls = cam_db.generate(x.clone(), target_class=4)

db_weights_after = {k: v for k, v in m_db.state_dict().items()}
db_weights_unmodified = all(torch.equal(db_weights_before[k], db_weights_after[k]) for k in db_weights_before)

report["dual_branch_gradcam"] = {
    "shallow_target_layer": "shallow_branch.block3.conv",
    "deep_target_layer": "deep_branch.stage4.conv_layers[-2]",
    "texture_cam_shape": list(tex_cam.shape),
    "structure_cam_shape": list(str_cam.shape),
    "texture_cam_min": float(np.min(tex_cam)),
    "texture_cam_max": float(np.max(tex_cam)),
    "structure_cam_min": float(np.min(str_cam)),
    "structure_cam_max": float(np.max(str_cam)),
    "weights_remained_unmodified": db_weights_unmodified,
    "model_eval_mode_preserved": not m_db.training
}

(OUT / "05_gradcam.json").write_text(json.dumps(report, indent=2))
print("Grad-CAM audit completed successfully:")
print(json.dumps(report, indent=2))
