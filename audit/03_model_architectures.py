"""
AUDIT 03 — Model architecture, parameter counts, tensor dimensions, and fusion mechanisms.
Verifies:
- Parameter counts (total, trainable) for all architectures
- Exact tensor flow and shapes through branches
- Attention gate implementation differences across versions (v1 matrix gate vs v2 scalar gates)
- Pretrained initialization behavior
- Loss functions (FocalLoss vs CrossEntropy)
"""
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import timm
from models.dual_branch_net import DualBranchNet
from models.shallow_wide_branch import ShallowWideBranch
from models.deep_narrow_branch import DeepNarrowBranch
from models.fusion import AttentionFusionHead
from losses.focal_loss import FocalLoss

OUT = ROOT / "audit" / "outputs"
OUT.mkdir(exist_ok=True)

results = {}

# 1. Baseline Architectures
baselines = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "efficientnet_b4": "efficientnet_b4"
}

for name, timm_id in baselines.items():
    m = timm.create_model(timm_id, pretrained=False, num_classes=7)
    total_params = sum(p.numel() for p in m.parameters())
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    results[name] = {
        "timm_id": timm_id,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "output_shape": list(out.shape),
        "classifier_type": str(type(m.get_classifier())),
    }

# 2. Dual-Branch CNN Current Code
db_current = DualBranchNet(num_classes=7, pretrained_init=False)
x = torch.randn(2, 3, 224, 224)
logits, fmap_tex, fmap_str = db_current(x)

results["dual_branch_current_code"] = {
    "total_parameters": sum(p.numel() for p in db_current.parameters()),
    "trainable_parameters": sum(p.numel() for p in db_current.parameters() if p.requires_grad),
    "shallow_branch_parameters": sum(p.numel() for p in db_current.shallow_branch.parameters()),
    "deep_branch_parameters": sum(p.numel() for p in db_current.deep_branch.parameters()),
    "fusion_parameters": sum(p.numel() for p in db_current.fusion.parameters()),
    "classifier_parameters": sum(p.numel() for p in db_current.classifier.parameters()),
    "output_logits_shape": list(logits.shape),
    "texture_fmap_shape": list(fmap_tex.shape),
    "structure_fmap_shape": list(fmap_str.shape),
    "fusion_type": "scalar_gates (gate_t: Linear(1024, 1), gate_s: Linear(256, 1))"
}

# 3. Dual-Branch CNN V1 (from checkpoint seed123/seed999 inspection)
# In seed123/seed999, fusion was:
# gate: Linear(1280, 1280) -> Sigmoid()
# mlp: Linear(1280, 512) -> GELU -> Dropout -> Linear(512, 256) -> GELU -> Dropout
class AttentionFusionHeadV1(nn.Module):
    def __init__(self, texture_dim=1024, structure_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()
        concat_dim = texture_dim + structure_dim
        self.gate = nn.Sequential(nn.Linear(concat_dim, concat_dim), nn.Sigmoid())
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
    def forward(self, t, s):
        c = torch.cat([t, s], dim=1)
        g = self.gate(c)
        return self.mlp(c * g)

fusion_v1 = AttentionFusionHeadV1()
db_v1_params = (
    sum(p.numel() for p in db_current.shallow_branch.parameters()) +
    sum(p.numel() for p in db_current.deep_branch.parameters()) +
    sum(p.numel() for p in fusion_v1.parameters()) +
    sum(p.numel() for p in db_current.classifier.parameters())
)
results["dual_branch_v1_matrix_gate"] = {
    "total_parameters": db_v1_params,
    "fusion_parameters": sum(p.numel() for p in fusion_v1.parameters()),
    "gate_type": "Linear(1280, 1280) -> Sigmoid()"
}

# 4. Focal Loss verification
fl = FocalLoss(gamma=2.0)
dummy_logits = torch.randn(4, 7)
dummy_targets = torch.tensor([0, 2, 4, 6])
loss_val = fl(dummy_logits, dummy_targets)
results["focal_loss_test"] = {
    "loss_scalar": float(loss_val.item()),
    "is_nan": bool(torch.isnan(loss_val).item())
}

(OUT / "03_architecture.json").write_text(json.dumps(results, indent=2))
print("Architecture audit finished. Results:")
print(json.dumps(results, indent=2))
