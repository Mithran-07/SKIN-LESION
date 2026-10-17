"""
AUDIT 04 — Test Set Re-evaluation and Verification of All Checkpoints.
Optimized with batch_size=8 and torch.mps.empty_cache() to prevent
unified memory exhaustion on Apple Silicon (16GB RAM) caused by
the shallow branch's massive 1024-channel x 112x112 spatial feature maps.
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dual_branch_net import DualBranchNet
from data.augmentations import get_val_transforms

OUT = ROOT / "audit" / "outputs"
OUT.mkdir(exist_ok=True)
CACHE_FILE = OUT / "04_evaluation_verified.json"

HAM = Path("/Users/mithran/Documents/My projects/ADL MAIN/ADL.  LOQ/datasets/HAM10000")
LOQ_CKPTS = Path("/Users/mithran/Documents/My projects/ADL MAIN/ADL.  LOQ/checkpoints")
MAC_CKPTS = ROOT / "checkpoints"

image_index = {p.stem: str(p) for d in ["HAM10000_images_part_1", "HAM10000_images_part_2"] for p in (HAM / d).glob("*.jpg")}
test_df = pd.read_csv(ROOT / "splits" / "test.csv")
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

class FastDermDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        self.paths = [image_index[iid] for iid in df["image_id"]]
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as img:
            img_np = np.array(img.convert("RGB"))
        augmented = self.transform(image=img_np)
        return augmented["image"], self.labels[idx]

val_transform = get_val_transforms(224)
test_dataset = FastDermDataset(test_df, val_transform)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using evaluation device: {device}")

# Helper for Dual Branch V1 instantiation
class DualBranchNetV1(DualBranchNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        self.fusion = AttentionFusionHeadV1(
            texture_dim=self.shallow_branch.output_dim,
            structure_dim=self.deep_branch.output_dim
        )

# Load existing evaluations if any
if CACHE_FILE.exists():
    eval_summary = json.loads(CACHE_FILE.read_text())
else:
    eval_summary = {}

models_to_eval = [
    {
        "name": "efficientnet_b4",
        "type": "timm",
        "timm_id": "efficientnet_b4",
        "batch_size": 16,
        "ckpt": MAC_CKPTS / "efficientnet_b4" / "best_checkpoint.pth"
    },
    {
        "name": "densenet121",
        "type": "timm",
        "timm_id": "densenet121",
        "batch_size": 16,
        "ckpt": LOQ_CKPTS / "densenet121" / "best_checkpoint.pth"
    },
    {
        "name": "resnet50",
        "type": "timm",
        "timm_id": "resnet50",
        "batch_size": 16,
        "ckpt": LOQ_CKPTS / "resnet50" / "best_checkpoint.pth"
    },
    {
        "name": "dual_branch_seed123",
        "type": "dual_branch_v1",
        "batch_size": 8,
        "ckpt": LOQ_CKPTS / "dual_branch_seed123" / "best_checkpoint.pth"
    },
    {
        "name": "dual_branch_seed999",
        "type": "dual_branch_v1",
        "batch_size": 8,
        "ckpt": LOQ_CKPTS / "dual_branch_seed999" / "best_checkpoint.pth"
    },
    {
        "name": "dual_branch_seed42",
        "type": "dual_branch_v2",
        "batch_size": 8,
        "ckpt": LOQ_CKPTS / "dual_branch_seed42" / "best_checkpoint.pth"
    }
]

for target in models_to_eval:
    m_name = target["name"]
    ckpt_path = target["ckpt"]

    if m_name in eval_summary:
        print(f"Skipping {m_name} (already evaluated).")
        continue

    print(f"\n==================================================")
    print(f"Evaluating {m_name} (batch_size={target['batch_size']}) from {ckpt_path.name}...")

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        continue

    if target["type"] == "timm":
        model = timm.create_model(target["timm_id"], pretrained=False, num_classes=7)
    elif target["type"] == "dual_branch_v1":
        model = DualBranchNetV1(num_classes=7, pretrained_init=False)
    elif target["type"] == "dual_branch_v2":
        model = DualBranchNet(num_classes=7, pretrained_init=False)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=target["batch_size"], shuffle=False, num_workers=0)

    all_preds = []
    all_probs = []
    all_targets = []

    # Warmup
    dummy = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy)

    start_time = time.perf_counter()
    with torch.no_grad():
        for b_idx, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(device)
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits.float(), dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(np.array(lbls))

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(loader):
                print(f"  [{m_name}] Batch {b_idx + 1}/{len(loader)} done.")

            # Prevent memory thrashing on MPS
            if device.type == "mps":
                torch.mps.empty_cache()

    elapsed = time.perf_counter() - start_time
    latency_ms_per_img = (elapsed / len(test_df)) * 1000

    y_probs = np.vstack(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    y_bin = label_binarize(y_true, classes=list(range(7)))
    try:
        macro_auc = roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr")
    except Exception:
        macro_auc = None

    try:
        pr_aucs = [average_precision_score(y_bin[:, c], y_probs[:, c]) for c in range(7)]
        macro_pr_auc = float(np.mean(pr_aucs))
    except Exception:
        macro_pr_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    clf_rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    res = {
        "model_name": m_name,
        "checkpoint_epoch": ckpt.get("epoch"),
        "best_val_bal_acc_in_ckpt": ckpt.get("best_val_bal_acc"),
        "test_n_samples": len(y_true),
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bal_acc),
        "test_macro_precision": float(p_macro),
        "test_macro_recall": float(r_macro),
        "test_macro_f1": float(f1_macro),
        "test_macro_roc_auc": float(macro_auc) if macro_auc is not None else None,
        "test_macro_pr_auc": float(macro_pr_auc) if macro_pr_auc is not None else None,
        "measured_latency_ms_per_img": float(latency_ms_per_img),
        "measured_device": str(device),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_rep
    }

    eval_summary[m_name] = res
    CACHE_FILE.write_text(json.dumps(eval_summary, indent=2))

    print(f"Results for {m_name}:")
    print(f"  Test Accuracy:     {acc*100:.2f}%")
    print(f"  Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"  Macro F1:          {f1_macro*100:.2f}%")
    print(f"  Macro ROC-AUC:     {macro_auc*100:.2f}%" if macro_auc else "  Macro ROC-AUC: N/A")
    print(f"  Latency ({device}): {latency_ms_per_img:.2f} ms/img")

    # Clean up model
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

print(f"\nAll evaluations complete. Saved to {CACHE_FILE}")
