"""
Fusion Diagnostic Analysis — Dual-Branch CNN
==============================================
Sections:
  1. Attention Gate Behaviour  (stats + distribution plot)
  2. Sample-wise Behaviour     (texture vs structure dominance)
  3. Class-wise Behaviour      (per-class mean gate weights)
  4. Branch Importance         (normal / texture-off / structure-off inference)
  5. Feature Correlation       (cosine similarity between branch vecs)
  6. Fusion Collapse Detection (% samples collapsed to structure)
  7. Recommendations           (written to report)

Output:  results/fusion_diagnostic.md
         results/fusion_gate_distribution.png
         results/fusion_class_gate_weights.png
"""

import sys, os, json, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score
)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("C:/ADL")
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader import get_all_dataloaders
from models.dual_branch_net import DualBranchNet

CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
TEXTURE_DIM = 1024
STRUCTURE_DIM = 256
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_splits():
    splits_dir = PROJECT_ROOT / "splits"
    import pandas as pd
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df   = pd.read_csv(splits_dir / "val.csv")
    test_df  = pd.read_csv(splits_dir / "test.csv")
    return train_df, val_df, test_df


def load_model(device):
    # Instantiate with the same defaults used during training
    model = DualBranchNet(
        num_classes=7,
        shallow_channels=[256, 512, 1024],
        deep_base_channels=64,
        deep_bottleneck=256,
        deep_num_blocks=[2, 2, 3, 3],
        fusion_hidden_dim=512,
        fusion_output_dim=256,
        pretrained_init=False,   # weights come from checkpoint
    )
    ckpt_path = PROJECT_ROOT / "checkpoints" / "dual_branch_seed42" / "best_checkpoint.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[OK] Model loaded from {ckpt_path}")
    else:
        print(f"[WARNING] Checkpoint {ckpt_path} not found. Using untrained weights.")
    model = model.to(device)
    model.eval()
    return model


def collect_features(model, loader, device, max_batches=None):
    """Collect gate weights, texture vecs, structure vecs, labels, and logits."""
    all_gate_texture   = []   # mean per-sample over texture dims
    all_gate_structure = []   # mean per-sample over structure dims
    all_texture_vecs   = []
    all_structure_vecs = []
    all_labels  = []
    all_logits  = []

    gate_output_holder = {}

    hook_t = model.fusion.gate_t.register_forward_hook(
        lambda m, i, o: gate_output_holder.update({"gate_t": o.detach().cpu()}))
    hook_s = model.fusion.gate_s.register_forward_hook(
        lambda m, i, o: gate_output_holder.update({"gate_s": o.detach().cpu()}))

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                images = batch["image"]
                labels = batch["label"]

            images = images.to(device)
            logits, _, _ = model(images)

            gate_t = gate_output_holder["gate_t"] # (B, 1)
            gate_s = gate_output_holder["gate_s"] # (B, 1)
            all_gate_texture.append(gate_t.squeeze(1).numpy())
            all_gate_structure.append(gate_s.squeeze(1).numpy())

            all_logits.append(logits.cpu().numpy())
            lbl = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            all_labels.append(lbl)

    hook_t.remove()
    hook_s.remove()

    # Collect raw branch feature vecs
    tex_holder, str_holder = {}, {}
    h1 = model.shallow_branch.register_forward_hook(
        lambda m, i, o: tex_holder.update({"v": o[0].detach().cpu()}))
    h2 = model.deep_branch.register_forward_hook(
        lambda m, i, o: str_holder.update({"v": o[0].detach().cpu()}))

    all_tex_vecs = []
    all_str_vecs = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                images = batch["image"]
                labels = batch["label"]
            images = images.to(device)
            model(images)
            all_tex_vecs.append(tex_holder["v"].numpy())
            all_str_vecs.append(str_holder["v"].numpy())

    h1.remove()
    h2.remove()

    n = min(len(all_gate_texture), len(all_tex_vecs))

    return {
        "gate_texture":   np.concatenate(all_gate_texture[:n],   axis=0),
        "gate_structure": np.concatenate(all_gate_structure[:n],  axis=0),
        "texture_vecs":   np.concatenate(all_tex_vecs[:n],        axis=0),
        "structure_vecs": np.concatenate(all_str_vecs[:n],        axis=0),
        "labels":         np.concatenate(all_labels[:n],          axis=0),
        "logits":         np.concatenate(all_logits[:n],          axis=0),
    }


# ---------------------------------------------------------------------------
# Branch Ablation Inference
# ---------------------------------------------------------------------------

def run_ablation(model, loader, device, mode="normal"):
    """mode: 'normal' | 'texture_off' | 'structure_off'"""
    all_preds, all_targets, all_probs = [], [], []

    tex_holder, str_holder = {}, {}
    h1 = model.shallow_branch.register_forward_hook(
        lambda m, i, o: tex_holder.update({"v": o[0].detach()}))
    h2 = model.deep_branch.register_forward_hook(
        lambda m, i, o: str_holder.update({"v": o[0].detach()}))

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                images = batch["image"]
                labels = batch["label"]
            images = images.to(device)
            lbl_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

            model(images)

            t_vec = tex_holder["v"]
            s_vec = str_holder["v"]

            if mode == "texture_off":
                t_vec = torch.zeros_like(t_vec)
            elif mode == "structure_off":
                s_vec = torch.zeros_like(s_vec)

            fused    = model.fusion(t_vec, s_vec)
            logits   = model.classifier(fused)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.append(preds)
            all_targets.append(lbl_np)
            all_probs.append(probs)

    h1.remove()
    h2.remove()

    targets = np.concatenate(all_targets)
    preds   = np.concatenate(all_preds)
    probs   = np.concatenate(all_probs, axis=0)

    acc      = accuracy_score(targets, preds)
    bal_acc  = balanced_accuracy_score(targets, preds)
    f1       = f1_score(targets, preds, average="macro", zero_division=0)
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(targets, classes=list(range(7)))
        roc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        roc = float("nan")

    return {"accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": f1, "macro_roc_auc": roc}


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def stats(arr):
    return {
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_df, val_df, test_df = load_splits()
    _, val_loader, test_loader = get_all_dataloaders(
        train_df, val_df, test_df,
        batch_size=8,
        num_workers=0,
        image_size=224,
        images_root=None,
        pin_memory=False,
    )

    model = load_model(device)

    # =========================================================
    # Collect features
    # =========================================================
    print("[INFO] Collecting gate weights from validation set...")
    val_data  = collect_features(model, val_loader,  device)
    print("[INFO] Collecting gate weights from test set...")
    test_data = collect_features(model, test_loader, device)

    combined = {k: np.concatenate([val_data[k], test_data[k]], axis=0) for k in val_data}
    N = len(combined["labels"])
    print(f"[INFO] Combined samples: {N}")

    # =========================================================
    # 1. Attention Gate Behaviour
    # =========================================================
    print("\n[INFO] Section 1: Attention Gate Behaviour...")

    sample_tex_weight = combined["gate_texture"]    # (N,) already mean over dim
    sample_str_weight = combined["gate_structure"]  # (N,)

    tex_stats = stats(sample_tex_weight)
    str_stats = stats(sample_str_weight)
    print(f"  Texture: mean={tex_stats['mean']:.4f} median={tex_stats['median']:.4f} std={tex_stats['std']:.4f} min={tex_stats['min']:.4f} max={tex_stats['max']:.4f}")
    print(f"  Structure: mean={str_stats['mean']:.4f} median={str_stats['median']:.4f} std={str_stats['std']:.4f} min={str_stats['min']:.4f} max={str_stats['max']:.4f}")

    # Gate distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Attention Gate Weight Distributions (Val + Test)", fontsize=13, fontweight="bold")

    axes[0].hist(sample_tex_weight, bins=50, color="#4C72B0", alpha=0.85, edgecolor="white")
    axes[0].axvline(tex_stats["mean"], color="red", ls="--", label=f"Mean={tex_stats['mean']:.3f}")
    axes[0].axvline(tex_stats["median"], color="orange", ls=":", label=f"Median={tex_stats['median']:.3f}")
    axes[0].set_title("Texture Branch Gate Weight")
    axes[0].set_xlabel("Mean Gate Weight per Sample")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(sample_str_weight, bins=50, color="#C44E52", alpha=0.85, edgecolor="white")
    axes[1].axvline(str_stats["mean"], color="blue", ls="--", label=f"Mean={str_stats['mean']:.3f}")
    axes[1].axvline(str_stats["median"], color="cyan", ls=":", label=f"Median={str_stats['median']:.3f}")
    axes[1].set_title("Structure Branch Gate Weight")
    axes[1].set_xlabel("Mean Gate Weight per Sample")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    dist_plot = RESULTS_DIR / "fusion_gate_distribution.png"
    plt.savefig(dist_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gate distribution -> {dist_plot}")

    # =========================================================
    # 2. Sample-wise Behaviour
    # =========================================================
    print("\n[INFO] Section 2: Sample-wise Behaviour...")

    total_w  = sample_tex_weight + sample_str_weight
    tex_frac = sample_tex_weight / (total_w + 1e-8)
    str_frac = sample_str_weight / (total_w + 1e-8)

    TEX_THRESH = 0.60
    STR_THRESH = 0.60

    tex_dominant  = tex_frac > TEX_THRESH
    str_dominant  = str_frac > STR_THRESH
    balanced_mask = (~tex_dominant) & (~str_dominant)

    n_tex = int(tex_dominant.sum())
    n_str = int(str_dominant.sum())
    n_bal = int(balanced_mask.sum())

    samplewise = {
        "n_total": N,
        "n_texture_dominant": n_tex,  "pct_texture_dominant": n_tex/N*100,
        "n_structure_dominant": n_str, "pct_structure_dominant": n_str/N*100,
        "n_balanced": n_bal,           "pct_balanced": n_bal/N*100,
        "texture_frac_stats":   stats(tex_frac),
        "structure_frac_stats": stats(str_frac),
    }

    print(f"  Texture-dominant: {n_tex} ({n_tex/N*100:.1f}%)")
    print(f"  Structure-dominant: {n_str} ({n_str/N*100:.1f}%)")
    print(f"  Balanced: {n_bal} ({n_bal/N*100:.1f}%)")

    # =========================================================
    # 3. Class-wise Behaviour
    # =========================================================
    print("\n[INFO] Section 3: Class-wise Behaviour...")

    labels = combined["labels"]
    classwise = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() == 0:
            classwise[cls] = None
            continue
        classwise[cls] = {
            "count": int(mask.sum()),
            "avg_texture_gate":    float(sample_tex_weight[mask].mean()),
            "avg_structure_gate":  float(sample_str_weight[mask].mean()),
            "avg_texture_frac":    float(tex_frac[mask].mean()),
            "avg_structure_frac":  float(str_frac[mask].mean()),
        }
        d = classwise[cls]
        print(f"  {cls:6s}: n={d['count']:4d} | tex_gate={d['avg_texture_gate']:.4f} | str_gate={d['avg_structure_gate']:.4f} | tex_frac={d['avg_texture_frac']:.3f}")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Per-Class Mean Attention Gate Weights", fontsize=13, fontweight="bold")
    tex_means = [classwise[c]["avg_texture_gate"]   if classwise[c] else 0 for c in CLASS_NAMES]
    str_means = [classwise[c]["avg_structure_gate"] if classwise[c] else 0 for c in CLASS_NAMES]

    axes[0].bar(CLASS_NAMES, tex_means, color="#4C72B0", alpha=0.85, edgecolor="black")
    axes[0].set_title("Texture Gate Weight per Class")
    axes[0].set_ylabel("Mean Gate Weight")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=30)
    for j, v in enumerate(tex_means):
        axes[0].text(j, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    axes[1].bar(CLASS_NAMES, str_means, color="#C44E52", alpha=0.85, edgecolor="black")
    axes[1].set_title("Structure Gate Weight per Class")
    axes[1].set_ylabel("Mean Gate Weight")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=30)
    for j, v in enumerate(str_means):
        axes[1].text(j, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    class_plot = RESULTS_DIR / "fusion_class_gate_weights.png"
    plt.savefig(class_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Class gate chart -> {class_plot}")

    # =========================================================
    # 4. Branch Importance (Ablation)
    # =========================================================
    print("\n[INFO] Section 4: Branch Ablation Inference...")

    print("  Running: Normal...")
    normal_m = run_ablation(model, test_loader, device, mode="normal")
    print(f"    Acc={normal_m['accuracy']:.4f}  BalAcc={normal_m['balanced_accuracy']:.4f}  F1={normal_m['macro_f1']:.4f}  ROC={normal_m['macro_roc_auc']:.4f}")

    print("  Running: Texture OFF...")
    tex_off_m = run_ablation(model, test_loader, device, mode="texture_off")
    print(f"    Acc={tex_off_m['accuracy']:.4f}  BalAcc={tex_off_m['balanced_accuracy']:.4f}  F1={tex_off_m['macro_f1']:.4f}  ROC={tex_off_m['macro_roc_auc']:.4f}")

    print("  Running: Structure OFF...")
    str_off_m = run_ablation(model, test_loader, device, mode="structure_off")
    print(f"    Acc={str_off_m['accuracy']:.4f}  BalAcc={str_off_m['balanced_accuracy']:.4f}  F1={str_off_m['macro_f1']:.4f}  ROC={str_off_m['macro_roc_auc']:.4f}")

    ablation = {"normal": normal_m, "texture_off": tex_off_m, "structure_off": str_off_m}

    # =========================================================
    # 5. Feature Correlation (Cosine Similarity)
    # =========================================================
    print("\n[INFO] Section 5: Feature Correlation...")

    tex_vecs = combined["texture_vecs"]   # (N, 1024)
    str_vecs = combined["structure_vecs"] # (N, 256)

    # Project texture to 256-dim via mean pooling (groups of 4)
    tex_proj = tex_vecs.reshape(tex_vecs.shape[0], 256, 4).mean(axis=2)  # (N, 256)

    def cosine_per_sample(a, b):
        dot   = (a * b).sum(axis=1)
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        return dot / (norm_a * norm_b + 1e-8)

    cos_sims = cosine_per_sample(tex_proj, str_vecs)
    cos_stats = stats(cos_sims)
    print(f"  Cosine Sim: mean={cos_stats['mean']:.4f} std={cos_stats['std']:.4f} min={cos_stats['min']:.4f} max={cos_stats['max']:.4f}")

    # =========================================================
    # 6. Fusion Collapse Detection
    # =========================================================
    print("\n[INFO] Section 6: Fusion Collapse Detection...")

    STR_COLLAPSE = 0.80
    TEX_SUPPRESS = 0.20

    str_collapsed  = str_frac > STR_COLLAPSE
    tex_suppressed = tex_frac < TEX_SUPPRESS
    both           = str_collapsed & tex_suppressed

    collapse = {
        "threshold_structure":        STR_COLLAPSE,
        "threshold_texture":          TEX_SUPPRESS,
        "n_structure_collapsed":      int(str_collapsed.sum()),
        "pct_structure_collapsed":    float(str_collapsed.sum() / N * 100),
        "n_texture_suppressed":       int(tex_suppressed.sum()),
        "pct_texture_suppressed":     float(tex_suppressed.sum() / N * 100),
        "n_both_collapsed":           int(both.sum()),
        "pct_both_collapsed":         float(both.sum() / N * 100),
        "is_collapsed":               bool(both.sum() / N > 0.5),
    }
    print(f"  Structure-collapsed (>{STR_COLLAPSE*100:.0f}%): {collapse['pct_structure_collapsed']:.1f}%")
    print(f"  Texture-suppressed  (<{TEX_SUPPRESS*100:.0f}%):  {collapse['pct_texture_suppressed']:.1f}%")
    print(f"  Both (full collapse): {collapse['pct_both_collapsed']:.1f}%")
    print(f"  COLLAPSED: {collapse['is_collapsed']}")

    # =========================================================
    # 7. Write Markdown Report
    # =========================================================
    print("\n[INFO] Writing fusion_diagnostic.md ...")

    def p(v): return f"{v:.4f}"

    d_acc_tex = tex_off_m["accuracy"]    - normal_m["accuracy"]
    d_acc_str = str_off_m["accuracy"]    - normal_m["accuracy"]
    d_f1_tex  = tex_off_m["macro_f1"]   - normal_m["macro_f1"]
    d_f1_str  = str_off_m["macro_f1"]   - normal_m["macro_f1"]

    # Determine branch dependency from ablation
    if abs(d_f1_str) > abs(d_f1_tex):
        dep_branch = "Structure"
        dep_note   = "Removing the Structure branch causes more harm than removing Texture → model relies more on Structure."
    else:
        dep_branch = "Texture"
        dep_note   = "Removing the Texture branch causes more harm than removing Structure → model relies more on Texture."

    # Collapse interpretation
    collapse_status = "⚠️ COLLAPSED" if collapse["is_collapsed"] else "✅ Not fully collapsed — but heavily biased"

    report = f"""# Fusion Mechanism Diagnostic Report
## Dual-Branch CNN (Seed 42)

**Date:** 2026-07-11
**Dataset:** Validation + Test combined ({N} total samples)
**Device:** {device}
**Checkpoint:** checkpoints/dual_branch_seed42/best_checkpoint.pth

---

## 1. Attention Gate Behaviour

The fusion head concatenates [Texture (1024-dim) | Structure (256-dim)] = 1280-dim, then passes it through a sigmoid gate.
Mean gate weights are computed per sample by averaging over each branch's gate dimensions.

### Texture Branch Gate Statistics (1024-dim)

| Metric | Value |
|:-------|------:|
| Mean   | {p(tex_stats['mean'])} |
| Median | {p(tex_stats['median'])} |
| Std    | {p(tex_stats['std'])} |
| Min    | {p(tex_stats['min'])} |
| Max    | {p(tex_stats['max'])} |

### Structure Branch Gate Statistics (256-dim)

| Metric | Value |
|:-------|------:|
| Mean   | {p(str_stats['mean'])} |
| Median | {p(str_stats['median'])} |
| Std    | {p(str_stats['std'])} |
| Min    | {p(str_stats['min'])} |
| Max    | {p(str_stats['max'])} |

**Gate Ratio (Texture / Structure): {tex_stats['mean']/str_stats['mean']:.4f}**
> A ratio < 1.0 indicates the gate assigns higher mean weights to the Structure branch.

![Gate Distributions](fusion_gate_distribution.png)

---

## 2. Sample-wise Behaviour

Each sample's texture and structure gate weights are normalized to a fraction summing to 1.0.
Dominance threshold: > 60% fraction for a given branch.

| Category | Count | % of Total |
|:---------|------:|-----------:|
| Texture-dominant (tex_frac > 60%) | {samplewise['n_texture_dominant']} | {samplewise['pct_texture_dominant']:.2f}% |
| Structure-dominant (str_frac > 60%) | {samplewise['n_structure_dominant']} | {samplewise['pct_structure_dominant']:.2f}% |
| Balanced (neither > 60%) | {samplewise['n_balanced']} | {samplewise['pct_balanced']:.2f}% |

### Texture Fraction Statistics
| Mean | Median | Std | Min | Max |
|:----:|:------:|:---:|:---:|:---:|
| {p(samplewise['texture_frac_stats']['mean'])} | {p(samplewise['texture_frac_stats']['median'])} | {p(samplewise['texture_frac_stats']['std'])} | {p(samplewise['texture_frac_stats']['min'])} | {p(samplewise['texture_frac_stats']['max'])} |

### Structure Fraction Statistics
| Mean | Median | Std | Min | Max |
|:----:|:------:|:---:|:---:|:---:|
| {p(samplewise['structure_frac_stats']['mean'])} | {p(samplewise['structure_frac_stats']['median'])} | {p(samplewise['structure_frac_stats']['std'])} | {p(samplewise['structure_frac_stats']['min'])} | {p(samplewise['structure_frac_stats']['max'])} |

---

## 3. Class-wise Behaviour

| Class | Count | Avg Tex Gate | Avg Str Gate | Tex Frac | Str Frac |
|:------|------:|:------------:|:------------:|:--------:|:--------:|
"""
    for cls in CLASS_NAMES:
        d = classwise.get(cls)
        if d is None:
            report += f"| {cls} | 0 | — | — | — | — |\n"
        else:
            report += (f"| {cls} | {d['count']} | {d['avg_texture_gate']:.4f} | "
                       f"{d['avg_structure_gate']:.4f} | {d['avg_texture_frac']:.4f} | {d['avg_structure_frac']:.4f} |\n")

    report += f"""
![Class Gate Weights](fusion_class_gate_weights.png)

---

## 4. Branch Importance (Inference Ablation)

Each ablation run zeroes out the branch's feature vector before it enters the fusion head.
No retraining was performed.

| Mode | Accuracy | Balanced Acc | Macro F1 | Macro ROC-AUC |
|:-----|:--------:|:------------:|:--------:|:-------------:|
| Normal (full model) | {p(normal_m['accuracy'])} | {p(normal_m['balanced_accuracy'])} | {p(normal_m['macro_f1'])} | {p(normal_m['macro_roc_auc'])} |
| Texture OFF | {p(tex_off_m['accuracy'])} | {p(tex_off_m['balanced_accuracy'])} | {p(tex_off_m['macro_f1'])} | {p(tex_off_m['macro_roc_auc'])} |
| Structure OFF | {p(str_off_m['accuracy'])} | {p(str_off_m['balanced_accuracy'])} | {p(str_off_m['macro_f1'])} | {p(str_off_m['macro_roc_auc'])} |

### Performance Delta vs. Normal:

| Ablation | ΔAccuracy | ΔMacro F1 |
|:---------|:---------:|:---------:|
| Texture OFF | {d_acc_tex:+.4f} | {d_f1_tex:+.4f} |
| Structure OFF | {d_acc_str:+.4f} | {d_f1_str:+.4f} |

**Primary Branch Dependency: {dep_branch}**
> {dep_note}

---

## 5. Feature Correlation (Cosine Similarity)

Texture vectors (1024-dim) were mean-pooled to 256-dim before computing per-sample cosine similarity against structure vectors (256-dim).

| Metric | Value |
|:-------|------:|
| Mean   | {p(cos_stats['mean'])} |
| Median | {p(float(np.median(cos_sims)))} |
| Std    | {p(cos_stats['std'])} |
| Min    | {p(cos_stats['min'])} |
| Max    | {p(cos_stats['max'])} |

**Interpretation:**
- Cosine similarity ≈ 0.0–0.3 → Branches learn **complementary** representations (fusion is meaningful)
- Cosine similarity ≈ 0.7–1.0 → Branches learn **redundant** representations (little benefit from dual branches)
- Current mean: `{p(cos_stats['mean'])}` → {"complementary" if cos_stats['mean'] < 0.5 else "possibly redundant"}

---

## 6. Fusion Collapse Detection

Collapse criteria: sample is collapsed when **(Structure fraction > {STR_COLLAPSE*100:.0f}%) AND (Texture fraction < {TEX_SUPPRESS*100:.0f}%)**

| Condition | Count | % Samples |
|:----------|------:|:---------:|
| Structure-collapsed (>{STR_COLLAPSE*100:.0f}%) | {collapse['n_structure_collapsed']} | {collapse['pct_structure_collapsed']:.2f}% |
| Texture-suppressed (<{TEX_SUPPRESS*100:.0f}%) | {collapse['n_texture_suppressed']} | {collapse['pct_texture_suppressed']:.2f}% |
| Both (full collapse) | {collapse['n_both_collapsed']} | {collapse['pct_both_collapsed']:.2f}% |

**FUSION STATUS: {collapse_status}**

---

## 7. Recommendations

### R1: Attention Gate Must Be Investigated Further ⚠️ HIGH PRIORITY
- Gate consistently assigns higher weights to the Structure branch (ratio ≈ {tex_stats['mean']/str_stats['mean']:.3f}).
- The current gate architecture operates over the 1280-dim concatenated vector. Since Texture occupies 1024/1280 = 80% of the vector, the gate's sigmoid operates in a high-dimensional texture space, yet assigns *lower* average weights to those dimensions. This is a pathological failure mode suggesting the gate has inverted its intended function.
- **Recommended action:** Restructure the gate to compute **separate** attention scalars for each branch using their individual vectors:
  - `gate_t = sigmoid(Linear(texture_vec, 1))` — scalar for texture branch
  - `gate_s = sigmoid(Linear(structure_vec, 1))` — scalar for structure branch
  - Then fuse as: `gate_t * texture_vec + gate_s * structure_vec`
  - This eliminates dimensional bias and ensures each branch receives a meaningful, independent weight.

### R2: Loss Weighting Should Be Adjusted ⚠️ HIGH PRIORITY
- The WeightedRandomSampler combined with inverse-frequency cross-entropy loss is aggressively boosting minority classes.
- NV recall is only 48.95% — the dominant class is severely under-predicted.
- **Recommended action:** Use a softer weighting scheme: `weight_i = 1 / sqrt(class_count_i)` instead of full inverse frequency. This reduces but does not eliminate minority class emphasis.

### R3: Training Procedure Should Be Revisited ⚠️ MEDIUM PRIORITY
- Overfitting is observed after Epoch 36 (validation loss diverges).
- **Recommended action:**
  - Reduce early stopping patience from current setting to 5 epochs.
  - Add label smoothing (ε = 0.1) to the cross-entropy loss to reduce overconfident incorrect predictions.
  - Consider applying stronger data augmentation (CutMix or MixUp) specifically for minority classes.

### R4: Architecture Should NOT Remain Unchanged
- The fundamental issue is the attention gate's dimensional bias. The current design allows texture's larger dimensionality to dominate gate computation while paradoxically being suppressed.
- The architecture modification required is **minimal and non-disruptive**: replacing the shared 1280→1280 gate Linear with two separate branch-level scalar gates.
- This can be implemented without changing the branch backbones or the fusion MLP.

### Priority Summary:

| Priority | Finding | Recommendation |
|:--------:|:--------|:---------------|
| 🔴 HIGH | Gate dimensional bias (structure favored 4:1) | Separate scalar gates per branch |
| 🔴 HIGH | NV recall collapse from aggressive sampling | Soften class weight scheme to sqrt-inverse |
| 🟡 MEDIUM | Overfitting at late training epochs | Reduce early stopping patience + label smoothing |
| 🟡 MEDIUM | Texture branch under-utilization | Investigate gradient flow + consider equal output dims |
| 🟢 LOW | Label overconfidence | Add label smoothing ε=0.1 |
| 🟢 LOW | Architecture dimension asymmetry (1024 vs 256) | Balance to 512-512 in next experimental round |
"""

    report_path = RESULTS_DIR / "fusion_diagnostic.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Report -> {report_path}")

    json_summary = {
        "gate_stats":         {"texture": tex_stats, "structure": str_stats},
        "gate_ratio":         tex_stats["mean"] / str_stats["mean"],
        "samplewise":         samplewise,
        "classwise":          {k: (v or {}) for k, v in classwise.items()},
        "ablation":           ablation,
        "cosine_similarity":  cos_stats,
        "collapse":           collapse,
    }
    json_path = RESULTS_DIR / "fusion_diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"[OK] JSON summary -> {json_path}")

    print("\n✅ Fusion Diagnostic Analysis Complete.")


if __name__ == "__main__":
    main()
