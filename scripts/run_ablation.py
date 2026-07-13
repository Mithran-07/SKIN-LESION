import os
import sys
import csv
import time
import yaml
import torch
import random
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.splitter  import load_splits
from data.dataloader import get_all_dataloaders, compute_class_weights
from models.dual_branch_net import build_model_from_config, DualBranchNet
from training.trainer    import BaselineTrainer
from training.visualizer import generate_all_outputs
from losses.focal_loss import FocalLoss

# ── Custom architectures for ablation
class TextureBranchOnly(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.shallow_branch = full_model.shallow_branch
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(full_model.shallow_branch.output_dim, full_model.num_classes)
        )
        self.num_classes = full_model.num_classes

    def forward(self, x):
        vec, fmap = self.shallow_branch(x)
        logits = self.classifier(vec)
        return logits, fmap, None

class StructureBranchOnly(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.deep_branch = full_model.deep_branch
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(full_model.deep_branch.output_dim, full_model.num_classes)
        )
        self.num_classes = full_model.num_classes

    def forward(self, x):
        vec, fmap = self.deep_branch(x)
        logits = self.classifier(vec)
        return logits, None, fmap

class NoAttentionFusionHead(torch.nn.Module):
    def __init__(self, texture_dim=1024, structure_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()
        concat_dim = texture_dim + structure_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(concat_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.3),
        )

    def forward(self, texture_vec, structure_vec):
        combined = torch.cat([texture_vec, structure_vec], dim=1)
        return self.mlp(combined)

class DualBranchNoAttention(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.shallow_branch = full_model.shallow_branch
        self.deep_branch = full_model.deep_branch
        self.fusion = NoAttentionFusionHead(
            texture_dim=self.shallow_branch.output_dim,
            structure_dim=self.deep_branch.output_dim,
            hidden_dim=512,
            output_dim=256
        )
        self.classifier = full_model.classifier
        self.num_classes = full_model.num_classes

    def forward(self, x):
        texture_vec, texture_fmap = self.shallow_branch(x)
        structure_vec, structure_fmap = self.deep_branch(x)
        fused = self.fusion(texture_vec, structure_vec)
        logits = self.classifier(fused)
        return logits, texture_fmap, structure_fmap


class AblationTrainer(BaselineTrainer):
    def __init__(self, *args, use_focal=True, **kwargs):
        super().__init__(*args, **kwargs)
        if use_focal:
            alpha = self.criterion.weight
            self.criterion = FocalLoss(
                alpha=alpha,
                gamma=2.0,
                label_smoothing=0.1
            )
            self.logger.info(f"Initialized FocalLoss for ablation training.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    with open(PROJECT_ROOT / "configs" / "baseline_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "repo_temp" / "SKIN-LESION-main" / "config" / "config.yaml", "r") as f:
        repo_cfg = yaml.safe_load(f)

    # Force sequential training configs
    batch_size = 8
    n_workers = 0
    img_size = base_cfg["data"].get("image_size", 224)
    num_classes = base_cfg["data"]["num_classes"]
    epochs = base_cfg["training"]["epochs"]
    seed = 42
    set_seed(seed)

    train_df, val_df, test_df = load_splits()
    class_weights = compute_class_weights(train_df["label"].tolist(), num_classes=num_classes)

    train_loader, val_loader, test_loader = get_all_dataloaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        num_workers=n_workers,
        image_size=img_size,
        images_root=PROJECT_ROOT / "datasets" / "HAM10000",
        pin_memory=False,
    )

    experiments = [
        ("ablation_exp_a", lambda m: m, True),  # Full with Focal
        ("ablation_exp_b", lambda m: TextureBranchOnly(m), True),
        ("ablation_exp_c", lambda m: StructureBranchOnly(m), True),
        ("ablation_exp_d", lambda m: DualBranchNoAttention(m), True),
    ]

    for name, model_fn, use_focal in experiments:
        print(f"\n==================================================")
        print(f"  RUNNING ABLATION: {name}")
        print(f"==================================================")

        full_model = build_model_from_config(repo_cfg)
        model = model_fn(full_model)

        trainer = AblationTrainer(
            model=model,
            model_name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            epochs=epochs,
            lr=float(base_cfg["training"]["lr"]),
            weight_decay=float(base_cfg["training"]["weight_decay"]),
            grad_clip=float(base_cfg["training"]["grad_clip"]),
            early_stop_patience=int(base_cfg["training"]["early_stop_patience"]),
            class_weights=class_weights,
            checkpoints_dir=PROJECT_ROOT / "checkpoints",
            results_dir=PROJECT_ROOT / "results",
            tensorboard_dir=PROJECT_ROOT / "tensorboard",
            logs_dir=PROJECT_ROOT / "logs",
            use_focal=use_focal,
        )

        best_val_metrics, history, training_time = trainer.train()
        test_metrics = trainer.evaluate(test_loader)

        timing = {
            "training_time_s":       training_time,
            "inference_per_img_ms":  test_metrics.get("inference_per_img_ms", 0),
            "peak_vram_mb":          test_metrics.get("peak_vram_mb", 0),
        }
        generate_all_outputs(
            model_name=name,
            history=history,
            test_metrics=test_metrics,
            output_dir=PROJECT_ROOT / "results" / name,
            timing=timing,
        )

        val_bal_acc_hist = history.get("val_bal_acc", [])
        best_epoch = int(max(range(len(val_bal_acc_hist)), key=lambda i: val_bal_acc_hist[i])) + 1 if val_bal_acc_hist else 0
        best_ckpt = PROJECT_ROOT / "checkpoints" / name / "best_checkpoint.pth"

        row = {
            "Model":                   name,
            "Parameters":              sum(p.numel() for p in model.parameters() if p.requires_grad),
            "Training_Time_s":         round(training_time, 1),
            "Val_Accuracy":            round(best_val_metrics.get("accuracy", 0), 4),
            "Test_Accuracy":           round(test_metrics.get("accuracy", 0), 4),
            "Val_Macro_F1":            round(best_val_metrics.get("f1_macro", 0), 4),
            "Test_Macro_F1":           round(test_metrics.get("f1_macro", 0), 4),
            "Val_Macro_AUC":           round(best_val_metrics.get("roc_auc", 0), 4),
            "Test_Macro_AUC":          round(test_metrics.get("roc_auc", 0), 4),
            "Val_Balanced_Accuracy":   round(best_val_metrics.get("balanced_accuracy", 0), 4),
            "Test_Balanced_Accuracy":  round(test_metrics.get("balanced_accuracy", 0), 4),
            "Inference_Time_ms_per_img": round(test_metrics.get("inference_per_img_ms", 0), 3),
            "Peak_VRAM_MB":            round(test_metrics.get("peak_vram_mb", 0), 1),
            "Best_Epoch":              best_epoch,
            "Checkpoint_Path":         str(best_ckpt),
        }

        with open(PROJECT_ROOT / "results" / "benchmark.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        print(f"[OK   ] Benchmark row saved for {name}")

        # Free GPU memory
        torch.cuda.empty_cache()

    # ── Experiment E: Without Focal Loss (copy from dual_branch_seed42)
    print("\nCopying dual_branch_seed42 as ablation_exp_e...")
    # Find dual_branch_seed42 row in benchmark.csv and duplicate it as ablation_exp_e
    rows = []
    with open(PROJECT_ROOT / "results" / "benchmark.csv", "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            if r["Model"] == "dual_branch_seed42":
                exp_e_row = r.copy()
                exp_e_row["Model"] = "ablation_exp_e"
                # Update checkpoint path reference
                exp_e_row["Checkpoint_Path"] = str(PROJECT_ROOT / "checkpoints" / "ablation_exp_e" / "best_checkpoint.pth")
                rows.append(exp_e_row)
                
                # Copy checkpoint folder
                import shutil
                src_ckpt = PROJECT_ROOT / "checkpoints" / "dual_branch_seed42"
                dst_ckpt = PROJECT_ROOT / "checkpoints" / "ablation_exp_e"
                if src_ckpt.exists() and not dst_ckpt.exists():
                    shutil.copytree(src_ckpt, dst_ckpt)
                
                # Copy results folder
                src_res = PROJECT_ROOT / "results" / "dual_branch_seed42"
                dst_res = PROJECT_ROOT / "results" / "ablation_exp_e"
                if src_res.exists() and not dst_res.exists():
                    shutil.copytree(src_res, dst_res)

    with open(PROJECT_ROOT / "results" / "benchmark.csv", "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print("✅ Ablation study complete.")

if __name__ == "__main__":
    main()
