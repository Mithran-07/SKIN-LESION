"""
Run Dual-Branch CNN — Phase 4
Trains the Dual-Branch CNN under exact baseline conditions.
"""

import sys
import time
import csv
import yaml
import torch
import random
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.splitter  import load_splits
from data.dataloader import get_all_dataloaders, compute_class_weights
from models.dual_branch_net import build_model_from_config
from training.trainer    import BaselineTrainer
from training.visualizer import generate_all_outputs

# ── Directories
DATASETS_ROOT   = PROJECT_ROOT / "datasets"
HAM10000_DIR    = DATASETS_ROOT / "HAM10000"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"
TENSORBOARD_DIR = PROJECT_ROOT / "tensorboard"
LOGS_DIR        = PROJECT_ROOT / "logs"

BENCHMARK_CSV = RESULTS_DIR / "benchmark.csv"
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline_config.yaml"
REPO_CONFIG = PROJECT_ROOT / "repo_temp" / "SKIN-LESION-main" / "config" / "config.yaml"

BENCHMARK_COLS = [
    "Model", "Parameters", "Training_Time_s",
    "Val_Accuracy", "Test_Accuracy",
    "Val_Macro_F1", "Test_Macro_F1",
    "Val_Macro_AUC", "Test_Macro_AUC",
    "Val_Balanced_Accuracy", "Test_Balanced_Accuracy",
    "Inference_Time_ms_per_img", "Peak_VRAM_MB",
    "Best_Epoch", "Checkpoint_Path",
]


def append_benchmark_row(row: dict):
    with open(BENCHMARK_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_COLS)
        writer.writerow(row)
    print(f"[OK   ] Benchmark row saved for {row['Model']}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_configs():
    with open(BASELINE_CONFIG, "r") as f:
        base_cfg = yaml.safe_load(f)
    
    with open(REPO_CONFIG, "r") as f:
        repo_cfg = yaml.safe_load(f)

    # Force correct types for base_cfg
    t = base_cfg.get("training", {})
    t["lr"]                  = float(t.get("lr", 1e-4))
    t["weight_decay"]        = float(t.get("weight_decay", 1e-2))
    t["grad_clip"]           = float(t.get("grad_clip", 1.0))
    t["epochs"]              = int(t.get("epochs", 50))
    t["batch_size"]          = 8 # Force 8 to avoid silent swapping on RTX 3050
    t["early_stop_patience"] = int(t.get("early_stop_patience", 10))

    return base_cfg, repo_cfg


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def main():
    print("\n" + "="*65)
    print("  PHASE 4 — DUAL-BRANCH CNN TRAINING")
    print("="*65)

    base_cfg, repo_cfg = load_configs()
    seed = sys.argv[1] if len(sys.argv) > 1 else str(base_cfg["project"].get("seed", 42))
    seed = int(seed)
    set_seed(seed)
    
    model_name = f"dual_branch_seed{seed}"
    batch_size = base_cfg["training"]["batch_size"]
    n_workers = 0 # 0 workers + pin_memory=False is the only safe combo on this machine
    img_size = base_cfg["data"].get("image_size", 224)
    num_classes = base_cfg["data"]["num_classes"]
    epochs = base_cfg["training"]["epochs"]

    print(f"\n[INFO ] Seed: {seed} | Batch: {batch_size} | Workers: {n_workers} | Epochs: {epochs}")

    # Load splits
    train_df, val_df, test_df = load_splits()
    
    class_weights = compute_class_weights(
        train_df["label"].tolist(),
        num_classes=num_classes
    )

    # Build model using the dual-branch config from the repo
    model = build_model_from_config(repo_cfg)
    param_info = count_parameters(model)
    print(f"[INFO ] Dual-Branch Model built. Params: {param_info['total']:,}")

    # DataLoaders
    train_loader, val_loader, test_loader = get_all_dataloaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        num_workers=n_workers,
        image_size=img_size,
        images_root=HAM10000_DIR,
        pin_memory=False,
    )

    # Trainer (Using exact baseline conditions)
    trainer = BaselineTrainer(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        lr=base_cfg["training"]["lr"],
        weight_decay=base_cfg["training"]["weight_decay"],
        grad_clip=base_cfg["training"]["grad_clip"],
        early_stop_patience=5,
        class_weights=class_weights,
        label_smoothing=0.1,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        tensorboard_dir=TENSORBOARD_DIR,
        logs_dir=LOGS_DIR,
    )

    # Train
    best_val_metrics, history, training_time = trainer.train()

    # Evaluate
    test_metrics = trainer.evaluate(test_loader)

    # Visualizations
    timing = {
        "training_time_s":       training_time,
        "inference_per_img_ms":  test_metrics.get("inference_per_img_ms", 0),
        "peak_vram_mb":          test_metrics.get("peak_vram_mb", 0),
    }
    generate_all_outputs(
        model_name=model_name,
        history=history,
        test_metrics=test_metrics,
        output_dir=RESULTS_DIR / model_name,
        timing=timing,
    )

    val_bal_acc_hist = history.get("val_bal_acc", [])
    best_epoch = int(max(range(len(val_bal_acc_hist)), key=lambda i: val_bal_acc_hist[i])) + 1 if val_bal_acc_hist else 0
    best_ckpt = CHECKPOINTS_DIR / model_name / "best_checkpoint.pth"

    row = {
        "Model":                   model_name,
        "Parameters":              param_info["total"],
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

    append_benchmark_row(row)
    print("\n" + "="*65)
    print(f"  FINISHED {model_name}")
    print("="*65)


if __name__ == "__main__":
    main()
