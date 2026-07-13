"""
Resume EfficientNet-B4 Training from checkpoint.
Disables persistent_workers and pin_memory to avoid CUDA host allocator crash on long runs.
"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.splitter    import load_splits
from data.dataloader  import get_all_dataloaders, compute_class_weights
from models.baseline  import get_model
from training.trainer    import BaselineTrainer
from training.visualizer import generate_all_outputs

DATASETS_ROOT   = PROJECT_ROOT / "datasets"
HAM10000_DIR    = DATASETS_ROOT / "HAM10000"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"
TENSORBOARD_DIR = PROJECT_ROOT / "tensorboard"
LOGS_DIR        = PROJECT_ROOT / "logs"

MODEL_NAME  = "efficientnet_b4"
BATCH_SIZE  = 16
NUM_WORKERS = 0          # 0 workers avoids Windows persistent_workers memory leak
PIN_MEMORY  = False      # Disable pinned memory to avoid CUDA host allocator crash
NUM_CLASSES = 7
EPOCHS      = 50
LR          = 1e-4
WEIGHT_DECAY= 1e-2


def main():
    print("\n" + "="*65)
    print(f"  RESUMING: {MODEL_NAME.upper()} (from last checkpoint)")
    print("="*65)

    # ── Load splits
    train_df, val_df, test_df = load_splits()
    print(f"[OK   ] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Class weights
    class_weights = compute_class_weights(train_df["label"].tolist(), NUM_CLASSES)

    # ── DataLoaders (stable settings for long runs)
    train_loader, val_loader, test_loader = get_all_dataloaders(
        train_df, val_df, test_df,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        images_root=HAM10000_DIR,
    )
    print(f"[OK   ] Loaders built | num_workers={NUM_WORKERS} | pin_memory={PIN_MEMORY}")

    # ── Model
    model = get_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True)

    # ── Trainer (auto-resume reads latest_checkpoint.pth)
    trainer = BaselineTrainer(
        model=model,
        model_name=MODEL_NAME,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=NUM_CLASSES,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=1.0,
        early_stop_patience=10,
        class_weights=class_weights,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        tensorboard_dir=TENSORBOARD_DIR,
        logs_dir=LOGS_DIR,
    )

    # ── Train (auto-resumes from epoch 28)
    best_val_metrics, history, training_time = trainer.train()

    # ── Test evaluation
    test_metrics = trainer.evaluate(test_loader)

    # ── Visualizations
    timing = {
        "training_time_s":      training_time,
        "inference_per_img_ms": test_metrics.get("inference_per_img_ms", 0),
        "peak_vram_mb":         test_metrics.get("peak_vram_mb", 0),
    }
    generate_all_outputs(
        model_name=MODEL_NAME,
        history=history,
        test_metrics=test_metrics,
        output_dir=RESULTS_DIR / MODEL_NAME,
        timing=timing,
    )

    # ── Append to benchmark.csv
    import csv
    BENCHMARK_CSV = RESULTS_DIR / "benchmark.csv"
    BENCHMARK_COLS = [
        "Model", "Parameters", "Training_Time_s",
        "Val_Accuracy", "Test_Accuracy",
        "Val_Macro_F1", "Test_Macro_F1",
        "Val_Macro_AUC", "Test_Macro_AUC",
        "Val_Balanced_Accuracy", "Test_Balanced_Accuracy",
        "Inference_Time_ms_per_img", "Peak_VRAM_MB",
        "Best_Epoch", "Checkpoint_Path",
    ]

    from models.baseline import count_parameters
    param_info = count_parameters(model)
    val_bal_acc_hist = history.get("val_bal_acc", [])
    best_epoch = int(max(range(len(val_bal_acc_hist)),
                         key=lambda i: val_bal_acc_hist[i])) + 1 if val_bal_acc_hist else 28

    row = {
        "Model":                   MODEL_NAME,
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
        "Checkpoint_Path":         str(CHECKPOINTS_DIR / MODEL_NAME / "best_checkpoint.pth"),
    }

    with open(BENCHMARK_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_COLS)
        writer.writerow(row)

    print(f"\n[DONE ] EfficientNet-B4 complete!")
    print(f"  Test Balanced Acc  : {row['Test_Balanced_Accuracy']:.4f}")
    print(f"  Test Macro F1      : {row['Test_Macro_F1']:.4f}")
    print(f"  Test Macro AUC     : {row['Test_Macro_AUC']:.4f}")
    print(f"  Benchmark CSV      : {BENCHMARK_CSV}")


if __name__ == "__main__":
    main()
