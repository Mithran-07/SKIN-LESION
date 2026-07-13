"""
Baseline Training Script — Phase 3
Trains ResNet50, DenseNet121, and EfficientNet-B4 sequentially.
Generates benchmark.csv with all metrics.
"""

import sys
import time
import csv
import yaml
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.splitter  import load_splits
from data.dataloader import get_all_dataloaders, compute_class_weights
from models.baseline import get_model, count_parameters
from training.trainer    import BaselineTrainer
from training.visualizer import generate_all_outputs

# ── Directories
DATASETS_ROOT   = PROJECT_ROOT / "datasets"
HAM10000_DIR    = DATASETS_ROOT / "HAM10000"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"
TENSORBOARD_DIR = PROJECT_ROOT / "tensorboard"
LOGS_DIR        = PROJECT_ROOT / "logs"
SPLITS_DIR      = PROJECT_ROOT / "splits"

BENCHMARK_CSV = RESULTS_DIR / "benchmark.csv"
CONFIG_PATH   = PROJECT_ROOT / "configs" / "baseline_config.yaml"

# ── Benchmark CSV columns
BENCHMARK_COLS = [
    "Model", "Parameters", "Training_Time_s",
    "Val_Accuracy", "Test_Accuracy",
    "Val_Macro_F1", "Test_Macro_F1",
    "Val_Macro_AUC", "Test_Macro_AUC",
    "Val_Balanced_Accuracy", "Test_Balanced_Accuracy",
    "Inference_Time_ms_per_img", "Peak_VRAM_MB",
    "Best_Epoch", "Checkpoint_Path",
]


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        # Fallback defaults
        cfg = {
            "training": {
                "epochs": 50, "lr": 1e-4, "weight_decay": 1e-2,
                "grad_clip": 1.0, "early_stop_patience": 10, "batch_size": 32,
            },
            "data": {"num_workers": 4, "image_size": 224, "num_classes": 7},
            "models": [
                {"name": "resnet50", "pretrained": True, "batch_size": 32},
                {"name": "densenet121", "pretrained": True, "batch_size": 32},
                {"name": "efficientnet_b4", "pretrained": True, "batch_size": 16},
            ],
        }

    # ── Force correct types (YAML may parse 1e-4 as string in some versions)
    t = cfg.get("training", {})
    t["lr"]                  = float(t.get("lr", 1e-4))
    t["weight_decay"]        = float(t.get("weight_decay", 1e-2))
    t["grad_clip"]           = float(t.get("grad_clip", 1.0))
    t["epochs"]              = int(t.get("epochs", 50))
    t["batch_size"]          = int(t.get("batch_size", 32))
    t["early_stop_patience"] = int(t.get("early_stop_patience", 10))
    t["eta_min"]             = float(t.get("eta_min", 1e-6))

    d = cfg.get("data", {})
    d["num_classes"]   = int(d.get("num_classes", 7))
    d["num_workers"]   = int(d.get("num_workers", 4))
    d["image_size"]    = int(d.get("image_size", 224))

    return cfg


def init_benchmark_csv():
    """Create or append to benchmark.csv."""
    if not BENCHMARK_CSV.exists():
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(BENCHMARK_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BENCHMARK_COLS)
            writer.writeheader()
        print(f"[OK   ] Benchmark CSV created: {BENCHMARK_CSV}")


def append_benchmark_row(row: dict):
    """Append one model's results to benchmark.csv."""
    with open(BENCHMARK_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_COLS)
        writer.writerow(row)
    print(f"[OK   ] Benchmark row saved for {row['Model']}")


def try_reduce_batch(original_batch: int) -> int:
    """Return half batch size for OOM-prone models."""
    return max(8, original_batch // 2)


def train_one_model(
    model_cfg:  dict,
    train_df,
    val_df,
    test_df,
    global_cfg: dict,
    class_weights: torch.Tensor,
) -> dict:
    """
    Full train + evaluate cycle for one model.
    Returns a benchmark dict row.
    """
    model_name  = model_cfg["name"]
    batch_size  = model_cfg.get("batch_size", global_cfg["training"]["batch_size"])
    n_workers   = global_cfg["data"]["num_workers"]
    img_size    = global_cfg["data"].get("image_size", 224)
    num_classes = global_cfg["data"]["num_classes"]
    epochs      = global_cfg["training"]["epochs"]

    print(f"\n{'='*65}")
    print(f"  TRAINING: {model_name.upper()}")
    print(f"  Batch: {batch_size} | Workers: {n_workers} | Epochs: {epochs}")
    print(f"{'='*65}")

    # ── Build model
    model = get_model(model_name, num_classes=num_classes, pretrained=model_cfg.get("pretrained", True))
    param_info = count_parameters(model)

    # ── DataLoaders
    train_loader, val_loader, test_loader = get_all_dataloaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        num_workers=n_workers,
        image_size=img_size,
        images_root=HAM10000_DIR,
    )

    # ── Trainer
    trainer = BaselineTrainer(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        lr=global_cfg["training"]["lr"],
        weight_decay=global_cfg["training"]["weight_decay"],
        grad_clip=global_cfg["training"]["grad_clip"],
        early_stop_patience=global_cfg["training"]["early_stop_patience"],
        class_weights=class_weights,
        checkpoints_dir=CHECKPOINTS_DIR,
        results_dir=RESULTS_DIR,
        tensorboard_dir=TENSORBOARD_DIR,
        logs_dir=LOGS_DIR,
    )

    # ── Train
    best_val_metrics, history, training_time = trainer.train()

    # ── Test evaluation
    test_metrics = trainer.evaluate(test_loader)

    # ── Visualizations
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

    # ── Best epoch
    val_bal_acc_hist = history.get("val_bal_acc", [])
    best_epoch = int(max(range(len(val_bal_acc_hist)),
                         key=lambda i: val_bal_acc_hist[i])) + 1 if val_bal_acc_hist else 0

    # ── Checkpoint path
    best_ckpt = CHECKPOINTS_DIR / model_name / "best_checkpoint.pth"

    # ── Build benchmark row
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

    return row


def main():
    print("\n" + "="*65)
    print("  PHASE 3 — BASELINE MODEL TRAINING")
    print("="*65)

    # ── Config
    cfg = load_config()
    print(f"[OK   ] Config loaded from {CONFIG_PATH}")

    # ── Load splits
    print("\n[INFO ] Loading data splits...")
    train_df, val_df, test_df = load_splits()
    print(f"[OK   ] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Class weights (from train set)
    class_weights = compute_class_weights(
        train_df["label"].tolist(),
        num_classes=cfg["data"]["num_classes"]
    )
    print(f"[OK   ] Class weights computed")

    # ── Initialize benchmark CSV
    init_benchmark_csv()

    # ── Train all models
    all_results = []
    models_cfg  = cfg.get("models", [])

    for model_cfg in models_cfg:
        try:
            row = train_one_model(
                model_cfg=model_cfg,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                global_cfg=cfg,
                class_weights=class_weights,
            )
            append_benchmark_row(row)
            all_results.append(row)

            # Free GPU memory between models
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"\n[OOM  ] Out of memory for {model_cfg['name']} "
                  f"at batch {model_cfg.get('batch_size')}. Reducing batch size and retrying...")
            torch.cuda.empty_cache()
            model_cfg["batch_size"] = try_reduce_batch(model_cfg.get("batch_size", 32))
            try:
                row = train_one_model(
                    model_cfg=model_cfg,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    global_cfg=cfg,
                    class_weights=class_weights,
                )
                append_benchmark_row(row)
                all_results.append(row)
                torch.cuda.empty_cache()
            except Exception as e2:
                print(f"[ERROR] {model_cfg['name']} failed again: {e2}")

        except Exception as e:
            print(f"[ERROR] {model_cfg['name']} failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Final summary
    print("\n" + "="*65)
    print("  BENCHMARK RESULTS")
    print("="*65)
    for row in all_results:
        print(f"\n  {row['Model'].upper()}")
        print(f"    Params         : {row['Parameters']:,}")
        print(f"    Test Accuracy  : {row['Test_Accuracy']:.4f}")
        print(f"    Test Bal. Acc  : {row['Test_Balanced_Accuracy']:.4f}")
        print(f"    Test Macro F1  : {row['Test_Macro_F1']:.4f}")
        print(f"    Test Macro AUC : {row['Test_Macro_AUC']:.4f}")
        print(f"    Training Time  : {row['Training_Time_s']:.0f}s")
        print(f"    Peak VRAM      : {row['Peak_VRAM_MB']:.0f} MB")

    print(f"\n[DONE ] Benchmark CSV: {BENCHMARK_CSV}")
    print(f"[DONE ] Results dir  : {RESULTS_DIR}")
    print("="*65)


if __name__ == "__main__":
    main()
