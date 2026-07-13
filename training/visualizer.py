"""
Result Visualizer — Phase 3
Generates all per-model output plots: ROC, PR, confusion matrix, learning curves.
"""

import os
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7

PALETTE = plt.cm.Set1(np.linspace(0, 1, NUM_CLASSES))


def save_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: Path):
    """Save a normalized + raw confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    for ax, (data, title) in zip(axes, [
        (cm, "Raw Counts"),
        (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1), "Row-Normalized")
    ]):
        sns.heatmap(
            data, annot=True,
            fmt=".0f" if data.dtype == int or data.max() > 1 else ".2f",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            linewidths=0.5,
            cbar=True,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    out = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK   ] Confusion matrix -> {out}")


def save_roc_curves(y_true: np.ndarray, y_probs: np.ndarray,
                    model_name: str, output_dir: Path):
    """Save per-class + macro-average ROC curves."""
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"ROC Curves — {model_name}", fontsize=13, fontweight="bold")

    all_fpr, all_tpr = [], []
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i], lw=1.8,
                label=f"{cls} (AUC={roc_auc:.3f})")
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    # Macro average
    all_fpr_cat = np.unique(np.concatenate(all_fpr))
    mean_tpr = np.zeros_like(all_fpr_cat)
    for fpr, tpr in zip(all_fpr, all_tpr):
        mean_tpr += np.interp(all_fpr_cat, fpr, tpr)
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr_cat, mean_tpr)
    ax.plot(all_fpr_cat, mean_tpr, "k--", lw=2.5,
            label=f"Macro avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "gray", linestyle=":", lw=1.2)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    out = output_dir / f"{model_name}_roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK   ] ROC curves -> {out}")


def save_pr_curves(y_true: np.ndarray, y_probs: np.ndarray,
                   model_name: str, output_dir: Path):
    """Save per-class Precision-Recall curves."""
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Precision-Recall Curves — {model_name}", fontsize=13, fontweight="bold")

    for i, cls in enumerate(CLASS_NAMES):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(rec, prec, color=PALETTE[i], lw=1.8,
                label=f"{cls} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    out = output_dir / f"{model_name}_pr_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK   ] PR curves -> {out}")


def save_learning_curves(history: dict, model_name: str, output_dir: Path):
    """
    Plot train/val loss and accuracy curves.
    history = {
        'train_loss': [...], 'val_loss': [...],
        'train_acc': [...],  'val_acc': [...],
        'train_f1': [...],   'val_f1': [...],
    }
    """
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Learning Curves — {model_name}", fontsize=13, fontweight="bold")

    pairs = [
        ("train_loss", "val_loss", "Loss", axes[0]),
        ("train_acc", "val_acc", "Accuracy", axes[1]),
        ("train_f1", "val_f1", "Macro F1", axes[2]),
    ]

    for train_key, val_key, ylabel, ax in pairs:
        if train_key in history:
            ax.plot(epochs, history[train_key], "b-o", markersize=3, label="Train")
        if val_key in history:
            ax.plot(epochs, history[val_key], "r-s", markersize=3, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

        # Mark best val
        if val_key in history:
            vals = history[val_key]
            best_idx = np.argmin(vals) if "loss" in val_key else np.argmax(vals)
            best_val = vals[best_idx]
            ax.axvline(x=epochs[best_idx], color="green", linestyle="--", alpha=0.5,
                       label=f"Best: {best_val:.4f}")

    plt.tight_layout()
    out = output_dir / f"{model_name}_learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK   ] Learning curves -> {out}")


def save_classification_report(clf_report: dict, model_name: str, output_dir: Path):
    """Save classification report as JSON."""
    out = output_dir / f"{model_name}_classification_report.json"
    with open(out, "w") as f:
        json.dump(clf_report, f, indent=2, default=str)
    print(f"[OK   ] Classification report -> {out}")


def generate_all_outputs(
    model_name: str,
    history: dict,
    test_metrics: dict,
    output_dir: Path,
    timing: dict = None,
):
    """
    Generate all outputs for a single model after training.

    Args:
        model_name: e.g. 'resnet50'
        history: dict of per-epoch metric lists
        test_metrics: dict from MetricTracker.compute() on test set
        output_dir: where to save all plots
        timing: optional dict with training_time, inference_time, peak_vram
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true  = test_metrics["y_true"]
    y_pred  = test_metrics["y_pred"]
    y_probs = test_metrics["y_probs"]
    cm      = test_metrics["confusion_matrix"]

    save_confusion_matrix(cm, model_name, output_dir)
    save_roc_curves(y_true, y_probs, model_name, output_dir)
    save_pr_curves(y_true, y_probs, model_name, output_dir)
    save_learning_curves(history, model_name, output_dir)
    save_classification_report(test_metrics["clf_report"], model_name, output_dir)

    # Save timing summary
    if timing:
        timing_path = output_dir / f"{model_name}_timing.json"
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)
        print(f"[OK   ] Timing info -> {timing_path}")

    print(f"\n[DONE ] All outputs for {model_name} saved to {output_dir}")
