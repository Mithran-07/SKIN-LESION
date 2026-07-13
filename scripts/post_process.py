import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.splitter  import load_splits
from data.dataloader import get_all_dataloaders
from models.dual_branch_net import build_model_from_config
from explainability.gradcam import DualBranchGradCAM
from explainability.visualize import plot_dual_cam
from uncertainty.conformal_prediction import SplitConformalPredictor

# Directories
RESULTS_DIR = PROJECT_ROOT / "results"
DB_RESULTS_DIR = RESULTS_DIR / "dual_branch"
DB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR = DB_RESULTS_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

def main():
    print("\n" + "="*65)
    print("  PHASE 4 — EVALUATION & POST-PROCESSING")
    print("="*65)

    # 1. Load config and datasets
    with open(PROJECT_ROOT / "configs" / "baseline_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "repo_temp" / "SKIN-LESION-main" / "config" / "config.yaml", "r") as f:
        repo_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO ] Using device: {device}")

    train_df, val_df, test_df = load_splits()
    train_loader, val_loader, test_loader = get_all_dataloaders(
        train_df, val_df, test_df,
        batch_size=8,
        num_workers=0,
        image_size=224,
        images_root=PROJECT_ROOT / "datasets" / "HAM10000",
        pin_memory=False,
    )

    # Load best checkpoint of Seed 42
    model = build_model_from_config(repo_cfg)
    ckpt_path = PROJECT_ROOT / "checkpoints" / "dual_branch_seed42" / "best_checkpoint.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[INFO ] Loaded best model from {ckpt_path}")

    # 2. Collect test predictions
    all_probs = []
    all_targets = []
    all_images = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            logits, *_ = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())
            # Save first few batches of images for Grad-CAM
            if batch_idx < 10:
                all_images.append(images.cpu())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds = probs.argmax(axis=1)

    # 3. Standard Evaluation Metrics
    report_dict = classification_report(targets, preds, target_names=CLASS_NAMES, output_dict=True)
    report_txt = classification_report(targets, preds, target_names=CLASS_NAMES)
    
    # Save classification report txt
    with open(DB_RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report_txt)
    print("[OK   ] Classification report saved.")

    # Calculate ROC-AUC and PR-AUC
    roc_aucs = {}
    pr_aucs = {}
    for i, name in enumerate(CLASS_NAMES):
        one_hot_targets = (targets == i).astype(int)
        if one_hot_targets.sum() > 0:
            fpr, tpr, _ = roc_curve(one_hot_targets, probs[:, i])
            roc_aucs[name] = auc(fpr, tpr)
            precision_val, recall_val, _ = precision_recall_curve(one_hot_targets, probs[:, i])
            pr_aucs[name] = auc(recall_val, precision_val)
        else:
            roc_aucs[name] = 0.0
            pr_aucs[name] = 0.0

    # Macro averages
    macro_roc = np.mean(list(roc_aucs.values()))
    macro_pr = np.mean(list(pr_aucs.values()))

    # Build metrics dict
    metrics_summary = {
        "accuracy": report_dict["accuracy"],
        "macro_avg": report_dict["macro avg"],
        "class_wise_precision": {name: report_dict[name]["precision"] for name in CLASS_NAMES},
        "class_wise_recall": {name: report_dict[name]["recall"] for name in CLASS_NAMES},
        "class_wise_f1": {name: report_dict[name]["f1-score"] for name in CLASS_NAMES},
        "class_wise_roc_auc": roc_aucs,
        "class_wise_pr_auc": pr_aucs,
        "macro_roc_auc": macro_roc,
        "macro_pr_auc": macro_pr,
    }

    with open(DB_RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
    
    # Write to metrics.csv
    pd.DataFrame([{
        "Accuracy": report_dict["accuracy"],
        "Macro_F1": report_dict["macro avg"]["f1-score"],
        "Macro_ROC_AUC": macro_roc,
        "Macro_PR_AUC": macro_pr,
    }]).to_csv(DB_RESULTS_DIR / "metrics.csv", index=False)
    print("[OK   ] Metrics JSON and CSV saved.")

    # 4. Confusion Matrices
    cm = confusion_matrix(targets, preds)
    cm_norm = confusion_matrix(targets, preds, normalize="true")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.savefig(DB_RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Normalized Confusion Matrix")
    plt.savefig(DB_RESULTS_DIR / "confusion_matrix_normalized.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK   ] Confusion matrices saved.")

    # 5. Error & Misclassification Analysis
    confidences = probs[np.arange(len(targets)), preds]
    correct_mask = (preds == targets)

    # Find top confused pairs
    confusions = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true_class": CLASS_NAMES[i],
                    "pred_class": CLASS_NAMES[j],
                    "count": int(cm[i, j])
                })
    confusions = sorted(confusions, key=lambda x: x["count"], reverse=True)

    # Worst predictions (incorrect with highest confidence)
    incorrect_indices = np.where(~correct_mask)[0]
    worst_indices = incorrect_indices[np.argsort(confidences[incorrect_indices])[-10:]]
    worst_predictions = []
    for idx in worst_indices:
        worst_predictions.append({
            "index": int(idx),
            "true_class": CLASS_NAMES[targets[idx]],
            "pred_class": CLASS_NAMES[preds[idx]],
            "confidence": float(confidences[idx])
        })

    error_analysis_data = {
        "top_confused_pairs": confusions[:5],
        "worst_predictions": worst_predictions,
    }
    with open(DB_RESULTS_DIR / "error_analysis.json", "w") as f:
        json.dump(error_analysis_data, f, indent=4)
    print("[OK   ] Error analysis saved.")

    # Confidence Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(confidences[correct_mask], bins=20, alpha=0.5, label="Correct", color="green")
    plt.hist(confidences[~correct_mask], bins=20, alpha=0.5, label="Incorrect", color="red")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Confidence Distribution")
    plt.savefig(DB_RESULTS_DIR / "confidence_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC / PR Curves
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(CLASS_NAMES):
        one_hot_targets = (targets == i).astype(int)
        if one_hot_targets.sum() > 0:
            fpr, tpr, _ = roc_curve(one_hot_targets, probs[:, i])
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Class")
    plt.legend()
    plt.savefig(DB_RESULTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(CLASS_NAMES):
        one_hot_targets = (targets == i).astype(int)
        if one_hot_targets.sum() > 0:
            precision_val, recall_val, _ = precision_recall_curve(one_hot_targets, probs[:, i])
            plt.plot(recall_val, precision_val, label=f"{name} (AUC={auc(recall_val, precision_val):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves per Class")
    plt.legend()
    plt.savefig(DB_RESULTS_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Conformal Prediction Evaluation
    print("[INFO ] Calibrating and evaluating Conformal Prediction...")
    cp = SplitConformalPredictor(alpha=0.1, class_names=CLASS_NAMES)
    
    # We calibrate using a 20% validation sub-sampler or the val_loader directly
    cp.calibrate(model, val_loader, device)
    conformal_metrics = cp.coverage_report(model, test_loader, device)
    
    with open(DB_RESULTS_DIR / "conformal_results.json", "w") as f:
        json.dump(conformal_metrics, f, indent=4)
    print("[OK   ] Conformal prediction metrics saved.")

    # 7. Grad-CAM Representative Examples
    print("[INFO ] Generating Grad-CAM visualizations...")
    gradcam = DualBranchGradCAM(
        model=model,
        shallow_target_layer=model.shallow_branch.block3.conv,
        deep_target_layer=model.deep_branch.stage4.conv_layers[-2]
    )

    flat_images = torch.cat(all_images, dim=0)
    
    # Generate examples for correct, incorrect, high-conf, low-conf
    # Correct + High Confidence
    correct_high = np.where(correct_mask & (confidences > 0.85))[0]
    # Correct + Low Confidence
    correct_low = np.where(correct_mask & (confidences < 0.45))[0]
    # Incorrect + High Confidence
    incorrect_high = np.where(~correct_mask & (confidences > 0.70))[0]
    # Incorrect + Low Confidence
    incorrect_low = np.where(~correct_mask & (confidences < 0.45))[0]

    examples = {
        "correct_high": correct_high,
        "correct_low": correct_low,
        "incorrect_high": incorrect_high,
        "incorrect_low": incorrect_low,
    }

    for key, indices in examples.items():
        if len(indices) > 0:
            idx = int(indices[0])
            img_tensor, _ = test_loader.dataset[idx]
            img_tensor = img_tensor.unsqueeze(0)
            
            with torch.no_grad():
                logits, _, _ = model(img_tensor.to(device))
                probs_val = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Re-generate with gradients enabled
            t_cam, s_cam, pred_cls = gradcam.generate(img_tensor, device=device)
            
            plot_dual_cam(
                image=img_tensor,
                texture_cam=t_cam,
                structure_cam=s_cam,
                probabilities=probs_val,
                pred_class=pred_cls,
                class_names=CLASS_NAMES,
                save_path=str(GRADCAM_DIR / f"{key}_example.png"),
                title=f"Grad-CAM Example: {key.replace('_', ' ').title()} Prediction"
            )

    gradcam.remove_hooks()
    print("✅ Post-processing phase completed.")

if __name__ == "__main__":
    main()
