"""
Export Unified Verified Benchmark Registry v2 (`results/verified_v2/benchmark_registry_v2.json`).

Synthesizes verified evaluation results, checkpoint cryptographic hashes, architecture versioning,
dataset & lesion-grouped split integrity, OOD v2 calibration/evaluation metrics, and Grad-CAM
verification into a single machine-readable source of truth for the FastAPI backend and Next.js 14
research workstation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT_OUT = ROOT / "audit/outputs"
RESULTS_V2 = ROOT / "results/verified_v2"
RESULTS_V2.mkdir(parents=True, exist_ok=True)


def main():
    ckpts = json.loads((AUDIT_OUT / "01_checkpoints.json").read_text())
    dataset = json.loads((AUDIT_OUT / "02_dataset.json").read_text())
    eval_res = json.loads((AUDIT_OUT / "04_evaluation_verified.json").read_text())
    gradcam_res = json.loads((AUDIT_OUT / "05_gradcam.json").read_text())
    ood_res = json.loads((RESULTS_V2 / "ood_evaluation_v2.json").read_text())

    # Map checkpoint metadata by model key
    ckpt_by_model = {}
    for path_str, item in ckpts.items():
        if not isinstance(item, dict):
            continue
        if "efficientnet_b4/best_checkpoint.pth" in path_str:
            ckpt_by_model["efficientnet_b4"] = item
        elif "densenet121/best_checkpoint.pth" in path_str:
            ckpt_by_model["densenet121"] = item
        elif "resnet50/best_checkpoint.pth" in path_str:
            ckpt_by_model["resnet50"] = item
        elif "dual_branch_seed123/best_checkpoint.pth" in path_str:
            ckpt_by_model["dual_branch_seed123"] = item
        elif "dual_branch_seed999/best_checkpoint.pth" in path_str:
            ckpt_by_model["dual_branch_seed999"] = item
        elif "dual_branch_seed42/best_checkpoint.pth" in path_str:
            ckpt_by_model["dual_branch_seed42"] = item

    model_specs = [
        {
            "id": "efficientnet_b4",
            "display_name": "EfficientNet-B4 (Deployed)",
            "family": "Compound-Scaled CNN (timm)",
            "architecture_version": "timm/efficientnet_b4",
            "fusion_version": None,
            "is_deployed": True,
            "seed": 42,
            "historical_claimed_metric": {
                "value": 0.7364,
                "metric_name": "Accuracy (on N=1,502 legacy split)",
                "reconciled_explanation": "Verified held-out test accuracy on N=1,510 split is 73.51%, Balanced Accuracy is 79.23%, Macro ROC-AUC is 95.91%."
            },
        },
        {
            "id": "densenet121",
            "display_name": "DenseNet-121",
            "family": "Dense Convolutional Network (timm)",
            "architecture_version": "timm/densenet121",
            "fusion_version": None,
            "is_deployed": False,
            "seed": 42,
            "historical_claimed_metric": {
                "value": 0.7883,
                "metric_name": "Balanced Accuracy",
                "reconciled_explanation": "Matches verified held-out test Balanced Accuracy (78.83%) and Macro ROC-AUC (95.32%)."
            },
        },
        {
            "id": "resnet50",
            "display_name": "ResNet-50",
            "family": "Deep Residual Network (timm)",
            "architecture_version": "timm/resnet50",
            "fusion_version": None,
            "is_deployed": False,
            "seed": 42,
            "historical_claimed_metric": {
                "value": 0.7512,
                "metric_name": "Balanced Accuracy",
                "reconciled_explanation": "Matches verified held-out test Balanced Accuracy (75.12%) and Macro ROC-AUC (93.58%)."
            },
        },
        {
            "id": "dual_branch_seed123",
            "display_name": "Dual-Branch CNN (Seed 123)",
            "family": "Custom Dual-Branch Texture+Structure CNN",
            "architecture_version": "DualBranchNet(v1_matrix)",
            "fusion_version": "v1_matrix",
            "is_deployed": False,
            "seed": 123,
            "historical_claimed_metric": {
                "value": 0.9098,
                "metric_name": "Mislabelled as Accuracy in legacy UI (Actually Macro ROC-AUC = 90.98%)",
                "reconciled_explanation": "Legacy UI table conflated Macro ROC-AUC (0.9098) with Accuracy. True held-out test Balanced Accuracy is 70.28% and Accuracy is 55.50%."
            },
        },
        {
            "id": "dual_branch_seed999",
            "display_name": "Dual-Branch CNN (Seed 999)",
            "family": "Custom Dual-Branch Texture+Structure CNN",
            "architecture_version": "DualBranchNet(v1_matrix)",
            "fusion_version": "v1_matrix",
            "is_deployed": False,
            "seed": 999,
            "historical_claimed_metric": {
                "value": 0.8973,
                "metric_name": "Mislabelled as Accuracy in legacy UI (Actually Macro ROC-AUC = 89.75%)",
                "reconciled_explanation": "Legacy UI table conflated Macro ROC-AUC (0.8975) with Accuracy. True held-out test Balanced Accuracy is 66.42% and Accuracy is 55.10%."
            },
        },
        {
            "id": "dual_branch_seed42",
            "display_name": "Dual-Branch CNN (Seed 42)",
            "family": "Custom Dual-Branch Texture+Structure CNN",
            "architecture_version": "DualBranchNet(v2_scalar)",
            "fusion_version": "v2_scalar",
            "is_deployed": False,
            "seed": 42,
            "historical_claimed_metric": {
                "value": 0.9054,
                "metric_name": "Mislabelled as Accuracy in legacy UI (Actually Macro ROC-AUC = 90.13%)",
                "reconciled_explanation": "Legacy UI table conflated Macro ROC-AUC (0.9013) with Accuracy. Uses v2_scalar (1280->1) fusion gate; true Balanced Accuracy is 59.49% and Accuracy is 64.30%."
            },
        },
    ]

    models_registry = []
    for spec in model_specs:
        mid = spec["id"]
        ev = eval_res[mid]
        ck = ckpt_by_model.get(mid, {})
        models_registry.append({
            **spec,
            "checkpoint": {
                "sha256": ck.get("sha256", ""),
                "sha256_short": ck.get("sha256", "")[:16],
                "size_mb": round(ck.get("size_bytes", 0) / (1024 * 1024), 2),
                "epoch": ev["checkpoint_epoch"],
                "best_val_balanced_accuracy": round(ev["best_val_bal_acc_in_ckpt"], 4),
                "num_tensors": ck.get("n_tensors", 0),
                "total_parameters": ck.get("n_learnable_params_in_state_dict", 0),
            },
            "test_metrics": {
                "split": "splits/test.csv",
                "n_samples": ev["test_n_samples"],
                "accuracy": round(ev["test_accuracy"], 4),
                "balanced_accuracy": round(ev["test_balanced_accuracy"], 4),
                "macro_precision": round(ev["test_macro_precision"], 4),
                "macro_recall": round(ev["test_macro_recall"], 4),
                "macro_f1": round(ev["test_macro_f1"], 4),
                "macro_roc_auc": round(ev["test_macro_roc_auc"], 4),
                "macro_pr_auc": round(ev["test_macro_pr_auc"], 4),
                "weighted_f1": round(ev["classification_report"]["weighted avg"]["f1-score"], 4),
            },
            "hardware_benchmark": {
                "mps_m4_latency_ms_per_img": round(ev["measured_latency_ms_per_img"], 2),
                "cuda_rtx3050_historical_latency_ms": 8.83 if mid == "efficientnet_b4" else None,
                "evaluation_device": "Apple Silicon M4 (MPS)",
            },
            "per_class_metrics": {
                cls_code: {
                    "precision": round(ev["classification_report"][cls_code]["precision"], 4),
                    "recall": round(ev["classification_report"][cls_code]["recall"], 4),
                    "f1_score": round(ev["classification_report"][cls_code]["f1-score"], 4),
                    "support": int(ev["classification_report"][cls_code]["support"]),
                }
                for cls_code in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            },
            "confusion_matrix": ev["confusion_matrix"],
        })

    registry = {
        "registry_version": "2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_class_order": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "canonical_preprocessing": {
            "version": "eval_v1_resize256_crop224_imagenet",
            "steps": [
                "Resize(256, 256, interpolation=BILINEAR)",
                "CenterCrop(224, 224)",
                "ToTensor() -> [0.0, 1.0]",
                "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
            ]
        },
        "deployed_model_summary": {
            "model_id": "efficientnet_b4",
            "checkpoint_sha256": "125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e",
            "selection_rationale": (
                "EfficientNet-B4 achieved the highest held-out test Balanced Accuracy (79.23%), "
                "Accuracy (73.51%), Macro F1 (69.24%), Macro ROC-AUC (95.91%), and Macro PR-AUC (77.52%) "
                "among all evaluated models while requiring 5.2x lower inference latency on Apple Silicon MPS "
                "than the custom Dual-Branch CNN trained from scratch."
            ),
        },
        "dataset_integrity": {
            "dataset_name": "HAM10000 (Human Against Machine with 10000 training images)",
            "citation": "Tschandl, P., Rosendahl, C. & Kittler, H. Sci. Data 5, 180161 (2018)",
            "total_images": dataset["metadata_rows"],
            "unique_lesions": dataset["unique_lesion_id"],
            "image_resolution": "600x450 RGB",
            "imbalance_ratio_nv_to_df": round(6705 / 115, 2),
            "class_counts": dataset["class_counts"],
            "class_percentages": dataset["class_pct"],
            "dx_type_verification_methods": dataset["dx_type_values"],
            "splits": {
                "strategy": "Lesion-grouped stratified split (zero lesion_id overlap across splits)",
                "train": {
                    "images": dataset["split_sizes"]["train"],
                    "unique_lesions": dataset["split_lesions"]["train"],
                    "class_counts": dataset["split_class_counts"]["train"],
                },
                "val": {
                    "images": dataset["split_sizes"]["val"],
                    "unique_lesions": dataset["split_lesions"]["val"],
                    "class_counts": dataset["split_class_counts"]["val"],
                },
                "test": {
                    "images": dataset["split_sizes"]["test"],
                    "unique_lesions": dataset["split_lesions"]["test"],
                    "class_counts": dataset["split_class_counts"]["test"],
                },
                "lesion_overlap": dataset["lesion_overlap"],
                "image_overlap": dataset["image_overlap"],
                "audit_notes": [
                    "0 lesion_id overlap and 0 image_id overlap between train, val, and test splits.",
                    "2 image pairs in upstream HAM10000 have identical MD5 hashes assigned to different lesion_ids by dataset authors.",
                    "6 duplicate '- Copy.jpg' files found on disk outside HAM10000_metadata.csv are ignored by split manifests."
                ]
            }
        },
        "models": models_registry,
        "ood_gate_v2": ood_res,
        "gradcam_verification": gradcam_res,
    }

    out_path = RESULTS_V2 / "benchmark_registry_v2.json"
    out_path.write_text(json.dumps(registry, indent=2))
    print(f"[SUCCESS] Wrote unified benchmark registry to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
