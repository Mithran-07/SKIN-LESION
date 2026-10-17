# DermAI Research Platform V2 — Scientific Audit & Remediation Report

**Repository:** `/Users/mithran/Documents/My projects/ADL` (`/Users/mithran/Documents/ADL`)  
**Branch:** `feat-dermai-platform-v2`  
**Audit & Remediation Date:** October 17, 2026  

---

## 1. Executive Summary

This document records the complete technical and scientific audit of the DermAI skin-lesion classification repository, the defects and ambiguities uncovered during baseline inspection, and the verified engineering remediations executed in **DermAI Platform V2**. Every claim in this document is backed by executable code, cryptographic checkpoint hashes (`SHA-256`), and reproducible JSON artifacts under `audit/outputs/` and `results/verified_v2/`.

| Audit Domain | Baseline Status (V1) | Remediated Status (V2) | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **Dual-Branch CNN Versioning** | `models/fusion.py` hardcoded `Linear(1280 -> 1280)`, causing `strict=True` failure on `seed_42` (`Linear(1280 -> 1)`) | Explicit `fusion_version: Literal["v1_matrix", "v2_scalar"]` in `AttentionFusionHead` and `DualBranchNet.from_checkpoint(..., strict=True)` | `models/fusion.py`, `models/dual_branch_net.py`, `tests/test_model_shapes.py` |
| **Historical Metric Claims (`90.98%`, `90.54%`, `89.73%`)** | Cited ambiguously as "Accuracy" in legacy summaries | Reconciled as **Macro ROC-AUC (%)**. True held-out test Accuracy (`55.50%`, `64.30%`, `55.10%`) and Balanced Accuracy (`70.28%`, `59.49%`, `66.42%`) published | `results/verified_v2/benchmark_registry_v2.json` |
| **OOD Domain Gate** | Built from 7 demo files in `data/samples/` and evaluated on the same 7 files (`N = 22` circular check); suffered **100% FRR** on unseen test images at `0.55` threshold | Fitted strictly on `splits/train.csv` (`N = 2,123`), calibrated on `splits/val.csv` (`τ = 0.2500`), evaluated once on `splits/test.csv` (`N = 1,510` ID + `200` OOD): **0.9985 AUROC**, **98.08% ID Acceptance**, **100.0% OOD Rejection** | `checkpoints/ood/efficientnet_b4_ood_calibration_v2.pt`, `results/verified_v2/ood_evaluation_v2.json` |
| **Demo Sample Integrity (`data/samples/`)** | All 7 `*_sample.jpg` files were cropped from a matplotlib figure (`results/sample_images.png`) with white padding borders | Replaced with 7 genuine `600 × 450` held-out test images from `splits/test.csv`; legacy crops archived in `audit/modernization_baseline/demo_samples_grid_crops_bak/` | `data/samples/*.jpg`, `app/skin-lesion-app/public/samples/*.jpg` |
| **Apple Silicon M4 MPS Latency & Memory** | Cold start `3,765.76 ms` (`/predict`) & `6,396.38 ms` (`/predict/explain`); duplicate backbone passes; `157.84 MB` active MPS memory | Startup shader warmup (`15.93×` faster first `/predict`: `236.38 ms`); single-pass OOD reuse (`27.64 ms` warm `/predict`); `69.95 MB` active MPS memory (`-55.7%`) | `results/verified_v2/post_optimization_performance_profile.json` |
| **Path Leakage in `/health`** | `weights_status` exposed local absolute path `/Users/mithran/...` | Sanitized status strings + `checkpoint_sha256_short` (`125340ffa59ca4a1`) | `api/main.py`, `api/inference_engine.py` |

---

## 2. Finding-by-Finding Audit & Remediation

### 2.1 Dual-Branch CNN Architecture Drift (`v1_matrix` vs. `v2_scalar`)
- **Defect Identified:** Inspecting the state dictionaries of the three historical Dual-Branch checkpoints revealed two incompatible attention gate parameterizations in `fusion.gate.0`:
  - `dual_branch_seed123` (`SHA-256: bcece2501c91947e...`) and `dual_branch_seed999` (`SHA-256: 1b5ca7fa42e9be2b...`) have `fusion.gate.0.weight` of shape `(1280, 1280)` and `fusion.gate.0.bias` of shape `(1280,)`, totaling **10,669,639** learnable parameters.
  - `dual_branch_seed42` (`SHA-256: 84d950e6cf30e1c9...`) has `fusion.gate.0.weight` of shape `(1, 1280)` and `fusion.gate.0.bias` of shape `(1,)`, totaling **9,031,241** learnable parameters.
- **Remediation Executed:**
  - Refactored `AttentionFusionHead` (`models/fusion.py`) to accept `fusion_version: Literal["v1_matrix", "v2_scalar"] = "v1_matrix"`.
  - In `v1_matrix`, $\mathbf{g} = \mathbf{z} \odot \sigma(\mathbf{W}_m \mathbf{z} + \mathbf{b}_m)$ applies element-wise gating across all 1,280 concatenated features.
  - In `v2_scalar`, $\alpha = \sigma(\mathbf{w}_s^\top \mathbf{z} + b_s) \in (0, 1)$ applies complementary branch-level scalar gating $[\alpha \cdot \mathbf{f}_{\text{tex}} \;\|\; (1 - \alpha) \cdot \mathbf{f}_{\text{struct}}]$.
  - Added `DualBranchNet.inspect_checkpoint_version(state_dict)` and `DualBranchNet.from_checkpoint(path, strict=True)` in `models/dual_branch_net.py`, verified by automated unit tests in `tests/test_model_shapes.py`.

### 2.2 Reconciliation of Historical Benchmark Claims
- **Defect Identified:** Historical project notes cited `90.98%` (Seed 123), `89.73%` (Seed 999), and `90.54%` (Seed 42) for Dual-Branch CNN alongside `73.64%` for EfficientNet-B4 without clarifying that the Dual-Branch figures were **Macro ROC-AUC (%)** whereas `73.64%` was **Top-1 Multi-Class Accuracy (%)**.
- **Remediation Executed:** Re-evaluated all 6 checkpoints on the canonical lesion-disjoint test split (`splits/test.csv`, $N = 1,510$ images across $1,121$ unique lesions) using identical deterministic preprocessing (`Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)`). All metrics (Balanced Accuracy, Macro ROC-AUC, Macro PR-AUC, Top-1 Accuracy, Macro F1, per-class Precision/Recall/F1, and $7 \times 7$ confusion matrices) are now published side-by-side in `results/verified_v2/benchmark_registry_v2.json`.

### 2.3 Root-Cause Analysis: Why Pretrained Compound Scaling Outperformed Custom Dual-Branch CNN
1. **Pretraining Deficit (99.88% Scratch Initialization):**
   - `ShallowWideBranch` (`6,035,456` params) expands directly from $3 \to 256$ channels (`3×3`), preventing weight transfer from `WideResNet-50-2` (`64×3×7×7`). Thus 100% of the texture branch was Kaiming-initialized from scratch.
   - `DeepNarrowBranch` (`1,961,216` params) only transferred its $7 \times 7$ stem (`9,536` params, or `0.49%` of branch weights) from `DenseNet-121`.
   - Overall, **99.88% of `DualBranchNet` parameters** were trained from scratch on 7,002 training images, whereas `EfficientNet-B4` (`17.56M` params) transferred 100% of its multi-scale MBConv + Squeeze-and-Excitation hierarchy from ImageNet-1K.
2. **Spatial Pooling Bottleneck:**
   - `ShallowWideBranch` applies `AdaptiveAvgPool2d(1, 1)` directly to a $112 \times 112$ feature map after only 3 convolutional layers ($9 \times 9$ receptive field), averaging out spatial localization of pigment network irregularities prior to fusion.

### 2.4 Remediation of Circular OOD Evaluation & Matplotlib-Cropped Demo Samples
- **Defect Identified:**
  1. The 7 demo images in `data/samples/*_sample.jpg` were discovered to be crops from a matplotlib visualization grid (`results/sample_images.png`) containing white border margins (`R=255, G=255, B=255`).
  2. Consequently, the legacy OOD prototype bank (`api/ood_gate.py` v1) had memorized white-bordered matplotlib crops as its "in-domain" reference, causing it to reject unseen genuine HAM10000 dermoscopic images while accepting bright office documents.
- **Remediation Executed:**
  1. Replaced all 7 demo images in `data/samples/` and `app/skin-lesion-app/public/samples/` with genuine, uncropped $600 \times 450$ HAM10000 held-out test images (`ISIC_0026149`, `ISIC_0024590`, `ISIC_0032417`, `ISIC_0033695`, `ISIC_0025751`, `ISIC_0027411`, `ISIC_0031215`).
  2. Implemented and calibrated **OODDomainGate v2.0.0** (`scripts/calibrate_and_eval_ood.py`, `api/ood_gate.py`) using a 4-signal ensemble reusing EfficientNet-B4's 1,792-dim penultimate features and 7-class logits:
     - **Signal 1:** Cosine similarity to 49 training-set class/sub-cluster prototypes ($7 \text{ classes} \times 7 \text{ k-means clusters}$ fitted strictly on `splits/train.csv`, $N = 2,123$).
     - **Signal 2:** Diagonal Mahalanobis log-density under per-class feature distributions fitted on `splits/train.csv`.
     - **Signal 3:** Logit Free Energy score $-T \log \sum_c \exp(z_c / T)$.
     - **Signal 4:** Dermoscopic photometric sanity envelope (rejecting blank/saturated frames, extreme cold-blue UI screenshots, and pure grayscale scans).
  3. Selected threshold $\tau = 0.2500$ strictly on `splits/val.csv` ($N = 1,503$ ID + $N = 100$ val OOD).
  4. Evaluated once on untouched `splits/test.csv` ($N = 1,510$ ID) + $N = 200$ held-out OOD images across 10 categories: achieved **0.9985 AUROC** (95% CI: `0.9972–0.9994`), **0.9998 AUPRC**, **98.08% ID Acceptance Rate** (`1.92% FRR`), and **100.0% OOD Rejection Rate** (`0.0% FAR`).
