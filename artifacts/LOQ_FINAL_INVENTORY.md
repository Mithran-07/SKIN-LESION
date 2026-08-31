# Lenovo LOQ — Final Inventory & Research Archive

**Workstation:** Lenovo LOQ (Windows 11, RTX 3050 6GB)  
**Project:** Dual-Branch CNN Framework for Non-Melanoma Dermoscopic Classification  
**Status:** FROZEN / ARCHIVE MODE  
**Date:** 2026-08-31  

---

## 1. Core Source Code Locations

| Component | Path | Description |
|---|---|---|
| Model Definitions | `models/` | Dual-Branch Net (V1/V2), Baselines (ResNet50, DenseNet121, EfficientNet-B4), Fusion layers, MTL head |
| Data Scaffold | `data/` | HAM10000 loader, patient-level splitter, dataset validator, albumentations transforms |
| Training Scaffold | `training/` | PyTorch trainer, metrics computation, learning curve visualizer |
| Losses | `losses/` | Focal loss, multi-task learning loss |
| Explainability | `explainability/` | DualBranchGradCAM, single-branch GradCAM, heatmap overlays |
| Uncertainty | `uncertainty/` | Conformal prediction, Monte Carlo Dropout |
| API Backend | `api/` | FastAPI main application (`main.py`) & timm inference engine (`inference_engine.py`) |
| Web Application | `app/skin-lesion-app/` | Next.js 15 app (Home, Classify, Dashboard, Research Story, Architecture) |

---

## 2. Checkpoint Locations

| Model | Path | Size | Status | SHA-256 |
|---|---|---|---|---|
| **EfficientNet-B4** | `checkpoints/efficientnet_b4/best_checkpoint.pth` | 202.05 MB | **FINAL DEPLOYMENT** | `125340FFA59CA4A18FB0CD0BDCA024EAE9CA2FE3CE54D1F230B465BD9AD9658E` |
| ResNet50 | `checkpoints/resnet50/best_checkpoint.pth` | 94.1 MB | Baseline Archive | Verified |
| DenseNet121 | `checkpoints/densenet121/best_checkpoint.pth` | 28.0 MB | Baseline Archive | Verified |
| Dual-Branch V1 (s42) | `checkpoints/dual_branch_seed42/best_checkpoint.pth` | 42.7 MB | Research Archive | Verified |
| Dual-Branch V1 (s123) | `checkpoints/dual_branch_seed123/best_checkpoint.pth` | 42.7 MB | Research Archive | Verified |
| Dual-Branch V1 (s999) | `checkpoints/dual_branch_seed999/best_checkpoint.pth` | 42.7 MB | Research Archive | Verified |

---

## 3. Results & Publication Artifacts

| Category | Path | Description |
|---|---|---|
| Final Benchmark | `results/final/benchmark_final.csv` | Full metric comparison across all 6 evaluated models |
| Comparison Table | `results/final/comparison_table.csv` / `.md` | Formatted comparison tables for paper/thesis |
| Metric Summary | `results/final/summary_metrics.json` | JSON format of all metrics |
| Figures | `results/final/figures/` | Publication-ready vector & raster figures (ROC, PR, CM, LC, Fusion, Class Dist) in PNG, SVG, and PDF formats |
| Final Summary | `results/final/FINAL_RESULTS.md` | Consolidated findings and statistical analysis |

---

## 4. Benchmark Verification Summary

| Model | Test Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC | Status |
|---|---|---|---|---|---|
| **EfficientNet-B4** | **73.64%** | **79.16%** | **69.19%** | **95.92%** | **DEPLOYED** |
| DenseNet121 | 66.36% | 79.14% | 62.42% | 95.31% | Baseline |
| ResNet50 | 56.62% | 75.13% | 53.52% | 93.52% | Baseline |
| Dual-Branch V1 (avg 3 seeds) | 54.79% | 68.44% | 46.41% | 90.41% | Experimental |
| Dual-Branch V1.1 (training) | 65.76% | 62.18% | 48.14% | 90.06% | Experimental |
| Dual-Branch V2 (fusion redesign)| 64.24% | 59.48% | 49.50% | 90.15% | Experimental |

---

## 5. Workstation Environment Specs

- **Python:** 3.13.2
- **PyTorch:** 2.7.1+cu118
- **CUDA:** 11.8 (NVIDIA GeForce RTX 3050 Laptop GPU)
- **timm:** 1.0.29
- **FastAPI:** 0.141.1
- **Node.js:** v22.17.0
- **Next.js:** 15.x

---

## 6. Exact Transfer Instructions for Mac

1. `git pull origin main` on MacBook to receive all code, frontend, API, docs, and results.
2. Manually copy `checkpoints/efficientnet_b4/best_checkpoint.pth` (202.05 MB) to `checkpoints/efficientnet_b4/` on the Mac.
3. Verify SHA-256 on Mac using `shasum -a 256 checkpoints/efficientnet_b4/best_checkpoint.pth`.
4. Run `python -m uvicorn api.main:app --port 8000` and `npm run dev` (in `app/skin-lesion-app`).
