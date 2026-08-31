# Final Project Status & Submission Summary

**Project Title**: Advanced Deep Learning for Non-Melanoma Dermoscopic Classification  
**Date of Completion**: August 31, 2026  
**Status**: Completed, Fully Validated, and Submission-Ready ✅  
**Workstation**: Apple MacBook Pro M4 (Inference, Demo, Documentation) & Lenovo LOQ (ML Archive)

---

## 1. Research Overview

### Dataset
- **HAM10000 Benchmark**: 10,015 dermatoscopic images across 7 diagnostic categories.
- **Diagnostic Categories**:
  - `MEL`: Melanoma (1,113 cases / 11.1%)
  - `NV`: Melanocytic Nevi (6,705 cases / 67.0%)
  - `BCC`: Basal Cell Carcinoma (514 cases / 5.1%)
  - `AKIEC`: Actinic Keratoses & Intraepithelial Carcinoma (327 cases / 3.3%)
  - `BKL`: Benign Keratosis-like Lesions (1,099 cases / 11.0%)
  - `DF`: Dermatofibroma (115 cases / 1.1%)
  - `VASC`: Vascular Lesions (142 cases / 1.4%)
- **Data Splitting**: Strict patient-aware stratified splitting partitioned on `lesion_id` (70% train, 15% val, 15% test) to prevent identical lesions contaminating test splits.

### Architectures Evaluated
1. **Decoupled Dual-Branch CNN** (Evaluated across seeds 42, 123, 999):
   - *Shallow-Wide Branch*: High channel width (1024 channels) with minimal spatial downsampling to preserve fine surface texture, arborizing vessels, and keratin pearls.
   - *Deep-Narrow Branch*: Base 64 channels with 4 residual stages to expand receptive field and distill macroscopic asymmetry and border irregularity.
   - *Attention-Gated Fusion Module*: Sigmoid gate with MLP projection.
2. **Compound-Scaled Single-Branch Baselines**:
   - *EfficientNet-B4* (Depthwise separable convolutions + compound scaling)
   - *DenseNet-121* (Dense connectivity feature reuse)
   - *ResNet-50* (Residual network baseline)

### Objective Findings & Research Conclusion
- The Dual-Branch CNN attained respectable discrimination (**ROC-AUC ~90.98%** across seeds).
- However, **EfficientNet-B4 compound scaling consistently outperformed all multi-branch and single-branch models** across all key classification metrics.
- Attention gate diagnostics revealed that gate weights favored structural representations (>78%), while texture branch gradients experienced higher variance.
- In accordance with rigorous scientific practice, **EfficientNet-B4 was selected as the verified production model for deployment**.

---

## 2. Best Model & Verified Benchmark Performance

| Evaluation Metric | EfficientNet-B4 (Best Model) | DenseNet-121 | Dual-Branch CNN (Seed 123) | ResNet-50 |
|---|---|---|---|---|
| **Macro ROC-AUC** | **95.92%** | 95.31% | 90.98% | 93.52% |
| **Balanced Accuracy** | **79.16%** | 79.14% | 70.31% | 75.13% |
| **Overall Test Accuracy** | **73.64%** | 66.36% | 55.50% | 56.62% |
| **Macro F1 Score** | **69.19%** | 62.42% | 48.55% | 53.52% |
| **Inference Latency** | **8.83 ms** | 20.37 ms | 27.23 ms | 20.96 ms |
| **Peak VRAM** | **677.7 MB** | 393.7 MB | 1561.1 MB | 900.3 MB |
| **Total Parameters** | **17.56 M** | 6.96 M | 10.67 M | 23.52 M |
| **Benchmark Rank** | **Rank 1 (Champion)** | Rank 2 | Rank 3 | Rank 6 |

---

## 3. Production Software Architecture

```
                                USER / BROWSER
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Next.js 14 Web Client   │
                        │ (Tailwind CSS, React 18)  │
                        └─────────────┬─────────────┘
                                      │ HTTP / Multipart
                                      ▼
                        ┌───────────────────────────┐
                        │      FastAPI Backend      │
                        │    (api/main.py:8000)     │
                        └─────────────┬─────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
            ┌───────────────────────────┐ ┌───────────────────────────┐
            │   Inference Engine (MPS)  │ │     Grad-CAM Visualizer   │
            │      EfficientNet-B4      │ │  (Final Conv Feature Map) │
            │   Top-3 Probabilities     │ │   Attribution Base64 PNG  │
            └───────────────────────────┘ └───────────────────────────┘
```

- **Backend**: FastAPI server with 5 endpoints (`GET /health`, `GET /benchmark`, `GET /classes`, `POST /predict`, `POST /predict/explain`).
- **Frontend**: Next.js 14 App Router application with 5 complete views:
  - `/` — Project Overview, Clinical Problem, Metric Highlights
  - `/classify` — Live Dermoscopic Image Classifier, Top-3 Predictions, Grad-CAM Overlay
  - `/dashboard` — Interactive Empirical Benchmark Explorer & Metrics Comparison
  - `/research` — HAM10000 Dataset Breakdown, Class Imbalance, Methodology
  - `/architecture` — Decoupled Dual-Branch CNN Blueprint & Gate Diagnostics
- **Explainability**: Integrated Grad-CAM generating spatial attribution heatmaps over 224×224 dermoscopic inputs.

---

## 4. Verification & Testing Matrix

| Test Suite | Total Assertions | Passed | Success Rate |
|---|---|---|---|
| **Deep Learning Shapes & Forward Pass** | 8 | 8 | 100% |
| **Focal Loss Weighting & Reduction** | 5 | 5 | 100% |
| **Split Conformal Prediction** | 4 | 4 | 100% |
| **FastAPI Backend Suite (All 5 Endpoints)** | 30 | 30 | 100% |
| **Real HAM10000 7-Class Image Validation** | 7 | 7 | 100% |
| **Next.js Production Build** | 8 routes | 8 | 100% |
| **Overall Project Verification** | **62 Assertions** | **62 Passed** | **100% PASSED ✅** |

---

## 5. Limitations & Ethical Considerations

1. **HAM10000 Dataset Bias**: Images originate primarily from fair-skinned European patient cohorts (Fitzpatrick skin types I–III); performance on darker skin phenotypes remains unvalidated.
2. **Severe Class Imbalance**: Nevi (NV) represents 67.0% of images, while Dermatofibroma (DF) accounts for only 1.1%.
3. **No External Clinical Trial**: The system has not undergone clinical trials with practicing dermatologists in prospective hospital settings.
4. **Academic Research Prototype**: The system is designed strictly for academic evaluation and educational demonstrations. It is not approved as a medical device (SaMD).
5. **Dual-Branch Hypothesis Outcome**: Handcrafted two-branch decoupling did not outperform unified compound scaling in EfficientNet-B4.

---

## 6. Exact Launch Commands for Demonstration

### Start Backend (Terminal 1):
```bash
cd "/Users/mithran/Documents/My projects/ADL"
source .venv/bin/activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend (Terminal 2):
```bash
cd "/Users/mithran/Documents/My projects/ADL/app/skin-lesion-app"
npm run dev
```

*Open browser at*: `http://localhost:3000`
