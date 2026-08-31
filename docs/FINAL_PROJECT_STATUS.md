# Final Project Status & Submission Document

**Project**: Dual-Branch CNN Framework for Non-Melanoma Dermoscopic Classification  
**Date**: August 31, 2026  
**Status**: Fully Integrated, Validated, and Submission-Ready ✅  
**Deployment Model**: EfficientNet-B4 (HAM10000 7-Class Dermoscopic Classifier)  

---

## 1. Project Objective

Automate the classification of cutaneous lesions into 7 diagnostic categories using the HAM10000 dataset, specifically targeting the detection and distinction of Non-Melanoma Skin Cancers (Basal Cell Carcinoma, Actinic Keratosis) and Melanoma from benign mimickers. The research investigated a novel Decoupled Dual-Branch CNN architecture separating texture and structure representations, benchmarked against state-of-the-art single-branch CNN baselines.

---

## 2. Dataset Formulation & Handling

- **HAM10000 Benchmark**: 10,015 multi-source dermatoscopic images across 7 classes:
  - `MEL`: Melanoma (1,113 / 11.1%)
  - `NV`: Melanocytic Nevi (6,705 / 67.0%)
  - `BCC`: Basal Cell Carcinoma (514 / 5.1%)
  - `AKIEC`: Actinic Keratosis / Intraepithelial Carcinoma (327 / 3.3%)
  - `BKL`: Benign Keratosis-like Lesions (1,099 / 11.0%)
  - `DF`: Dermatofibroma (115 / 1.1%)
  - `VASC`: Vascular Lesions (142 / 1.4%)
- **Patient-Aware Stratified Splitting**: Strict grouping by `lesion_id` (70% train, 15% val, 15% test) to prevent identical lesions contaminating train and test distributions.
- **Class-Weighted Focal Loss**: Applied inverse class frequency \(\alpha\)-weighting with \(\gamma=2.0\) focusing parameter to counteract the 58:1 class imbalance.

---

## 3. Evaluated Architectures & Empirical Findings

### Dual-Branch Research Progression
1. **Dual-Branch V1 (Original)**: Multi-seed average test accuracy **54.79%** (Seed 42: 53.91%, Seed 123: 55.50%, Seed 999: 54.97%), ROC-AUC ~90.98%.
2. **Dual-Branch V1.1 (Optimized Training)**: Test accuracy **65.76%**, ROC-AUC **90.06%**, Macro F1 **48.14%**.
3. **Dual-Branch V2 (Refined Architecture)**: Test accuracy **64.24%**, ROC-AUC **90.15%**, Macro F1 **49.50%**.

### Comparative Baseline Benchmark

| Model Architecture | Parameters | Macro ROC-AUC | Balanced Accuracy | Test Accuracy | Macro F1 | Latency (ms) | Peak VRAM | Status |
|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B4** | **17.56M** | **95.92%** | **79.16%** | **73.64%** | **69.19%** | **8.83 ms** | **677.7 MB** | **Champion (Deployed)** |
| **DenseNet-121** | 6.96M | 95.31% | 79.14% | 66.36% | 62.42% | 20.37 ms | 393.7 MB | Baseline |
| **Dual-Branch V1.1** | 10.67M | 90.06% | 62.18% | 65.76% | 48.14% | 24.56 ms | 1561.1 MB | Experimental |
| **Dual-Branch V2** | 9.03M | 90.15% | 59.48% | 64.24% | 49.50% | 25.54 ms | 1516.7 MB | Experimental |
| **Dual-Branch V1 (Seed 123)**| 10.67M | 90.98% | 70.31% | 55.50% | 48.55% | 27.23 ms | 1561.1 MB | Experimental |
| **ResNet-50** | 23.52M | 93.52% | 75.13% | 56.62% | 53.52% | 20.96 ms | 900.3 MB | Baseline |

### Research Conclusion
EfficientNet-B4 achieved the strongest discrimination across all evaluation criteria. Attention gate diagnostics showed that the Dual-Branch attention gate favored structural features (>78%), while texture gradients suffered higher variance. In adherence to scientific integrity, EfficientNet-B4 is deployed as the production model.

---

## 4. Production Application Architecture

```
                    Browser Client (Next.js 14, React 18)
                                  │
                   HTTP / Multipart POST Requests
                                  ▼
                     FastAPI Backend (Port 8000)
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
        EfficientNet-B4 Engine          Grad-CAM Visualizer
        (MPS / CUDA / CPU)              (Final Conv Feature Map)
        Top-3 Probabilities Array       Base64 PNG Attribution Overlay
```

- **Backend (`api/`)**: FastAPI with 5 endpoints (`/health`, `/benchmark`, `/classes`, `/predict`, `/predict/explain`).
- **Frontend (`app/skin-lesion-app`)**: Next.js 14 App Router application with 5 interactive pages:
  - `/` (Overview & Problem Statement)
  - `/classify` (Live Classification, Top-3 Predictions, Grad-CAM Overlay)
  - `/dashboard` (Empirical Benchmarks Explorer)
  - `/research` (Dataset & Methodology Reference)
  - `/architecture` (Dual-Branch Design & Diagnostic Findings)

---

## 5. Limitations & Future Work

### Limitations
1. **Cohort Demographics**: HAM10000 is predominantly sampled from fair-skinned populations (Fitzpatrick I–III).
2. **Dataset Imbalance**: Nevi (NV) comprise 67.0% of images, while Dermatofibroma (DF) constitutes only 1.1%.
3. **Non-Clinical Deployment**: Academic research prototype only; not certified for clinical diagnostic decision-making.

### Future Work
1. Incorporate prospective multi-center clinical validation across diverse ethnic demographics.
2. Explore Vision Transformers (ViT) and Swin Transformer backbones with spatial self-attention.
3. Integrate multimodal clinical metadata (patient age, anatomical site, lesion history) into the decision pipeline.

---

## 6. Execution Runbook

### Start Backend:
```bash
# In project root
source .venv/bin/activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend:
```bash
cd app/skin-lesion-app
npm run dev
```

*Open*: `http://localhost:3000` (API Docs at `http://127.0.0.1:8000/docs`).

---

## 7. Medical Safety Disclaimer

> **ACADEMIC RESEARCH PROTOTYPE ONLY**: This system is strictly an academic research prototype and is not intended to provide medical diagnosis or replace professional medical advice. Always consult a board-certified dermatologist for clinical evaluation.
