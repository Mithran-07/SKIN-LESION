# Dual-Branch CNN Framework for Non-Melanoma Dermoscopic Classification

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical deep learning investigation evaluating a novel **Decoupled Dual-Branch Convolutional Neural Network** against compound-scaled single-branch architectures on the 7-class **HAM10000** dermoscopic skin lesion dataset, featuring an interactive **Next.js 14** frontend, **FastAPI** inference backend, and **Grad-CAM** model attribution.

---

## 1. Research Overview & Problem Statement

Skin cancer represents a global public health crisis. While **Melanoma (MEL)** carries the highest mortality, **Non-Melanoma Skin Cancers (NMSC)**—specifically **Basal Cell Carcinoma (BCC)** and **Actinic Keratoses / Intraepithelial Carcinoma (AKIEC)**—account for the overwhelming majority of cutaneous malignancies.

### Research Question
> *Can physically decoupling high-frequency textural representation (micro-vessels, surface keratinization) from macroscopic morphological structure (asymmetry, border irregularity) in a Dual-Branch CNN outperform state-of-the-art single-branch compound-scaling networks in dermoscopic classification under extreme class imbalance?*

---

## 2. Architectural Design

```
                     Dermoscopic Image (224×224 RGB)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      Shallow-Wide Branch                  Deep-Narrow Branch
    (Textural Extraction)                (Structural Morphology)
    Conv(3→256) + BN + ReLU              Stem: Conv(3→64, s=2) + Pool
    Conv(256→512) + MaxPool              4 Residual Stages (64→128)
    Conv(512→1024) + BN + ReLU           Bottleneck: Conv(128→256)
    AdaptiveAvgPool(1,1)                 AdaptiveAvgPool(1,1)
          1024-dim                             256-dim
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                         Attention-Gated Fusion
                    [Texture(1024) ‖ Structure(256)]
                    Sigmoid Attention Gate (1280-dim)
                    MLP: 1280 → 512 → 256 (GELU + Dropout)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        7-Class Logits Output               Grad-CAM Attribution
     (MEL, NV, BCC, AKIEC, BKL, DF, VASC)     (Final Conv Heads)
```

1. **Shallow-Wide Branch**: Large channel capacity (1024-dim) with minimal spatial downsampling to preserve fine spatial details (arborizing telangiectasia, dotted vessels, keratin scales).
2. **Deep-Narrow Branch**: Narrow channel width (base 64-dim) across 4 residual stages to maximize the receptive field for global lesion asymmetry, border notch patterns, and pigment contrast.
3. **Attention-Gated Fusion**: Concatenates representations and computes a learned sigmoid attention mask before two-stage MLP projection.

---

## 3. Empirical Benchmark Results

All models were evaluated on the HAM10000 dataset using patient-aware stratified splitting and class-weighted Focal Loss (\(\gamma=2.0\)):

| Model Architecture | Parameters | Macro ROC-AUC | Balanced Accuracy | Test Accuracy | Macro F1 | Latency (ms) | Peak VRAM | Status |
|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B4** | **17.56M** | **95.92%** | **79.16%** | **73.64%** | **69.19%** | **8.83 ms** | **677.7 MB** | **Champion (Deployed)** |
| **DenseNet-121** | 6.96M | 95.31% | 79.14% | 66.36% | 62.42% | 20.37 ms | 393.7 MB | Baseline |
| **Dual-Branch CNN (Seed 123)** | 10.67M | 90.98% | 70.31% | 55.50% | 48.55% | 27.23 ms | 1561.1 MB | Experimental |
| **Dual-Branch CNN (Seed 999)** | 10.67M | 89.73% | 66.39% | 54.97% | 45.77% | 26.32 ms | 1561.1 MB | Experimental |
| **Dual-Branch CNN (Seed 42)** | 10.67M | 90.54% | 68.62% | 53.91% | 44.90% | 28.47 ms | 1561.1 MB | Experimental |
| **ResNet-50** | 23.52M | 93.52% | 75.13% | 56.62% | 53.52% | 20.96 ms | 900.3 MB | Baseline |

### Key Scientific Findings
- **Dual-Branch Viability**: The Dual-Branch CNN achieved strong discriminatory capability (**~91% ROC-AUC**).
- **Compound Scaling Superiority**: **EfficientNet-B4 outperformed all architectures** across all diagnostic metrics due to principled joint scaling of depth, width, and resolution.
- **Fusion Gate Diagnostics**: Fusion gates skewed predominantly toward the structural branch (>78%), resulting in suboptimal gradient propagation into high-dimensional texture layers.
- **Deployment Decision**: Following rigorous scientific integrity, **EfficientNet-B4 was selected as the verified production model for the live application**.

---

## 4. Full-Stack Software Architecture

```
                    USER BROWSER / CLINICAL INTERFACE
                                   │
                                   ▼
                     Next.js 14 Frontend Application
               (Tailwind CSS, React 18, Responsive Design)
                     http://localhost:3000
                                   │
                    HTTP REST Multipart Form Requests
                                   ▼
                         FastAPI Backend Server
                        (Python 3.11+, Uvicorn)
                     http://127.0.0.1:8000
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
          EfficientNet-B4 Engine        Grad-CAM Attribution
          (MPS / CUDA / CPU)            (Final Conv Feature Head)
          Top-3 Probability Array       Base64 Overlaid Heatmap
```

### Application Features
- **`/` (Overview)**: Clinical problem statement, methodology summary, and empirical benchmark highlights.
- **`/classify` (Classifier)**: Live image upload, instant demo presets for all 7 classes, top-3 probabilities, and side-by-side Grad-CAM spatial attribution visualizer.
- **`/dashboard` (Benchmark)**: Interactive benchmark comparison tables, bar charts, latency vs VRAM trade-offs.
- **`/research` (Dataset)**: HAM10000 dataset analysis, class imbalance metrics, patient-aware splitting controls.
- **`/architecture` (Deep Dive)**: Dual-Branch topology diagrams, fusion gate diagnostics, and empirical comparison rationale.

---

## 5. Quick Start & Execution

### Prerequisites
- Python 3.11+
- Node.js v18+ & npm
- PyTorch with MPS (Apple Silicon), CUDA (NVIDIA), or CPU support

### 1. Backend Setup & Startup
```bash
# Clone the repository
git clone https://github.com/Mithran-07/SKIN-LESION.git
cd SKIN-LESION

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start FastAPI backend server
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
*Backend active at*: `http://127.0.0.1:8000`  
*Interactive Swagger API documentation*: `http://127.0.0.1:8000/docs`

---

### 2. Frontend Setup & Startup
```bash
# In a new terminal window
cd app/skin-lesion-app
npm install
npm run dev
```
*Open application in browser*: `http://localhost:3000`

---

## 6. Running Automated Test Suites

```bash
# 1. Run Core PyTorch Test Suite (17 tests: Shapes, Focal Loss, Conformal Prediction)
.venv/bin/python -m pytest tests/test_model_shapes.py tests/test_focal_loss.py tests/test_conformal.py -v

# 2. Run Complete FastAPI Backend Suite (30 test assertions across all 5 endpoints)
.venv/bin/python scripts/test_api_suite.py

# 3. Run Real HAM10000 7-Class Image Validation Matrix
.venv/bin/python scripts/validate_real_images.py

# 4. Verify Frontend Production Compilation
cd app/skin-lesion-app && npm run build
```

---

## 7. Diagnostic Class Taxonomy (HAM10000)

| Index | Code | Full Clinical Name | Pathological Category | Class Count |
|---|---|---|---|---|
| 0 | **MEL** | Melanoma | Malignant | 1,113 (11.1%) |
| 1 | **NV** | Melanocytic Nevi | Benign | 6,705 (67.0%) |
| 2 | **BCC** | Basal Cell Carcinoma | Malignant (NMSC) | 514 (5.1%) |
| 3 | **AKIEC** | Actinic Keratosis / Intraepithelial Carcinoma | Pre-malignant | 327 (3.3%) |
| 4 | **BKL** | Benign Keratosis-like Lesions | Benign | 1,099 (11.0%) |
| 5 | **DF** | Dermatofibroma | Benign | 115 (1.1%) |
| 6 | **VASC** | Vascular Lesions | Benign | 142 (1.4%) |

---

## 8. Limitations & Ethical Considerations

1. **Demographic Bias**: HAM10000 predominantly features fair-skinned European populations (Fitzpatrick skin types I–III). Efficacy on darker skin phototypes requires external validation.
2. **Extreme Class Imbalance**: Benign Nevi comprise 67% of data, requiring specialized loss weighting.
3. **Research Prototype**: The application is an academic research prototype and is not a certified Medical Device (SaMD).

---

## 9. Important Medical Disclaimer

> **ACADEMIC RESEARCH PROTOTYPE ONLY**: This system is strictly an academic research artifact intended for educational and demonstration purposes. It does not provide medical diagnosis, clinical certainty, or replace professional dermatological consultation.

---

## 10. Citation

```bibtex
@misc{mithran2026skinlesion,
  author = {Mithran, A.},
  title = {Dual-Branch CNN Framework for Non-Melanoma Dermoscopic Classification},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Mithran-07/SKIN-LESION}}
}
```

---

## 11. License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
