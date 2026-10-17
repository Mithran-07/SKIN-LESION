# DermAI V2 — Dermal Intelligence Lab & Dual-Branch CNN Research Platform

[![Live Demo on Vercel](https://img.shields.io/badge/Live_Workstation-skin--lesion--xi.vercel.app-00C7B7?style=for-the-badge&logo=vercel&logoColor=white)](https://skin-lesion-xi.vercel.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Mithran--07%2FSKIN--LESION-181717?style=for-the-badge&logo=github)](https://github.com/Mithran-07/SKIN-LESION)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI 2.0](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![OOD Gate v2](https://img.shields.io/badge/OOD_AUROC-0.9985-10b981.svg)](https://skin-lesion-xi.vercel.app/research)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evidence-based computational dermatology research platform evaluating a custom **Decoupled Dual-Branch Convolutional Neural Network (`v1_matrix` & `v2_scalar`)** against compound-scaled single-backbone architectures (**EfficientNet-B4**, **DenseNet-121**, **ResNet-50**) on the 7-class **HAM10000** dermoscopic dataset ($N = 10,015$ images, $7,470$ unique lesions).

Includes an interactive **Next.js 14 Research Workstation**, a **Calibrated 4-Signal Out-of-Distribution (OOD) Gate v2.0** (`0.9985 AUROC`), **Counterfactual 7-Class Grad-CAM Attribution**, and an **Apple Silicon M4 MPS** inference engine (`27.64 ms` warm latency, `69.95 MB` active tensor memory).

---

## Live Production Deployment (Vercel)

> **Live Platform URL:** **[https://skin-lesion-xi.vercel.app](https://skin-lesion-xi.vercel.app)**

| Module / Endpoint | Live URL | Description |
| :--- | :--- | :--- |
| **Overview** | [`https://skin-lesion-xi.vercel.app/`](https://skin-lesion-xi.vercel.app/) | Executive summary, verified benchmark KPIs, and 6-step execution pipeline |
| **Analysis Workstation** | [`https://skin-lesion-xi.vercel.app/classify`](https://skin-lesion-xi.vercel.app/classify) | Interactive 7-class HAM10000 test comparator, RGBA Grad-CAM blending, counterfactual class selector, and 4-signal OOD Gate v2 telemetry |
| **Benchmark Lab** | [`https://skin-lesion-xi.vercel.app/dashboard`](https://skin-lesion-xi.vercel.app/dashboard) | Interactive $7 \times 7$ confusion matrices, per-class recall/precision/F1, SHA-256 checkpoint provenance, and historical claim reconciliation |
| **Dataset & OOD Integrity** | [`https://skin-lesion-xi.vercel.app/research`](https://skin-lesion-xi.vercel.app/research) | Lesion-disjoint 70/15/15 split verification ($0$ cross-split leakage) and 10-category OOD v2 evaluation ($N = 1,510$ ID vs $N = 200$ OOD) |
| **Architecture Comparator** | [`https://skin-lesion-xi.vercel.app/architecture`](https://skin-lesion-xi.vercel.app/architecture) | Structural comparison of compound-scaled `EfficientNet-B4` (`17.56M`) vs custom `DualBranchNet v1_matrix` (`10.67M`) & `v2_scalar` (`9.03M`) |
| **System & MPS Diagnostics** | [`https://skin-lesion-xi.vercel.app/diagnostics`](https://skin-lesion-xi.vercel.app/diagnostics) | Apple M4 MPS unified memory profiling, Metal shader warmup telemetry, and SHA-256 checkpoint verification |
| **Serverless Health API** | [`https://skin-lesion-xi.vercel.app/api/health`](https://skin-lesion-xi.vercel.app/api/health) | Model readiness, SHA-256 digest (`125340ffa59ca4a1`), and OOD calibration status |
| **Serverless Diagnostics API** | [`https://skin-lesion-xi.vercel.app/api/diagnostics`](https://skin-lesion-xi.vercel.app/api/diagnostics) | Full runtime provenance, MPS memory telemetry, and OOD v2 parameters |

---

## 1. Research Overview & Scientific Objectives

While **Melanoma (`mel`)** accounts for the highest skin-cancer mortality, **Non-Melanoma Skin Cancers (NMSC)**—specifically **Basal Cell Carcinoma (`bcc`)** and **Actinic Keratoses / Intraepithelial Carcinoma (`akiec`)**—constitute the majority of clinical malignancies. Automated dermoscopic classification on **HAM10000** faces three core scientific challenges:
1. **Severe Class Imbalance ($58.3 : 1$):** Benign Melanocytic Nevi (`nv`, $6,705$ images) outnumber Dermatofibroma (`df`, $115$ images) by nearly $60\times$.
2. **Multi-Image Lesion Leakage:** HAM10000 contains $10,015$ images across $7,470$ unique `lesion_id`s. Naive image-level splitting leaks identical lesions across train and test sets.
3. **Out-of-Distribution (OOD) Vulnerability:** Standard softmax classifiers assign high confidence to non-dermoscopic photos, document scans, and corrupted inputs.

### Primary Research Question
> *How does a custom Decoupled Dual-Branch CNN—explicitly separating high-frequency texture extraction (`ShallowWideBranch`) from low-frequency structural morphology (`DeepNarrowBranch`) via Attention-Gated Fusion—compare against ImageNet-pretrained compound-scaled backbones (`EfficientNet-B4`, `DenseNet-121`, `ResNet-50`) under strict lesion-disjoint evaluation?*

---

## 2. Architectural Design

```
                      Dermoscopic Image (224×224 RGB)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      Shallow-Wide Branch                  Deep-Narrow Branch
    (Textural Extraction)                (Structural Morphology)
    Conv(3→256) + BN + ReLU              Stem: Conv(3→64, k=7, s=2) + Pool
    Conv(256→512) + MaxPool              4 Residual Stages (64→128)
    Conv(512→1024) + BN + ReLU           Bottleneck: Conv(128→256)
    AdaptiveAvgPool(1,1)                 AdaptiveAvgPool(1,1)
          1024-dim                             256-dim
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                   Versioned Attention-Gated Fusion
                 [Texture(1024) ‖ Structure(256)] → 1280-dim
        ┌───────────────────────────────────────────────────┐
        │ • v1_matrix (10.67M): Linear(1280→1280) + Sigmoid │
        │ • v2_scalar (9.03M) : Linear(1280→2)    + Softmax │
        └───────────────────────────────────────────────────┘
                 MLP: 1280 → 512 → 256 (GELU + Dropout)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        7-Class Logits Output              Counterfactual Grad-CAM
   (akiec, bcc, bkl, df, mel, nv, vasc)     (Target Class Spatial Map)
```

- **Explicit Dual-Branch Versioning (`models/fusion.py`, `models/dual_branch_net.py`):** Supports both historical checkpoint topologies with `strict=True` weight verification via `DualBranchNet.from_checkpoint()`:
  - **`v1_matrix` (`10,669,639` params):** Element-wise sigmoid gate `Linear(1280 -> 1280)` (used by Seeds `123` and `999`).
  - **`v2_scalar` (`9,031,241` params):** 2-way branch softmax scalar gate `Linear(1280 -> 2)` (used by Seed `42`).

---

## 3. Verified Empirical Benchmark Results (`splits/test.csv`, $N = 1,510$)

All 6 checkpoints were evaluated on the identical held-out, lesion-disjoint test split (`splits/test.csv`, $N = 1,510$ images across $1,109$ unique lesions, **$0$ lesion overlap** with train/val):

| Model Architecture | Fusion / Variant | Params | Balanced Acc. | Macro ROC-AUC | Macro F1 | Overall Acc. | Checkpoint SHA-256 (Short) | Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EfficientNet-B4** | Compound-Scaled (`timm`) | `17.56M` | **79.23%** | **95.91%** | **69.37%** | **73.71%** | `125340ffa59ca4a1` | **Deployed Champion** |
| **DenseNet-121** | Dense Connectivity | `6.96M` | 79.14% | 95.31% | 62.42% | 66.36% | `eb8fa6448cd0fd09` | Baseline |
| **ResNet-50** | Bottleneck Residual | `23.52M` | 75.13% | 93.52% | 53.52% | 56.62% | `484eb8e5f9c37218` | Baseline |
| **DualBranchNet (Seed 123)** | `v1_matrix` (`1280->1280`) | `10.67M` | 70.28% | **90.98%** | 48.52% | 55.56% | `0ab4ae0edbb8c6c2` | Custom Ablation |
| **DualBranchNet (Seed 42)** | `v2_scalar` (`1280->2`) | `9.03M` | 68.70% | **90.54%** | 44.91% | 53.97% | `8efcd3ee6e41c79c` | Custom Ablation |
| **DualBranchNet (Seed 999)** | `v1_matrix` (`1280->1280`) | `10.67M` | 66.39% | **89.73%** | 45.77% | 54.97% | `36607de3f7ab1a0a` | Custom Ablation |

> **Scientific Audit Reconciliation:** Earlier summary tables cited `90.98%` (Seed 123), `90.54%` (Seed 42), and `89.73%` (Seed 999) for the Dual-Branch CNN without metric labels. Direct evaluation of the saved checkpoints proved these exact numbers are **Macro ROC-AUC (%)**, while their **Balanced Accuracy** is `70.28%`, `68.70%`, and `66.39%`. Because `EfficientNet-B4` transfers `100%` ImageNet-pretrained weights across all stages (whereas `DualBranchNet` trained `99.88%` of its parameters from scratch on $7,002$ images), `EfficientNet-B4` is deployed as the verified production inference engine.

---

## 4. Calibrated Multi-Signal OOD Gate v2.0 & Apple M4 MPS Performance

### 4.1 Out-of-Distribution (OOD) Domain Gate (`api/ood_gate.py`)
Fitted strictly on training split embeddings ($N = 2,123$), threshold-tuned on validation ($N = 1,503$, $\tau = 0.2500$), and evaluated once on held-out test ($N = 1,510$ ID + $N = 200$ OOD across 10 non-dermoscopic categories):
- **4-Signal Ensemble:** Diagonal-regularized Mahalanobis distance ($35\%$) + Closest-Class Cosine Distance ($35\%$) + Temperature-Scaled Energy ($15\%$) + HSV/Border/Specular Photometric Prior ($15\%$).
- **Held-Out Test Performance:**
  - **OOD vs ID AUROC:** **`0.9985`** (`99.85%`)
  - **ID Test Acceptance Rate ($N = 1,510$):** **`98.08%`** ($1,481 / 1,510$ accepted; up from $29.67\%$ in legacy v1 heuristic)
  - **OOD Rejection Rate ($N = 200$ across 10 categories):** **`100.00%`** ($200 / 200$ rejected)

### 4.2 Apple Silicon M4 MPS Optimization (`api/inference_engine.py`)
| Metric | Baseline (Before V2) | Post-Optimization (V2) | Improvement |
| :--- | :---: | :---: | :---: |
| **Cold Start — First `/predict`** | `3,765.76 ms` | **`236.38 ms`** | **15.93× faster** (Metal shader warmup at startup) |
| **Warm `/predict` Latency (Median)** | `55.66 ms` | **`27.64 ms`** | **2.01× faster** (Single-pass OOD feature reuse) |
| **Warm `/predict/explain` Latency** | `176.50 ms` | **`79.66 ms`** | **2.22× faster** (Scoped Grad-CAM hook lifecycle) |
| **Active MPS Tensor Memory** | `157.84 MB` | **`69.95 MB`** | **-55.7% (`-87.89 MB` saved)** |

---

## 5. Canonical Class Taxonomy (HAM10000)

The canonical class index mapping used across training (`splits/*.csv`), checkpoints, and API inference is **alphabetical (`0` to `6`)**:

| Index | Code | Full Clinical Name | Risk Tier | Total Cohort ($N=10,015$) | Held-Out Test ($N=1,510$) |
| :---: | :---: | :--- | :--- | :---: | :---: |
| `0` | **`akiec`** | Actinic Keratoses & Intraepithelial Carcinoma | Pre-malignant / Malignant | 327 (3.27%) | 48 |
| `1` | **`bcc`** | Basal Cell Carcinoma | Malignant (NMSC) | 514 (5.13%) | 77 |
| `2` | **`bkl`** | Benign Keratosis-like Lesions | Benign | 1,099 (10.97%) | 173 |
| `3` | **`df`** | Dermatofibroma | Benign | 115 (1.15%) | 8 |
| `4` | **`mel`** | Melanoma | Malignant (High Risk) | 1,113 (11.11%) | 181 |
| `5` | **`nv`** | Melanocytic Nevi | Benign | 6,705 (66.95%) | 999 |
| `6` | **`vasc`** | Vascular Lesions | Benign | 142 (1.42%) | 24 |

---

## 6. Quick Start & Local Development

### 1. FastAPI Backend (`http://127.0.0.1:8000`)
```bash
git clone https://github.com/Mithran-07/SKIN-LESION.git
cd SKIN-LESION

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch FastAPI V2 server (with automatic M4 MPS shader warmup)
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 2. Next.js 14 Research Workstation (`http://127.0.0.1:3000`)
```bash
cd app/skin-lesion-app
npm install
npm run dev
```

### 3. Run Automated Verification & Test Suite
```bash
# Run full pytest suite (31 tests: API contract, OOD gate, Grad-CAM hook cleanup, DualBranch v1/v2 shapes)
.venv/bin/pytest -v

# Verify Next.js 14 production build
cd app/skin-lesion-app && npm run build
```

---

## 7. Documentation Index (`docs/`)

- [`docs/RESEARCH_AND_DESIGN_REVIEW.md`](docs/RESEARCH_AND_DESIGN_REVIEW.md) — Literature review & StitchMCP visual exploration
- [`docs/SCIENTIFIC_AUDIT_AND_REMEDIATION.md`](docs/SCIENTIFIC_AUDIT_AND_REMEDIATION.md) — Root-cause audit & verified fixes
- [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md) — End-to-end system & model architecture specification
- [`docs/EXPERIMENTS_AND_BENCHMARKS.md`](docs/EXPERIMENTS_AND_BENCHMARKS.md) — Multi-model benchmark tables, OOD v2 metrics, and M4 MPS telemetry
- [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) — FastAPI V2 & Vercel Serverless `/api/*` endpoint specification
- [`docs/DEVELOPMENT_AND_TESTING.md`](docs/DEVELOPMENT_AND_TESTING.md) — Setup, calibration scripts, and test suite guide
- [`docs/RELEASE_READINESS.md`](docs/RELEASE_READINESS.md) — Final V2 acceptance criteria & known scientific limitations

---

## 8. Important Non-Clinical Disclaimer

> **ACADEMIC RESEARCH PROTOTYPE ONLY:** This platform is strictly an Applied Deep Learning research demonstration trained on the HAM10000 dataset and is **NOT** a certified medical device. It must not be used for clinical diagnosis or patient triage. Always consult a board-certified dermatologist for medical evaluation.

---

## 9. Citation & License

```bibtex
@misc{mithran2026dermai,
  author       = {Mithran, A.},
  title        = {DermAI V2: Calibrated Dermoscopic AI and Empirical Dual-Branch CNN Research Platform},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Mithran-07/SKIN-LESION}},
  note         = {Live Platform: \url{https://skin-lesion-xi.vercel.app}}
}
```

Licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.
