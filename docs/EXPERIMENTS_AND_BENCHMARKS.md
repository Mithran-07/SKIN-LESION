# DermAI Research Platform V2 — Experiments, Benchmarks & Statistical Evaluation

**Evaluation Split:** `splits/test.csv` ($N = 1,510$ images across $1,121$ unique lesions; $0$ lesion overlap with train/val)  
**Machine-Readable Registry:** `results/verified_v2/benchmark_registry_v2.json`  
**Primary Ranking Metric:** Balanced Accuracy (mean per-class recall / sensitivity, ISIC Challenge standard)  

---

## 1. Canonical 6-Checkpoint Benchmark Table ($N = 1,510$ Held-Out Test Split)

All six trained checkpoints were evaluated deterministically using identical preprocessing (`Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)`):

| Rank | Model / Checkpoint | Architecture / Fusion Version | Params | Epoch | Balanced Acc (%) | Macro ROC-AUC (%) | Macro PR-AUC (%) | Top-1 Acc (%) | Macro F1 (%) | M4 MPS Latency (ms) | CUDA RTX 3050 Latency (ms) | Checkpoint SHA-256 Prefix |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **EfficientNet-B4 (Champion)** | `timm/efficientnet_b4` | `17,561,167` | 34 | **79.23%** | **95.91%** | **77.52%** | **73.51%** | **69.24%** | `25.86 ms` | `8.83 ms` | `125340ffa59ca4a1` |
| **#2** | **DenseNet-121 Baseline** | `timm/densenet121` | `6,961,031` | 18 | **78.83%** | **95.32%** | **75.23%** | `66.42%` | `61.99%` | `27.55 ms` | `20.37 ms` | `d6eae64e2065e097` |
| **#3** | **ResNet-50 Baseline** | `timm/resnet50` | `23,522,375` | 29 | **75.12%** | **93.58%** | **73.64%** | `56.56%` | `53.83%` | **16.09 ms** | `20.96 ms` | `84bf6ab90ca227fb` |
| **#4** | **Dual-Branch CNN (Seed 123)** | `DualBranchNet(v1_matrix)` | `10,669,639` | 48 | **70.28%** | **90.98%** | `60.45%` | `55.50%` | `48.45%` | `135.33 ms` | `27.23 ms` | `bcece2501c91947e` |
| **#5** | **Dual-Branch CNN (Seed 999)** | `DualBranchNet(v1_matrix)` | `10,669,639` | 28 | **66.42%** | **89.75%** | `58.37%` | `55.10%` | `45.72%` | `145.71 ms` | `26.32 ms` | `1b5ca7fa42e9be2b` |
| **#6** | **Dual-Branch CNN (Seed 42)** | `DualBranchNet(v2_scalar)` | `9,031,241` | 16 | **59.49%** | **90.13%** | `57.01%` | `64.30%` | `49.58%` | `145.13 ms` | `28.47 ms` | `84d950e6cf30e1c9` |

---

## 2. Per-Class Precision, Recall & F1 Comparison (Champion vs. Best Dual-Branch)

### 2.1 `EfficientNet-B4` (Rank #1 — Deployed Champion)

| Class Code | Diagnostic Category | Test Support ($N$) | Precision (%) | Recall / Sensitivity (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AKIEC** | Actinic Keratoses / Bowen's Disease | 53 | `64.81%` | `66.04%` | `65.42%` |
| **BCC** | Basal Cell Carcinoma | 59 | `63.53%` | **91.53%** | `75.00%` |
| **BKL** | Benign Keratosis-like Lesions | 186 | `50.68%` | `79.57%` | `61.92%` |
| **DF** | Dermatofibroma | 8 | `53.85%` | `87.50%` | `66.67%` |
| **MEL** | Melanoma | 181 | `42.81%` | `75.69%` | `54.69%` |
| **NV** | Melanocytic Nevi | 999 | **98.61%** | `70.97%` | **82.54%** |
| **VASC** | Vascular Lesions | 24 | `74.07%` | `83.33%` | `78.43%` |

### 2.2 `Dual-Branch CNN (Seed 123, v1_matrix)` (Rank #4 — Best Experimental Seed)

| Class Code | Diagnostic Category | Test Support ($N$) | Precision (%) | Recall / Sensitivity (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AKIEC** | Actinic Keratoses / Bowen's Disease | 53 | `36.26%` | `62.26%` | `45.83%` |
| **BCC** | Basal Cell Carcinoma | 59 | `48.11%` | `86.44%` | `61.82%` |
| **BKL** | Benign Keratosis-like Lesions | 186 | `43.30%` | `60.75%` | `50.56%` |
| **DF** | Dermatofibroma | 8 | `11.86%` | `87.50%` | `20.90%` |
| **MEL** | Melanoma | 181 | `26.91%` | `66.30%` | `38.28%` |
| **NV** | Melanocytic Nevi | 999 | `98.41%` | `49.55%` | `65.91%` |
| **VASC** | Vascular Lesions | 24 | `43.18%` | `79.17%` | `55.88%` |

---

## 3. Calibrated Out-of-Distribution (OOD) Gate v2 Evaluation (`results/verified_v2/ood_evaluation_v2.json`)

- **Fitting Set:** `splits/train.csv` ($N = 2,123$ stratified training images; 49 sub-cluster prototypes + diagonal Mahalanobis statistics).
- **Calibration Set:** `splits/val.csv` ($N = 1,503$ ID images + $N = 100$ validation OOD images) $\to$ selected threshold $\tau = 0.2500$.
- **Held-Out Evaluation Set:** Untouched `splits/test.csv` ($N = 1,510$ ID images) vs. $N = 200$ independent OOD test images across 10 categories:
  - **Held-Out AUROC:** `0.9985` (95% Bootstrap CI: `0.9972 – 0.9994`)
  - **Held-Out AUPRC:** `0.9998`
  - **In-Domain Acceptance Rate:** `98.08%` ($1,481 / 1,510$, 95% CI: `97.29% – 98.68%`; `1.92%` False Rejection Rate)
  - **OOD Rejection Rate:** `100.00%` ($200 / 200$ across all 10 categories; `0.00%` False Acceptance Rate)

| OOD Test Category | Samples ($N$) | Rejected | Rejection Rate (%) | Mean Domain Score | Max Domain Score ($\tau = 0.2500$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Office Documents & UI Screenshots | 20 | 20 | `100.0%` | `0.0339` | `0.0387` |
| Natural Landscapes & Sky | 20 | 20 | `100.0%` | `0.0465` | `0.0519` |
| Everyday Objects & Geometric | 20 | 20 | `100.0%` | `0.0759` | `0.1802` |
| Grayscale Radiograph / Ultrasound | 20 | 20 | `100.0%` | `0.0449` | `0.0633` |
| Non-Dermoscopic Macro Textures | 20 | 20 | `100.0%` | `0.0998` | `0.1908` |
| Gaussian & Impulse Noise | 20 | 20 | `100.0%` | `0.0572` | `0.1495` |
| Solid & Blank Frames | 20 | 20 | `100.0%` | `0.0167` | `0.0205` |
| Color Gradients & Digital Art | 20 | 20 | `100.0%` | `0.0477` | `0.0484` |
| Near-OOD: Warm Organic / Leather | 20 | 20 | `100.0%` | `0.1383` | `0.1782` |
| Near-OOD: Non-Dermoscopic Skin | 20 | 20 | `100.0%` | `0.1855` | `0.1965` |

---

## 4. Apple Silicon M4 MPS Latency & Memory Benchmarks (`results/verified_v2/post_optimization_performance_profile.json`)

| Metric | Legacy V1 Engine | Optimized V2 Engine | Improvement |
| :--- | :---: | :---: | :---: |
| **First User `/predict` Latency (Post-Startup)** | `3,765.76 ms` | **236.38 ms** | **15.93× faster** |
| **First User `/predict/explain` Latency** | `6,396.38 ms` | **462.34 ms** | **13.83× faster** |
| **Warm `/predict` Median Latency (P50 / P95)** | `36.64 ms` (`41.82 ms`) | **27.64 ms** (`32.06 ms`) | **24.6% faster** |
| **Warm `/predict/explain` Median Latency (P50 / P95)** | `165.71 ms` (`185.12 ms`) | **136.81 ms** (`158.97 ms`) | **17.4% faster** |
| **Active MPS Tensor Memory (After 25 Requests)** | `157.84 MB` | **69.95 MB** | **-87.89 MB (-55.7%)** |
