# Chapter 4: Experimental Results and Evaluation

## 4.1 Overview of Experimental Protocol

All experiments were executed on the HAM10000 benchmark dataset comprising 10,015 dermoscopic images categorized into 7 diagnostic classes. To ensure scientific rigor and prevent patient-level data leakage, datasets were partitioned using patient-aware stratified sampling grouped on `lesion_id` into training (70%), validation (15%), and testing (15%) subsets.

Models were optimized using AdamW ($\text{lr}=10^{-4}$, $\text{weight\_decay}=10^{-4}$) with a cosine annealing learning rate scheduler with linear warmup. To address the severe 58:1 class imbalance between Melanocytic Nevi (NV) and rare classes (DF, VASC), class-weighted Focal Loss ($\gamma=2.0$) was employed across all architectures.

---

## 4.2 Comprehensive Benchmark Results

The empirical performance across standard single-branch transfer learning baselines and the proposed Decoupled Dual-Branch CNN architectures is summarized below:

| Architecture | Model Parameters | Macro ROC-AUC | Balanced Accuracy | Top-1 Test Accuracy | Macro F1 Score | Inference Latency (ms) | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|
| **EfficientNet-B4** | **17.56M** | **95.92%** | **79.16%** | **73.64%** | **69.19%** | **8.83** | **677.7** |
| **DenseNet-121** | 6.96M | 95.31% | 79.14% | 66.36% | 62.42% | 20.37 | 393.7 |
| **Dual-Branch V1.1 (Optimized Training)** | 10.67M | 90.06% | 62.18% | 65.76% | 48.14% | 24.56 | 1561.1 |
| **Dual-Branch V2 (Refined Topology)** | 9.03M | 90.15% | 59.48% | 64.24% | 49.50% | 25.54 | 1516.7 |
| **Dual-Branch V1 (Seed 123)** | 10.67M | 90.98% | 70.31% | 55.50% | 48.55% | 27.23 | 1561.1 |
| **Dual-Branch V1 (Seed 999)** | 10.67M | 89.73% | 66.39% | 54.97% | 45.77% | 26.32 | 1561.1 |
| **Dual-Branch V1 (Seed 42)** | 10.67M | 90.54% | 68.62% | 53.91% | 44.90% | 28.47 | 1561.1 |
| **ResNet-50** | 23.52M | 93.52% | 75.13% | 56.62% | 53.52% | 20.96 | 900.3 |

---

## 4.3 Multi-Seed Statistical Validation

To ensure that the performance of the Dual-Branch CNN was not an artifact of random weight initialization or batch ordering, Dual-Branch V1 was trained across 3 independent random seeds:
- Seed 42: Top-1 Accuracy = 53.91%, Balanced Accuracy = 68.62%, Macro F1 = 44.90%, ROC-AUC = 90.54%
- Seed 123: Top-1 Accuracy = 55.50%, Balanced Accuracy = 70.31%, Macro F1 = 48.55%, ROC-AUC = 90.98%
- Seed 999: Top-1 Accuracy = 54.97%, Balanced Accuracy = 66.39%, Macro F1 = 45.77%, ROC-AUC = 89.73%

**Mean Performance**:
- **Accuracy**: $54.79\% \pm 0.81\%$
- **Balanced Accuracy**: $68.44\% \pm 1.97\%$
- **Macro ROC-AUC**: $90.42\% \pm 0.63\%$

---

## 4.4 Diagnostic Evaluation of the Negative Finding

A core contribution of this thesis is the rigorous scientific investigation of why the Decoupled Dual-Branch CNN hypothesis did not surpass the compound-scaled EfficientNet-B4 baseline:

1. **Attention Gate Imbalance (Fusion Collapse)**:
   - Saliency analysis of the attention fusion gate demonstrated that over 78% of gate weights were allocated to the deep structural branch across all diagnostic classes.
   - Consequently, the high-dimensional shallow texture branch (1024 channels) received diluted gradient feedback, limiting the model's ability to learn fine dermoscopic patterns.
2. **Compound Scaling Advantages**:
   - EfficientNet-B4 uniformly scales network depth, channel width, and image resolution using principled compound scaling coefficients.
   - This enables EfficientNet-B4 to capture both micro-scale textures and macro-scale lesion boundaries within a unified, computationally efficient feature hierarchy.
3. **Deployment Strategy**:
   - Adhering to empirical truth, EfficientNet-B4 was selected as the deployment engine for the production clinical screening prototype, paired with Grad-CAM attribution overlays.
