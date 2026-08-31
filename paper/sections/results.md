# Results & Empirical Findings

## 1. Quantitative Benchmark Comparison

We evaluated the Decoupled Dual-Branch CNN framework and standard single-branch transfer learning baselines on the HAM10000 7-class dermoscopic classification benchmark under identical patient-aware stratified train/val/test splits (70/15/15) and class-weighted Focal Loss ($\gamma=2.0$).

### Overall Performance Comparison Table

| Architecture | Total Parameters | Macro ROC-AUC | Balanced Accuracy | Top-1 Accuracy | Macro F1 | Inference Latency | Peak VRAM |
|---|---|---|---|---|---|---|---|
| **EfficientNet-B4** | **17.56M** | **95.92%** | **79.16%** | **73.64%** | **69.19%** | **8.83 ms** | **677.7 MB** |
| **DenseNet-121** | 6.96M | 95.31% | 79.14% | 66.36% | 62.42% | 20.37 ms | 393.7 MB |
| **Dual-Branch V1.1 (Optimized Training)** | 10.67M | 90.06% | 62.18% | 65.76% | 48.14% | 24.56 ms | 1561.1 MB |
| **Dual-Branch V2 (Refined Architecture)** | 9.03M | 90.15% | 59.48% | 64.24% | 49.50% | 25.54 ms | 1516.7 MB |
| **Dual-Branch V1 (Seed 123)** | 10.67M | 90.98% | 70.31% | 55.50% | 48.55% | 27.23 ms | 1561.1 MB |
| **Dual-Branch V1 (Seed 999)** | 10.67M | 89.73% | 66.39% | 54.97% | 45.77% | 26.32 ms | 1561.1 MB |
| **Dual-Branch V1 (Seed 42)** | 10.67M | 90.54% | 68.62% | 53.91% | 44.90% | 28.47 ms | 1561.1 MB |
| **ResNet-50** | 23.52M | 93.52% | 75.13% | 56.62% | 53.52% | 20.96 ms | 900.3 MB |

---

## 2. Multi-Seed Robustness Analysis (Dual-Branch V1)

To evaluate statistical stability, Dual-Branch V1 was trained across three random seeds:
- **Seed 42**: Accuracy = 53.91%, Balanced Accuracy = 68.62%, Macro F1 = 44.90%, ROC-AUC = 90.54%
- **Seed 123**: Accuracy = 55.50%, Balanced Accuracy = 70.31%, Macro F1 = 48.55%, ROC-AUC = 90.98%
- **Seed 999**: Accuracy = 54.97%, Balanced Accuracy = 66.39%, Macro F1 = 45.77%, ROC-AUC = 89.73%
- **Multi-Seed Mean**: Accuracy = **54.79% ± 0.81%**, Balanced Accuracy = **68.44% ± 1.97%**, ROC-AUC = **90.42% ± 0.63%**

---

## 3. Analysis of the Negative Research Finding

### Why EfficientNet-B4 Outperformed Dual-Branch Architectures
1. **Compound Scaling Superiority**: EfficientNet-B4 utilizes principled joint scaling of depth, width, and input resolution ($224 \times 224$ with MBConv depthwise separable blocks), which preserves hierarchical spatial features across both fine texture and global lesion context without requiring artificial topological decoupling.
2. **Attention Gate Imbalance**: Empirical gate diagnostics revealed that the attention fusion module consistently assigned over 78% of learned gate weights to the deep structural branch, leaving the high-dimensional shallow texture branch (1024-dim) under-leveraged during gradient backpropagation.
3. **Inference Latency & Efficiency**: EfficientNet-B4 achieved a 3x faster inference latency (8.83 ms vs 27.23 ms) and 2.3x lower peak VRAM consumption (677.7 MB vs 1561.1 MB) compared to the Dual-Branch network.

---

## 4. Production Deployment Decision

Based on strict scientific and engineering criteria:
- **Deployed Inference Engine**: **EfficientNet-B4**
- **Explainability**: Integrated Grad-CAM attribution overlay on the final convolutional feature head.
- **Serving Architecture**: FastAPI REST backend paired with Next.js 14 web client.
