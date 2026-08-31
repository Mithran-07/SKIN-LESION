# Results & Empirical Analysis

Please refer to the comprehensive [paper/sections/results.md](sections/results.md) for full benchmark tables, multi-seed statistical analyses, and gate diagnostics.

### Summary Ground Truth Metrics

| Model | ROC-AUC | Balanced Accuracy | Test Accuracy | Macro F1 | Status |
|---|---|---|---|---|---|
| **EfficientNet-B4** | **95.92%** | **79.16%** | **73.64%** | **69.19%** | **Best Baseline (Deployed)** |
| **DenseNet-121** | 95.31% | 79.14% | 66.36% | 62.42% | Baseline |
| **Dual-Branch V1.1** | 90.06% | 62.18% | 65.76% | 48.14% | Research Variant |
| **Dual-Branch V2** | 90.15% | 59.48% | 64.24% | 49.50% | Research Variant |
| **Dual-Branch V1** | 90.42% ± 0.63% | 68.44% ± 1.97% | 54.79% ± 0.81% | 46.41% | Research Baseline |
| **ResNet-50** | 93.52% | 75.13% | 56.62% | 53.52% | Baseline |
