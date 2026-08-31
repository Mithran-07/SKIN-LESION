# Chapter 5: Conclusion, Limitations, and Future Work

## 5.1 Summary of Contributions

This dissertation investigated the design, empirical evaluation, and clinical translation of deep convolutional neural networks for automated dermoscopic skin lesion diagnosis on the HAM10000 benchmark dataset.

The key contributions of this research are:
1. **Architectural Formulation**: Designed and implemented a novel Decoupled Dual-Branch CNN architecture intended to separate high-frequency surface texture features (Shallow-Wide branch, 1024 channels) from macroscopic structural morphology (Deep-Narrow branch, 256 channels) using attention-gated fusion.
2. **Empirical Benchmarking**: Conducted comprehensive comparisons against strong single-branch baselines (EfficientNet-B4, DenseNet-121, ResNet-50) using patient-aware stratified splitting and class-weighted Focal Loss.
3. **Rigorous Negative Finding Analysis**: Documented that compound-scaled EfficientNet-B4 attained superior classification performance (**95.92% ROC-AUC, 79.16% Balanced Accuracy, 73.64% Overall Accuracy**) compared to Dual-Branch variants, explaining the failure mechanism through fusion gate diagnostic analysis.
4. **End-to-End Clinical Screening Prototype**: Developed and deployed a fully functional FastAPI backend and Next.js 14 web client featuring real-time inference and Grad-CAM spatial model attribution visualizations.

---

## 5.2 Limitations

1. **Dataset Demographics**: HAM10000 predominantly comprises images from European, fair-skinned populations (Fitzpatrick skin types I–III). Generalization to skin types IV–VI requires prospective multi-ethnic cohort validation.
2. **Severe Class Imbalance**: The high prevalence of Melanocytic Nevi (67.0%) relative to rare malignancies such as Dermatofibroma (1.1%) presents persistent optimization challenges.
3. **Non-Clinical Clearance**: The deployed system is an academic research prototype and has not been cleared as a Software as a Medical Device (SaMD).

---

## 5.3 Future Work

1. **Vision Transformers & Swin Backbones**: Exploring shifted-window self-attention mechanisms to dynamically capture local textures and global context without static branch decoupling.
2. **Multimodal Metadata Integration**: Fusing patient demographic data, anatomical location, and clinical lesion history directly into the deep feature representation.
3. **Prospective Clinical Trials**: Conducting reader studies with practicing dermatologists to assess diagnostic accuracy improvements in real-world clinical workflows.
