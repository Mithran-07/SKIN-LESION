# MAC Final Audit — Dual-Branch CNN Research Project
Date: 2026-08-31

## 1. Repository Overview
This is the Lenovo LOQ workstation repository containing:
- Research code (models, training, data, losses, scripts)
- Trained checkpoints (efficientnet_b4: 201MB, resnet50, densenet121, dual_branch variants)
- Results: benchmark.csv, classification reports, confusion matrices, ROC/PR curves, learning curves
- Explainability infrastructure: gradcam.py (Dual-Branch) + newly added EfficientNet Grad-CAM
- GradCAM note: Dual-Branch Grad-CAM exists. EfficientNet Grad-CAM added to api/inference_engine.py

## 2. MISSING — Being Created Now
- api/main.py         FastAPI backend [CREATED]
- api/inference_engine.py  EfficientNet inference + Grad-CAM [CREATED]
- app/skin-lesion-app/     Next.js frontend [IN PROGRESS]
- docs/               Documentation [IN PROGRESS]

## 3. Checkpoint Status
- EfficientNet-B4: checkpoints/efficientnet_b4/best_checkpoint.pth (201MB) OK
- Checkpoint key: 'state_dict' (verified)
- Device: CUDA on LOQ, CPU/MPS on Mac

## 4. GradCAM Status
The repository contains Grad-CAM infrastructure, but explainability experiments
were intentionally excluded from the final experimental protocol and are reserved
for future work. EfficientNet-B4 Grad-CAM has been implemented in api/inference_engine.py
as part of the application demonstration layer.

## 5. Final Benchmark (Verified)
| Model | Accuracy | Balanced Acc | F1 | AUC |
|---|---|---|---|---|
| EfficientNet-B4 | 73.64% | 79.16% | 69.19% | 95.92% |
| DenseNet121 | 66.36% | 79.14% | 62.42% | 95.31% |
| ResNet50 | 56.62% | 75.13% | 53.52% | 93.52% |
| Dual-Branch V1.1 | 65.76% | 62.18% | 48.14% | 90.06% |
| Dual-Branch V2 | 64.24% | 59.48% | 49.50% | 90.15% |
