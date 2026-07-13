# ADL — Advanced Deep Learning for Non-Melanoma Dermoscopic Classification

A sophisticated dual-branch CNN framework for automated dermoscopic skin lesion classification, targeting BCC, SCC, and AKIEC, optimized for clinical deployment.

## Architecture Overview

```
Dermoscopic Image (224×224 RGB)
         │
    ┌────┴────┐
    │         │
Shallow-Wide  Deep-Narrow
 (Texture)    (Structure)
 1024-dim     256-dim
    │         │
    └────┬────┘
  Attention Fusion
      256-dim
    ┌────┴────┐
    │         │
  Class    Seg Mask
 (7 cls)  (pixel-wise)
```

## Key Features

- **Dual-Branch CNN**: Physically decoupled texture (shallow-wide) and structure (deep-narrow) branches
- **Focal Loss**: Per-class α-weighted, γ-modulated to handle extreme class imbalance in HAM10000
- **Grad-CAM**: Dual heatmaps per branch for clinical interpretability
- **Conformal Prediction**: Mathematically guaranteed prediction sets (90% coverage by default)
- **Multi-Task Learning**: Joint segmentation + classification head
- **Federated Learning Scaffold**: Privacy-preserving training architecture (stub)

## Setup

### MacBook M4 (MPS)
```bash
conda create -n adl python=3.11 -y
conda activate adl
pip install -r requirements.txt
```

### Lenovo LOQ (CUDA)
```bash
conda create -n adl python=3.11 -y
conda activate adl
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Dataset Setup

```bash
# Download HAM10000 via ISIC API
python data/download.py --dataset ham10000 --output data/raw/

# Or manually from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# Extract to: data/raw/HAM10000/
```

## Quick Start

```bash
# Train dual-branch model
python scripts/train.py --config config/config.yaml --model dual_branch

# Evaluate all models
python scripts/evaluate.py --checkpoint results/best_dual_branch.pt

# Single image inference with Grad-CAM + Conformal Prediction
python scripts/infer.py --image path/to/lesion.jpg --checkpoint results/best_dual_branch.pt
```

## Project Structure

```
ADL/
├── config/config.yaml          # All hyperparameters
├── data/                       # Dataset classes & augmentations
├── models/                     # All model architectures
│   ├── shallow_wide_branch.py  # Texture branch
│   ├── deep_narrow_branch.py   # Structure branch
│   ├── fusion.py               # Attention-gated fusion
│   ├── dual_branch_net.py      # Full classifier
│   ├── mtl_head.py             # Multi-task head
│   └── baselines/              # ResNet50, DenseNet201, EfficientNet
├── losses/                     # Focal loss, MTL loss
├── training/                   # Trainer, scheduler, metrics
├── explainability/             # Grad-CAM, visualization
├── uncertainty/                # Conformal prediction, MC-Dropout
├── federated/                  # FL scaffold (stub)
├── scripts/                    # train.py, evaluate.py, infer.py
├── notebooks/                  # 5 Jupyter notebooks
└── tests/                      # pytest test suite
```

## Class Labels (HAM10000)

| Index | Code | Full Name | Type |
|-------|------|-----------|------|
| 0 | MEL | Melanoma | Malignant |
| 1 | NV | Melanocytic Nevi | Benign |
| 2 | BCC | Basal Cell Carcinoma | Malignant (NMSC) |
| 3 | AKIEC | Actinic Keratosis / Intraepithelial Carcinoma | Pre-malignant |
| 4 | BKL | Benign Keratosis-like Lesions | Benign |
| 5 | DF | Dermatofibroma | Benign |
| 6 | VASC | Vascular Lesions | Benign |

## Citation

If using this framework in academic work, please cite the HAM10000 dataset:
> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci. Data* 5, 180161 (2018).
