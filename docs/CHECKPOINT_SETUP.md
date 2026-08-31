# EfficientNet-B4 Checkpoint Installation Guide

**Model**: EfficientNet-B4 (HAM10000 7-Class Dermoscopic Classifier)  
**Expected Checkpoint Location**: `checkpoints/efficientnet_b4/best_checkpoint.pth`  
**Model Parameters**: 17,561,167 (~68 MB file size)  
**Verification SHA-256 Checksum**: `artifacts/checksums.txt`

---

## 1. Checkpoint Status
Large PyTorch model binary weights (`*.pth`, `*.pt`, `*.ckpt`) are excluded from Git repository history by design to maintain a lightweight codebase.

- **Current State**: When running on the MacBook without the local binary checkpoint, the inference engine automatically initializes the compound-scaled **EfficientNet-B4** architecture with pre-trained feature extraction weights, permitting 100% functional validation of the API, Grad-CAM attribution pipeline, and frontend UI without errors.
- **Production State**: When copying the trained checkpoint from the Lenovo LOQ training workstation, place the file in the designated path below.

---

## 2. Checkpoint Placement Instructions

1. On the Lenovo LOQ, locate the trained checkpoint file:
   ```
   C:\ADL\checkpoints\efficientnet_b4\best_checkpoint.pth
   ```
2. Copy the file to the MacBook into the following directory:
   ```bash
   mkdir -p checkpoints/efficientnet_b4
   cp /path/to/copied/best_checkpoint.pth checkpoints/efficientnet_b4/best_checkpoint.pth
   ```
3. Verify file placement and integrity:
   ```bash
   ls -lh checkpoints/efficientnet_b4/best_checkpoint.pth
   shasum -a 256 checkpoints/efficientnet_b4/best_checkpoint.pth
   ```

---

## 3. Automated Device Loading on MacBook M4
The inference engine in `api/inference_engine.py` automatically maps weights across heterogeneous hardware:
- **Apple Silicon (MacBook M4)**: `map_location=torch.device("mps")`
- **NVIDIA GPU (Lenovo LOQ)**: `map_location=torch.device("cuda")`
- **Standard CPU**: `map_location=torch.device("cpu")`
