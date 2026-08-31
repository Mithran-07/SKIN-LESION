# Mac Transfer Manifest

This document specifies exactly how to transfer the research project from the Lenovo LOQ workstation to the MacBook development environment.

---

## 1. Transfer Categories

### Category A: Git Synchronization (Automatic via `git pull origin main`)
The following lightweight directories and files are committed to Git and will automatically sync when pulling the repository:

- `api/` — FastAPI backend (`main.py`, `inference_engine.py`)
- `app/skin-lesion-app/` — Next.js web application (source, components, pages, config)
- `models/` — PyTorch neural network definitions (`baseline.py`, `dual_branch_net.py`, `fusion.py`, `deep_narrow_branch.py`, `shallow_wide_branch.py`, `mtl_head.py`)
- `data/` — Data loading, dataset splitting, validation, and augmentation code
- `training/` — Training scaffold (`trainer.py`, `metrics.py`, `visualizer.py`)
- `losses/` — Loss functions (`focal_loss.py`, `mtl_loss.py`)
- `configs/` — YAML configurations (`baseline_config.yaml`)
- `scripts/` — Audit, evaluation, diagnostic, and visualization scripts
- `explainability/` — Dual-Branch and EfficientNet Grad-CAM modules (`gradcam.py`, `visualize.py`)
- `uncertainty/` — Conformal prediction and MC Dropout modules
- `tests/` — Unit and integration tests (`test_api.py`, `test_model_shapes.py`, etc.)
- `results/final/` — Final publication figures, CSV benchmark tables, and markdown reports
- `docs/` — Architecture specs, deployment guide, demo script, MAC audit report
- `requirements.txt`, `environment.yml`, `README.md`, `CITATION.cff`, `MODEL_REGISTRY.md`, `FINAL_RESULTS.md`, `REPRODUCIBILITY.md`

---

### Category B: Manual Transfer (Copy via AirDrop / USB / Cloud Drive)
The following binary checkpoint file is too large for Git history (202 MB) and **must be copied manually**:

| File | Size | Destination on Mac | SHA-256 Checksum |
|---|---|---|---|
| `checkpoints/efficientnet_b4/best_checkpoint.pth` | 202.05 MB | `checkpoints/efficientnet_b4/best_checkpoint.pth` | `125340FFA59CA4A18FB0CD0BDCA024EAE9CA2FE3CE54D1F230B465BD9AD9658E` |

*Verification command on Mac:*
```bash
shasum -a 256 checkpoints/efficientnet_b4/best_checkpoint.pth
```
Compare output against `artifacts/efficientnet_b4_checkpoint.sha256`.

---

### Category C: Never Transfer (Excluded via `.gitignore`)
Do NOT copy or commit the following dynamic, environment-specific, or massive data directories:

- `datasets/` — Raw HAM10000, ISIC2019, ISIC2018 image datasets (10GB+)
- `checkpoints/` — Other baseline checkpoints (ResNet50, DenseNet121, Dual-Branch seeds) unless specifically needed
- `logs/` & `tensorboard/` — Training run logs and event files
- `.venv/` / `__pycache__/` / `.next/` / `node_modules/` — Local environment & build caches
- `cache/` & `tmp/` — Temporary files

---

## 2. MacBook Setup Instructions

1. **Clone repository:**
   ```bash
   git clone https://github.com/Mithran-07/SKIN-LESION.git
   cd SKIN-LESION
   ```

2. **Copy checkpoint:**
   Place `best_checkpoint.pth` into `checkpoints/efficientnet_b4/best_checkpoint.pth`.

3. **Verify Checkpoint:**
   ```bash
   shasum -a 256 checkpoints/efficientnet_b4/best_checkpoint.pth
   # Output must match: 125340FFA59CA4A18FB0CD0BDCA024EAE9CA2FE3CE54D1F230B465BD9AD9658E
   ```

4. **Install Python Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Install Node.js Frontend Dependencies:**
   ```bash
   cd app/skin-lesion-app
   npm install
   ```

6. **Run Demonstration Application:**
   - **Terminal 1 (Backend):**
     ```bash
     python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
     ```
   - **Terminal 2 (Frontend):**
     ```bash
     cd app/skin-lesion-app
     npm run dev
     ```
   - Open browser at `http://localhost:3000`.
