# Duplicate File Audit Report

**Repository**: `Mithran-07/SKIN-LESION`  
**Date**: August 31, 2026  
**Methodology**: Automated SHA-256 cryptographic hash computation across all 450 tracked files.

---

## 1. Duplicate Sets Identified via Cryptographic Hash

### Set 1: Nested Duplicate Directory (`repo_temp/SKIN-LESION-main/`)
* **Matched Files**: 23 identical files across `models/`, `losses/`, `tests/`, `explainability/`, `uncertainty/`, `utils/`, `training/`, and `data/`.
* **Example Hashes**:
  * `7514c9f9...`: `models/deep_narrow_branch.py` ↔ `repo_temp/SKIN-LESION-main/models/deep_narrow_branch.py`
  * `b87948be...`: `models/dual_branch_net.py` ↔ `repo_temp/SKIN-LESION-main/models/dual_branch_net.py`
  * `adfcaa71...`: `losses/focal_loss.py` ↔ `repo_temp/SKIN-LESION-main/losses/focal_loss.py`
  * `7bad07c8...`: `tests/test_model_shapes.py` ↔ `repo_temp/SKIN-LESION-main/tests/test_model_shapes.py`
* **Root Cause**: An unzipped archive from an earlier release was committed to `repo_temp/`.
* **Recommendation**: **DELETE `repo_temp/` completely**. All primary files are maintained and actively tested in top-level packages.

---

### Set 2: Frontend Static Assets vs Data Samples
* **Matched Files**: 7 sample images (`akiec_sample.jpg`, `bcc_sample.jpg`, `bkl_sample.jpg`, `df_sample.jpg`, `mel_sample.jpg`, `nv_sample.jpg`, `vasc_sample.jpg`).
* **Locations**: `data/samples/` (Python backend evaluation) ↔ `app/skin-lesion-app/public/samples/` (Next.js public assets for instant demo).
* **Evaluation**: **INTENTIONAL REPLICATION**. Next.js serves static assets directly from `public/`, while Python CLI evaluation scripts reference `data/samples/`. Keeping both ensures full decoupling between frontend and backend.
* **Recommendation**: **RETAIN BOTH**.

---

### Set 3: Benchmark CSV Files
* **Matched Files**: `results/benchmark.csv` ↔ `results/final/benchmark_final.csv` ↔ `results/final/comparison_table.csv`.
* **SHA-256 Hash**: `97a56ac6110f01198c258d4a46083f2dc5a7f9a888c3a165337e6f3d1eecfa34` (1,652 bytes).
* **Evaluation**: `results/benchmark.csv` is the canonical runtime file loaded by `api/main.py` and `scripts/compare_models.py`, while `results/final/` serves as the historical publication snapshot referenced by LaTeX paper tables.
* **Recommendation**: **RETAIN BOTH**.

---

### Set 4: 0-Byte Redundant Files
* **Matched Files**: `LICENSE` (0 bytes), `environment.yml` (0 bytes), root `manifest.json` (0 bytes).
* **Recommendation**:
  * Populate `LICENSE` with standard MIT License.
  * Populate `environment.yml` with clean Conda environment specification.
  * Remove redundant 0-byte root `manifest.json` (canonical JSON manifest exists at `results/final/manifest.json`).

---

## 2. Summary of Actionable Removals
- **Directory to Remove**: `repo_temp/` (saves clutter, removes 23 duplicate files).
- **Redundant Root Files to Archive / Remove**: `SKIN-LESION.zip`, `manifest.json` (0-byte in root).
