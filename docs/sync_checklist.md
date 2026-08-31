# Repository Synchronization Checklist

This document details the exact synchronization procedure for handoff from the Lenovo LOQ (Compute Archive) to the MacBook M4 (Master Research Repository).

## 1. Source Code (Copy)
- [ ] `config/`
- [ ] `models/`
- [ ] `training/`
- [ ] `losses/`
- [ ] `data/` (source code only, strictly no datasets)
- [ ] `utils/`
- [ ] `scripts/`
- [ ] `tests/`
- [ ] `visualization/`
- [ ] `uncertainty/`
- [ ] `explainability/`
- [ ] `federated/`

## 2. Documentation (Copy)
- [ ] `docs/`
- [ ] `paper/`
- [ ] `thesis/`
- [ ] `research_journal/`

## 3. Configuration (Copy)
- [ ] `requirements.txt`
- [ ] `requirements-lock.txt`
- [ ] `environment.yml`
- [ ] `pyproject.toml`
- [ ] `.gitignore`
- [ ] `config/experiments/`

## 4. Research Results (Copy from `results/final/`)
- [ ] `benchmark_final.csv`
- [ ] `comparison_table.csv`
- [ ] `comparison_table.md`
- [ ] `summary_metrics.json`
- [ ] `experiment_summary.md`
- [ ] `FINAL_RESULTS.md`
- [ ] `README.md`
- [ ] All publication figures (PNG/SVG/PDF)

## 5. Metadata (Copy)
- [ ] `MODEL_REGISTRY.md`
- [ ] `PROJECT_HANDOFF.md`
- [ ] `REPRODUCIBILITY.md`
- [ ] `manifest.json`
- [ ] `CITATION.cff`
- [ ] `LICENSE`
- [ ] `README.md`

## 6. Optional Backup (Recommended Checkpoints)
**Recommendation**: *We highly recommend copying the final `.pth` weights for the core baseline and our proposed variants to enable rapid inference and Grad-CAM generation without needing access to the LOQ archive.*
If copied, place them in `archive/checkpoints/`:
- [ ] `best_resnet50.pth`
- [ ] `best_densenet121.pth`
- [ ] `best_efficientnet_b4.pth`
- [ ] `best_dual_branch_v1.pth`
- [ ] `best_dual_branch_v1_1.pth`
- [ ] `best_dual_branch_v2.pth`
*(Note: Do NOT copy intermediate epoch checkpoints to conserve disk space.)*

## 7. Explicit Ignore List
The following directories MUST NOT be copied to the MacBook to preserve repository hygiene:
- 🚫 `datasets/`
- 🚫 `tensorboard/`
- 🚫 `cache/`
- 🚫 `logs/`
- 🚫 `.venv/`
- 🚫 `__pycache__/`
- 🚫 Temporary files
- 🚫 `artifacts/checkpoints.zip` (unless generating a full redundant backup)
