# Complete Repository Audit & File Classification

**Repository**: `Mithran-07/SKIN-LESION`  
**Project**: Dual-Branch CNN Framework for Non-Melanoma Dermoscopic Classification  
**Date**: August 31, 2026  
**Auditor**: Senior Software Architect & Repository Maintainer  

---

## 1. Executive Summary

This comprehensive audit evaluates all directories and files across the repository to prepare it for a clean, professional, publication- and college-submission-grade release.

- **Total Tracked & Source Files**: 450 files across 27 directories.
- **Identified Redundant / Duplicate Directories**: `repo_temp/` (unpacked clone of old July commit; 100% duplicate of top-level source files).
- **Identified Root Artifacts for Archival**: `SKIN-LESION.zip`, `GIT_PUSH_CHECKLIST.md`, `env_report.md`, `MASTER_REPOSITORY.md`.
- **Zero-Byte Files Requiring Resolution**: `LICENSE` (populated with MIT License), `environment.yml` (populated with clean Conda spec), root `manifest.json` (redundant; canonical exists in `results/final/manifest.json`).
- **Secrets / Sensitive Tokens**: Zero secrets or credentials detected.

---

## 2. Directory Purpose & Structure Inventory

| Directory | Purpose | Category | Action |
|---|---|---|---|
| `api/` | Production FastAPI backend, EfficientNet-B4 inference engine & Grad-CAM attribution generator. | Source Code (Core) | **MUST KEEP** |
| `app/skin-lesion-app/` | Next.js 14 interactive web application (Home, Classify, Dashboard, Research, Architecture). | Frontend Application | **MUST KEEP** |
| `config/` & `configs/` | YAML experiment configurations for baseline and dual-branch training. | Configurations | **MUST KEEP** |
| `data/` | Dataset loading, patient-aware splitting, augmentations, and sample evaluation assets. | Source Code | **MUST KEEP** |
| `explainability/` | Grad-CAM hooks and visualization utilities. | Source Code | **MUST KEEP** |
| `losses/` | Focal Loss and Multi-Task Loss implementations. | Source Code | **MUST KEEP** |
| `models/` | DualBranchNet, ShallowWideBranch, DeepNarrowBranch, AttentionFusion, and Baselines. | Source Code (Core) | **MUST KEEP** |
| `notebooks/` | Interactive Jupyter notebooks for EDA and model exploration. | Research Tool | **MUST KEEP** |
| `paper/` | Complete academic research paper sections, figures, and bibliography (`references.bib`). | Academic Publication | **MUST KEEP** |
| `thesis/` | Comprehensive 5-chapter thesis manuscript and dissertation frontmatter. | Academic Thesis | **MUST KEEP** |
| `results/` & `results/final/` | Verified benchmark CSVs, training curves, confusion matrices, ROC/PR figures. | Empirical Research | **MUST KEEP** |
| `scripts/` | CLI utilities for training, evaluation, benchmark analysis, and test suites. | Source Code | **MUST KEEP** |
| `splits/` | Patient-aware train/val/test CSV index splits. | Reproducibility Data | **MUST KEEP** |
| `tests/` | Automated unit tests (shapes, loss, conformal prediction, API tests). | Quality Assurance | **MUST KEEP** |
| `training/` | Training loop, metric calculation, and learning rate schedulers. | Source Code | **MUST KEEP** |
| `uncertainty/` | Split Conformal Prediction and MC-Dropout uncertainty quantification. | Source Code | **MUST KEEP** |
| `utils/` | Device management, config loader, logging, checkpointing, and statistics. | Source Code | **MUST KEEP** |
| `visualization/` | Architecture, feature map, and curve generation scripts. | Source Code | **MUST KEEP** |
| `docs/` | Architectural guides, runbooks, checklists, and viva demonstration guides. | Documentation | **MUST KEEP** |
| `repo_temp/` | Temporary unpacked duplicate copy of old July commit. | Redundant Duplicate | **ARCHIVE / REMOVE** |
| `artifacts/` | Checksum files and local release verification hashes. | Verification Metadata| **KEEP** |

---

## 3. Comprehensive File Classification Table

| Path | Type | Purpose | Used By | Category | Recommendation | Reason |
|---|---|---|---|---|---|---|
| `api/main.py` | Python (FastAPI) | Core API server & endpoint definitions | Next.js Frontend / CLI | A. MUST KEEP | Keep | Primary backend entry point |
| `api/inference_engine.py` | Python | EfficientNet-B4 inference & Grad-CAM | `api/main.py`, `tests/` | A. MUST KEEP | Keep | Core classification engine |
| `app/skin-lesion-app/` | TypeScript/React | Next.js 14 web client | User Browser | A. MUST KEEP | Keep | Production frontend |
| `models/dual_branch_net.py`| Python (PyTorch) | Decoupled Dual-Branch CNN model | `scripts/train.py`, `tests/`| A. MUST KEEP | Keep | Core research architecture |
| `models/shallow_wide_branch.py`| Python | Texture extraction branch (1024ch)| `models/dual_branch_net.py` | A. MUST KEEP | Keep | Research contribution |
| `models/deep_narrow_branch.py` | Python | Morphological structure branch | `models/dual_branch_net.py` | A. MUST KEEP | Keep | Research contribution |
| `models/fusion.py` | Python | Attention-gated fusion head | `models/dual_branch_net.py` | A. MUST KEEP | Keep | Research contribution |
| `models/baselines/*.py` | Python | ResNet50, DenseNet121/201, EfficientNet | Benchmarks & evaluation | A. MUST KEEP | Keep | Baseline comparisons |
| `losses/focal_loss.py` | Python | Focal Loss with $\alpha/\gamma$ | Training scripts | A. MUST KEEP | Keep | Handles class imbalance |
| `results/benchmark.csv` | CSV | Ground-truth empirical metrics | Dashboard, API, Paper | A. MUST KEEP | Keep | Empirical proof |
| `results/final/` | Markdown/CSV/PDF | Archived publication figures | Paper, Thesis | A. MUST KEEP | Keep | Research archive |
| `tests/test_*.py` | Python (pytest) | Test suites | CI/CD, local testing | A. MUST KEEP | Keep | Verifies code health |
| `SKIN-LESION.zip` | ZIP Archive | Old repository archive | None | C. ARCHIVE | Move to `archive/` | Redundant archive in root |
| `GIT_PUSH_CHECKLIST.md` | Markdown | Temporary pre-push notes | None | C. ARCHIVE | Move to `docs/archive/` | Superseded by final checklists |
| `env_report.md` | Markdown | Temporary machine spec note | None | C. ARCHIVE | Move to `docs/archive/` | Superseded by `RUN_PROJECT.md` |
| `repo_temp/` | Directory | Duplicate unpacked repo | None | D. REMOVE | Remove | 100% duplicate of top-level code |
| `LICENSE` | Text | Open-source license | GitHub | A. MUST KEEP | Populate MIT License | Empty 0-byte file |
| `environment.yml` | YAML | Conda environment specification | Conda users | A. MUST KEEP | Populate Clean Spec | Empty 0-byte file |
| `README.md` | Markdown | Primary repository landing page | All Users / Evaluators | A. MUST KEEP | Rebuild | Primary documentation |
| `pyproject.toml` | TOML | Build & linter configuration | Flake8, pytest, ruff | A. MUST KEEP | Keep | Standard Python metadata |
| `requirements.txt` | Text | Python dependencies list | pip install | A. MUST KEEP | Keep | Core dependency list |
| `requirements-lock.txt` | Text | Pinned exact dependencies | Reproducibility | A. MUST KEEP | Keep | Exact package lock |

---

## 4. Key Findings & Next Actions
1. **Preserve Research History**: Keep all Dual-Branch files, ablation scripts, and multi-seed results intact.
2. **Move Temporary Reports**: Relocate one-off root diagnostic reports into `docs/archive/` to keep the root directory pristine.
3. **Eliminate `repo_temp/`**: Delete the redundant nested duplicate folder.
4. **Populate Empty Files**: Add proper MIT License to `LICENSE` and valid Conda YAML to `environment.yml`.
