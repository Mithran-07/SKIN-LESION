# Repository Cleanup & Organization Plan

**Project**: `Mithran-07/SKIN-LESION`  
**Date**: August 31, 2026  
**Status**: Formulated & Ready for Safe Execution ✅

---

## 1. Principles of Repository Cleanup
1. **Zero Destructive Deletion**: Core research source code, experiment outputs, and paper/thesis manuscripts will NOT be deleted.
2. **Dual-Branch Preservation**: The Dual-Branch CNN source files and multi-seed results are permanently retained as the core scientific contribution.
3. **Pristine Root Directory**: Root will contain only standard repository entry points (`README.md`, `LICENSE`, `CITATION.cff`, `requirements.txt`, `pyproject.toml`, `.gitignore`).
4. **Archival over Deletion**: Historical machine logs and temporary notes are organized into `docs/archive/` or `archive/`.

---

## 2. Itemized Action Plan

### A. Files & Directories to Remove (Clearly Obsolete / 100% Duplicate)
- `repo_temp/` — Redundant unpacked clone of old July commit (23 duplicate files).
- Root `manifest.json` — 0-byte placeholder (canonical manifest is in `results/final/manifest.json`).

### B. Files to Move to `docs/archive/` (Historical Documentation)
- `AUDIT_REPORT.md` → `docs/archive/AUDIT_REPORT_JULY.md`
- `FIX_SUMMARY.md` → `docs/archive/FIX_SUMMARY.md`
- `PROJECT_COMPLETION_STATUS.md` → `docs/archive/PROJECT_COMPLETION_STATUS.md`
- `READINESS_REPORT.md` → `docs/archive/READINESS_REPORT.md`
- `SETUP_COMPLETE.md` → `docs/archive/SETUP_COMPLETE.md`
- `WORKFLOW.md` → `docs/archive/WORKFLOW.md`
- `FINAL_REPOSITORY_REPORT.md` → `docs/archive/FINAL_REPOSITORY_REPORT.md`
- `MASTER_REPOSITORY.md` → `docs/archive/MASTER_REPOSITORY.md`
- `GIT_PUSH_CHECKLIST.md` → `docs/archive/GIT_PUSH_CHECKLIST.md`
- `env_report.md` → `docs/archive/env_report.md`
- `PROJECT_HANDOFF.md` → `docs/archive/PROJECT_HANDOFF.md`

### C. Files to Move to `archive/`
- `SKIN-LESION.zip` → `archive/SKIN-LESION-v1.0.zip`

### D. Files to Populate / Update
- `LICENSE` → Populate with standard MIT License text.
- `environment.yml` → Populate with clean, valid Conda environment YAML matching `requirements.txt`.
- `README.md` → Comprehensive, professional rebuild with architecture diagrams, benchmark tables, API runbook, viva notes, and disclaimers.
- `.gitignore` → Complete coverage of `.venv/`, `node_modules/`, `.next/`, `*.pth`, `datasets/`, `logs/`, `.DS_Store`.

---

## 3. Final Proposed Repository Structure

```
SKIN-LESION/
│
├── api/                           # Production FastAPI Backend
│   ├── __init__.py
│   ├── inference_engine.py       # EfficientNet-B4 + Grad-CAM Pipeline
│   └── main.py                   # 5 REST Endpoints (/health, /benchmark, /predict, etc.)
│
├── app/                           # Production Next.js 14 Web Frontend
│   └── skin-lesion-app/          # React 18, Tailwind CSS, Lucide Icons
│       ├── public/samples/       # 7-Class Demo Sample Images
│       └── src/app/              # Pages: /, /classify, /dashboard, /research, /architecture
│
├── config/ & configs/             # YAML Training & Model Hyperparameters
├── data/                          # HAM10000 Dataset Loader, Samplers, Augmentations
│   └── samples/                  # Representative 7-Class Sample Images
│
├── explainability/                # Grad-CAM Hooks & Visualization Engine
├── losses/                        # Class-Weighted Focal Loss & Multi-Task Loss
├── models/                        # DualBranchNet, Shallow-Wide, Deep-Narrow, Baselines
├── notebooks/                     # Interactive Jupyter Research Notebooks
├── paper/                         # Full Academic Paper Sections & references.bib
├── thesis/                        # 5-Chapter Thesis Manuscript & Dissertation Frontmatter
├── results/                       # Verified Benchmark CSVs & Figure Plots
│   └── final/                    # Canonical Publication Figures & Metrics
│
├── scripts/                       # Training, Evaluation, Testing & Diagnostic CLI Scripts
├── splits/                        # Patient-Aware Stratified Train/Val/Test Split Indices
├── tests/                         # Automated Pytest Suite (Shapes, Losses, Conformal, API)
├── training/                      # Trainer, Metrics Tracker & Cosine Warmup Scheduler
├── uncertainty/                   # Split Conformal Prediction & MC-Dropout
├── utils/                         # Config Loader, Device Selector, Logging, Checkpointing
├── visualization/                 # Architecture, ROC/PR & Confusion Matrix Plotters
│
├── docs/                          # Comprehensive Guides & Specifications
│   ├── COLLEGE_DEMO_GUIDE.md     # 16-Step Demonstration Script & Viva Q&A
│   ├── FINAL_TEST_REPORT.md      # Automated Test Results (62/62 Passed)
│   ├── RUN_PROJECT.md            # Exact Execution & Troubleshooting Runbook
│   ├── CHECKPOINT_SETUP.md       # Checkpoint Placement Guide
│   ├── REPOSITORY_AUDIT.md       # Complete File Audit & Classification
│   ├── DUPLICATE_FILE_REPORT.md  # Duplicate Analysis Report
│   └── archive/                  # Historical Development Reports
│
├── archive/                       # Archived Legacy ZIP Packages
│
├── README.md                      # Primary Project Documentation
├── LICENSE                        # MIT Open Source License
├── CITATION.cff                   # Academic Citation Metadata
├── MODEL_REGISTRY.md              # Model Architecture & Parameter Registry
├── REPRODUCIBILITY.md             # Scientific Reproducibility Protocol
├── FINAL_PROJECT_STATUS.md        # Comprehensive Final Project Status
├── FINAL_SUBMISSION_CHECKLIST.md  # Final Defense & Submission Checklist
├── pyproject.toml                 # Build & Linter Metadata
├── requirements.txt               # Primary Python Dependencies
├── requirements-lock.txt          # Exact Pinned Dependencies
├── environment.yml                # Clean Conda Environment Specification
└── .gitignore                     # Git Exclusion Rules
```
