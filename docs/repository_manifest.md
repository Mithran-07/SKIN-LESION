# Repository Manifest

## Scope
This repository contains the source code, configuration, documentation, figures, tests, and research notes for a dual-branch dermoscopic classification project.

The Lenovo LOQ should use this repository for training, evaluation, inference, and uncertainty analysis.
The MacBook R&D workstation should remain limited to static research, inspection, and documentation tasks.

## Complete folder tree

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── experiments/
│       ├── baseline_resnet50.yaml
│       ├── baseline_densenet121.yaml
│       ├── baseline_efficientnet.yaml
│       ├── dual_branch.yaml
│       ├── dual_branch_focal.yaml
│       ├── dual_branch_mtl.yaml
│       └── dual_branch_ablation_no_attention.yaml
├── data/
│   ├── __init__.py
│   ├── augmentations.py
│   ├── dataset.py
│   ├── download.py
│   ├── sampler.py
│   └── README.md
├── docs/
│   ├── __init__.py
│   ├── architecture.md
│   ├── methodology.md
│   ├── experiment_plan.md
│   ├── reproducibility_guide.md
│   ├── code_review.md
│   ├── dataset_manifest.md
│   ├── literature_review.md
│   ├── model_card.md
│   ├── presentation.md
│   └── reproducibility_report.md
├── explainability/
│   ├── __init__.py
│   ├── gradcam.py
│   └── visualize.py
├── federated/
│   ├── __init__.py
│   ├── client.py
│   └── server.py
├── figures/
│   ├── README.md
│   ├── architecture_overview.svg / .pdf
│   ├── conformal_workflow.svg / .pdf
│   ├── dual_branch_pipeline.svg / .pdf
│   └── templates/
│       ├── confusion_matrix.svg / .png / .pdf
│       ├── pr_curve.svg / .png / .pdf
│       ├── roc_curve.svg / .png / .pdf
│       └── training_curves.svg / .png / .pdf
├── losses/
│   ├── __init__.py
│   ├── focal_loss.py
│   └── mtl_loss.py
├── models/
│   ├── __init__.py
│   ├── dual_branch_net.py
│   ├── deep_narrow_branch.py
│   ├── shallow_wide_branch.py
│   ├── fusion.py
│   ├── mtl_head.py
│   ├── model_output.py
│   └── baselines/
│       ├── __init__.py
│       ├── resnet50_baseline.py
│       ├── densenet201_baseline.py
│       └── efficientnet_baseline.py
├── notebooks/
│   ├── __init__.py
│   └── .gitkeep
├── paper/
│   ├── abstract.md
│   ├── introduction.md
│   ├── related_work.md
│   ├── methodology.md
│   ├── experimental_setup.md
│   ├── results.md
│   ├── discussion.md
│   ├── limitations.md
│   ├── future_work.md
│   ├── conclusion.md
│   └── references.bib
├── research_journal/
│   └── 2026-07-10.md
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── analyze_results.py
│   ├── error_analysis.py
│   └── generate_figure_templates.py
├── tests/
│   ├── __init__.py
│   ├── test_conformal.py
│   ├── test_focal_loss.py
│   └── test_model_shapes.py
├── thesis/
│   ├── 00_frontmatter.md
│   ├── 01_introduction.md
│   ├── 02_literature_review.md
│   ├── 03_methodology.md
│   ├── 04_results.md
│   └── 05_conclusion.md
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── metrics.py
│   └── scheduler.py
├── uncertainty/
│   ├── __init__.py
│   ├── conformal_prediction.py
│   └── mc_dropout.py
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── config_loader.py
│   ├── device.py
│   ├── logger.py
│   ├── reproducibility.py
│   ├── ablation.py
│   ├── model_analysis.py
│   ├── experiment_tracker.py
│   └── reproducibility_report.py
└── visualization/
    ├── __init__.py
    ├── architecture_diagram.py
    ├── confusion_matrix.py
    ├── feature_maps.py
    ├── pr_curves.py
    ├── roc_curves.py
    └── training_curves.py
```

## Purpose of major directories

- config/: runtime configuration and experiment YAMLs.
- data/: dataset loading, augmentation, splitting, and download scaffolding.
- docs/: research methodology, architecture notes, reproducibility, and audit material.
- explainability/: Grad-CAM and visualization helpers.
- federated/: federated-learning scaffold and server/client logic.
- figures/: paper-ready static figures and templates.
- losses/: custom loss functions.
- models/: all classifiers, branches, fusion, and output containers.
- notebooks/: notebook package marker and local exploratory work.
- paper/: manuscript drafting material.
- research_journal/: dated research log entries.
- scripts/: CLI entry points for train, evaluate, infer, and analysis utilities.
- tests/: automated verification.
- thesis/: thesis or report chapter drafts.
- training/: trainer, metrics, and scheduler.
- uncertainty/: conformal prediction and MC dropout.
- utils/: shared utilities such as logging, configuration, checkpointing, analysis, and reproducibility.
- visualization/: reusable plotting helpers for analysis and publication graphics.

## Required files for Lenovo LOQ training and evaluation

### Core entry points
- scripts/train.py
- scripts/evaluate.py
- scripts/infer.py

### Model code
- models/dual_branch_net.py
- models/deep_narrow_branch.py
- models/shallow_wide_branch.py
- models/fusion.py
- models/mtl_head.py
- models/model_output.py
- models/baselines/__init__.py
- models/baselines/resnet50_baseline.py
- models/baselines/densenet201_baseline.py
- models/baselines/efficientnet_baseline.py

### Training and metrics
- training/trainer.py
- training/metrics.py
- training/scheduler.py

### Losses
- losses/focal_loss.py
- losses/mtl_loss.py

### Data pipeline
- data/__init__.py
- data/dataset.py
- data/augmentations.py
- data/sampler.py
- data/download.py
- data/README.md

### Explainability and uncertainty
- explainability/gradcam.py
- explainability/visualize.py
- uncertainty/conformal_prediction.py
- uncertainty/mc_dropout.py

### Shared utilities
- utils/checkpoint.py
- utils/config_loader.py
- utils/device.py
- utils/logger.py
- utils/reproducibility.py

### Configuration and verification
- config/config.yaml
- config/experiments/*.yaml
- tests/*.py
- requirements.txt
- pyproject.toml
- README.md

## Optional files

These are useful but not required for a clean training/evaluation clone:
- docs/code_review.md
- docs/dataset_manifest.md
- docs/literature_review.md
- docs/model_card.md
- docs/presentation.md
- docs/reproducibility_report.md
- generate_docs.py
- generate_paper_thesis.py
- paper/
- thesis/
- research_journal/
- notebooks/
- figures/templates/
- scripts/analyze_results.py
- scripts/error_analysis.py
- scripts/generate_figure_templates.py
- utils/experiment_tracker.py
- utils/model_analysis.py
- utils/ablation.py
- visualization/
- federated/

## Files intentionally excluded from Git

The following are local-only or generated artifacts and should not be committed:
- datasets/
- data/raw/
- data/processed/
- checkpoints/
- tensorboard/
- logs/
- results/
- cache/
- experiments/
- .venv/
- __pycache__/
- *.pt
- *.pth
- *.ckpt
- *.pyc
- *.csv
- .DS_Store
- .pytest_cache/

## Audit summary

- Core source files exist.
- Required model, training, and uncertainty modules exist.
- No missing imports were found in the training/evaluation/inference import paths after verification.
- The repository contains additional research and documentation material that is optional for training.
- Generated outputs and local runtime artifacts are excluded via .gitignore.
