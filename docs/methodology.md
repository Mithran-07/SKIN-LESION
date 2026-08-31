# Methodology

## Research Goal
Evaluate whether a texture/structure decomposition improves dermoscopic
classification, uncertainty calibration, and explanation quality.

## Experimental Design
1. Baseline single-backbone classifiers.
2. Dual-branch classifier with bottleneck gating.
3. Dual-branch classifier without the attention gate.
4. Multi-task variant with segmentation regularisation.
5. Loss-function ablations such as focal loss versus cross-entropy.

## Evaluation Metrics
- Macro AUC
- Macro F1
- Balanced accuracy
- Per-class recall
- Conformal set size and coverage
- Grad-CAM qualitative inspection

## Static Verification Policy
No training runs are executed on the MacBook workstation.
Only code validation, configuration validation, and figure generation are used.

## Reproducibility Controls
- Fixed random seed
- Deterministic backend settings where supported
- Explicit config files for each experiment
- Centralised logging and checkpointing
