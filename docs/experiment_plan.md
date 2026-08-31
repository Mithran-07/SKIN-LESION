# Experiment Plan

## Baselines
- ResNet50
- DenseNet121/201 family baseline
- EfficientNet baseline

## Dual-Branch Ablations
- Full dual-branch model
- Dual-branch with focal loss
- Dual-branch with multi-task decoder
- Dual-branch without attention gate

## Hypotheses
1. The dual-branch model will improve macro F1 on texture-heavy classes.
2. The bottleneck gate will reduce parameter cost without harming performance.
3. Conformal prediction will provide conservative prediction sets with target coverage.
4. The multi-task decoder will improve structural regularity.

## Static Outputs
- Parameter counts
- FLOPs estimates
- VRAM estimates
- Architecture diagrams
- ROC, PR, and confusion-matrix figures
