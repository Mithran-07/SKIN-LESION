# Overfitting Analysis

## Overview
This template documents investigation into whether models are overfitting to the training set, particularly the Dual-Branch variants with dual feature extractors.

## Status
**Awaiting experimental results from Lenovo LOQ**

## Key Metrics to Compare

### Training vs. Validation Performance

| Metric | Training | Validation | Gap | Status |
|--------|----------|------------|-----|--------|
| AUC | _______ | _______ | _______ | [ ] OK / [ ] WARNING / [ ] CRITICAL |
| F1 Score | _______ | _______ | _______ | [ ] OK / [ ] WARNING / [ ] CRITICAL |
| Accuracy | _______ | _______ | _______ | [ ] OK / [ ] WARNING / [ ] CRITICAL |
| Loss | _______ | _______ | _______ | [ ] OK / [ ] WARNING / [ ] CRITICAL |

### Overfitting Severity Classification
- **Healthy**: Validation gap < 0.02 for all metrics
- **Mild**: Gap 0.02-0.05 (acceptable for small datasets)
- **Moderate**: Gap 0.05-0.10 (requires regularization)
- **Severe**: Gap > 0.10 (model failing to generalize)

**Classification**: ______________________________

## Investigation Checklist

### 1. Per-Class Overfitting
- [ ] Does overfitting vary by class? (collect per-class validation gaps)
  - Class 1: Training _______ → Validation _______ (gap: _______)
  - Class 2: Training _______ → Validation _______ (gap: _______)
  - Class 3: Training _______ → Validation _______ (gap: _______)
  - *[Continue for all 7 classes]*

- [ ] Are underrepresented classes overfitting more severely?

### 2. Early Stopping Behavior
- [ ] What epoch did validation loss stop improving? ________
- [ ] How many epochs before overfitting became critical? ________
- [ ] Did model continue training past optimal checkpoint? ________

### 3. Regularization Effects
- [ ] Dropout: Enabled [ ] / Disabled [ ]. Effect on validation gap: _______
- [ ] L2 Regularization: Strength _______ . Effect on validation gap: _______
- [ ] Data Augmentation intensity: _______ . Effect on validation gap: _______
- [ ] Focal Loss (alpha, gamma): _______ . Effect on validation gap: _______

### 4. Model Complexity Analysis
- [ ] Total parameters: _________
- [ ] Parameters in each branch:
  - Texture branch: _________
  - Structure branch: _________
  - Fusion head: _________
- [ ] Is model too complex for dataset size (9015 images)?

### 5. Data Leakage Investigation
- [ ] Training set: Any duplicates? ________
- [ ] Train/Val/Test split: Properly stratified? ________
- [ ] Data augmentation: Different between train and val? ________

## Visualizations to Generate

### Learning Curves
```
[To be populated with training loss, validation loss, training AUC, validation AUC]
```

### Overfitting Severity by Epoch
```
[To be populated with gap between train and validation metrics over time]
```

### Per-Class Generalization Gap
```
[To be populated with bar chart showing overfitting severity per lesion type]
```

## Root Cause Diagnosis

When results arrive, identify root cause:

- [ ] **Data insufficiency**: Model capacity too high for 9015 samples
  - Recommended action: Reduce model size or increase augmentation
- [ ] **Inadequate regularization**: Dropout/L2 insufficient
  - Recommended action: Increase regularization strength
- [ ] **Class imbalance**: Overfitting on underrepresented classes
  - Recommended action: Increase focal loss alpha for minority classes
- [ ] **Noisy labels**: Annotation errors in training set
  - Recommended action: Manual review of flagged samples
- [ ] **Other**: ____________________________

## Mitigation Strategies (If Overfitting Detected)

Ranked by effectiveness for Dual-Branch architecture:

1. **Increase Data Augmentation**
   - [ ] Add geometric: Rotation, shear, zoom
   - [ ] Add color: Hue, saturation, brightness
   - [ ] Add affine: Elastic deformation, grid distortion

2. **Regularization Adjustment**
   - [ ] Increase Dropout to 0.5+ in feature extractors
   - [ ] Increase L2 lambda to 1e-3 or higher
   - [ ] Add LayerNorm between blocks

3. **Training Procedure**
   - [ ] Implement early stopping with patience=5-10
   - [ ] Use learning rate warmup
   - [ ] Reduce batch size (more frequent updates, less batch normalization bias)

4. **Architecture Changes** (if above fails)
   - [ ] Reduce feature map sizes in branches
   - [ ] Reduce depth of branches
   - [ ] Simplify fusion head

## Conclusions

**To be filled in once results arrive:**

- Overfitting severity: **____________** (HEALTHY / MILD / MODERATE / SEVERE)
- Root cause: ____________________________________________
- Mitigation applied: ____________________________________________
- Post-mitigation validation gap: ____________

## References
- Related sections: Model Architecture (Section 3), Training Procedure (Section 4)
- HAM10000 dataset characteristics: data/README.md
