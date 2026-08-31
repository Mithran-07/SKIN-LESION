# Class Imbalance Analysis

## Overview
HAM10000 dataset contains significant class imbalance (e.g., melanoma is ~8%, acanthosis nigricans is <1%). This template documents how class imbalance affects model performance and whether mitigation strategies (focal loss, weighted sampling) are effective.

## Status
**Awaiting experimental results from Lenovo LOQ**

## Class Distribution Summary

### Expected Class Distribution (from HAM10000)
| Class | Label | Expected % | Count |
|-------|-------|-----------|-------|
| MEL | 0 | 7-10% | ~700-900 |
| NV | 1 | 60-70% | ~5000-6300 |
| BCC | 2 | 5-10% | ~500-800 |
| AKIEC | 3 | 3-4% | ~300-400 |
| BKL | 4 | 10-15% | ~900-1300 |
| DF | 5 | 1-2% | ~100-200 |
| VASC | 6 | 0.5-1% | ~50-100 |

### Actual Distribution in Test Set (results to populate)
| Class | Count | % | Representation |
|-------|-------|---|-----------------|
| MEL | _____ | _____ | |
| NV | _____ | _____ | |
| BCC | _____ | _____ | |
| AKIEC | _____ | _____ | |
| BKL | _____ | _____ | |
| DF | _____ | _____ | |
| VASC | _____ | _____ | |

## Performance Impact Analysis

### Per-Class Metrics from Best Model

| Class | Precision | Recall | F1 | AUC | Notes |
|-------|-----------|--------|----|----|-------|
| MEL | _______ | _______ | _______ | _______ | |
| NV | _______ | _______ | _______ | _______ | |
| BCC | _______ | _______ | _______ | _______ | |
| AKIEC | _______ | _______ | _______ | _______ | |
| BKL | _______ | _______ | _______ | _______ | |
| DF | _______ | _______ | _______ | _______ | |
| VASC | _______ | _______ | _______ | _______ | |

### Imbalance Impact Assessment
- [ ] Does recall decrease for minority classes? (expected: YES for severe imbalance)
- [ ] Is F1 score lower for minority classes? (expected: YES)
- [ ] What is largest performance gap between majority and minority? _______ points
- [ ] **Imbalance severity**: [ ] MILD | [ ] MODERATE | [ ] SEVERE

## Mitigation Strategies Employed

### Strategy 1: Focal Loss
- [ ] Implemented
- [ ] Alpha (class weighting): ________________
- [ ] Gamma (focusing parameter): ________________
- [ ] Effect on minority class recall: ________________

### Strategy 2: Class-Weighted Sampling
- [ ] Implemented
- [ ] Sampling weights calculated: ________________
- [ ] Oversampling underrepresented classes: ________________
- [ ] Effect on minority class recall: ________________

### Strategy 3: Data Augmentation
- [ ] Intensity for minority classes increased: ________________
- [ ] Effect on model robustness: ________________

### Strategy 4: Balanced Metrics
- [ ] Macro F1 computed (unweighted average across classes)
- [ ] Macro F1 vs. Weighted F1 difference: ________________
- [ ] This indicates degree of class-wise performance variation

## Analysis Questions

### 1. Does Model Favor Majority Class?
- [ ] Does model predict majority class (NV) too often? ________
- [ ] Baseline accuracy if always predicting NV: _______ %
- [ ] Actual model accuracy: _______ % (should be much higher)
- [ ] Is improvement statistically significant? (Chi-squared test) ________

### 2. Are Minority Classes Ignored?
- [ ] Recall for DF and VASC (rarest classes): ________
- [ ] If recall < 0.3 for any class, model is likely ignoring it
- [ ] Which classes have critical low recall? ____________________

### 3. Is Focal Loss Working?
- [ ] Comparison: Focal Loss vs. Standard CrossEntropy
  - Focal Loss macro F1: _________
  - CrossEntropy macro F1: _________
  - Improvement: _________
- [ ] Did focal loss increase minority class recall? ________

### 4. Trade-off Analysis
- [ ] Weighted F1 (emphasizes accuracy): _________
- [ ] Macro F1 (equal class emphasis): _________
- [ ] Gap indicates class-wise performance variation
- [ ] Is gap acceptable? _______ (should be < 0.05 ideally)

## Visualizations

### Confusion Matrix Normalized by True Label
```
[To be populated - shows recall per class]
```

### Per-Class Performance Ranking
```
[To be populated - bar chart of F1 by class]
```

### Class Imbalance vs. Performance Gap
```
[To be populated - scatter plot: class frequency vs. recall]
```

## Clinical Significance

**Important for medical imaging**: Minority classes often include dangerous lesions (e.g., melanoma).

- [ ] Which minority class is most clinically important? ____________________
- [ ] What is recall for this class? _______ (target: > 0.9 for clinical use)
- [ ] Is this recall clinically acceptable? _______ (YES / NO / BORDERLINE)

If NO:
- **Recommended action**: Retrain with higher focal loss alpha for clinically important classes
- **Target minimum recall**: _________

## Root Cause and Recommendations

### If Imbalance Problem is Severe:

1. **Increase Focal Loss Alpha for Minority Classes**
   - [ ] Current alpha: _________
   - [ ] Recommended alpha: 0.5-0.9
   - [ ] Expected improvement in minority recall: ~5-15%

2. **Under-sample Majority Class**
   - [ ] Current NV samples in training: _________
   - [ ] Proposed reduction to: _________ (target ratio: 3:1 with next class)
   - [ ] Trade-off: May lose generalization on NV

3. **Generate Synthetic Samples for Minority Classes**
   - [ ] Use SMOTE or StyleGAN for underrepresented lesions
   - [ ] Expected new minority class size: _________

4. **Use Class-Weighted Loss Throughout**
   - [ ] CrossEntropy weights: ____________________
   - [ ] Ensure applied to all loss terms

## Conclusions

**To be filled in once results arrive:**

- Class imbalance severity: **____________** (MILD / MODERATE / SEVERE)
- Most affected class: ____________ (recall: ______)
- Mitigation effectiveness (if applied): ____________
- Recommendation for production deployment: ________________________

## References
- HAM10000 dataset: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
- Related sections: data/README.md, losses/focal_loss.py, data/sampler.py
