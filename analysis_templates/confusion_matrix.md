# Confusion Matrix Interpretation

## Overview
The confusion matrix provides fine-grained insight into per-class performance, error patterns, and misclassification relationships. This template documents systematic interpretation.

## Status
**Awaiting experimental results from Lenovo LOQ**

## Confusion Matrix Template

### Raw Counts (Rows = True Label, Columns = Predicted Label)
```
Predicted:    MEL    NV   BCC  AKIEC  BKL   DF  VASC
True MEL:      ___    ___   ___   ___   ___  ___   ___
True NV:       ___    ___   ___   ___   ___  ___   ___
True BCC:      ___    ___   ___   ___   ___  ___   ___
True AKIEC:    ___    ___   ___   ___   ___  ___   ___
True BKL:      ___    ___   ___   ___   ___  ___   ___
True DF:       ___    ___   ___   ___   ___  ___   ___
True VASC:     ___    ___   ___   ___   ___  ___   ___
```

### Normalized by True Label (Row Percentages = Recall per Class)
```
Predicted:    MEL    NV   BCC  AKIEC  BKL   DF  VASC
True MEL:    ____%  ___%  ___%  ___%  ___%  __%  ___%
True NV:     ____%  ___%  ___%  ___%  ___%  __%  ___%
True BCC:    ____%  ___%  ___%  ___%  ___%  __%  ___%
True AKIEC:  ____%  ___%  ___%  ___%  ___%  __%  ___%
True BKL:    ____%  ___%  ___%  ___%  ___%  __%  ___%
True DF:     ____%  ___%  ___%  ___%  ___%  __%  ___%
True VASC:   ____%  ___%  ___%  ___%  ___%  __%  ___%
```

**Note**: Diagonal values are recall. Off-diagonal values show where misclassifications occur.

## Per-Class Analysis

### Class: MEL (Melanoma) - Most Clinically Important

**Recall (Sensitivity)**: _______% - Can model detect melanoma? ________

- [ ] Recall > 0.9: Excellent (suitable for screening)
- [ ] Recall 0.8-0.9: Good (acceptable with confirmation)
- [ ] Recall 0.7-0.8: Moderate (requires careful review)
- [ ] Recall < 0.7: Poor (not suitable for clinical use)

**Common Misclassifications**:
- When true class is MEL, predicted as NV: _______% (probably benign nevi)
- When true class is MEL, predicted as BCC: _______% (another malignancy)
- When true class is MEL, predicted as BKL: _______% (keratosis confusion)

**Clinical Implications**:
- [ ] False negatives (MEL → Benign): _______ cases. **CRITICAL - missed malignancy**
- [ ] False positives (other → MEL): _______ cases. Acceptable (causes follow-up)

**Assessment**: ____________________________________________________

---

### Class: NV (Benign Nevus) - Most Frequent

**Recall (Specificity wrt other classes)**: _______% - How often is benign correctly identified?

**Common Misclassifications**:
- When true class is NV, predicted as MEL: _______% (false alarm - overdiagnosis)
- When true class is NV, predicted as BKL: _______% (misidentified as keratosis)

**Assessment**: ____________________________________________________

---

### Class: BCC (Basal Cell Carcinoma) - Second Malignancy

**Recall**: _______%

**Common Misclassifications**:
- When true class is BCC, predicted as NV: _______% (false negative - missed cancer)
- When true class is BCC, predicted as MEL: _______% (misidentified malignancy)
- When true class is BCC, predicted as AKIEC: _______% (confusion with inflamed)

**Assessment**: ____________________________________________________

---

### Class: AKIEC (Actinic Keratosis/Intraepithelial Carcinoma) - Precancerous

**Recall**: _______%

**Common Misclassifications**:
- When true class is AKIEC, predicted as BKL: _______% (confused with keratosis)
- When true class is AKIEC, predicted as MEL: _______% (overestimated severity)

**Assessment**: ____________________________________________________

---

### Class: BKL (Seborrheic Keratosis) - Benign

**Recall**: _______%

**Common Misclassifications**:
- When true class is BKL, predicted as NV: _______% (confused with nevus)
- When true class is BKL, predicted as AKIEC: _______% (overestimated severity)

**Assessment**: ____________________________________________________

---

### Class: DF (Dermatofibroma) - Rare Benign

**Recall**: _______%

**Common Misclassifications** (list all):
- ___________________________________________________________________

**Assessment**: ____________________________________________________

---

### Class: VASC (Vascular Lesion) - Rare Benign

**Recall**: _______%

**Common Misclassifications** (list all):
- ___________________________________________________________________

**Assessment**: ____________________________________________________

---

## Cross-Class Confusion Patterns

### Question 1: Do Malignancies Cluster Together?
- Do malignancies (MEL, BCC, AKIEC) misclassify as each other?
- MEL→BCC or BCC→MEL: _______% (acceptable - all malignant)
- But MEL→BKL: _______% (worse - benign misclassification)

**Finding**: ____________________________________________________

### Question 2: Do Benign Lesions Cluster Together?
- Do benign classes (NV, BKL, DF, VASC) misclassify as each other?
- NV→BKL or BKL→NV: _______% (less critical than cancer misclassification)

**Finding**: ____________________________________________________

### Question 3: Are Rare Classes Underrepresented?
- DF (Dermatofibroma) recall: _______% (should be > 0.6 for acceptable performance)
- VASC (Vascular) recall: _______% (should be > 0.6)
- If both < 0.6: Model is struggling with rare classes

**Finding**: ____________________________________________________

### Question 4: Is Model Overly Conservative?
- What is the rate of "safe" predictions (predicting MEL when uncertain)?
- Number of times other classes predicted as MEL: _______

**Finding**: ____________________________________________________

## Model Behavior Insights

### Diagonal Dominance
- Average recall (main diagonal): _______% (target: > 0.85 for good model)
- Model is: [ ] Excellent | [ ] Good | [ ] Acceptable | [ ] Poor

### Off-Diagonal Concentration
- Where do most errors occur?
  - Most common single misclassification: _______ → _______ (_______%)
  - Second most common: _______ → _______ (_______%)
  - Third most common: _______ → _______ (_______%)

### Precision vs. Recall Trade-offs
- Classes with high recall but low precision (overpredicted):
  - __________________ (model too liberal)
- Classes with high precision but low recall (underpredicted):
  - __________________ (model too conservative)

## Error Analysis

### Critical Errors (High Clinical Impact)
- [ ] Melanoma (MEL) misclassified as benign:  _______  cases
  - **ACTION REQUIRED**: Investigate samples; consider retraining with class weights
- [ ] Basal cell carcinoma (BCC) misclassified as benign: _______ cases
  - **ACTION REQUIRED**: Similar investigation

### Acceptable Errors (Low Clinical Impact)
- [ ] Benign lesion misclassified as cancer (false alarm): _______ cases
  - **ACTION**: Acceptable (leads to follow-up, not harm)

## Comparison: Dual-Branch vs. Baselines

When benchmarks arrive, compare confusion matrices:

| Misclassification | Dual-Branch | ResNet50 | DenseNet | EfficientNet | Winner |
|------------------|-------------|----------|----------|--------------|--------|
| MEL → Benign | _______% | _______% | _______% | _______% | |
| BCC → Benign | _______% | _______% | _______% | _______% | |
| Rare class errors | _______% | _______% | _______% | _______% | |

**Conclusion**: Which model has best confusion matrix pattern?

## Recommendations for Improvement

**If high error rate for specific class pair:**

1. Identify visually similar pairs (e.g., why does model confuse X with Y?)
2. Review 10-20 misclassified samples manually
3. Potential fixes:
   - [ ] Increase focal loss weight for confusing class pair
   - [ ] Add more augmentation for visually similar lesions
   - [ ] Increase feature extractor capacity for discriminative features
   - [ ] Collect more high-quality samples for confusing pair

## Conclusions

**To be filled in once results arrive:**

- Best performing class: ____________ (recall: ____%)
- Worst performing class: ____________ (recall: ____%)
- Critical errors identified: ________________________
- Recommended action: ________________________

## References
- Related: Medical imaging interpretation, HAM10000 class descriptions
- Scripts: scripts/evaluate.py (generates confusion matrices)
