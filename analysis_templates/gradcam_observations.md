# Grad-CAM Observations

## Overview
Grad-CAM (Gradient-weighted Class Activation Maps) visualizes which image regions the model attends to for classification decisions. For the dual-branch architecture, we analyze attention separately in the texture (shallow-wide) and structure (deep-narrow) branches.

## Status
**Awaiting experimental results from Lenovo LOQ**

## Methodology

### Dual-Branch Visualization
The model has two parallel branches:
- **Texture Branch** (shallow, wide): Captures color, texture, fine details
- **Structure Branch** (deep, narrow): Captures shape, morphology, spatial structure
- **Fusion Head**: Combines both via attention-gated mechanism

Grad-CAM applied to:
- Last convolutional layer of texture branch
- Last convolutional layer of structure branch
- Visualization: Side-by-side heatmaps on same image

### Expected Behavior
- **Healthy model**: Each branch focuses on complementary aspects
  - Texture: Color variations, surface patterns
  - Structure: Lesion boundaries, morphological structures
- **Collapsed model**: Both branches focus on same features (wasted capacity)
- **Uninformative**: Attention spread everywhere (model confused)

## Sample Visualizations

### Sample 1: Melanoma (True Class) - Correctly Classified

**Image**: [Original lesion photo]

**Texture Branch Activation**:
```
[Grad-CAM heatmap showing color/pigmentation attention]
```
- Primary attention region: ____________________
- Pattern observations: ____________________
- Interpretability: [ ] Clear | [ ] Fuzzy | [ ] Uninformative

**Structure Branch Activation**:
```
[Grad-CAM heatmap showing boundary/shape attention]
```
- Primary attention region: ____________________
- Pattern observations: ____________________
- Interpretability: [ ] Clear | [ ] Fuzzy | [ ] Uninformative

**Fusion Decision**:
- Model prediction: _________ (correct: ✓)
- Attention gate weight (texture): _______%
- Attention gate weight (structure): _______%
- **Assessment**: Both branches contributed meaningfully? ________

**Clinical Validation**:
- Do attended regions correspond to clinically suspicious areas? ________
- Would dermatologist agree with attention pattern? ________

---

### Sample 2: Benign Nevus (True Class) - Correctly Classified

**Image**: [Original lesion photo]

**Texture Branch Activation**:
```
[Grad-CAM heatmap]
```
- Observations: ____________________

**Structure Branch Activation**:
```
[Grad-CAM heatmap]
```
- Observations: ____________________

**Assessment**: ____________________________________________________

---

### Sample 3: Misclassified Sample - Error Analysis

**Image**: [Problematic lesion]

**True Class**: ____________ | **Predicted Class**: ____________ | **Confidence**: ______%

**Texture Branch Activation**:
```
[Grad-CAM heatmap]
```
- Attended regions: ____________________
- Problem?: ____________________

**Structure Branch Activation**:
```
[Grad-CAM heatmap]
```
- Attended regions: ____________________
- Problem?: ____________________

**Why was it misclassified?**
- [ ] Model attended to wrong region entirely
- [ ] Model attended to correct region but made wrong decision
- [ ] Attention scattered (model confused)
- [ ] Other: ____________________

**Clinical Assessment**: Is model's error understandable? ________

---

## Pattern Analysis: Across Cohorts

### Melanoma Samples (All True Melanomas)
Generate Grad-CAM for 5-10 correctly classified melanomas:

**Common Attention Patterns**:
- Texture branch consistently attends to: ____________________
- Structure branch consistently attends to: ____________________
- Both branches agree on attention region?: ____________________

**Interpretability Score**: 
- 1 = All samples show clear, interpretable patterns
- 5 = Patterns are confused/unclear

Score: _______ / 5

---

### Benign Nevus Samples (All True Benign)
Generate Grad-CAM for 5-10 correctly classified nevi:

**Common Attention Patterns**:
- Texture branch consistently attends to: ____________________
- Structure branch consistently attends to: ____________________

**Interpretability Score**: _______ / 5

---

### Misclassified Samples (All Wrong Predictions)
Generate Grad-CAM for misclassified samples:

**Error Pattern Summary**:
- [ ] Most errors: Model attended to correct region but made wrong decision
- [ ] Most errors: Model attended to wrong region
- [ ] Most errors: Attention pattern is unclear/scattered
- [ ] Most errors: Attention heavily imbalanced (one branch dominates)

**Common Failure Mode**: ____________________________________________________

---

## Branch Specialization Analysis

### Question 1: Do Branches Specialize?

For correctly classified samples, measure:
- **Overlap**: Compute IoU (Intersection over Union) of texture vs. structure attention
  - Average overlap: _______% 
  - If > 70%: Branches not specializing; redundant
  - If < 30%: Branches complementary; good specialization

**Finding**: ____________________________________________________

### Question 2: Does One Branch Dominate?

For each sample, record fusion gate weights:
- Average texture branch weight: _______%
- Average structure branch weight: _______%
- Standard deviation across samples: _______%

- [ ] Weights balanced (45-55% / 45-55%): Healthy
- [ ] Weights imbalanced (70/30 or worse): Potential fusion collapse
- [ ] Weights highly variable (high std): Model inconsistent

**Finding**: ____________________________________________________

### Question 3: Branch Confidence Disagreement

When branches disagree:
- What is the distribution of fusion gate weights? ____________________
- Does model still make correct predictions? _______%
- When branches disagree AND model is correct: Is fusion head making good decision? ________

**Finding**: ____________________________________________________

---

## Comparison: Dual-Branch vs. Single-Branch Baselines

### Grad-CAM Interpretability
When baseline results arrive:

| Architecture | Attention Clarity | Localization Accuracy | Interpretability |
|-------------|------------------|----------------------|------------------|
| Dual-Branch | ________________ | ________________ | ________________ |
| ResNet50 | ________________ | ________________ | ________________ |
| DenseNet121 | ________________ | ________________ | ________________ |
| EfficientNet-B4 | ________________ | ________________ | ________________ |

**Winner for Interpretability**: ____________

### Clinical Validation
- Do clinicians find dual-branch attention more interpretable? ________
- Do attended regions align better with clinical diagnosis rationale? ________

---

## Quality Control Checklist

- [ ] Do all attention heatmaps have reasonable magnitude (not all white/black)?
- [ ] Are activation maps spatially smooth (not random noise)?
- [ ] Do activations concentrate on lesion (not background)?
- [ ] Is there class-specific consistency (melanomas look similar to each other)?

**Quality Assessment**: [ ] Good | [ ] Acceptable | [ ] Poor

If Poor: Possible reasons:
- [ ] Model not well-trained (low validation accuracy)
- [ ] Grad-CAM layer selection inappropriate
- [ ] Feature maps have insufficient information
- [ ] Other: ____________________

---

## Clinical Insights

### Strengths of Model Attention
- Model consistently attends to clinically relevant features? ________
- Examples of insightful attention patterns: ____________________

### Weaknesses of Model Attention
- Model attends to irrelevant features? ________
- Examples of problematic attention: ____________________

### Potential for Clinical Deployment
- Could radiologists trust this model's decisions based on attention patterns? ________
- What additional validation is needed? ____________________

---

## Recommendations

### If Attention is Interpretable and Correct:
- [ ] Model suitable for clinical deployment with attention visualization
- [ ] Include Grad-CAM in clinical decision support interface

### If Attention is Confusing or Incorrect:
- [ ] Model not suitable for clinical use (black box)
- [ ] Recommend additional feature extraction layers
- [ ] Consider simpler, more interpretable architectures

### If Branches Don't Specialize:
- [ ] Potential fusion collapse or unnecessary architecture complexity
- [ ] Consider simplified single-branch architecture
- [ ] Or: Increase branch independence (different initializations, separate optimizers)

---

## Conclusions

**To be filled in once results arrive:**

- Dual-branch specialization: [ ] Clear | [ ] Partial | [ ] None
- Overall interpretability: **____________** (Poor / Acceptable / Good / Excellent)
- Clinical applicability: ________________________
- Recommendations: ________________________

## References
- Grad-CAM implementation: explainability/gradcam.py
- Related paper: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Medical interpretability: Medical AI deployment guidelines
