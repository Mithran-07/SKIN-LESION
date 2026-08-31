# Introduction

## Background & Motivation

Melanoma and other skin cancers represent a significant global health burden, with annual incidence increasing worldwide. Early detection is crucial for improving patient outcomes, but dermoscopic image analysis requires specialized expertise that is not universally available in all clinical settings.

### Clinical Challenge
- **Variability**: Skin lesions exhibit high visual variability due to patient factors (skin tone, age, lesion location)
- **Rarity**: Some lesion types (e.g., actinic keratosis, dermatofibroma) are underrepresented in training data
- **Similarity**: Benign and malignant lesions can have overlapping features
- **Interpretability**: Clinicians require understanding of classification decisions for liability and validation

### Technical Challenge
Deep learning models, while achieving high accuracy, often act as "black boxes" that clinicians cannot interpret or trust for diagnostic support.

## State of the Art

Current approaches fall into two categories:

### Single-Branch Architectures
**Advantages**:
- Simpler, faster inference
- Easier to train and deploy

**Disadvantages**:
- Single feature representation may miss important diagnostic cues
- No built-in interpretability

**Examples**: ResNet, DenseNet, EfficientNet

### Multi-Task Learning
**Advantages**:
- Can leverage auxiliary tasks (e.g., segmentation) to improve performance
- Encourages learning multiple representations

**Disadvantages**:
- Increased complexity
- May require annotated auxiliary data

## Our Contribution

We propose a **dual-branch architecture** that:
1. **Exploits complementary information**: Texture and structural branches process different aspects of the image
2. **Provides interpretability**: Attention mechanism shows which branch contribution is more important
3. **Handles class imbalance**: Focal loss addresses underrepresented lesion classes
4. **Maintains clinical practicality**: Reasonable inference time for clinical workflow integration

### Key Innovation
The **attention-gated fusion head** dynamically weights texture and structure branches, allowing the model to:
- Emphasize color patterns for well-structured lesions
- Emphasize morphology for poorly-defined lesions
- Provide visual evidence through Grad-CAM attention maps

## Research Questions

1. **Performance**: Does dual-branch architecture outperform single-branch baselines?
2. **Interpretability**: Do attention mechanisms reflect clinically meaningful features?
3. **Robustness**: Does the architecture handle class imbalance better than standard approaches?
4. **Generalization**: Can the model perform well on underrepresented lesion types?

## Expected Outcomes

This work will demonstrate:
- State-of-the-art performance on HAM10000 benchmark
- Interpretable decision-making suitable for clinical use
- Robust handling of imbalanced multi-class classification
- Potential for clinical deployment with radiologist confidence

