# Abstract

## Advanced Deep Learning for Skin Lesion Diagnosis: A Dual-Branch CNN Architecture with Attention-Gated Fusion

### Objective
Skin cancer diagnosis from dermoscopic images remains a critical challenge in automated medical imaging. This work presents a novel dual-branch convolutional neural network architecture that exploits complementary texture and structural information to improve classification accuracy for melanoma and other skin lesions.

### Methods
We propose a dual-branch architecture consisting of:
- **Texture Branch** (shallow, wide): Captures color, pigmentation, and fine surface details
- **Structure Branch** (deep, narrow): Captures morphological features and boundary information
- **Attention-Gated Fusion Head**: Dynamically weights branch contributions based on input features

The model is trained on the HAM10000 dataset with 9,015 dermoscopic images across 7 lesion classes using focal loss for class imbalance handling and data augmentation for regularization.

### Expected Results
[Auto-populated with benchmark results from Lenovo LOQ]

Compared to baseline architectures (ResNet50, DenseNet121, EfficientNet-B4), the dual-branch approach is expected to:
- Achieve improved AUC through better feature utilization
- Provide interpretable attention maps showing diagnostic reasoning
- Maintain strong performance on underrepresented lesion classes

### Clinical Significance
This work advances automated screening systems by combining precision with interpretability, supporting clinician decision-making rather than replacing expert judgment.

### Conclusion
[To be populated with final results and recommendations]

---

**Keywords**: Skin cancer detection, convolutional neural networks, multi-task learning, attention mechanisms, medical image analysis

