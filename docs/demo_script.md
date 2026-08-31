# Demo Script — 5-Minute Faculty Demonstration

## Overview
This is a structured 5-minute demonstration script for presenting the
Dual-Branch CNN for Non-Melanoma Dermoscopic Classification project.

---

## [0:00-0:30] Introduction
"This project applies deep learning to dermoscopic skin lesion classification —
a challenging computer vision problem where the goal is to classify 7 types of
skin lesions from close-up camera images used by dermatologists."

"The dataset is HAM10000: 10,015 dermoscopic images. We trained and evaluated
multiple models, and today I'll walk through the research findings and show you
the final working demo."

---

## [0:30-1:00] Show Dataset & Problem
Navigate to: /research

"Here you can see the complete research story. The dataset has extreme class
imbalance — 67% of images are benign Melanocytic Nevi. We used patient-level
splits to prevent data leakage."

---

## [1:00-2:00] Show Architecture & Experiments
Navigate to: /architecture

"Our core research hypothesis was that a Dual-Branch CNN — one branch focusing
on texture via WideResNet, another on structure via DenseNet — would outperform
single-branch baselines."

"As you can see in the pipeline, we evaluate three baselines and three versions
of the Dual-Branch framework."

---

## [2:00-2:30] Show Baseline Comparison
Navigate to: /dashboard

"Looking at the full benchmark — EfficientNet-B4 achieves 73.64% test accuracy
and 95.92% ROC-AUC. The Dual-Branch V2 reaches 64.24% — roughly 8 points lower."

"The radar chart here shows the performance gap across all 4 metrics clearly."

---

## [2:30-3:30] Live Classification Demo
Navigate to: /classify

"Let me now upload a dermoscopic image and show you the prediction in real time."

1. Enable "Include Grad-CAM Explainability"
2. Upload a sample dermoscopic image
3. Click "Classify Image"

"The model predicts [CLASS] with [X]% confidence. Here are the top 3 predictions."
"The Grad-CAM overlay shows which parts of the image the model focused on."

---

## [3:30-4:00] Explain Key Finding
"The most interesting research finding is what we call Fusion Collapse.
Despite designing the Dual-Branch architecture to balance texture and structure,
the training optimizer consistently pushed the structure gate to 91% and
suppressed the texture gate to 35%. The network rejected the textural features."

"This suggests that a modern single-branch network like EfficientNet-B4 naturally
learns multimodal representations more effectively than our forced separation."

---

## [4:00-4:30] Limitations
"We want to be honest about what this system cannot do.
HAM10000 is limited to one imaging center. The model has not been clinically
validated. Class imbalance affects rare-class recall. Predictions are
NOT medical diagnosis."

---

## [4:30-5:00] Conclusion
"EfficientNet-B4 is the final deployment model. The Dual-Branch CNN is our
research contribution — a documented negative finding that adds value to the field
by demonstrating that explicit feature separation does not guarantee improvement
over strong single-branch baselines."

"Thank you. Any questions?"

---

## Sample Images to Use
Place dermoscopic images in: app/skin-lesion-app/public/samples/
Suggested: one NV image, one MEL image, one BCC image
