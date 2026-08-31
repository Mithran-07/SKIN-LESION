# Presentation Outline

## 1. Motivation
Skin cancer is rising. Early detection saves lives. Dermoscopy improves diagnosis but requires significant expertise.
*Notes: Emphasize the mortality difference between early and late detection.*

## 2. Problem Statement
Current CNNs collapse structural and textural features into a single bottleneck, losing fine-grained details critical for differentiating lesions.
*Notes: Define the difference between macroscopic structure and microscopic texture.*

## 3. Existing Methods
Standard ResNet/DenseNet backbones trained with cross-entropy.
*Notes: Briefly mention their limitations (black-box, poor on rare classes).*

## 4. Research Gap
Lack of architectures explicitly designed to decouple high-frequency texture from low-frequency structure. Lack of uncertainty quantification.
*Notes: This is where we justify the dual-branch approach.*

## 5. Objectives
Design a dual-branch network, integrate attention fusion, handle class imbalance, and provide conformal uncertainty metrics.
*Notes: State the 4 main goals clearly.*

## 6. Dataset
HAM10000. 7 classes. Highly imbalanced (67% Nevi).
*Notes: Show class distribution chart.*

## 7. Proposed Architecture
High-level overview of the pipeline (Image -> Two Branches -> Fusion -> Classifier).
*Notes: Use the architecture diagram here.*

## 8. Dual Branch CNN
Shallow-Wide for texture, Deep-Narrow for structure.
*Notes: Explain the receptive field difference.*

## 9. Attention Fusion
Squeeze-and-Excitation gate to dynamically weigh texture vs structure.
*Notes: Explain why some lesions need texture more than structure.*

## 10. Focal Loss
Dynamically scales down well-classified examples.
*Notes: Explain the gamma parameter.*

## 11. Grad-CAM
Post-hoc explainability hooked to both branches.
*Notes: Show examples of heatmaps on specific lesions.*

## 12. Conformal Prediction
Mathematical guarantees for prediction sets.
*Notes: "I don't know" is a safe answer.*

## 13. Experimental Setup
PyTorch, Lenovo LOQ (CUDA), strict stratified splits.
*Notes: Briefly mention reproducible seeds.*

## 14. Results
[Placeholder for metrics, Macro-AUC, F1]
*Notes: Highlight the improvement over baseline.*

## 15. Ablation Study
[Placeholder for with/without attention, focal loss]
*Notes: Prove that every component contributes to the final score.*

## 16. Future Work
Multi-modal (clinical data + image), federated learning.
*Notes: Vision for the next 5 years.*

## 17. Conclusion
Decoupled feature extraction improves both accuracy and interpretability.
*Notes: Strong closing statement.*
