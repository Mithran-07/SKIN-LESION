# Model Card: Dual-Branch CNN

## Purpose
Classify non-melanoma and melanoma dermoscopic images using a dual-branch neural architecture.

## Architecture
Dual-Branch CNN (Shallow-Wide Branch for Texture + Deep-Narrow Branch for Structure) fused via Squeeze-and-Excitation Attention Gate.

## Dataset
HAM10000 / ISIC (Class distribution highly imbalanced, mitigated via Focal Loss).

## Training Procedure
- **Optimizer**: AdamW
- **Scheduler**: Cosine with Warmup
- **Loss**: Focal Loss (with dynamic alpha scaling)
- **Hardware**: Lenovo LOQ (CUDA)

## Evaluation Metrics
- Macro-AUC
- Macro-F1
- Balanced Accuracy
- Class-wise Recall

## Known Limitations
- Model performance highly dependent on lighting and background artifacts.
- Low performance on heavily hair-occluded lesions if augmentation fails.

## Ethical Considerations
- Bias toward lighter skin tones in the HAM10000 dataset.
- Not a replacement for a dermatologist's biopsy.

## Clinical Use Disclaimer
This tool is strictly for research purposes. It is **not** an FDA-approved diagnostic tool and should not be used in a clinical setting without physician oversight.

## Future Improvements
- Federated learning implementation across multiple clinics to mitigate bias.
- Real-time video stream support for live dermoscopy.
