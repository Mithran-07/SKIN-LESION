# Master Repository Report

## Repository Structure
This repository serves as the definitive, frozen master copy of the **Dual-Branch CNN Framework for Non-Melanoma Dermoscopic Classification** research project. All architecture development, hyperparameter tuning, and dataset processing have been concluded. 

## Final Project Status
- **Architecture**: Frozen.
- **Training**: Completed.
- **Analysis**: Ready for final review.
- **Publication**: Drafts prepared in `paper/` and `thesis/`.

## Archived Experiments
All experimental logs and intermediate checkpoints have been permanently archived on the Lenovo LOQ. Only the final aggregated benchmarks, metrics, and core weights have been synchronized to this repository.

## Models Synced
- **Best Baseline**: ResNet50 (`best_resnet50.pth`)
- **Dual-Branch V1**: Core architecture (`best_dual_branch_v1.pth`)
- **Dual-Branch V1.1**: Added SE-Net Attention Fusion (`best_dual_branch_v1_1.pth`)
- **Dual-Branch V2**: Integrated Multi-Task Learning Segmentation Head (`best_dual_branch_v2.pth`)

## Final Research Conclusion
Decoupling structural and textural feature extraction via the Dual-Branch CNN yields measurable improvements in diagnostic confidence and Grad-CAM interpretability for non-melanoma dermoscopic classification. Conformal prediction successfully bounds uncertainty for out-of-distribution artifacts.

## Future Work
- Expanding to federated learning across clinical silos.
- Incorporating EHR (Electronic Health Record) tabular data.
- Live deployment pipeline for real-time dermoscopy video inference.
