# Final Results: Dual-Branch CNN Research

## 1. Experimental Setup
We evaluated the efficacy of a novel Dual-Branch CNN framework (combining WideResNet-50-2 for textural extraction and DenseNet121 for structural extraction) on an imbalanced dermoscopic dataset (AKIEC, BCC, BKL, DF, MEL, NV, VASC). All models were trained using PyTorch with AMP, early stopping, and consistent augmentations. 

## 2. Baseline Summary
- **ResNet50:** Achieved 56.62% Test Accuracy.
- **DenseNet121:** Achieved 66.36% Test Accuracy.
- **EfficientNet-B4:** Achieved **73.64% Test Accuracy** (Our strongest baseline).

## 3. Dual-Branch V1
The initial Dual-Branch framework utilized an attention gate to merge representations. It achieved an average accuracy of **54.79%**. Diagnostics revealed severe branch collapse, where the Texture branch was aggressively suppressed in favor of the Structure branch.

## 4. Dual-Branch V1.1
We introduced class-balanced focal loss and label smoothing to fix the optimization environment. Performance improved to **65.76%**, but the underlying fusion gate collapse remained unchanged.

## 5. Dual-Branch V2
We hypothesized that the $1280 	imes 1280$ cross-dimensional gate was mathematically biased towards the 1024-d Texture branch, causing optimization instability. V2 replaced this with independent scalar gates. However, performance degraded slightly to **64.24%**, and the network STILL collapsed the texture branch.

## 6. Statistical Comparison
Compared to EfficientNet-B4, Dual-Branch V1 exhibits a massive performance deficit (Cohen's d = -23.28 for accuracy). Even the optimal V1.1 variant sits roughly 8% below the EfficientNet-B4 baseline.

## 7. Key Observations
- Explicitly forcing the extraction of texture and structure into separate backbone networks is inefficient for this dataset.
- A well-designed single-branch network (EfficientNet-B4) naturally captures multimodal feature interactions much better than our bipartite fusion design.
- The fusion gate collapse is not a bug; it is the network rejecting unhelpful textural representations.

## 8. Limitations
- The dataset is heavily imbalanced (mostly NV).
- We did not utilize pre-trained weights for the fusion block.

## 9. Final Conclusion
The core hypothesis is falsified. The Dual-Branch framework introduces massive computational overhead while strictly degrading diagnostic correctness. The Lenovo LOQ research project is officially frozen, and no further development on this architecture will proceed.
