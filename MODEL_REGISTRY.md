# Model Registry
| Model | Status | Parameters | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC | Training Time | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| ResNet50 | Frozen | 23.5M | 0.5662 | 0.7513 | 0.5352 | 0.9352 | 1852.7s | best_checkpoint.pth |
| DenseNet121 | Frozen | 6.9M | 0.6636 | 0.7914 | 0.6242 | 0.9531 | 1476.1s | best_checkpoint.pth |
| EfficientNet-B4 | Frozen | 17.5M | 0.7364 | 0.7916 | 0.6919 | 0.9592 | 2337.3s | best_checkpoint.pth |
| Dual-Branch V1 | Frozen | 10.6M | 0.5479 (avg) | 0.6844 (avg)| 0.4641 (avg)| 0.9041 (avg)| ~23131s | best_checkpoint.pth |
| Dual-Branch V1.1 | Frozen | 10.6M | 0.6576 | 0.6218 | 0.4814 | 0.9006 | 10272.5s | best_checkpoint.pth |
| Dual-Branch V2 | Frozen | 9.0M | 0.6424 | 0.5948 | 0.4950 | 0.9015 | 10869.5s | best_checkpoint.pth |
