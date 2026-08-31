# Methods

## Dataset

### HAM10000 Dataset
The study uses the **HAM10000** (Human Against Machine with 10,000 training images) dataset:
- **Total samples**: 10,015 dermoscopic images
- **Image size**: 600×450 pixels (standardized)
- **Classes**: 7 skin lesion types
  - Melanoma (MEL)
  - Melanocytic Nevus (NV)
  - Basal Cell Carcinoma (BCC)
  - Actinic Keratosis (AK)
  - Benign Keratosis (BKL)
  - Dermatofibroma (DF)
  - Vascular Lesion (VASC)

### Class Distribution
|Class | Count | Percentage | Imbalance Ratio |
|------|-------|-----------|-----------------|
| NV | 6,705 | 66.8% | 48.6x |
| MEL | 1,113 | 11.1% | 7.6x |
| BKL | 1,099 | 11.0% | 7.8x |
| BCC | 514 | 5.1% | 18.4x |
| AK | 327 | 3.3% | 30.6x |
| DF | 115 | 1.1% | 86.6x |
| VASC | 142 | 1.4% | 70.5x |

**Challenge**: Extreme class imbalance (VASC/DF to NV ratio ~1:47-86), requiring specialized loss functions and sampling strategies.

## Data Augmentation

To address limited data and improve robustness:

```python
# Training-time augmentations
- Random horizontal/vertical flip
- Random rotation (-30° to +30°)
- Random color jitter (±20% brightness, contrast, saturation)
- Random elastic deformations
- Gaussian blur with σ ∈ [0.5, 1.5]

# Validation/Test: Only center crop
```

### Implementation Details
- **Augmentation library**: torchvision.transforms + albumentations
- **Probability**: 0.8 for augmentation application
- **Seed**: Fixed for reproducibility across runs

## Model Architecture

### Dual-Branch Design Philosophy

The architecture is motivated by clinical dermatology:
- **Texture Branch** ("Shallow-Wide"): Captures color patterns, pigmentation (primary feature)
- **Structure Branch** ("Deep-Narrow"): Captures morphology, borders (supporting feature)

### Texture Branch (Shallow-Wide)
```
Input (3, 224, 224)
  ↓
Conv 3×3, 64 filters
  ↓
Conv 3×3, 128 filters
  ↓
Conv 3×3, 256 filters
  ↓
AdaptiveAvgPool2D
  ↓
Feature Map (256)
```

**Rationale**: Fewer layers preserve fine color details; wider receptive field at each level.

### Structure Branch (Deep-Narrow)
```
Input (3, 224, 224)
  ↓
ResNet-34 backbone (pretrained on ImageNet)
  ↓
Remove final FC layer
  ↓
Feature Map (512)
```

**Rationale**: Deeper architecture better suited for morphological features; transfer learning from ImageNet.

### Fusion Head (Attention-Gated)

```python
# Features: texture_features (B, 256), struct_features (B, 512)

# Channel attention
texture_attention = SoftMax(Dense(256))  # Weights texture
struct_attention = SoftMax(Dense(512))   # Weights structure

# Weighted combination
combined = (texture_attention * texture_features) + 
           (struct_attention * struct_features)

# Classification
logits = Dense(7)(combined)
```

**Key feature**: Attention weights are interpretable - high texture_attention suggests color-pattern-dominant lesion.

### Multi-Task Learning Head (Optional)

For auxiliary task learning:
```python
# Shared backbone features
segmentation_decoder = UpsampleConvBlock(features → 1 channel)
seg_mask = Sigmoid(segmentation_decoder)

# Main task
class_logits = SoftMax(classification_head)
```

## Loss Functions

### Primary Task: Focal Loss

Standard cross-entropy is ineffective for imbalanced data:

$$\text{Focal Loss} = -\sum_{t=1}^{N} (1-p_t)^{\gamma} \log(p_t)$$

Where:
- $p_t$ = model's probability for true class
- $\gamma = 2.0$ (focusing parameter)
- $(1-p_t)^{\gamma}$ = modulation factor emphasizing hard negatives

**Effect**: Easy examples (high $p_t$) contribute minimally; hard examples (low $p_t$) drive training.

### Class Weights
$\alpha = \frac{\text{total_samples}}{\text{class_count}}$ for each class

Typical values:
- Vascular lesion (DF, VASC): $\alpha = 75-90$
- Rare lesions (AK): $\alpha = 30$  
- Common lesions (NV): $\alpha = 1.5$

### Auxiliary Task: Segmentation Loss (MTL variant)

$$\text{Total Loss} = \text{CE}(\text{class}) + \lambda_{\text{seg}} \cdot \text{Dice}(\text{segmentation})$$

Where $\lambda_{\text{seg}} = 0.3$ balances main and auxiliary tasks.

## Training Procedure

### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Weight decay regularization |
| Learning Rate | 1×10⁻⁴ | Low for transfer learning stability |
| Batch Size | 32 | Trade-off between memory and gradient estimates |
| Epochs | 100 | With early stopping if val loss plateaus |
| Weight Decay | 1×10⁻⁵ | L2 regularization to prevent overfitting |
| LR Scheduler | CosineAnnealing | Gradual warmup and decay |

### Class Sampling Strategy

To handle imbalance during training:

**Stratified K-Fold Split**:
- 5-fold stratification by class
- Train: 70% (7,010 images)
- Validation: 15% (1,502 images)
- Test: 15% (1,503 images)

**Weighted Sampling** (training only):
- Probability of selecting sample ∝ 1/class_count
- Ensures rare classes see sufficient training exposure

### Early Stopping
- Monitor: Validation AUC
- Patience: 15 epochs
- Best model saved to checkpoints/best_model.pth

## Evaluation Metrics

### Primary Metrics
For multi-class classification with imbalanced data:

1. **Balanced Accuracy**
$$\text{BA} = \frac{1}{N} \sum_{c=1}^{N} \text{Recall}_c$$
(Unbiased by class distribution)

2. **Macro F1-Score**
$$\text{F1}_{\text{macro}} = \frac{1}{N} \sum_{c=1}^{N} \text{F1}_c$$
(Equal weight to each class)

3. **Weighted AUC (One-vs-Rest)**
- OvR AUC for each class weighted by class frequency

### Secondary Metrics
- Accuracy (overall)
- Per-class Recall (sensitivity) and Precision
- Confusion matrix
- Grad-CAM attention maps for interpretability

### Uncertainty Quantification

**Monte Carlo Dropout**:
- Enable dropout at test time: T=20 stochastic forward passes
- Uncertainty estimate: std(predictions) across T passes
- High uncertainty → low confidence classification (flag for review)

**Conformal Prediction**:
- Build calibration set on validation data
- For new test samples, construct prediction sets with ≥90% coverage
- Clinical output: "Most likely class: MEL with supporting classes: {BCC, AK}"

## Explainability

### Grad-CAM Visualization
For each misclassification or low-confidence prediction:

$$L^c = \frac{1}{Z} \sum_k a_k^c \cdot A^k$$

Where:
- $a_k^c$ = gradient of class score w.r.t. activation $k$
- $A^k$ = activation map $k$
- Heatmap overlaid on image highlighting discriminative regions

### Attention Weight Visualization
Plot texture_attention vs struct_attention scores to show which branch dominated the decision.

## Reproducibility

### Environment
- Python 3.10.x
- PyTorch 1.13+
- CUDA 11.8 (GPU training)
- See requirements.txt for complete dependency list

### Seeds & Random State
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### Configuration
All hyperparameters stored in config/config.yaml with override support:
```bash
python scripts/train.py \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --epochs 150
```

### Model Checkpoints
- Saved every 5 epochs to checkpoints/
- Best model (max validation AUC) saved separately
- Enables checkpoint restart for interrupted training

