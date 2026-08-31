# Model Architecture Documentation

This document provides a comprehensive theoretical and mathematical breakdown of the Dual-Branch CNN framework and its associated components.

## 1. ShallowWideBranch

### Mathematical Intuition
The Shallow-Wide Branch is designed to act as a dense, high-frequency filter bank. By minimizing spatial pooling and maximizing channel width early in the network, it avoids the spatial compression that typically destroys fine-grained textural cues (e.g., arborizing vessels, pigment networks).

Let $X \in \mathbb{R}^{3 \times H \times W}$ be the input image. The branch applies a series of convolutions without deep bottlenecks:
$H_1 = \text{ReLU}(\text{BN}(W_1 * X)) \quad \text{where } W_1 \in \mathbb{R}^{256 \times 3 \times 3 \times 3}$
$H_2 = \text{Pool}(\text{ReLU}(\text{BN}(W_2 * H_1))) \quad \text{where } W_2 \in \mathbb{R}^{512 \times 256 \times 3 \times 3}$
$H_3 = \text{ReLU}(\text{BN}(W_3 * H_2)) \quad \text{where } W_3 \in \mathbb{R}^{1024 \times 512 \times 3 \times 3}$

The final texture embedding is obtained via Global Average Pooling (GAP):
$t = \frac{1}{H'W'} \sum_{i,j} H_3(:, i, j) \in \mathbb{R}^{1024}$

### Tensor Shapes
*   **Input**: $(B, 3, 224, 224)$
*   **Block 1**: $(B, 256, 224, 224)$
*   **Block 2**: $(B, 512, 112, 112)$
*   **Block 3**: $(B, 1024, 112, 112)$
*   **Output Vector**: $(B, 1024)$

### Computational Complexity & Parameters
*   **Parameters**: ~5.9M (mostly in Block 3: $512 \times 1024 \times 9$)
*   **FLOPs**: Extremely high due to wide channels at high resolution ($224\times224$ and $112\times112$).

### Advantages & Limitations
*   **Advantages**: Preserves micro-textures essential for distinguishing melanoma from nevi based on dermatoscopic structures.
*   **Limitations**: Computationally expensive and highly memory-bound during the forward pass.

---

## 2. DeepNarrowBranch

### Mathematical Intuition
The Deep-Narrow Branch captures macroscopic, global structural morphology (e.g., asymmetry, border irregularity). By enforcing a channel bottleneck (base 64 channels), it operates as a low-pass filter, forcing the network to distill abstract semantic features rather than memorizing local textures.

It utilizes residual connections $F(x) + x$ over 4 stages to safely increase the receptive field without vanishing gradients.

### Tensor Shapes
*   **Input**: $(B, 3, 224, 224)$
*   **Stem**: $(B, 64, 56, 56)$
*   **Stage 1**: $(B, 64, 56, 56)$
*   **Stage 2**: $(B, 128, 28, 28)$
*   **Stage 3**: $(B, 128, 28, 28)$
*   **Stage 4**: $(B, 256, 14, 14)$
*   **Bottleneck/Output**: $(B, 256)$

### Computational Complexity & Parameters
*   **Parameters**: ~2.5M
*   **FLOPs**: Low, due to early spatial downsampling (stride 2 stem, stride 2 pooling) and narrow channel widths.

### Advantages & Limitations
*   **Advantages**: Highly efficient; effectively captures global shape descriptors.
*   **Limitations**: Blind to fine textures (e.g., keratinization) due to the low-resolution feature maps ($14 \times 14$).

---

## 3. AttentionFusionHead

### Mathematical Intuition
The fusion module dynamically weighs the importance of texture versus structure depending on the lesion. It uses an SE-style squeeze-and-excitation bottleneck gate.

Given texture $t \in \mathbb{R}^{1024}$ and structure $s \in \mathbb{R}^{256}$:
1.  **Concatenation**: $z = [t \Vert s] \in \mathbb{R}^{1280}$
2.  **Squeeze-Excitation Gate**:
    $g = \sigma(W_{up} \cdot \text{ReLU}(W_{down} \cdot z))$
    where $W_{down} \in \mathbb{R}^{128 \times 1280}$ and $W_{up} \in \mathbb{R}^{1280 \times 128}$.
3.  **Gating**: $z' = z \odot g$
4.  **MLP Projection**: $f = \text{MLP}(z') \in \mathbb{R}^{256}$

### Computational Complexity & Parameters
*   **Gate Parameters**: $1280 \times 128 + 128 \times 1280 = 327,680$
*   **Advantages**: The bottleneck design saves ~1.3M parameters compared to a naive linear layer while providing non-linear channel-wise attention.

---

## 4. MTL Head (Multi-Task Learning)

### Mathematical Intuition
The MTL head introduces a U-Net style segmentation decoder attached to the Shallow-Wide Branch. This acts as a spatial regularizer.

Loss is computed as a weighted sum:
$\mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{focal} + \lambda_{seg} \mathcal{L}_{dice/bce}$

### Advantages & Limitations
*   **Advantages**: Forces the shallow encoder to attend strictly to the lesion boundary, eliminating background artifact bias (rulers, hair).
*   **Limitations**: Requires pixel-level ground truth masks for training.

---

## 5. Focal Loss

### Mathematical Intuition
Focal Loss dynamically scales cross-entropy based on prediction confidence, addressing class imbalance (e.g., rare vascular lesions vs. common nevi).
$\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$
where $p_t$ is the model's estimated probability for the true class. The modulating factor $(1 - p_t)^\gamma$ down-weights well-classified examples.

---

## 6. Grad-CAM

### Mathematical Intuition
Grad-CAM uses the gradients of a target concept (class logit $y^c$) flowing into the final convolutional layer to produce a coarse localization map $L^c_{Grad-CAM}$.
Neuron importance weights $\alpha_k^c$:
$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$
Heatmap generation:
$L^c_{Grad-CAM} = \text{ReLU} \left( \sum_k \alpha_k^c A^k \right)$

### Advantages
Provides post-hoc interpretability without requiring architectural changes or bounding box annotations.

---

## 7. Conformal Prediction

### Mathematical Intuition
Split Conformal Prediction guarantees marginal coverage without distribution assumptions.
1.  **Calibration**: On a hold-out set, compute non-conformity scores $s_i = 1 - \hat{p}(y_i | x_i)$.
2.  **Quantile Calculation**: Find the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile of $\{s_i\}$, denoted $\hat{q}$.
3.  **Prediction Set**: For a new test point, the prediction set includes all classes where $1 - \hat{p}(y | x) \le \hat{q}$.

### Advantages
Provides mathematically guaranteed uncertainty bounds, critical for clinical deployment where "I don't know" is a safer output than an overconfident wrong diagnosis.
