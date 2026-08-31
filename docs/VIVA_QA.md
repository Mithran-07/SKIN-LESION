# College Viva Preparation & Technical Defense Guide

**Project**: Dual-Branch CNN Framework for Non-Melanoma Dermoscopic Classification  
**Target Examination**: Final Semester Project Viva & Review Committee  

---

## 1. Dataset & Preprocessing

### Q1: What dataset was used and what are its key characteristics?
**Answer**: We used the **HAM10000** (*Human Against Machine with 10,000 training images*) benchmark dataset collected from the Medical University of Vienna and Cliff Rosendahl's practice in Australia. It contains 10,015 high-resolution dermatoscopic images across 7 diagnostic categories: Melanoma (MEL, 1,113), Melanocytic Nevi (NV, 6,705), Basal Cell Carcinoma (BCC, 514), Actinic Keratoses (AKIEC, 327), Benign Keratosis (BKL, 1,099), Dermatofibroma (DF, 115), and Vascular Lesions (VASC, 142).

### Q2: Why is random image splitting catastrophic in dermoscopy, and how did you prevent it?
**Answer**: In clinical dermatology datasets, multiple photographs are frequently taken of the same physical lesion under different angles, zoom levels, or follow-up dates. If you perform a random image split, images of the exact same lesion appear in both train and test splits. The network then achieves artificially inflated accuracy by memorizing patient-specific skin characteristics (hair patterns, background lighting, skin pigmentation) rather than pathological biomarkers. We strictly enforced **patient-aware splitting grouped on `lesion_id`** (70% train, 15% val, 15% test) to guarantee complete lesion independence.

### Q3: How did you address the 58:1 class imbalance between NV and DF?
**Answer**: We used a three-tier strategy:
1. **Class-Weighted Focal Loss**: Applied inverse class frequency weighting \(\alpha_c\) and a focusing parameter \(\gamma=2.0\) to dynamically scale down the loss contribution from easy majority samples (NV) and amplify gradients from rare minority classes (DF, VASC, AKIEC).
2. **Balanced Weighted Random Sampling**: Upsampled rare classes during DataLoader mini-batch creation.
3. **Imbalance-Aware Metrics**: Evaluated models using Balanced Accuracy, Macro F1, and Multi-Class Macro ROC-AUC rather than raw Top-1 Accuracy.

### Q4: What dermoscopy-specific augmentations were used?
**Answer**: We built an Albumentations pipeline incorporating:
- **`HairAugmentation`**: Synthesizes realistic Bezier-curve hair strands overlaid across lesions to regularize against hair artifact spurious correlations.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances subtle vascular and pigment network contrast.
- **Elastic Transformations & Random Resized Crops**: Simulates biological deformation and varying dermatoscope contact pressure.

---

## 2. Model Architectures & Deep Learning Theory

### Q5: What was the motivation behind the Decoupled Dual-Branch CNN?
**Answer**: Clinical dermoscopy evaluation relies on two distinct visual scales:
1. **Micro-Textures**: High-frequency localized patterns (arborizing telangiectasia in BCC, keratin pearls in SCC, pigment network mesh regularity).
2. **Macro-Morphology**: Low-frequency global structural geometry (border irregularity, asymmetry, overall lesion diameter).
Standard CNNs continuously pool away spatial dimensions as they deepen, destroying high-frequency textures. We designed a **Shallow-Wide Branch** (1024 channels, minimal pooling) to preserve texture resolution, and a **Deep-Narrow Branch** (256 channels, 4 residual stages) to evaluate global morphology, combined via **Attention-Gated Fusion**.

### Q6: Why did EfficientNet-B4 outperform the Dual-Branch CNN?
**Answer**: Our empirical diagnostics identified two fundamental mechanisms:
1. **Compound Scaling**: EfficientNet-B4 uses neural architecture search (NAS) to uniformly balance network depth ($\alpha^\phi$), width ($\beta^\phi$), and resolution ($\gamma^\phi$). This compound scaling captures both fine local textures and high-level semantics within a unified MBConv depthwise separable hierarchy without artificial branch boundaries.
2. **Attention Fusion Collapse**: Saliency diagnostics on the attention gate revealed that gate weights skewed heavily toward the deep structural branch (>78%). This starved the shallow-wide texture branch of gradient flow, resulting in high gradient variance and suboptimal texture feature learning.

### Q7: What is Focal Loss and how does it mathematically operate?
**Answer**: Standard Cross-Entropy loss is:
\[
\text{CE}(p_t) = -\log(p_t)
\]
Focal Loss introduces a modulating factor \((1 - p_t)^\gamma\) and class weights \(\alpha_t\):
\[
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
\]
When a sample is well-classified ($p_t \to 1$), the modulating factor \((1 - p_t)^\gamma \to 0\), suppressing the loss. When a minority or difficult sample is misclassified ($p_t \to 0$), the factor approaches 1, forcing the network parameters to update based on hard samples. We set \(\gamma = 2.0\).

---

## 3. Explainability, Uncertainty & Deployment

### Q8: How does Grad-CAM work in your deployed system?
**Answer**: Gradient-weighted Class Activation Mapping (Grad-CAM) computes the gradient of the predicted class score $y^c$ with respect to the feature activation maps $A^k$ of the final convolutional layer:
\[
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}
\]
A weighted combination followed by a ReLU operation produces the spatial heatmap:
\[
L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
\]
The heatmap is bilinearly upsampled to $224 \times 224$, color-mapped using Jet colormap, and blended over the original dermoscopic image to highlight the spatial regions most influential to the model's score.

### Q9: Why is raw Top-1 Accuracy misleading in medical imaging?
**Answer**: If a model simply predicts "Melanocytic Nevus" for every image in HAM10000, it would achieve 67.0% accuracy while failing to detect 100% of Melanomas and Basal Cell Carcinomas. In clinical settings, **Balanced Accuracy** (the unweighted mean of per-class recalls) and **Macro ROC-AUC** are essential because they penalize models that sacrifice sensitivity on rare, life-threatening malignancies.

### Q10: What is Split Conformal Prediction?
**Answer**: Standard deep learning outputs point probabilities that can be overconfident. Split Conformal Prediction uses a held-out calibration set to construct a mathematically guaranteed prediction set $C(X)$ such that the true label $Y$ is contained within $C(X)$ with a user-defined coverage probability:
\[
P(Y \in C(X)) \ge 1 - \alpha \quad (\text{e.g., } 90\% \text{ coverage for } \alpha=0.1)
\]
If a lesion is ambiguous, the prediction set outputs multiple candidate classes, alerting clinicians to clinical uncertainty.

### Q11: What is the production application stack?
**Answer**:
- **Backend**: FastAPI REST server with CORS, request validation (15MB limit, MIME verification), and Apple Silicon MPS / CUDA / CPU auto-detection.
- **Frontend**: Next.js 14 App Router application with React 18, Tailwind CSS, and Lucide React icons.
- **Deployment Model**: EfficientNet-B4 (17.56M parameters, 8.83 ms inference latency on Apple Silicon).

### Q12: What are the primary clinical limitations of this work?
**Answer**:
1. **Demographic Bias**: HAM10000 was collected primarily from fair-skinned European populations (Fitzpatrick skin types I–III). Performance on darker skin phototypes requires external validation.
2. **Prototype Status**: The application is an academic research prototype intended for educational and decision-support exploration, not a certified Software as a Medical Device (SaMD).
