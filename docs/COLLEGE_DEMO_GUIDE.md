# College Project Demonstration & Viva Guide

**Project Title**: Advanced Deep Learning for Non-Melanoma Dermoscopic Classification  
**Target Audience**: Academic Review Committee, Faculty Evaluators & Viva Panel  
**Primary Demonstrator Workstation**: Apple MacBook Pro M4 (Localhost Demo)  

---

## 1. Executive Summary for Evaluators

This project presents a rigorous empirical study addressing the automated classification of skin lesions into 7 diagnostic categories using the HAM10000 benchmark dataset. We designed and implemented a novel **Decoupled Dual-Branch CNN** architecture intended to separate high-frequency surface texture features (Shallow-Wide branch) from macroscopic lesion morphology (Deep-Narrow branch). Through systematic experimentation against strong single-branch baselines (EfficientNet-B4, DenseNet-121, ResNet-50), we found that **EfficientNet-B4 compound scaling achieved superior classification performance (95.92% ROC-AUC, 79.16% Balanced Accuracy, 73.64% Overall Accuracy)**. Adhering to scientific integrity, EfficientNet-B4 was deployed in our real-time interactive web application featuring **Grad-CAM model attribution visualizations**.

---

## 2. Step-by-Step 16-Point Demonstration Script

### Step 1: Open the Application (`http://localhost:3000`)
- **Action**: Show the Home Page on the projector/screen.
- **Talking Point**: *"Good morning, esteemed committee members. Today we present our project on AI-assisted dermoscopic lesion classification. Our system focuses on both non-melanoma malignancies (BCC, AKIEC) and melanoma."*

### Step 2: Explain the Clinical Problem
- **Talking Point**: *"Skin cancer is among the most prevalent human cancers globally. While Melanoma is the deadliest, Non-Melanoma Skin Cancers (BCC, SCC) account for over 80% of all cases. Early non-invasive detection significantly improves patient survival and reduces disfigurement."*

### Step 3: Navigate to the Research Page (`/research`)
- **Talking Point**: *"Let us examine the dataset underpinning this research: the HAM10000 benchmark comprising 10,015 dermoscopic images."*

### Step 4: Address Class Imbalance & Data Leakage
- **Talking Point**: *"Notice the severe real-world class imbalance: Melanocytic Nevi (NV) represents 67.0% of images, while rare lesions like Dermatofibroma (DF) constitute just 1.1% (a 58:1 ratio). To prevent artificial over-optimism, we implemented two key controls: (1) Patient-aware stratified splitting by `lesion_id` to prevent identical lesions appearing in both train and test sets, and (2) Class-weighted Focal Loss (\(\gamma=2.0\)) to force the gradient to focus on hard, rare malignancies."*

### Step 5: Navigate to Architecture Page (`/architecture`)
- **Talking Point**: *"Here is our core architectural contribution: the Decoupled Dual-Branch CNN."*

### Step 6: Explain the Dual-Branch Hypothesis
- **Talking Point**: *"Standard CNNs collapse all visual signals into a single pathway. We hypothesized that dermatologists naturally decouple two types of visual cues: local micro-textures (such as arborizing vessels in BCC or keratin pearls in SCC) versus macroscopic lesion geometry (such as border asymmetry and lesion diameter)."*

### Step 7: Detail the Branch Topology
- **Talking Point**: *"To mirror this clinical reasoning:
  1. The **Shallow-Wide Branch** maintains a 1024-channel width with minimal downsampling to preserve fine spatial details.
  2. The **Deep-Narrow Branch** uses 4 residual stages to expand the receptive field while restricting channel width to 64/128/256 dimensions.
  3. The **Attention-Gated Fusion Module** concatenates both representations and applies a learned sigmoid gate before MLP classification."*

### Step 8: Navigate to the Dashboard Page (`/dashboard`)
- **Talking Point**: *"Now let us review the empirical results recorded during training on our dedicated workstation."*

### Step 9: Present the Benchmark Comparison
- **Talking Point**: *"We benchmarked 6 configurations:
  - **EfficientNet-B4**: 95.92% ROC-AUC | 79.16% Balanced Accuracy | 73.64% Accuracy | 8.83 ms latency (Rank 1)
  - **DenseNet-121**: 95.31% ROC-AUC | 79.14% Balanced Accuracy | 66.36% Accuracy (Rank 2)
  - **Dual-Branch CNN (Seed 123)**: 90.98% ROC-AUC | 70.31% Balanced Accuracy | 55.50% Accuracy (Rank 3)
  - **ResNet-50**: 93.52% ROC-AUC | 75.13% Balanced Accuracy | 56.62% Accuracy (Rank 6)"*

### Step 10: Explain Why EfficientNet-B4 Was Selected for Deployment
- **Talking Point**: *"Scientific integrity is central to our work. Although the Dual-Branch CNN attained respectable discrimination (AUC ~91%), our diagnostics revealed that compound scaling in EfficientNet-B4 balanced depth, width, and resolution more effectively across rare classes with 3x lower inference latency. Therefore, we deployed the highest-performing model for our live application."*

### Step 11: Navigate to the Classify Page (`/classify`)
- **Talking Point**: *"Let us now test the live interactive classification engine running on this MacBook M4 with Apple MPS hardware acceleration."*

### Step 12: Demonstrate Sample Preset Classification
- **Action**: Click the **MEL (Melanoma)** or **BCC (Basal Cell Carcinoma)** demo preset.
- **Talking Point**: *"The client uploads the image to our FastAPI backend, which preprocesses it to 224x224 and feeds it through EfficientNet-B4 in under 20 milliseconds."*

### Step 13: Review Prediction & Top-3 Probabilities
- **Talking Point**: *"The model outputs a full probability distribution and highlights the top-3 diagnostic candidates. Rather than presenting a single absolute label, top-3 probabilities provide clinician-friendly differential support."*

### Step 14: Demonstrate Grad-CAM Model Attribution
- **Talking Point**: *"To prevent 'black-box' opacity, we compute Gradient-weighted Class Activation Mapping (Grad-CAM) at the final convolutional feature layer. Warmer red/yellow regions illustrate the specific spatial features that drove the model's confidence."*

### Step 15: Highlight Limitations & Medical Safety
- **Talking Point**: *"We emphasize strict medical safety: our user interface and API outputs enforce an explicit disclaimer stating this is an academic research prototype, not a medical device."*

### Step 16: Conclude Presentation & Open for Questions
- **Talking Point**: *"In conclusion, we have built a complete, reproducible, and verifiable deep learning pipeline spanning architectural research, empirical ablation, and real-time explainable deployment. Thank you, and we welcome your questions."*

---

## 3. High-Yield Viva Questions & Answers

**Q1: Why did you use Focal Loss instead of standard Cross-Entropy?**  
*Answer*: *"Standard Cross-Entropy treats all samples equally. In HAM10000, where 67% of images are benign Nevi (NV), Cross-Entropy allows the vast majority class to overwhelm the gradient, resulting in poor sensitivity on rare life-threatening cancers (like Dermatofibroma or Vascular lesions). Focal Loss introduces a dynamic scaling factor \((1 - p_t)^\gamma\) that down-weights easy examples and concentrates gradients on hard minority classes."*

**Q2: What is the clinical reason behind the Dual-Branch architecture?**  
*Answer*: *"Dermatologists assess lesions on two physical scales: macroscopic morphology (asymmetry, border irregularity, global color variegation) and microscopic textures (arborizing telangiectasia, pigment networks, blue-white veils). Standard CNNs pool away spatial textures as they deepen. Our Dual-Branch physically decoupled a shallow-wide pathway (to preserve spatial texture resolution) from a deep-narrow pathway (for global receptive field morphology)."*

**Q3: Why did EfficientNet-B4 outperform the Dual-Branch CNN?**  
*Answer*: *"Our fusion diagnostics revealed two key factors: (1) Gate saliency imbalance: the attention gate assigned over 78% weight to the deep structural branch, leaving texture features under-leveraged; (2) Compound scaling: EfficientNet-B4 uniformly scales depth, width, and image resolution using principled neural architecture search coefficients, which proved more resilient to high intra-class variance than our handcrafted two-branch topology."*

**Q4: How does Grad-CAM work in your system?**  
*Answer*: *"Grad-CAM computes the gradient of the predicted class score with respect to the feature maps of the model's final convolutional layer. These gradients are globally pooled to obtain importance weights for each feature channel, followed by a ReLU operation to isolate positive contributing activations. The resulting heatmap is resized and blended over the original image."*

**Q5: What is patient-aware splitting and why is it mandatory?**  
*Answer*: *"In clinical dermoscopy datasets, a patient often has multiple photographs of the same lesion taken under different lighting or angles. If you perform random image splitting, images of the same patient end up in both train and test splits, causing severe data leakage where the model memorizes patient skin tone rather than pathological features. We strictly partitioned images by `lesion_id` to guarantee patient independence across splits."*
