# Final College Live Demonstration Checklist & Script (5–10 Minutes)

**Project**: Dual-Branch CNN Framework for Non-Melanoma Dermoscopic Classification  
**Target Evaluation**: Final College Viva & Review Committee  
**Workstation**: MacBook Pro (Apple Silicon MPS / Localhost Demo)  

---

## 1. Pre-Demo Setup Checklist (2 Minutes Before Evaluation)

- [ ] **Terminal 1 (Backend)**:
  ```bash
  source .venv/bin/activate
  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
  ```
  *Verify*: Open `http://127.0.0.1:8000/health` → Returns `{"status": "healthy", "model_name": "EfficientNet-B4"}`.
- [ ] **Terminal 2 (Frontend)**:
  ```bash
  cd app/skin-lesion-app
  npm run dev
  ```
  *Verify*: Open `http://localhost:3000` → Loads cleanly.
- [ ] **Browser Tabs Prepared**:
  1. `http://localhost:3000` (Home Page)
  2. `http://localhost:3000/classify` (Live Classifier)
  3. `http://localhost:3000/dashboard` (Empirical Benchmarks)
  4. `http://localhost:3000/architecture` (Dual-Branch Design)

---

## 2. Step-by-Step 11-Stage Live Presentation Script (5–10 Minutes)

### 1. Introduce the Problem (0:00 – 1:00)
- **Action**: Display Home Page (`/`).
- **Talking Point**: *"Good morning, esteemed committee members. Skin cancer is one of the most prevalent malignancies worldwide. While Melanoma is the deadliest form, Non-Melanoma Skin Cancers (Basal Cell Carcinoma and Actinic Keratoses) account for the vast majority of cases. Our objective is automated dermoscopic classification across 7 lesion categories to aid early non-invasive screening."*

### 2. Explain Dataset & Class Imbalance (1:00 – 2:00)
- **Action**: Navigate to Research Page (`/research`).
- **Talking Point**: *"We benchmarked on HAM10000 (10,015 images). Notice the severe real-world class imbalance: 67% of images are benign Melanocytic Nevi, while rare lesions like Dermatofibroma account for just 1.1%. To ensure scientific rigor, we implemented two key controls: (1) Patient-aware stratified splitting by `lesion_id` to prevent identical lesion leakage, and (2) Class-weighted Focal Loss (\(\gamma=2.0\)) to prioritize hard minority classes."*

### 3. Explain the Dual-Branch Research Hypothesis (2:00 – 3:30)
- **Action**: Navigate to Architecture Page (`/architecture`).
- **Talking Point**: *"Our core research hypothesis asked: can we mirror clinical dermatological assessment by decoupling visual analysis into two distinct physical pathways?
  - A **Shallow-Wide Branch** (1024 channels, minimal downsampling) to preserve fine micro-textures like arborizing vessels and keratin pearls.
  - A **Deep-Narrow Branch** (256 channels, 4 residual stages) to evaluate global asymmetry and border irregularity.
  - An **Attention-Gated Fusion Module** to adaptively weight texture versus structure."*

### 4. Present Benchmark Comparison & Empirical Results (3:30 – 5:00)
- **Action**: Navigate to Dashboard Page (`/dashboard`).
- **Talking Point**: *"We trained and benchmarked 6 configurations:
  - **EfficientNet-B4**: 95.92% ROC-AUC | 79.16% Balanced Accuracy | 73.64% Top-1 Accuracy (Rank 1 Champion)
  - **DenseNet-121**: 95.31% ROC-AUC | 79.14% Balanced Accuracy | 66.36% Accuracy (Rank 2)
  - **Dual-Branch CNN**: 90.98% ROC-AUC | 70.31% Balanced Accuracy | 55.50% Accuracy (Rank 3)
  - **ResNet-50**: 93.52% ROC-AUC | 75.13% Balanced Accuracy | 56.62% Accuracy (Rank 6)
  Our fusion diagnostics revealed that the attention gate assigned over 78% weight to structure, causing fusion collapse. EfficientNet-B4 compound scaling uniformly scaled depth, width, and resolution, capturing both scales more effectively. Therefore, adhering to scientific integrity, **EfficientNet-B4 was chosen as our production deployment model**."*

### 5. Live Image Upload & Classification (5:00 – 6:30)
- **Action**: Navigate to Classify Page (`/classify`). Click demo preset **MEL (Melanoma)** or upload an image.
- **Talking Point**: *"Let us demonstrate the live classification pipeline. The client sends the image to our FastAPI backend, which preprocesses it and runs inference in under 20 milliseconds on Apple Silicon MPS."*

### 6. Review Top-3 Probabilities (6:30 – 7:30)
- **Talking Point**: *"Rather than providing a single black-box decision, our system outputs a complete probability distribution and highlights the top-3 ranked differential diagnoses, giving clinicians nuanced multi-class support."*

### 7. Demonstrate Grad-CAM Attribution Overlay (7:30 – 8:30)
- **Action**: Point to the Grad-CAM visualization beside the original image.
- **Talking Point**: *"For clinical interpretability, we compute Gradient-weighted Class Activation Mapping (Grad-CAM) at the final convolutional feature head. Warmer regions show the precise spatial features driving the model's prediction."*

### 8. Address Limitations & Medical Disclaimer (8:30 – 9:15)
- **Talking Point**: *"We enforce strict ethical boundaries: our application displays an explicit academic disclaimer. HAM10000 is biased toward fair skin types (Fitzpatrick I–III), so prospective multi-center validation is essential before any clinical deployment."*

### 9. Conclude & Take Questions (9:15 – 10:00)
- **Talking Point**: *"In summary, we conducted a rigorous deep learning study, documented an insightful negative architectural finding, and built a complete, fully tested, explainable deployment application. Thank you, and we welcome your questions."*
