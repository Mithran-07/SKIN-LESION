# Project Launch & Execution Manual (MacBook M4)

**Project Root**: `/Users/mithran/Documents/My projects/ADL`  
**Operating System**: macOS (Apple Silicon M4, MPS Hardware Acceleration)  
**Python Environment**: `/Users/mithran/Documents/My projects/ADL/.venv`  
**Node.js**: v24.19.0 / npm 11.17.0  

---

## 1. Quick Start (Running Backend & Frontend)

### Step 1: Start the FastAPI Backend Server
Open **Terminal Tab 1** and run:
```bash
cd "/Users/mithran/Documents/My projects/ADL"
source .venv/bin/activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
*Backend will be live at*: `http://127.0.0.1:8000`  
*Interactive Swagger Documentation*: `http://127.0.0.1:8000/docs`

---

### Step 2: Start the Next.js Frontend
Open **Terminal Tab 2** and run:
```bash
cd "/Users/mithran/Documents/My projects/ADL/app/skin-lesion-app"
npm run dev
```
*Frontend application will be accessible at*: `http://localhost:3000`

---

## 2. Running Automated Tests

### Run Full PyTorch Deep Learning Test Suite:
```bash
cd "/Users/mithran/Documents/My projects/ADL"
.venv/bin/python -m pytest tests/test_model_shapes.py tests/test_focal_loss.py tests/test_conformal.py -v
```

### Run FastAPI Backend Test Suite (30 Assertions):
```bash
cd "/Users/mithran/Documents/My projects/ADL"
.venv/bin/python scripts/test_api_suite.py
```

### Run Real HAM10000 7-Class Image Validation Matrix:
```bash
cd "/Users/mithran/Documents/My projects/ADL"
.venv/bin/python scripts/validate_real_images.py
```

---

## 3. Frontend Production Build & Deployment

To verify or run the optimized Next.js production build:
```bash
cd "/Users/mithran/Documents/My projects/ADL/app/skin-lesion-app"
npm run build
npm run start
```

---

## 4. Troubleshooting Guide

- **Port 8000 Already in Use**:
  ```bash
  lsof -i :8000 | awk 'NR>1 {print $2}' | xargs kill -9 2>/dev/null || true
  ```
- **Port 3000 Already in Use**:
  ```bash
  lsof -i :3000 | awk 'NR>1 {print $2}' | xargs kill -9 2>/dev/null || true
  ```
- **Installing / Re-installing Backend Dependencies**:
  ```bash
  cd "/Users/mithran/Documents/My projects/ADL"
  .venv/bin/pip install fastapi uvicorn python-multipart timm pytest requests pydantic torch torchvision
  ```
