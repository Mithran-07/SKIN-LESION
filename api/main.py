"""
FastAPI Backend — Skin Lesion Classification API
Research demo application for EfficientNet-B4 dermoscopic classifier.
"""

import io
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "efficientnet_b4" / "best_checkpoint.pth"

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Skin Lesion Classification API",
    description=(
        "Academic research prototype for dermoscopic skin lesion classification "
        "using EfficientNet-B4 trained on HAM10000. NOT for clinical use."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model loading ─────────────────────────────────────────────────────────
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from api.inference_engine import InferenceEngine
        if not CHECKPOINT_PATH.exists():
            raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        logger.info(f"Loading model from {CHECKPOINT_PATH}")
        _engine = InferenceEngine(CHECKPOINT_PATH)
    return _engine


# ── Validation ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


def validate_image(file: UploadFile, data: bytes) -> Image.Image:
    # Size check
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )
    # Extension check
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{suffix}'. Allowed: JPG, JPEG, PNG.",
        )
    # Open and validate image
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))  # Re-open after verify()
        if img.mode not in ("RGB", "RGBA", "L"):
            raise ValueError("Unsupported image mode")
        return img
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}",
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        engine = get_engine()
        return {
            "status": "ok",
            "model": "EfficientNet-B4",
            "device": str(engine.device),
            "checkpoint": CHECKPOINT_PATH.name,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "error": str(e)},
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify a dermoscopic image.

    Returns predicted class, probability, top-3 predictions, and all class probabilities.

    **Disclaimer:** This is an academic research prototype. Results are NOT medical advice.
    """
    start = time.time()
    data = await file.read()
    img = validate_image(file, data)

    try:
        engine = get_engine()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {str(e)}")

    try:
        result = engine.predict(img)
        result["inference_time_ms"] = round((time.time() - start) * 1000, 2)
        result["disclaimer"] = (
            "This system is an academic research prototype and is not intended "
            "to provide medical diagnosis or replace professional medical advice."
        )
        return result
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/explain")
async def predict_with_explain(file: UploadFile = File(...)):
    """
    Classify a dermoscopic image and return Grad-CAM explainability heatmap.

    Returns prediction + base64-encoded heatmap overlay image.
    Grad-CAM highlights regions that contributed most to the prediction.
    """
    start = time.time()
    data = await file.read()
    img = validate_image(file, data)

    try:
        engine = get_engine()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {str(e)}")

    try:
        result = engine.predict_with_gradcam(img)
        result["inference_time_ms"] = round((time.time() - start) * 1000, 2)
        result["disclaimer"] = (
            "This system is an academic research prototype and is not intended "
            "to provide medical diagnosis or replace professional medical advice."
        )
        result["gradcam_note"] = (
            "Highlighted regions represent areas that contributed strongly to "
            "the model's prediction. This is not clinical evidence."
        )
        return result
    except Exception as e:
        logger.exception("Grad-CAM inference failed")
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")


@app.get("/benchmark")
async def benchmark_data():
    """Return benchmark results for all evaluated models."""
    return {
        "models": [
            {
                "name": "ResNet50",
                "version": "Baseline",
                "parameters": 23522375,
                "test_accuracy": 0.5662,
                "balanced_accuracy": 0.7513,
                "macro_f1": 0.5352,
                "roc_auc": 0.9352,
                "training_time_s": 1852.7,
                "inference_ms": 20.96,
            },
            {
                "name": "DenseNet121",
                "version": "Baseline",
                "parameters": 6961031,
                "test_accuracy": 0.6636,
                "balanced_accuracy": 0.7914,
                "macro_f1": 0.6242,
                "roc_auc": 0.9531,
                "training_time_s": 1476.1,
                "inference_ms": 20.37,
            },
            {
                "name": "EfficientNet-B4",
                "version": "Baseline (Final)",
                "parameters": 17561167,
                "test_accuracy": 0.7364,
                "balanced_accuracy": 0.7916,
                "macro_f1": 0.6919,
                "roc_auc": 0.9592,
                "training_time_s": 2337.3,
                "inference_ms": 8.83,
            },
            {
                "name": "Dual-Branch CNN",
                "version": "V1 (avg 3 seeds)",
                "parameters": 10669639,
                "test_accuracy": 0.5479,
                "balanced_accuracy": 0.6844,
                "macro_f1": 0.4641,
                "roc_auc": 0.9041,
                "training_time_s": 23131,
                "inference_ms": 27.3,
            },
            {
                "name": "Dual-Branch CNN",
                "version": "V1.1 (Training Improvements)",
                "parameters": 10669639,
                "test_accuracy": 0.6576,
                "balanced_accuracy": 0.6218,
                "macro_f1": 0.4814,
                "roc_auc": 0.9006,
                "training_time_s": 10272.5,
                "inference_ms": 24.56,
            },
            {
                "name": "Dual-Branch CNN",
                "version": "V2 (Fusion Redesign)",
                "parameters": 9031241,
                "test_accuracy": 0.6424,
                "balanced_accuracy": 0.5948,
                "macro_f1": 0.4950,
                "roc_auc": 0.9015,
                "training_time_s": 10869.5,
                "inference_ms": 25.54,
            },
        ],
        "final_model": "EfficientNet-B4",
        "dataset": "HAM10000",
        "classes": 7,
    }


@app.get("/classes")
async def class_info():
    """Return class definitions and descriptions."""
    from api.inference_engine import CLASS_NAMES, CLASS_DISPLAY_NAMES, CLASS_DESCRIPTIONS
    return {
        CLASS_NAMES[i]: {
            "index": i,
            "label": CLASS_NAMES[i],
            "display_name": CLASS_DISPLAY_NAMES[CLASS_NAMES[i]],
            "description": CLASS_DESCRIPTIONS[CLASS_NAMES[i]],
        }
        for i in range(7)
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
