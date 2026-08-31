"""
FastAPI Server for Skin Lesion Classification & Research Benchmarks.

Endpoints:
- GET  /health           : System health, compute device, and model status
- GET  /benchmark        : Verified LOQ experimental benchmark metrics
- GET  /classes          : HAM10000 7-class taxonomy and clinical reference data
- POST /predict          : Dermoscopic image inference with top-3 probabilities
- POST /predict/explain  : Inference paired with Grad-CAM model attribution visualization
"""

import io
import logging
from typing import Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from api.inference_engine import (
    get_inference_engine,
    HAM10000_CLASSES,
    CLASS_METADATA,
    ACADEMIC_DISCLAIMER
)

logger = logging.getLogger("api_main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Skin Lesion Classification & Research API",
    description="Production-grade API for EfficientNet-B4 Dermoscopic Classification with Grad-CAM Explainability.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware allowing Next.js dev server & local environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp", "image/jpg", "image/bmp"]

# Verified LOQ Experimental Benchmark Archive
VERIFIED_BENCHMARKS = {
    "evaluation_dataset": "HAM10000 (Human Against Machine with 10,000 training images)",
    "split_strategy": "Patient-aware stratified split (70% train, 15% validation, 15% test)",
    "total_images": 10015,
    "num_classes": 7,
    "best_overall_model": {
        "model_name": "EfficientNet-B4",
        "architecture_family": "Compound Scaled CNN",
        "parameters": 17561167,
        "test_accuracy": 0.7364,
        "test_accuracy_pct": "73.64%",
        "test_balanced_accuracy": 0.7916,
        "test_balanced_accuracy_pct": "79.16%",
        "test_macro_f1": 0.6919,
        "test_macro_f1_pct": "69.19%",
        "test_macro_auc": 0.9592,
        "test_macro_auc_pct": "95.92%",
        "inference_latency_ms": 8.83,
        "peak_vram_mb": 677.7,
        "selection_rationale": (
            "EfficientNet-B4 achieved the highest ROC-AUC (95.92%), Macro F1 (69.19%), "
            "and Test Accuracy (73.64%) among all evaluated single and multi-branch architectures "
            "while maintaining the lowest per-image inference latency (8.83 ms)."
        )
    },
    "full_model_comparison": [
        {
            "model": "EfficientNet-B4 (Best Model)",
            "type": "Single-Branch Compound Scale",
            "parameters": "17.56M",
            "accuracy": "73.64%",
            "balanced_accuracy": "79.16%",
            "macro_f1": "69.19%",
            "macro_auc": "95.92%",
            "latency_ms": 8.83,
            "vram_mb": 677.7,
            "rank": 1
        },
        {
            "model": "DenseNet-121 Baseline",
            "type": "Single-Branch Dense Feature Reuse",
            "parameters": "6.96M",
            "accuracy": "66.36%",
            "balanced_accuracy": "79.14%",
            "macro_f1": "62.42%",
            "macro_auc": "95.31%",
            "latency_ms": 20.37,
            "vram_mb": 393.7,
            "rank": 2
        },
        {
            "model": "Dual-Branch CNN (Seed 123)",
            "type": "Decoupled Texture + Structure CNN",
            "parameters": "10.67M",
            "accuracy": "55.50%",
            "balanced_accuracy": "70.31%",
            "macro_f1": "48.55%",
            "macro_auc": "90.98%",
            "latency_ms": 27.23,
            "vram_mb": 1561.1,
            "rank": 3
        },
        {
            "model": "Dual-Branch CNN (Seed 999)",
            "type": "Decoupled Texture + Structure CNN",
            "parameters": "10.67M",
            "accuracy": "54.97%",
            "balanced_accuracy": "66.39%",
            "macro_f1": "45.77%",
            "macro_auc": "89.73%",
            "latency_ms": 26.32,
            "vram_mb": 1561.1,
            "rank": 4
        },
        {
            "model": "Dual-Branch CNN (Seed 42)",
            "type": "Decoupled Texture + Structure CNN",
            "parameters": "10.67M",
            "accuracy": "53.91%",
            "balanced_accuracy": "68.62%",
            "macro_f1": "44.90%",
            "macro_auc": "90.54%",
            "latency_ms": 28.47,
            "vram_mb": 1561.1,
            "rank": 5
        },
        {
            "model": "ResNet-50 Baseline",
            "type": "Single-Branch Residual Network",
            "parameters": "23.52M",
            "accuracy": "56.62%",
            "balanced_accuracy": "75.13%",
            "macro_f1": "53.52%",
            "macro_auc": "93.52%",
            "latency_ms": 20.96,
            "vram_mb": 900.3,
            "rank": 6
        }
    ],
    "dual_branch_research_findings": {
        "hypothesis": (
            "Decoupling high-frequency textural features (Shallow-Wide branch) from macroscopic "
            "morphological structures (Deep-Narrow branch) would improve diagnostic discernment."
        ),
        "empirical_outcome": (
            "The Dual-Branch architecture achieved respectable discrimination (AUC ~90.98%), "
            "but did not surpass pre-trained single-branch compound-scaling networks (EfficientNet-B4: 95.92% AUC). "
            "Fusion gate diagnostics revealed that the attention gate predominantly favored "
            "deep structural representations while texture gradients suffered higher variance."
        ),
        "scientific_integrity_statement": (
            "We adhere to rigorous empirical reporting: the Dual-Branch CNN is documented as an "
            "experimental architecture, while the superior EfficientNet-B4 model was chosen for the "
            "deployed inference application."
        )
    }
}


def validate_image_upload(file: UploadFile) -> bytes:
    """Validate content type and payload size."""
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{file.content_type}'. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )

    contents = file.file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
        )

    return contents


def load_pil_image(image_bytes: bytes) -> Image.Image:
    """Parse byte stream into verified PIL Image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Verify image integrity
        # Reopen because verify() closes image buffer
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Corrupted or unreadable image file: {str(e)}"
        )


@app.get("/health", summary="Health and Hardware Status Check")
def health_check():
    """
    Returns hardware device, model initialization status, and API availability.
    """
    engine = get_inference_engine()
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "hardware_device": str(engine.device),
        "model_loaded": True,
        "model_name": "EfficientNet-B4",
        "checkpoint_status": engine.checkpoint_message,
        "supported_classes": HAM10000_CLASSES,
        "disclaimer": ACADEMIC_DISCLAIMER
    }


@app.get("/benchmark", summary="Experimental Benchmark Archive")
def get_benchmarks():
    """
    Returns verified LOQ empirical benchmark results across all tested architectures.
    """
    return VERIFIED_BENCHMARKS


@app.get("/classes", summary="Diagnostic Class Taxonomy")
def get_class_taxonomy():
    """
    Returns metadata, malignancy categorization, and clinical descriptions for the 7 HAM10000 classes.
    """
    return {
        "num_classes": len(HAM10000_CLASSES),
        "taxonomy": CLASS_METADATA,
        "disclaimer": ACADEMIC_DISCLAIMER
    }


@app.post("/predict", summary="Classify Dermoscopic Image")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts a dermoscopic image file and returns top-3 predictions and full probability distribution.
    """
    contents = validate_image_upload(file)
    pil_img = load_pil_image(contents)
    
    engine = get_inference_engine()
    try:
        result = engine.predict(pil_img)
        result["filename"] = file.filename
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        logger.error(f"Inference exception: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


@app.post("/predict/explain", summary="Classify Image with Grad-CAM Attribution")
async def explain_image(file: UploadFile = File(...)):
    """
    Accepts a dermoscopic image and returns classification paired with a Grad-CAM overlay heatmap (Base64 PNG).
    """
    contents = validate_image_upload(file)
    pil_img = load_pil_image(contents)

    engine = get_inference_engine()
    try:
        result = engine.predict_and_explain(pil_img)
        result["filename"] = file.filename
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        logger.error(f"Explainability exception: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grad-CAM generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
