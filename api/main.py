"""
FastAPI Server v2.0.0 for DermAI Skin Lesion Classification & Research Platform.

Endpoints:
- GET  /health           : System health, compute device, model & OOD gate readiness
- GET  /diagnostics      : Live Apple Silicon MPS memory telemetry, provenance hashes, & request counters
- GET  /benchmark        : Verified empirical benchmark registry & historical claim reconciliation
- GET  /classes          : HAM10000 7-class taxonomy and clinical reference data
- POST /predict          : Single-pass dermoscopic classification + calibrated OOD validation
- POST /predict/explain  : Single-pass classification + scoped Grad-CAM attribution (supports counterfactual target_class & alpha)
"""

import io
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from api.inference_engine import (
    get_inference_engine,
    HAM10000_CLASSES,
    CLASS_METADATA,
    ACADEMIC_DISCLAIMER,
    PREPROCESSING_VERSION,
    CLASS_MAPPING_VERSION,
)

logger = logging.getLogger("api_main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status ('healthy' or 'degraded')")
    api_version: str = Field(..., description="API contract version")
    hardware_device: str = Field(..., description="Active PyTorch compute device (mps, cuda, cpu)")
    model_loaded: bool = Field(..., description="True if trained checkpoint passed strict=True verification")
    model_name: str = Field(..., description="Active deployed architecture name")
    model_version: str = Field(..., description="Deployed model version identifier")
    checkpoint_sha256_short: str = Field(..., description="First 16 hex chars of checkpoint SHA-256")
    checkpoint_status: str = Field(..., description="Sanitized checkpoint verification status")
    preprocessing_version: str = Field(..., description="Inference preprocessing pipeline identifier")
    class_mapping_version: str = Field(..., description="Canonical 7-class mapping identifier")
    ood_version: str = Field(..., description="Calibrated OOD domain gate version")
    ood_calibrated: bool = Field(..., description="True if OOD v2 calibration artifact is loaded")
    ood_threshold: float = Field(..., description="Calibrated OOD domain score threshold")
    supported_classes: List[str] = Field(..., description="Canonical HAM10000 7-class codes")
    disclaimer: str = Field(..., description="Mandatory non-clinical academic research disclaimer")


app = FastAPI(
    title="DermAI — Dermal Intelligence Lab Research API",
    description=(
        "Research-grade API for 7-class HAM10000 dermoscopic skin-lesion classification, "
        "calibrated out-of-distribution (OOD) gating, counterfactual Grad-CAM explainability, "
        "and reproducible multi-architecture benchmark provenance."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp", "image/jpg", "image/bmp"]

ROOT_DIR = Path(__file__).resolve().parent.parent
REGISTRY_V2_PATH = ROOT_DIR / "results/verified_v2/benchmark_registry_v2.json"


def _load_verified_registry() -> Dict[str, Any]:
    if REGISTRY_V2_PATH.exists():
        try:
            return json.loads(REGISTRY_V2_PATH.read_text())
        except Exception as e:
            logger.error(f"Failed to read benchmark_registry_v2.json: {e}")
    return {}


def _build_benchmark_payload() -> Dict[str, Any]:
    """
    Combines the backward-compatible benchmark summary fields with the complete
    machine-verified v2 registry (`models`, `dataset_integrity`, `ood_gate_v2`,
    `gradcam_verification`) from `results/verified_v2/benchmark_registry_v2.json`.
    """
    reg = _load_verified_registry()
    return {
        "registry_version": reg.get("registry_version", "2.0.0"),
        "evaluation_dataset": "HAM10000 (Human Against Machine with 10,000 training images)",
        "split_strategy": "Lesion-grouped stratified split (7,002 train / 1,503 val / 1,510 test; zero lesion overlap)",
        "total_images": 10015,
        "unique_lesions": 7470,
        "num_classes": 7,
        "best_overall_model": {
            "model_name": "EfficientNet-B4",
            "architecture_family": "Compound-Scaled CNN (timm)",
            "parameters": 17561167,
            "checkpoint_sha256": "125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e",
            "checkpoint_epoch": 34,
            "test_n_samples": 1510,
            "test_accuracy": 0.7351,
            "test_accuracy_pct": "73.51%",
            "test_balanced_accuracy": 0.7923,
            "test_balanced_accuracy_pct": "79.23%",
            "test_macro_f1": 0.6924,
            "test_macro_f1_pct": "69.24%",
            "test_macro_auc": 0.9591,
            "test_macro_auc_pct": "95.91%",
            "test_macro_pr_auc": 0.7752,
            "test_macro_pr_auc_pct": "77.52%",
            "inference_latency_ms": 21.8,
            "historical_cuda_rtx3050_latency_ms": 8.83,
            "peak_vram_mb": 677.7,
            "selection_rationale": (
                "EfficientNet-B4 achieved the highest held-out test Balanced Accuracy (79.23%), "
                "Accuracy (73.51%), Macro F1 (69.24%), Macro ROC-AUC (95.91%), and Macro PR-AUC (77.52%) "
                "on the 1,510-image held-out test split while requiring 5.2x lower inference latency on "
                "Apple Silicon MPS than the custom Dual-Branch CNN."
            ),
        },
        "full_model_comparison": [
            {
                "id": "efficientnet_b4",
                "model": "EfficientNet-B4 (Deployed)",
                "type": "Single-Branch Compound Scale (timm)",
                "fusion_version": None,
                "parameters": "17.56M",
                "epoch": 34,
                "accuracy": "73.51%",
                "balanced_accuracy": "79.23%",
                "macro_precision": "64.05%",
                "macro_f1": "69.24%",
                "macro_auc": "95.91%",
                "macro_pr_auc": "77.52%",
                "latency_ms": 25.86,
                "cuda_latency_ms": 8.83,
                "vram_mb": 677.7,
                "rank": 1,
            },
            {
                "id": "densenet121",
                "model": "DenseNet-121 Baseline",
                "type": "Single-Branch Dense Feature Reuse (timm)",
                "fusion_version": None,
                "parameters": "6.96M",
                "epoch": 18,
                "accuracy": "66.42%",
                "balanced_accuracy": "78.83%",
                "macro_precision": "55.52%",
                "macro_f1": "61.99%",
                "macro_auc": "95.32%",
                "macro_pr_auc": "75.23%",
                "latency_ms": 27.55,
                "cuda_latency_ms": 20.37,
                "vram_mb": 393.7,
                "rank": 2,
            },
            {
                "id": "resnet50",
                "model": "ResNet-50 Baseline",
                "type": "Single-Branch Residual Network (timm)",
                "fusion_version": None,
                "parameters": "23.52M",
                "epoch": 29,
                "accuracy": "56.56%",
                "balanced_accuracy": "75.12%",
                "macro_precision": "49.54%",
                "macro_f1": "53.83%",
                "macro_auc": "93.58%",
                "macro_pr_auc": "73.64%",
                "latency_ms": 16.09,
                "cuda_latency_ms": 20.96,
                "vram_mb": 900.3,
                "rank": 3,
            },
            {
                "id": "dual_branch_seed123",
                "model": "Dual-Branch CNN (Seed 123)",
                "type": "Decoupled Texture + Structure CNN (v1_matrix)",
                "fusion_version": "v1_matrix",
                "parameters": "10.67M",
                "epoch": 48,
                "accuracy": "55.50%",
                "balanced_accuracy": "70.28%",
                "macro_precision": "44.00%",
                "macro_f1": "48.45%",
                "macro_auc": "90.98%",
                "macro_pr_auc": "60.45%",
                "latency_ms": 135.33,
                "cuda_latency_ms": 27.23,
                "vram_mb": 1561.1,
                "rank": 4,
            },
            {
                "id": "dual_branch_seed999",
                "model": "Dual-Branch CNN (Seed 999)",
                "type": "Decoupled Texture + Structure CNN (v1_matrix)",
                "fusion_version": "v1_matrix",
                "parameters": "10.67M",
                "epoch": 28,
                "accuracy": "55.10%",
                "balanced_accuracy": "66.42%",
                "macro_precision": "40.47%",
                "macro_f1": "45.72%",
                "macro_auc": "89.75%",
                "macro_pr_auc": "58.37%",
                "latency_ms": 145.71,
                "cuda_latency_ms": 26.32,
                "vram_mb": 1561.1,
                "rank": 5,
            },
            {
                "id": "dual_branch_seed42",
                "model": "Dual-Branch CNN (Seed 42)",
                "type": "Decoupled Texture + Structure CNN (v2_scalar)",
                "fusion_version": "v2_scalar",
                "parameters": "9.03M",
                "epoch": 16,
                "accuracy": "64.30%",
                "balanced_accuracy": "59.49%",
                "macro_precision": "46.68%",
                "macro_f1": "49.58%",
                "macro_auc": "90.13%",
                "macro_pr_auc": "57.01%",
                "latency_ms": 145.13,
                "cuda_latency_ms": 28.47,
                "vram_mb": 1420.0,
                "rank": 6,
            },
        ],
        "dual_branch_research_findings": {
            "hypothesis": (
                "Decoupling high-frequency textural features (Shallow-Wide branch: 3->256->512->1024 at 112x112) "
                "from macroscopic morphological structures (Deep-Narrow branch: 4-stage residual 64->128->256 at 14x14) "
                "via an attention-gated fusion head would improve minority-class diagnostic discernment."
            ),
            "empirical_outcome": (
                "While the custom Dual-Branch CNN achieved 89.75%-90.98% Macro ROC-AUC and up to 70.28% Balanced Accuracy, "
                "it did not surpass fully ImageNet-pretrained compound-scaled backbones (EfficientNet-B4: 79.23% Balanced Accuracy, "
                "95.91% Macro ROC-AUC). Because the Shallow-Wide branch (6.17M params) and Deep-Narrow stages 1-4 (2.20M params) "
                "are initialized via Kaiming normal weights rather than full pretrained backbones, training from scratch on "
                "7,002 images exhibits higher variance on minority classes. Furthermore, historical documentation that cited "
                "90.98% / 89.73% / 90.54% as 'Accuracy' actually reported Macro ROC-AUC."
            ),
            "versioning_remediation": (
                "Both historical fusion variants are explicitly supported and verified with strict=True: "
                "v1_matrix (Linear 1280->1280 gate, 10,669,639 params, used by seeds 123 & 999) and "
                "v2_scalar (Linear 1280->1 scalar gate, 9,031,241 params, used by seed 42)."
            ),
            "scientific_integrity_statement": (
                "All benchmark metrics are directly reproduced from verified checkpoint evaluation on the "
                "1,510-image lesion-grouped test split (splits/test.csv). EfficientNet-B4 is deployed for "
                "live inference, while Dual-Branch CNN variants are preserved as auditable research artifacts."
            ),
        },
        "verified_registry": reg,
    }


def validate_image_upload(file: UploadFile) -> bytes:
    """Validate content type and payload size."""
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{file.content_type}'. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}",
        )

    contents = file.file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    return contents


def load_pil_image(image_bytes: bytes) -> Image.Image:
    """Parse byte stream into verified PIL Image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Corrupted or unreadable image file: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse, summary="Health and Hardware Status Check")
def health_check():
    """
    Returns hardware device, strict checkpoint verification status, OOD calibration status,
    and canonical pipeline version tags without leaking local filesystem paths.
    """
    engine = get_inference_engine()
    return {
        "status": "healthy" if engine.checkpoint_loaded else "degraded",
        "api_version": "2.0.0",
        "hardware_device": str(engine.device),
        "model_loaded": engine.checkpoint_loaded,
        "model_name": "EfficientNet-B4",
        "model_version": "timm/efficientnet_b4_ham10000_v2",
        "checkpoint_sha256_short": engine.checkpoint_sha256_short,
        "checkpoint_status": engine.checkpoint_message,
        "preprocessing_version": PREPROCESSING_VERSION,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "ood_version": engine.ood_gate.VERSION,
        "ood_calibrated": engine.ood_gate.calibrated,
        "ood_threshold": round(float(engine.ood_gate.threshold), 4),
        "supported_classes": HAM10000_CLASSES,
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


@app.get("/diagnostics", summary="Runtime Memory, Hardware & Provenance Diagnostics")
def runtime_diagnostics():
    """
    Returns live Apple Silicon MPS tensor memory (`current_allocated_mb`, `driver_allocated_mb`),
    process RSS, checkpoint SHA-256 verification, OOD gate calibration parameters, and live
    request telemetry.
    """
    engine = get_inference_engine()
    return engine.get_diagnostics()


@app.get("/benchmark", summary="Verified Experimental Benchmark Registry")
def get_benchmarks():
    """
    Returns verified empirical benchmark metrics across all 6 trained checkpoints on `splits/test.csv`
    (N=1,510), including per-class Precision/Recall/F1, 7x7 confusion matrices, dataset split
    integrity verification, and OOD v2 evaluation results.
    """
    return _build_benchmark_payload()


@app.get("/classes", summary="Diagnostic Class Taxonomy")
def get_class_taxonomy():
    """
    Returns metadata, malignancy categorization, and clinical descriptions for the 7 HAM10000 classes.
    """
    return {
        "num_classes": len(HAM10000_CLASSES),
        "canonical_order": HAM10000_CLASSES,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "taxonomy": CLASS_METADATA,
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


@app.post("/predict", summary="Classify Dermoscopic Image")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts a dermoscopic image file and returns calibrated OOD validation, top-3 predictions,
    full 7-class probability distribution, normalized entropy, and stage-by-stage latency breakdown.
    """
    contents = validate_image_upload(file)
    pil_img = load_pil_image(contents)

    engine = get_inference_engine()
    if not engine.checkpoint_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error_code": "MODEL_UNAVAILABLE",
                "detail": "Trained EfficientNet-B4 checkpoint is not loaded.",
            },
        )

    try:
        result = engine.predict(pil_img)
        result["filename"] = file.filename
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        logger.error(f"Inference exception: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.post("/predict/explain", summary="Classify Image with Scoped Grad-CAM Attribution")
async def explain_image(
    file: UploadFile = File(...),
    target_class: Optional[str] = Query(
        default=None,
        description="Optional class code (akiec, bcc, bkl, df, mel, nv, vasc) for counterfactual Grad-CAM inspection.",
    ),
    alpha: float = Query(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Overlay blend opacity in [0.0, 1.0].",
    ),
):
    """
    Accepts a dermoscopic image and returns single-pass classification paired with both
    blended (`overlay_base64`) and standalone RGBA (`heatmap_base64`) Grad-CAM attribution maps.
    Supports optional `target_class` query parameter for counterfactual class activation inspection.
    """
    contents = validate_image_upload(file)
    pil_img = load_pil_image(contents)

    engine = get_inference_engine()
    if not engine.checkpoint_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error_code": "MODEL_UNAVAILABLE",
                "detail": "Trained EfficientNet-B4 checkpoint is not loaded.",
            },
        )

    try:
        result = engine.predict_and_explain(pil_img, target_class=target_class, alpha=alpha)
        result["filename"] = file.filename
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Explainability exception: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grad-CAM generation failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
