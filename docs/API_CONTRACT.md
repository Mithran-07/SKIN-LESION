# DermAI Research Platform V2 — REST API Contract Specification

**Base URL (Local Development):** `http://127.0.0.1:8000`  
**API Version:** `2.0.0`  
**OpenAPI Interactive Docs:** `http://127.0.0.1:8000/docs`  

---

## 1. Endpoints Overview

| Method | Path | Description |
| :---: | :--- | :--- |
| `GET` | `/health` | Service health, compute device (`mps`/`cuda`/`cpu`), strict checkpoint SHA-256 verification status, OOD v2 readiness, and sanitized status message. |
| `GET` | `/diagnostics` | Live Apple Silicon MPS unified memory telemetry (`mps_current_allocated_mb`, `mps_driver_allocated_mb`, `process_peak_rss_mb`), shader warmup timing, and request counters. |
| `GET` | `/benchmark` | Complete machine-verified 6-checkpoint evaluation registry (`N=1,510`), per-class metrics, $7 \times 7$ confusion matrices, and OOD v2 evaluation summary. |
| `GET` | `/classes` | Canonical 7-class HAM10000 ordering (`["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]`), malignancy categories, and clinical descriptions. |
| `POST` | `/predict` | Single-pass dermoscopic image classification with calibrated 4-signal OOD domain gating and stage-by-stage latency breakdown. |
| `POST` | `/predict/explain` | Single-pass classification + scoped Grad-CAM spatial attribution (`overlay_base64` + standalone `heatmap_base64`), supporting optional `?target_class=...&alpha=...` counterfactual queries. |

---

## 2. Detailed Endpoint Contracts

### 2.1 `GET /health`

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "api_version": "2.0.0",
  "hardware_device": "mps",
  "model_loaded": true,
  "model_name": "EfficientNet-B4",
  "model_version": "timm/efficientnet_b4_ham10000_v2",
  "checkpoint_sha256_short": "125340ffa59ca4a1",
  "checkpoint_status": "Verified trained EfficientNet-B4 checkpoint (strict=True, epoch=34, sha256=125340ffa59ca4a1)",
  "preprocessing_version": "ham10000_eval_r256_c224_imagenet_v1",
  "class_mapping_version": "ham10000_alphabetical_7class_v1",
  "ood_version": "2.0.0",
  "ood_calibrated": true,
  "ood_threshold": 0.25,
  "supported_classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
  "disclaimer": "ACADEMIC RESEARCH DEMONSTRATION ONLY..."
}
```

### 2.2 `GET /diagnostics`

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "uptime_seconds": 142.5,
  "device": "mps",
  "warmup_completed": true,
  "warmup_time_ms": 412.8,
  "memory": {
    "mps_current_allocated_mb": 69.95,
    "mps_driver_allocated_mb": 657.64,
    "process_peak_rss_mb": 556.8,
    "active_gradcam_hooks": 0
  },
  "model_provenance": {
    "model_name": "EfficientNet-B4",
    "model_version": "timm/efficientnet_b4_ham10000_v2",
    "checkpoint_loaded": true,
    "checkpoint_sha256": "125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e",
    "checkpoint_sha256_short": "125340ffa59ca4a1",
    "checkpoint_sha256_matches_canonical": true,
    "checkpoint_epoch": 34,
    "checkpoint_best_val_balanced_accuracy": 0.7859,
    "preprocessing_version": "ham10000_eval_r256_c224_imagenet_v1",
    "class_mapping_version": "ham10000_alphabetical_7class_v1",
    "canonical_classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
  },
  "ood_gate": {
    "version": "2.0.0",
    "calibrated": true,
    "threshold": 0.25,
    "architecture": "Shared EfficientNet-B4 Penultimate Embedding + Mahalanobis + Energy + Photometric Gate"
  },
  "telemetry": {
    "total_predictions_served": 12,
    "total_explanations_served": 8,
    "total_ood_rejections": 2,
    "last_inference_latency_ms": 27.41
  }
}
```

### 2.3 `POST /predict` & `POST /predict/explain`

**Request:**
- `Content-Type: multipart/form-data`
- Field `file`: Image file (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`, max `15 MB`).
- Optional Query Parameters on `/predict/explain`:
  - `target_class`: One of `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` (case-insensitive). Defaults to top-1 predicted class.
  - `alpha`: Float in `[0.0, 1.0]` controlling server-rendered `overlay_base64` blend opacity (default `0.45`).

**In-Domain Success Response (`200 OK`, `status: "success"`):**
- `input_validation`: `{ "valid": true, "domain_score": 0.6842, "threshold": 0.25, "signals": { "prototype_cosine": ..., "mahalanobis_score": ..., "energy_score": ..., "photometric_penalty": 1.0 }, ... }`
- `predicted_class`: `"MEL"`, `confidence`: `0.7812`, `normalized_entropy`: `0.3412`
- `probabilities`: `{ "akiec": ..., "bcc": ..., "bkl": ..., "df": ..., "mel": ..., "nv": ..., "vasc": ... }` (sums to `1.0`)
- `top3_predictions`: Top-3 ranked classes with clinical descriptions and urgency metadata.
- `stage_timings_ms`: `{ "preprocessing_ms": 1.4, "forward_pass_ms": 23.1, "ood_gate_ms": 2.6, "gradcam_backward_ms": 109.2 }`
- `explainability` (on `/predict/explain`):
  - `target_class_code`: `"BCC"`
  - `is_counterfactual`: `true`
  - `overlay_base64`: `"data:image/png;base64,..."`
  - `heatmap_base64`: `"data:image/png;base64,..."` (standalone RGBA heatmap for client-side opacity blending)
  - `peak_activation_coords`: `{ "x": 118, "y": 104 }`
  - `high_activation_area_fraction`: `0.2145`

**Out-of-Domain Rejection Response (`200 OK`, `status: "rejected"`):**
- `status`: `"rejected"`
- `input_validation.valid`: `false`
- `input_validation.rejection_reason`: Detailed explanation of why the input fell outside the calibrated dermoscopic manifold.
- No classification probabilities or Grad-CAM overlays are emitted for rejected OOD inputs.

**Error Responses:**
- `400 Bad Request`: Empty (`0`-byte) file upload.
- `413 Payload Too Large`: Upload exceeds `15 MB`.
- `415 Unsupported Media Type`: Non-image MIME type (e.g., `text/plain`, `application/pdf`).
- `422 Unprocessable Entity`: Corrupted/unreadable image bytes or invalid `target_class` query parameter.
- `503 Service Unavailable`: Trained model checkpoint is missing (`MODEL_UNAVAILABLE`).
