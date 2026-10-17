"""
Comprehensive Automated API Test Suite for DermAI V2 Classification Server.

Validates:
- GET  /health (checkpoint SHA-256 verification, sanitized status, OOD v2 readiness)
- GET  /diagnostics (Apple Silicon M4 MPS memory telemetry, warmup status, request counters)
- GET  /benchmark (verified 6-checkpoint benchmark registry, confusion matrices, OOD v2 summary)
- GET  /classes (canonical 7-class HAM10000 ordering: akiec, bcc, bkl, df, mel, nv, vasc)
- POST /predict (in-domain dermoscopic classification, probability simplex sum=1.0, OOD rejection)
- POST /predict/explain (Grad-CAM standalone heatmap + overlay, counterfactual target_class, hook cleanup)
- Edge cases (400 empty upload, 415 unsupported MIME, 422 corrupted bytes, 422 invalid target_class)
"""

import io
from pathlib import Path
import pytest
from PIL import Image
from starlette.testclient import TestClient

from api.main import app
from api.inference_engine import get_inference_engine, HAM10000_CLASSES, ACADEMIC_DISCLAIMER

SAMPLE_MEL_PATH = Path(__file__).resolve().parent.parent / "data" / "samples" / "mel_sample.jpg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def load_real_dermoscopic_bytes(fmt: str = "JPEG") -> bytes:
    """Load a genuine held-out HAM10000 dermoscopic sample from data/samples/."""
    assert SAMPLE_MEL_PATH.exists(), f"Sample image missing at {SAMPLE_MEL_PATH}"
    if fmt.upper() == "JPEG":
        return SAMPLE_MEL_PATH.read_bytes()
    img = Image.open(SAMPLE_MEL_PATH).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def create_ood_solid_image(fmt: str = "JPEG", size=(224, 224), color=(30, 120, 220)) -> bytes:
    """Generate an out-of-distribution non-dermoscopic solid blue image."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestHealthDiagnosticsAndMetadata:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "EfficientNet-B4"
        assert data["checkpoint_sha256_short"] == "125340ffa59ca4a1"
        assert data["ood_version"] == "2.0.0"
        assert data["ood_calibrated"] is True
        assert data["ood_threshold"] == pytest.approx(0.25, abs=0.01)
        # Ensure no absolute local filesystem paths leak in checkpoint_status
        assert "/Users/" not in data["checkpoint_status"]
        assert data["disclaimer"] == ACADEMIC_DISCLAIMER

    def test_diagnostics_endpoint(self, client):
        response = client.get("/diagnostics")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["warmup_completed"] is True
        assert "memory" in data
        assert data["memory"]["active_gradcam_hooks"] == 0
        assert "model_provenance" in data
        assert data["model_provenance"]["checkpoint_sha256_matches_canonical"] is True
        assert "ood_gate" in data
        assert data["ood_gate"]["version"] == "2.0.0"
        assert data["ood_gate"]["threshold"] == pytest.approx(0.25, abs=0.01)

    def test_benchmark_endpoint_accuracy_consistency(self, client):
        response = client.get("/benchmark")
        assert response.status_code == 200
        data = response.json()

        best_model = data["best_overall_model"]
        assert best_model["model_name"] == "EfficientNet-B4"
        assert best_model["test_accuracy"] == pytest.approx(0.7351, abs=0.005)
        assert best_model["test_balanced_accuracy"] == pytest.approx(0.7923, abs=0.005)
        assert best_model["test_macro_f1"] == pytest.approx(0.6924, abs=0.005)
        assert best_model["test_macro_auc"] == pytest.approx(0.9591, abs=0.005)

        comparison = data["full_model_comparison"]
        assert len(comparison) == 6
        model_names = [m["model"] for m in comparison]
        assert any("EfficientNet-B4" in n for n in model_names)
        assert any("Dual-Branch" in n for n in model_names)

    def test_classes_endpoint(self, client):
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == 7
        assert data["canonical_order"] == [
            "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
        ]
        taxonomy = data["taxonomy"]
        for cls_name in HAM10000_CLASSES:
            assert cls_name in taxonomy
            assert "category" in taxonomy[cls_name]
            assert "description" in taxonomy[cls_name]
            assert "urgency" in taxonomy[cls_name]


class TestInferenceAndExplainability:
    def test_predict_valid_dermoscopic_jpeg(self, client):
        img_bytes = load_real_dermoscopic_bytes("JPEG")
        response = client.post(
            "/predict",
            files={"file": ("mel_sample.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["predicted_class"].lower() in HAM10000_CLASSES
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["top3_predictions"]) == 3
        assert data["input_validation"]["valid"] is True

        prob_sum = sum(data["probabilities"].values())
        assert pytest.approx(prob_sum, 0.01) == 1.0
        assert data["inference_time_ms"] > 0
        assert data["disclaimer"] == ACADEMIC_DISCLAIMER

    def test_predict_valid_dermoscopic_png(self, client):
        img_bytes = load_real_dermoscopic_bytes("PNG")
        response = client.post(
            "/predict",
            files={"file": ("mel_sample.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["predicted_class"].lower() in HAM10000_CLASSES
        assert len(data["top3_predictions"]) == 3

    def test_predict_explain_default_and_counterfactual(self, client):
        img_bytes = load_real_dermoscopic_bytes("JPEG")
        response = client.post(
            "/predict/explain?target_class=bcc&alpha=0.5",
            files={"file": ("mel_sample.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "explainability" in data and data["explainability"] is not None
        expl = data["explainability"]
        assert expl["method"].startswith("Grad-CAM")
        assert expl["target_class_code"] == "BCC"
        assert expl["is_counterfactual"] is True
        assert expl["overlay_base64"].startswith("data:image/png;base64,")
        assert expl["heatmap_base64"].startswith("data:image/png;base64,")
        assert len(expl["overlay_base64"]) > 100
        assert len(expl["heatmap_base64"]) > 100
        assert data["disclaimer"] == ACADEMIC_DISCLAIMER

        # Verify that scoped Grad-CAM hooks were cleanly removed from the target layer
        engine = get_inference_engine()
        target_layer = engine.grad_cam.target_layer
        assert len(target_layer._forward_hooks) == 0
        assert len(target_layer._backward_hooks) == 0

    def test_ood_rejection_on_non_dermoscopic_image(self, client):
        ood_bytes = create_ood_solid_image("JPEG")
        response = client.post(
            "/predict/explain",
            files={"file": ("solid_blue.jpg", ood_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["input_validation"]["valid"] is False
        assert data.get("explainability") is None


class TestEdgeCasesAndValidation:
    def test_unsupported_file_type(self, client):
        response = client.post(
            "/predict",
            files={"file": ("data.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 415
        assert "Unsupported media type" in response.json()["detail"]

    def test_empty_file_upload(self, client):
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "0 bytes" in response.json()["detail"]

    def test_corrupted_image_bytes(self, client):
        response = client.post(
            "/predict",
            files={"file": ("corrupted.jpg", b"\xFF\xD8\xFFcorrupted_invalid_data", "image/jpeg")},
        )
        assert response.status_code in [400, 422]

    def test_invalid_counterfactual_target_class(self, client):
        img_bytes = load_real_dermoscopic_bytes("JPEG")
        response = client.post(
            "/predict/explain?target_class=invalid_lesion",
            files={"file": ("mel_sample.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 422
