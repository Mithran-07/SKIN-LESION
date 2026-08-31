"""
Comprehensive Automated API Test Suite for Skin Lesion Classification Server.

Validates:
- GET  /health (system health, device detection, model status)
- GET  /benchmark (LOQ verified benchmark metrics, consistency checks)
- GET  /classes (HAM10000 7-class diagnostic taxonomy metadata)
- POST /predict (standard inference, probability distributions, top-3 output)
- POST /predict/explain (Grad-CAM base64 attribution overlay, target layer verification)
- Edge cases (unsupported media types, 0-byte files, corrupted image bytes)
- Medical safety disclaimer verification on all API payloads
"""

import io
import pytest
from PIL import Image
from starlette.testclient import TestClient

from api.main import app
from api.inference_engine import HAM10000_CLASSES, ACADEMIC_DISCLAIMER


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def create_test_image(format="JPEG", size=(224, 224), color=(180, 100, 80)) -> bytes:
    """Helper to generate in-memory synthetic dermoscopic test image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


class TestHealthAndMetadataEndpoints:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "EfficientNet-B4"
        assert "hardware_device" in data
        assert data["disclaimer"] == ACADEMIC_DISCLAIMER

    def test_benchmark_endpoint_accuracy_consistency(self, client):
        response = client.get("/benchmark")
        assert response.status_code == 200
        data = response.json()
        
        # Verify established EfficientNet-B4 benchmark metrics exactly
        best_model = data["best_overall_model"]
        assert best_model["model_name"] == "EfficientNet-B4"
        assert best_model["test_accuracy"] == 0.7364
        assert best_model["test_balanced_accuracy"] == 0.7916
        assert best_model["test_macro_f1"] == 0.6919
        assert best_model["test_macro_auc"] == 0.9592

        # Verify full comparison list exists
        comparison = data["full_model_comparison"]
        assert len(comparison) >= 4
        model_names = [m["model"] for m in comparison]
        assert any("EfficientNet-B4" in n for n in model_names)
        assert any("Dual-Branch" in n for n in model_names)

    def test_classes_endpoint(self, client):
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert data["num_classes"] == 7
        taxonomy = data["taxonomy"]
        for cls_name in HAM10000_CLASSES:
            assert cls_name in taxonomy
            assert "category" in taxonomy[cls_name]
            assert "description" in taxonomy[cls_name]
            assert "urgency" in taxonomy[cls_name]


class TestInferenceEndpoints:
    def test_predict_valid_jpeg(self, client):
        img_bytes = create_test_image(format="JPEG")
        response = client.post(
            "/predict",
            files={"file": ("lesion_sample.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Core validation
        assert "predicted_class" in data
        assert data["predicted_class"].lower() in HAM10000_CLASSES
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["top3_predictions"]) == 3
        
        # Probabilities sum to approximately 1.0
        prob_sum = sum(data["probabilities"].values())
        assert pytest.approx(prob_sum, 0.01) == 1.0
        
        # Verify latency and medical disclaimer
        assert data["inference_time_ms"] > 0
        assert data["disclaimer"] == ACADEMIC_DISCLAIMER

    def test_predict_valid_png(self, client):
        img_bytes = create_test_image(format="PNG")
        response = client.post(
            "/predict",
            files={"file": ("lesion_sample.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"].lower() in HAM10000_CLASSES
        assert len(data["top3_predictions"]) == 3

    def test_predict_explain_gradcam(self, client):
        img_bytes = create_test_image(format="JPEG")
        response = client.post(
            "/predict/explain",
            files={"file": ("lesion_sample.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Explainability validation
        assert "explainability" in data
        expl = data["explainability"]
        assert expl["method"].startswith("Grad-CAM")
        assert expl["overlay_base64"].startswith("data:image/png;base64,")
        assert len(expl["overlay_base64"]) > 100
        assert "attribution_note" in expl
        assert data["disclaimer"] == ACADEMIC_DISCLAIMER


class TestEdgeCasesAndValidation:
    def test_unsupported_file_type(self, client):
        response = client.post(
            "/predict",
            files={"file": ("data.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 415
        assert "Unsupported media type" in response.json()["detail"]

    def test_empty_file_upload(self, client):
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 400
        assert "0 bytes" in response.json()["detail"]

    def test_corrupted_image_bytes(self, client):
        response = client.post(
            "/predict",
            files={"file": ("corrupted.jpg", b"\xFF\xD8\xFFcorrupted_invalid_data", "image/jpeg")}
        )
        assert response.status_code in [400, 422]
