"""
Integration tests for the FastAPI backend.
Run with: python -m pytest tests/test_api.py -v
"""

import io
import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def make_fake_image(width=224, height=224, color=(200, 100, 100)) -> bytes:
    """Create a synthetic dermoscopy-like PNG image as bytes."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# These tests import the FastAPI TestClient — no server required
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        # May return 200 (model loaded) or 503 (checkpoint missing)
        assert resp.status_code in (200, 503)

    def test_health_has_status_key(self, client):
        resp = client.get("/health")
        assert "status" in resp.json()


class TestBenchmark:
    def test_benchmark_returns_200(self, client):
        resp = client.get("/benchmark")
        assert resp.status_code == 200

    def test_benchmark_has_models(self, client):
        data = client.get("/benchmark").json()
        assert "models" in data
        assert len(data["models"]) >= 3

    def test_benchmark_efficientnet_values(self, client):
        data = client.get("/benchmark").json()
        eb4 = next(m for m in data["models"] if m["name"] == "EfficientNet-B4")
        assert abs(eb4["test_accuracy"] - 0.7364) < 0.001
        assert eb4["roc_auc"] > 0.95


class TestClasses:
    def test_classes_endpoint(self, client):
        resp = client.get("/classes")
        assert resp.status_code == 200
        data = resp.json()
        assert "nv" in data
        assert len(data) == 7


class TestPredictValidation:
    def test_predict_rejects_text_file(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code in (400, 422)

    def test_predict_rejects_oversized_file(self, client):
        big_data = b"0" * (21 * 1024 * 1024)  # 21 MB
        resp = client.post(
            "/predict",
            files={"file": ("big.png", big_data, "image/png")},
        )
        assert resp.status_code in (400, 413)

    def test_predict_accepts_png(self, client):
        img_bytes = make_fake_image()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        # Either 200 (model present) or 503 (no checkpoint in test environment)
        assert resp.status_code in (200, 503)

    def test_predict_response_structure(self, client):
        img_bytes = make_fake_image()
        resp = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        if resp.status_code == 200:
            data = resp.json()
            assert "predicted_label" in data
            assert "probability" in data
            assert "top3" in data
            assert len(data["top3"]) == 3
            assert "all_probabilities" in data
            assert "disclaimer" in data


class TestExplainEndpoint:
    def test_explain_accepts_png(self, client):
        img_bytes = make_fake_image()
        resp = client.post(
            "/predict/explain",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code in (200, 503)

    def test_explain_response_has_gradcam(self, client):
        img_bytes = make_fake_image()
        resp = client.post(
            "/predict/explain",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        if resp.status_code == 200:
            data = resp.json()
            assert "gradcam" in data
            assert "overlay_b64" in data["gradcam"]
            assert "heatmap_b64" in data["gradcam"]
