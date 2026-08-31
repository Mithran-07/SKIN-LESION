"""
API Test Suite Runner.
Executes comprehensive validation of all endpoints on the FastAPI server.
"""

import io
import sys
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app, VERIFIED_BENCHMARKS
from api.inference_engine import HAM10000_CLASSES, ACADEMIC_DISCLAIMER


def run_api_tests():
    print("=" * 70)
    print("       RUNNING FASTAPI BACKEND TEST SUITE")
    print("=" * 70)
    
    client = TestClient(app)
    passed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ [PASS] {name} {detail}")
        else:
            print(f"  ❌ [FAIL] {name} {detail}")
            assert False, f"Test failed: {name} {detail}"

    # 1. Health endpoint
    r = client.get("/health")
    check("GET /health status == 200", r.status_code == 200)
    d = r.json()
    check("GET /health status field", d.get("status") == "healthy")
    check("GET /health model loaded", d.get("model_loaded") is True)
    check("GET /health disclaimer present", d.get("disclaimer") == ACADEMIC_DISCLAIMER)

    # 2. Benchmark endpoint
    r = client.get("/benchmark")
    check("GET /benchmark status == 200", r.status_code == 200)
    d = r.json()
    best = d.get("best_overall_model", {})
    check("GET /benchmark best model is EfficientNet-B4", best.get("model_name") == "EfficientNet-B4")
    check("GET /benchmark accuracy == 73.64%", best.get("test_accuracy") == 0.7364)
    check("GET /benchmark balanced accuracy == 79.16%", best.get("test_balanced_accuracy") == 0.7916)
    check("GET /benchmark macro F1 == 69.19%", best.get("test_macro_f1") == 0.6919)
    check("GET /benchmark macro AUC == 95.92%", best.get("test_macro_auc") == 0.9592)

    # 3. Classes endpoint
    r = client.get("/classes")
    check("GET /classes status == 200", r.status_code == 200)
    d = r.json()
    check("GET /classes count == 7", d.get("num_classes") == 7)
    taxonomy = d.get("taxonomy", {})
    for c in HAM10000_CLASSES:
        check(f"Class '{c}' in taxonomy", c in taxonomy)

    # 4. Predict endpoint (JPEG)
    img = Image.new("RGB", (224, 224), (180, 90, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    r = client.post("/predict", files={"file": ("sample.jpg", buf.getvalue(), "image/jpeg")})
    check("POST /predict status == 200", r.status_code == 200)
    d = r.json()
    check("POST /predict returns predicted_class", d.get("predicted_class", "").lower() in HAM10000_CLASSES)
    check("POST /predict returns top3 list", len(d.get("top3_predictions", [])) == 3)
    prob_sum = sum(d.get("probabilities", {}).values())
    check("POST /predict probabilities sum ~ 1.0", abs(prob_sum - 1.0) < 0.05, f"(sum={prob_sum:.4f})")
    check("POST /predict disclaimer present", d.get("disclaimer") == ACADEMIC_DISCLAIMER)

    # 5. Predict / Explain endpoint (Grad-CAM)
    r = client.post("/predict/explain", files={"file": ("sample.jpg", buf.getvalue(), "image/jpeg")})
    check("POST /predict/explain status == 200", r.status_code == 200)
    d = r.json()
    expl = d.get("explainability", {})
    check("POST /predict/explain returns Grad-CAM overlay", expl.get("overlay_base64", "").startswith("data:image/png;base64,"))
    check("POST /predict/explain attribution note present", "attribution_note" in expl)

    # 6. Edge cases
    # Unsupported media type
    r = client.post("/predict", files={"file": ("doc.txt", b"plain text content", "text/plain")})
    check("POST /predict rejects unsupported media type (415)", r.status_code == 415)

    # Empty file
    r = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    check("POST /predict rejects empty file (400)", r.status_code == 400)

    # Corrupted image bytes
    r = client.post("/predict", files={"file": ("corrupted.jpg", b"\xFF\xD8\xFFnotanimage", "image/jpeg")})
    check("POST /predict handles corrupted image (400 or 422)", r.status_code in [400, 422])

    print("=" * 70)
    print(f"       RESULT: {passed}/{total} API TESTS PASSED (100% SUCCESS)")
    print("=" * 70)
    return passed == total


if __name__ == "__main__":
    success = run_api_tests()
    sys.exit(0 if success else 1)
