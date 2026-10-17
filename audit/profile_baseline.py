"""
Baseline Performance & Memory Profiler for DermAI on MacBook M4 (MPS).
Measures cold/warm latency, preprocessing, model-only, Grad-CAM, end-to-end API,
and MPS allocated / driver memory across repeated requests BEFORE optimization.
"""
import io
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision
import timm
import fastapi
import pydantic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import api.inference_engine as ie
from api.main import app


def get_mps_mem():
    if torch.backends.mps.is_available():
        return {
            "current_allocated_mb": round(torch.mps.current_allocated_memory() / (1024 * 1024), 3),
            "driver_allocated_mb": round(torch.mps.driver_allocated_memory() / (1024 * 1024), 3),
        }
    return {"current_allocated_mb": 0.0, "driver_allocated_mb": 0.0}


def get_rss_mb():
    # On macOS ru_maxrss is in bytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(usage / (1024 * 1024), 2)


def sync_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def stats(arr):
    a = np.array(arr, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(a)), 3),
        "median_ms": round(float(np.median(a)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "min_ms": round(float(np.min(a)), 3),
        "max_ms": round(float(np.max(a)), 3),
        "std_ms": round(float(np.std(a)), 3),
        "n": int(len(a)),
    }


def main():
    sample_img_path = ROOT / "data/samples/mel_sample.jpg"
    pil_img = Image.open(sample_img_path).convert("RGB")
    img_bytes = sample_img_path.read_bytes()

    mem_before_load = get_mps_mem()
    rss_before_load = get_rss_mb()

    # 1. Measure model loading time
    ie._engine_instance = None
    t0 = time.perf_counter()
    engine = ie.get_inference_engine()
    sync_mps()
    model_load_ms = round((time.perf_counter() - t0) * 1000, 2)

    mem_after_load = get_mps_mem()
    rss_after_load = get_rss_mb()

    # 2. First-request (cold) latency
    t0 = time.perf_counter()
    first_pred = engine.predict(pil_img)
    sync_mps()
    cold_predict_ms = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    first_explain = engine.predict_and_explain(pil_img)
    sync_mps()
    cold_explain_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 3. Component breakdown (20 warm iterations)
    preproc_times = []
    model_only_times = []
    ood_times = []
    gradcam_only_times = []
    warm_predict_times = []
    warm_explain_times = []

    for _ in range(20):
        # OOD
        t0 = time.perf_counter()
        _ = engine.ood_gate.validate_image(pil_img)
        sync_mps()
        ood_times.append((time.perf_counter() - t0) * 1000)

        # Preprocessing
        t0 = time.perf_counter()
        tensor = engine.preprocess_image(pil_img)
        sync_mps()
        preproc_times.append((time.perf_counter() - t0) * 1000)

        # Model-only forward
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = engine.model(tensor)
            _ = torch.softmax(logits, dim=1).cpu()
        sync_mps()
        model_only_times.append((time.perf_counter() - t0) * 1000)

        # Grad-CAM only
        t0 = time.perf_counter()
        hm = engine.grad_cam.generate_heatmap(tensor, 4)
        _ = engine.grad_cam.overlay_heatmap(pil_img, hm)
        sync_mps()
        gradcam_only_times.append((time.perf_counter() - t0) * 1000)

        # Full predict
        t0 = time.perf_counter()
        _ = engine.predict(pil_img)
        sync_mps()
        warm_predict_times.append((time.perf_counter() - t0) * 1000)

        # Full predict_and_explain
        t0 = time.perf_counter()
        _ = engine.predict_and_explain(pil_img)
        sync_mps()
        warm_explain_times.append((time.perf_counter() - t0) * 1000)

    # 4. End-to-end HTTP API via TestClient + memory across 25 repeated requests
    client = TestClient(app)
    api_predict_times = []
    api_explain_times = []
    mem_trace = []

    for i in range(25):
        t0 = time.perf_counter()
        r1 = client.post("/predict", files={"file": ("mel_sample.jpg", img_bytes, "image/jpeg")})
        sync_mps()
        api_predict_times.append((time.perf_counter() - t0) * 1000)
        assert r1.status_code == 200

        t0 = time.perf_counter()
        r2 = client.post("/predict/explain", files={"file": ("mel_sample.jpg", img_bytes, "image/jpeg")})
        sync_mps()
        api_explain_times.append((time.perf_counter() - t0) * 1000)
        assert r2.status_code == 200

        if i in (0, 4, 9, 14, 19, 24):
            mem_trace.append({
                "iteration": i + 1,
                **get_mps_mem(),
                "max_rss_mb": get_rss_mb(),
            })

    report = {
        "phase": "baseline_pre_optimization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "fastapi": fastapi.__version__,
            "pydantic": pydantic.__version__,
            "device": str(engine.device),
            "mps_available": torch.backends.mps.is_available(),
        },
        "model_loading": {
            "load_time_ms": model_load_ms,
            "mps_before_mb": mem_before_load,
            "mps_after_mb": mem_after_load,
            "rss_before_mb": rss_before_load,
            "rss_after_mb": rss_after_load,
        },
        "cold_start": {
            "first_predict_ms": cold_predict_ms,
            "first_explain_ms": cold_explain_ms,
        },
        "component_latency_20_runs": {
            "ood_gate": stats(ood_times),
            "preprocessing": stats(preproc_times),
            "model_forward_only": stats(model_only_times),
            "gradcam_only": stats(gradcam_only_times),
            "engine_predict_total": stats(warm_predict_times),
            "engine_explain_total": stats(warm_explain_times),
        },
        "end_to_end_api_25_runs": {
            "post_predict": stats(api_predict_times),
            "post_predict_explain": stats(api_explain_times),
        },
        "memory_trace_over_25_requests": mem_trace,
        "retained_gradcam_tensors_after_request": {
            "activations_retained": engine.grad_cam.activations is not None,
            "gradients_retained": engine.grad_cam.gradients is not None,
        },
    }

    out_path = ROOT / "audit/modernization_baseline/baseline_performance_profile.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
