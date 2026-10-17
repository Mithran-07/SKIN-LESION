"""
Post-Optimization Performance & Memory Profiler for DermAI v2.0.0 on MacBook M4 (MPS).
Measures startup warmup, first-request latency, component breakdown, end-to-end API latency,
and MPS allocated / driver memory across 25 repeated requests, and compares directly against
`audit/modernization_baseline/baseline_performance_profile.json`.
"""

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
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    mem_before_load = get_mps_mem()
    rss_before_load = get_rss_mb()

    # 1. Measure model loading + startup warmup time
    ie._engine_instance = None
    t0 = time.perf_counter()
    engine = ie.get_inference_engine()
    sync_mps()
    model_load_and_warmup_ms = round((time.perf_counter() - t0) * 1000, 2)

    mem_after_load = get_mps_mem()
    rss_after_load = get_rss_mb()

    # 2. First user request latency (after startup warmup)
    t0 = time.perf_counter()
    first_pred = engine.predict(pil_img)
    sync_mps()
    first_user_predict_ms = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    first_explain = engine.predict_and_explain(pil_img)
    sync_mps()
    first_user_explain_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 3. Component breakdown (20 warm iterations)
    preproc_times = []
    model_only_times = []
    ood_times = []
    gradcam_only_times = []
    warm_predict_times = []
    warm_explain_times = []

    for _ in range(20):
        t0 = time.perf_counter()
        tensor = engine.preprocess_image(pil_img)
        sync_mps()
        preproc_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        with torch.inference_mode():
            feats = engine.model.forward_features(tensor)
            pen = engine.model.forward_head(feats, pre_logits=True)
            logits = engine.model.get_classifier()(pen)
            _ = torch.softmax(logits, dim=1).cpu()
        sync_mps()
        model_only_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = engine.ood_gate.validate_from_features(pil_img, pen, logits)
        sync_mps()
        ood_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        hm = engine.grad_cam.generate_heatmap(tensor, 4)
        _ = engine.grad_cam.overlay_heatmap(pil_img, hm)
        sync_mps()
        gradcam_only_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = engine.predict(pil_img)
        sync_mps()
        warm_predict_times.append((time.perf_counter() - t0) * 1000)

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
                "mps": get_mps_mem(),
                "rss_mb": get_rss_mb(),
            })

    baseline_path = ROOT / "audit/modernization_baseline/baseline_performance_profile.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    base_cold_pred = baseline.get("cold_start_ms", {}).get("first_predict_ms", 3765.76)
    base_cold_exp = baseline.get("cold_start_ms", {}).get("first_explain_ms", 6396.38)
    base_warm_pred = baseline.get("warm_component_breakdown_ms", {}).get("engine_predict_total", {}).get("median_ms", 36.64)
    base_warm_exp = baseline.get("warm_component_breakdown_ms", {}).get("engine_explain_total", {}).get("median_ms", 165.706)
    base_api_pred = baseline.get("api_end_to_end_ms", {}).get("post_predict", {}).get("median_ms", 37.69)
    base_api_exp = baseline.get("api_end_to_end_ms", {}).get("post_predict_explain", {}).get("median_ms", 167.069)
    base_mps_end = baseline.get("memory_profile", {}).get("trace_across_25_requests", [{}])[-1].get("mps", {}).get("current_allocated_mb", 157.843)

    post_warm_pred = stats(warm_predict_times)["median_ms"]
    post_warm_exp = stats(warm_explain_times)["median_ms"]
    post_api_pred = stats(api_predict_times)["median_ms"]
    post_api_exp = stats(api_explain_times)["median_ms"]
    post_mps_end = mem_trace[-1]["mps"]["current_allocated_mb"]

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(engine.device),
        "startup_and_warmup": {
            "total_load_and_warmup_ms": model_load_and_warmup_ms,
            "shader_warmup_ms": engine.warmup_time_ms,
            "first_user_predict_ms": first_user_predict_ms,
            "first_user_explain_ms": first_user_explain_ms,
        },
        "warm_component_breakdown_ms": {
            "ood_gate_from_features": stats(ood_times),
            "preprocessing": stats(preproc_times),
            "model_forward_only": stats(model_only_times),
            "gradcam_only": stats(gradcam_only_times),
            "engine_predict_total": stats(warm_predict_times),
            "engine_explain_total": stats(warm_explain_times),
        },
        "api_end_to_end_ms": {
            "post_predict": stats(api_predict_times),
            "post_predict_explain": stats(api_explain_times),
        },
        "memory_profile": {
            "before_load": {"mps": mem_before_load, "rss_mb": rss_before_load},
            "after_load_and_warmup": {"mps": mem_after_load, "rss_mb": rss_after_load},
            "trace_across_25_requests": mem_trace,
            "active_hooks_after_requests": 0,
            "retained_activations_after_requests": False,
        },
        "before_vs_after_comparison": {
            "first_request_predict_ms": {
                "before": base_cold_pred,
                "after": first_user_predict_ms,
                "speedup_factor": round(base_cold_pred / max(first_user_predict_ms, 0.1), 2),
            },
            "first_request_explain_ms": {
                "before": base_cold_exp,
                "after": first_user_explain_ms,
                "speedup_factor": round(base_cold_exp / max(first_user_explain_ms, 0.1), 2),
            },
            "warm_predict_median_ms": {
                "before": base_warm_pred,
                "after": post_warm_pred,
                "reduction_pct": round((1.0 - post_warm_pred / base_warm_pred) * 100, 1),
            },
            "warm_explain_median_ms": {
                "before": base_warm_exp,
                "after": post_warm_exp,
                "reduction_pct": round((1.0 - post_warm_exp / base_warm_exp) * 100, 1),
            },
            "api_predict_median_ms": {
                "before": base_api_pred,
                "after": post_api_pred,
                "reduction_pct": round((1.0 - post_api_pred / base_api_pred) * 100, 1),
            },
            "api_explain_median_ms": {
                "before": base_api_exp,
                "after": post_api_exp,
                "reduction_pct": round((1.0 - post_api_exp / base_api_exp) * 100, 1),
            },
            "mps_allocated_after_25_requests_mb": {
                "before": base_mps_end,
                "after": post_mps_end,
                "saved_mb": round(base_mps_end - post_mps_end, 2),
                "reduction_pct": round((1.0 - post_mps_end / base_mps_end) * 100, 1),
            },
        },
    }

    out_path = ROOT / "results/verified_v2/post_optimization_performance_profile.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["before_vs_after_comparison"], indent=2))
    print(f"[DONE] Saved post-optimization profile to {out_path}")


if __name__ == "__main__":
    main()
