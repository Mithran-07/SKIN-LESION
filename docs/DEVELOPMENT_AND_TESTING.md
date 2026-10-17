# DermAI Research Platform V2 — Development, Execution & Testing Guide

**Target Hardware:** Apple Silicon MacBook M4 (`16 GB` Unified Memory, macOS, PyTorch `mps`) or Linux/Windows CUDA (`cuda`) / CPU (`cpu`).  
**Repository Root:** `/Users/mithran/Documents/My projects/ADL` (`/Users/mithran/Documents/ADL`)  

---

## 1. Local Environment Setup & Startup Commands

### 1.1 FastAPI Backend (`127.0.0.1:8000`)

Using the existing project virtual environment (`.venv`):

```bash
# From repository root (/Users/mithran/Documents/ADL)
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- On startup, `SkinLesionInferenceEngine` automatically:
  1. Verifies the SHA-256 hash (`125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e`) and loads `checkpoints/efficientnet_b4/best_checkpoint.pth` with `strict=True`.
  2. Loads `checkpoints/ood/efficientnet_b4_ood_calibration_v2.pt` into `OODDomainGate` v2.0.0.
  3. Runs a Metal/MPS forward + backward shader warmup pass (`~410 ms`) so the first interactive user request completes in `~236 ms` instead of `3.8s – 6.4s`.

### 1.2 Next.js 14 Research Workstation (`http://localhost:3000`)

```bash
# From frontend directory (/Users/mithran/Documents/ADL/app/skin-lesion-app)
npm run dev
```

For production build verification:

```bash
npm run build
npm run start
```

---

## 2. Reproducing All Verification & Calibration Artifacts

### 2.1 Calibrate & Evaluate OOD Domain Gate v2
Fits class/sub-cluster prototypes, diagonal Mahalanobis statistics, and logit energy distribution strictly on `splits/train.csv` ($N = 2,123$), selects threshold $\tau = 0.2500$ on `splits/val.csv` ($N = 1,503$), and evaluates once on `splits/test.csv` ($N = 1,510$ ID + $N = 200$ OOD across 10 categories):

```bash
.venv/bin/python scripts/calibrate_and_eval_ood.py
```

Outputs:
- `checkpoints/ood/efficientnet_b4_ood_calibration_v2.pt`
- `results/verified_v2/ood_evaluation_v2.json`

### 2.2 Export Unified Machine-Verified Benchmark Registry v2
Consolidates all 6 model evaluations on `splits/test.csv`, SHA-256 checkpoint digests, per-class metrics, $7 \times 7$ confusion matrices, and OOD v2 results:

```bash
.venv/bin/python scripts/export_verified_benchmarks_v2.py
```

Output:
- `results/verified_v2/benchmark_registry_v2.json`

### 2.3 Run Apple Silicon M4 MPS Latency & Memory Profiler
Measures cold-start vs. warm-request latency (`P50`, `P95`, `mean`, `std`) and active MPS tensor memory across 25 back-to-back `/predict` and `/predict/explain` requests:

```bash
.venv/bin/python audit/profile_post_optimization.py
```

Output:
- `results/verified_v2/post_optimization_performance_profile.json`

---

## 3. Automated Test Suite (`pytest`)

Run the complete 31-test automated verification suite from the repository root:

```bash
.venv/bin/pytest -v
```

### Test Coverage Breakdown (`31 passed in 17.81s`)
1. `tests/test_api.py` (12 tests):
   - `test_health_endpoint`: Verifies `/health` status, `strict=True` checkpoint loading, `125340ffa59ca4a1` SHA-256 prefix, OOD v2 readiness, and absence of local path leakage.
   - `test_diagnostics_endpoint`: Verifies `/diagnostics` MPS memory telemetry, zero active Grad-CAM hooks at rest, and canonical checkpoint SHA-256 verification.
   - `test_benchmark_endpoint_accuracy_consistency`: Verifies `/benchmark` 6-checkpoint registry and champion metrics.
   - `test_classes_endpoint`: Verifies canonical 7-class alphabetical ordering (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
   - `test_predict_valid_dermoscopic_jpeg` & `test_predict_valid_dermoscopic_png`: Verifies in-domain classification on genuine HAM10000 held-out test sample (`data/samples/mel_sample.jpg`) and probability simplex $\sum_c p_c = 1.0$.
   - `test_predict_explain_default_and_counterfactual`: Verifies `/predict/explain?target_class=bcc&alpha=0.5`, standalone `heatmap_base64`, `overlay_base64`, `is_counterfactual=True`, and confirms zero lingering forward/backward hooks on `conv_head`.
   - `test_ood_rejection_on_non_dermoscopic_image`: Verifies that non-dermoscopic inputs return `status="rejected"`, `input_validation.valid=False`, and `explainability=None`.
   - `test_unsupported_file_type` (`415`), `test_empty_file_upload` (`400`), `test_corrupted_image_bytes` (`422`), and `test_invalid_counterfactual_target_class` (`422`).
2. `tests/test_model_shapes.py` (10 tests):
   - Verifies tensor shapes for `ShallowWideBranch`, `DeepNarrowBranch`, `AttentionFusionHead` (`v1_matrix` and `v2_scalar`), `DualBranchNet` (`10,669,639` params for `v1_matrix` and `9,031,241` params for `v2_scalar`), `MTLDualBranchNet`, `ResNet50Baseline`, `DenseNet201Baseline`, and `EfficientNetBaseline`.
   - Verifies `DualBranchNet.from_checkpoint(..., strict=True)` round-trip loading and version mismatch detection (`ValueError`).
3. `tests/test_focal_loss.py` (5 tests) & `tests/test_conformal.py` (4 tests):
   - Verifies Focal Loss $\gamma$-focusing, $\alpha$-balancing, numerical stability, and RAPS conformal prediction coverage guarantees.
