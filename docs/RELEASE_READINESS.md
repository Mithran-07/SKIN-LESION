# DermAI Research Platform V2 — Release Readiness & Acceptance Checklist

**Release Version:** `2.0.0`  
**Branch:** `main` (`feat-dermai-platform-v2`)  
**Live Production URL:** [`https://skin-lesion-xi.vercel.app`](https://skin-lesion-xi.vercel.app)  
**GitHub Repository:** [`https://github.com/Mithran-07/SKIN-LESION`](https://github.com/Mithran-07/SKIN-LESION)  
**Date:** October 17, 2026  

---

## 1. Master Acceptance Criteria Verification

| Requirement Category | Criterion | Status | Evidence / Artifact |
| :--- | :--- | :---: | :--- |
| **Phase 1: Baseline Preservation** | Working branch created; original files backed up; baseline M4 MPS latency & memory profiled | **PASSED** | `audit/modernization_baseline/baseline_performance_profile.json`, `docs/RESEARCH_AND_DESIGN_REVIEW.md` |
| **Phase 2A: Dual-Branch Versioning** | Both `v1_matrix` (`10,669,639` params) and `v2_scalar` (`9,031,241` params) explicitly supported with `strict=True` loading | **PASSED** | `models/fusion.py`, `models/dual_branch_net.py`, `tests/test_model_shapes.py` |
| **Phase 2B: Benchmark Registry** | All 6 checkpoints evaluated on `splits/test.csv` ($N = 1,510$, $0$ lesion overlap); historical `90.98%`/`89.73%`/`90.54%` reconciled as Macro ROC-AUC | **PASSED** | `results/verified_v2/benchmark_registry_v2.json`, `docs/EXPERIMENTS_AND_BENCHMARKS.md` |
| **Phase 2C: Calibrated OOD Gate v2** | Strict Train ($N = 2,123$) / Val ($N = 1,503$, $\tau = 0.2500$) / Test ($N = 1,510$ ID + $200$ OOD) separation; **0.9985 AUROC**, **98.08% ID accept**, **100% OOD reject** | **PASSED** | `checkpoints/ood/efficientnet_b4_ood_calibration_v2.pt`, `results/verified_v2/ood_evaluation_v2.json` |
| **Phase 2D: Demo Sample Integrity** | Replaced legacy matplotlib grid crops (`*_sample.jpg`) with 7 genuine $600 \times 450$ HAM10000 held-out test images | **PASSED** | `data/samples/*.jpg`, `app/skin-lesion-app/public/samples/*.jpg` |
| **Phase 3: Backend Reliability** | Sanitized `/health` (no path leak), `/diagnostics`, counterfactual `/predict/explain?target_class=...&alpha=...`, strict input validation (`400`/`413`/`415`/`422`) | **PASSED** | `api/main.py`, `api/inference_engine.py`, `api/ood_gate.py`, `tests/test_api.py` |
| **Phase 4: M4 MPS Optimization** | Startup shader warmup (`15.93×` faster first `/predict`), single-pass OOD reuse (`27.64 ms` warm `/predict`), scoped Grad-CAM hooks, `-55.7%` active MPS memory (`69.95 MB`) | **PASSED** | `results/verified_v2/post_optimization_performance_profile.json` |
| **Phase 5: StitchMCP Design Exploration** | 3 distinct visual directions explored via `StitchMCP` in Project `13244743375257226262` and documented before frontend build | **PASSED** | Screens `f82ed26d31024f3d90dce6349875b817` & `3fcfb47b6368445b84a147e58df5ec7c`, `docs/RESEARCH_AND_DESIGN_REVIEW.md` |
| **Phase 6: Next.js 14 Workstation** | All 6 routes (`/`, `/classify`, `/dashboard`, `/research`, `/architecture`, `/diagnostics`) updated with real backend integration and `npm run build` passing | **PASSED** | `app/skin-lesion-app/src/app/**` (`next build` exit code `0`) |
| **Phase 7: Automated Tests** | Full `pytest` suite (`31 passed, 0 failed, 0 skipped`) covering API, OOD, Grad-CAM hook cleanup, and Dual-Branch versioning | **PASSED** | `tests/test_api.py`, `tests/test_model_shapes.py` |
| **Phase 8: Documentation Suite** | All 7 engineering and scientific documentation deliverables completed in `docs/` | **PASSED** | `docs/*.md` |

---

## 2. Documented Scientific Limitations & Future Research Extensions

1. **External Multi-Center Generalization:**
   - All six models and the calibrated OOD gate were trained and evaluated on partitions of **HAM10000** (ViDIR Group, Medical University of Vienna & Cliff Rosendahl practice, Queensland). Cross-dataset generalization on external cohorts (e.g., BCN20000,MSK, PAD-UFES-20, DDI) should be evaluated before claiming multi-institutional robustness.
2. **Pretrained Backbones for Dual-Branch CNN:**
   - The custom `DualBranchNet` trained `99.88%` of its weights from scratch on `7,002` images, which limited its Balanced Accuracy (`70.28%` on Seed 123) relative to pretrained `EfficientNet-B4` (`79.23%`). A natural future extension is replacing the scratch-initialized branches with pretrained multi-scale backbones (e.g., ConvNeXt-Tiny + Swin-Tiny) while retaining the versioned attention-gated fusion head.
3. **Rare-Class Sample Size (`DF` and `VASC`):**
   - In the canonical 15% lesion-disjoint test split ($N = 1,510$), Dermatofibroma (`df`) has $N = 8$ test images and Vascular Lesions (`vasc`) has $N = 24$ test images. Per-class recall on these rare classes carries wider binomial confidence intervals than `nv` ($N = 999$) or `mel` ($N = 181$).
4. **Academic Non-Clinical Scope:**
   - DermAI V2 is strictly an academic Applied Deep Learning research platform and is **not** a certified medical device. All UI screens and API responses enforce explicit non-clinical disclaimers.
