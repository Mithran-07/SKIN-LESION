# Research & Design Review — DermAI Platform V2

**Document Version:** 2.0.0  
**Date:** October 17, 2026  
**Target Platform:** Apple MacBook M4 (16 GB Unified Memory, Metal Performance Shaders / MPS)  
**Repository:** `Mithran-07/SKIN-LESION` (`feat-dermai-platform-v2`)

---

## 1. Executive Purpose

Before modifying model architectures, Out-of-Distribution (OOD) rejection mechanisms, Apple Silicon MPS memory handling, or the Next.js 14 research workstation interface, we conducted a targeted review of current authoritative engineering documentation and peer-reviewed medical imaging AI literature (2024–2026).

This document separates **externally sourced empirical/technical facts** from **DermAI engineering and design judgments**, and defines how every decision is empirically tested.

---

## 2. External Research Brief

### A. Out-of-Distribution (OOD) Detection & Dataset Shift in Medical Imaging

| Field | Detail |
|---|---|
| **Primary Sources** | 1. *"Empirical data drift detection experiments on real-world medical imaging data"*, *Nature Communications* (`s41467-024-46142-w`, 2024).<br>2. *"Automatic dataset shift identification to support safe deployment of medical imaging AI"*, *MICCAI* (`0082-Paper4612`, 2025).<br>3. Post-hoc feature-space distance studies in dermoscopy (*Mahalanobis++* / Cosine distance on ISIC & Derm7pt benchmarks, 2024–2025). |
| **Main Findings** | • Raw softmax confidence from deep classifiers is notoriously overconfident on out-of-domain inputs (e.g., text documents, natural objects, non-dermoscopic photos).<br>• Tiny prototype sets in generic ImageNet feature space fail to capture intra-class illumination, skin-tone, and acquisition-device variance in dermoscopy.<br>• Combining **L2-normalized penultimate-layer embeddings** (Cosine similarity to class-conditional centroids and regularized Mahalanobis distance fitted on the full training distribution) with low-level **acquisition-domain image statistics** (color-channel structure, high-frequency document/edge density, saturation distribution) provides robust separation between dermoscopic lesions and non-dermoscopic inputs without requiring synthetic OOD training data.<br>• Threshold calibration must be performed strictly on a held-out validation split (e.g., targeting $\ge 97\%$ True Acceptance Rate on validation in-domain images) and evaluated on an untouched test split plus diverse OOD categories. |
| **Relevance to DermAI** | Our baseline audit (`audit/outputs/06_ood_audit.json`) proved that `api/ood_gate.py` used only 7 demo images (`checkpoints/ood/mobilenet_v2_derm_prototypes.pt`), causing a **100% False Rejection Rate (50/50 rejected)** on unseen HAM10000 test images at `threshold=0.55` while falsely accepting a synthetic office text document (`0.5817`). |
| **Engineering Decision** | Replace the 7-sample MobileNetV2 prototype gate with a **Calibrated Multi-Signal Dermoscopic Domain Gate (`OODDomainGate v2`)**:<br>1. Extract penultimate 1,792-dim embeddings from the trained EfficientNet-B4 model across the `splits/train.csv` set to compute class-conditional normalized centroids, global mean, and regularized diagonal/shrinkage variance, plus training logit energy statistics.<br>2. Compute fast image-level domain heuristics (biomedical chromaticity/saturation profile, extreme document/screenshot bimodal luminance & sharp-edge grid penalty, extreme low-variance blank/solid-color check).<br>3. Calibrate the composite domain score threshold strictly on `splits/val.csv` ($N = 1,503$), never touching `splits/test.csv`.<br>4. Evaluate on all $N = 1,510$ held-out test images and a multi-category OOD benchmark suite (documents/screenshots, natural objects/landscapes, non-dermoscopic patterns, synthetic noise, corrupted files). |
| **Verification Plan** | Run `scripts/calibrate_and_eval_ood.py` and `pytest tests/test_ood_and_gradcam.py` verifying: (a) $\ge 96\%$ acceptance rate on unseen `splits/test.csv`, (b) $100\%$ rejection of text documents, noise, and non-skin images, (c) full AUROC, FAR, FRR, and 95% bootstrap confidence intervals recorded in `results/verified_v2/ood_evaluation_v2.json`. |
| **Limitations** | Close-up smartphone macro photographs of skin or clinical dermatology photographs with similar lesion pigmentation remain "near-OOD" challenges unless trained with an explicit smartphone-vs-dermatoscope binary discriminator. This limitation is explicitly surfaced in the UI and API metadata. |

---

### B. Explainability Evaluation & Grad-CAM Faithfulness in Dermatology

| Field | Detail |
|---|---|
| **Primary Sources** | 1. FDA / academic reviews on evaluating explainability in medical imaging (*MDPI Bioengineering*, `2306-5354/13/1/111`, 2025–2026).<br>2. Dermoscopy XAI literature on Grad-CAM spatial coarseness and spurious artifact attribution (2024–2025). |
| **Main Findings** | • Grad-CAM computes gradients with respect to the final convolutional feature map (`conv_head` in EfficientNet-B4, $7 \times 7$ spatial resolution before bilinear upsampling to $224 \times 224$).<br>• Consequently, heatmaps highlight **coarse regional localization** rather than pixel-exact cellular morphology.<br>• A visually plausible heatmap does **not** prove clinical correctness ("right for the wrong reasons" — e.g., attending to ruler markings, gel bubbles, or vignetting corners).<br>• Users need interactive controls (adjustable overlay opacity, side-by-side original vs. heatmap inspection, and explicit target-class selection) alongside clear methodological caveats. |
| **Relevance to DermAI** | Previously, `api/inference_engine.py` registered permanent hooks on `self.model`, retained computational graphs (`retain_graph=True`), only explained the top-1 predicted class, and returned a static baked overlay PNG at fixed $\alpha = 0.45$. |
| **Engineering Decision** | 1. Scope forward and backward hooks strictly inside `generate_heatmap()` using a `try...finally` block so hooks are removed and `activations`/`gradients` are cleared immediately after each explanation.<br>2. Return **both** the raw normalized heatmap (or transparent colorized heatmap layer PNG) and the blended overlay so the Next.js Explainability Lab can adjust overlay opacity ($0\%–100\%$) interactively in real time without re-querying the backend.<br>3. Support optional `target_class` parameter on `/predict/explain` so researchers can inspect counterfactual class attributions (e.g., *"Why did the model assign 18% probability to Melanoma vs 72% to Nevus?"*). |
| **Verification Plan** | Automated tests in `tests/test_ood_and_gradcam.py` verifying finite $[0, 1]$ heatmap values, exact $(224, 224)$ shape, bit-exact model parameter invariance before and after Grad-CAM, zero lingering forward/backward hooks on `model.conv_head`, and target-class sensitivity. |

---

### C. PyTorch on Apple Silicon M4 (MPS) Performance & Unified Memory Management

| Field | Detail |
|---|---|
| **Primary Sources** | 1. Official PyTorch MPS Backend Documentation (`https://docs.pytorch.org/docs/stable/notes/mps.html`, PyTorch 2.13/2.14, 2026).<br>2. Apple Developer Metal PyTorch Guide (`https://developer.apple.com/metal/pytorch/`). |
| **Main Findings** | • Apple Silicon uses Unified Memory shared between CPU, GPU (Metal), and macOS. Memory pressure above `PYTORCH_MPS_HIGH_WATERMARK_RATIO` causes severe SSD swap thrashing or OOM errors.<br>• `torch.mps.current_allocated_memory()` measures active tensor bytes, while `torch.mps.driver_allocated_memory()` measures total Metal driver pool memory.<br>• `torch.inference_mode()` is faster and more memory-efficient than `torch.no_grad()` because it completely disables autograd view tracking and version counters.<br>• First-time execution on MPS incurs a multi-second Metal shader compilation penalty unless warmed up at service initialization.<br>• Calling `retain_graph=True` during Grad-CAM or keeping tensor references on class attributes leaks MPS graph buffers across requests. |
| **Relevance to DermAI** | Our baseline profile (`audit/modernization_baseline/baseline_performance_profile.json`) measured:<br>• Cold first `/predict`: **3,765.76 ms**; cold first `/predict/explain`: **6,396.38 ms**.<br>• Warm `/predict`: **36.64 ms** (`9.52 ms` in separate MobileNetV2 OOD + `25.13 ms` in EfficientNet-B4).<br>• Warm `/predict/explain`: **165.71 ms** (ran image preprocessing twice and EfficientNet-B4 forward pass twice!).<br>• Retained Grad-CAM tensors (`activations_retained: true`, `gradients_retained: true`) increased active MPS memory from **82.64 MB** to **157.84 MB** (`547.33 MB` driver allocated). |
| **Engineering Decision** | 1. **Single-Pass Feature + Logit Extraction:** Extract penultimate features (`1792-dim`) and classification logits in a single `torch.inference_mode()` pass during `/predict`, reusing the penultimate embedding for OOD domain scoring (eliminating the separate MobileNetV2 forward pass and its memory footprint).<br>2. **Single-Pass Explainability:** In `/predict/explain`, reuse the preprocessed tensor and run a single forward/backward pass with scoped hooks, eliminating the redundant `predict()` forward pass.<br>3. **Startup MPS Warmup:** Run a dummy `(1, 3, 224, 224)` forward + Grad-CAM warmup pass during `SkinLesionInferenceEngine.__init__()` followed by `torch.mps.empty_cache()`, eliminating the 3.8s–6.4s cold-start lag for users.<br>4. **Zero-Leak Tensor Cleanup:** Drop `retain_graph=True`, clear hook buffers in `finally:`, and expose live `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()` in `/health` and `/diagnostics`. |
| **Verification Plan** | Re-run `audit/profile_post_optimization.py` using the identical 20-warm-run + 25-API-request protocol and compare median/p95 latencies and MPS memory footprint in `docs/RELEASE_READINESS.md`. |

---

### D. Next.js 14 App Router Production Architecture & Scientific Workstation UX

| Field | Detail |
|---|---|
| **Primary Sources** | 1. Next.js Official Production Checklist (`https://nextjs.org/docs/app/guides/production-checklist.md`, updated March 2026).<br>2. Healthcare & Scientific AI Interface Design Literature (Uncertainty communication, visual hierarchy, WCAG AA contrast, non-clinical framing). |
| **Main Findings** | • Keep data-heavy static structure in Server Components where possible and isolate interactive state in focused Client Components.<br>• Prevent memory leaks from `URL.createObjectURL()` by revoking object URLs on unmount or file replacement.<br>• Centralize API calls and TypeScript schemas so every UI component renders strictly validated backend/artifact data without hardcoded fallbacks.<br>• Scientific interfaces require high-contrast tabular numerals (`font-variant-numeric: tabular-nums`), explicit separation between **offline held-out test benchmarks** and **live inference telemetry**, and unambiguous status banners for rejected/OOD inputs. |
| **Relevance to DermAI** | Several existing frontend screens (`dashboard`, `architecture`, `research`) contained hardcoded summary strings, lacked per-class confusion matrices, omitted the V1 vs V2 Dual-Branch architectural distinction, and did not expose a dedicated System Diagnostics or interactive Explainability Lab view. |
| **Engineering Decision** | Use the integrated **StitchMCP** design workflow to evaluate three visual directions (*Research Instrument*, *Clinical Research Observatory*, *Computational Biology Lab*), select the strongest coherent direction, and implement a unified design system across all routes (`/`, `/classify`, `/dashboard`, `/research`, `/architecture`, `/diagnostics`) backed by real API endpoints (`/health`, `/benchmark`, `/classes`, `/diagnostics`, `/predict`, `/predict/explain`). |
| **Verification Plan** | `npm run build`, TypeScript strict check (`npx tsc --noEmit`), and live headless/browser verification across desktop and mobile viewports covering all operational states (idle, valid classification, counterfactual Grad-CAM, OOD rejection, corrupted image error). |

---

## 3. Prioritized 8-Phase Execution Plan

1. **Phase 1 — Baseline & Research (Completed):**
   - Branch `feat-dermai-platform-v2` created; baseline files backed up in `audit/modernization_baseline/`.
   - Pre-optimization MPS latency and memory profile recorded in `audit/modernization_baseline/baseline_performance_profile.json`.
2. **Phase 2 — Scientific Correctness Fixes:**
   - Update `models/fusion.py` and `models/dual_branch_net.py` with explicit `fusion_version: Literal["v1_matrix", "v2_scalar"]` support so all historical checkpoints (`seed123`, `seed999` V1 and `seed42` V2) load with `strict=True` and raise clear version-mismatch diagnostics.
   - Build, calibrate (on `splits/train.csv` + `splits/val.csv` only), and evaluate (on untouched `splits/test.csv` + multi-category OOD benchmark) the new **Calibrated Multi-Signal Dermoscopic Domain Gate (`OODDomainGate v2`)**.
   - Generate unified, machine-readable benchmark and provenance artifacts in `results/verified_v2/`.
3. **Phase 3 & 4 — Backend Reliability & Apple Silicon M4 MPS Optimization:**
   - Upgrade `api/inference_engine.py` and `api/main.py` with `strict=True` checkpoint verification, SHA-256 identity reporting, single-pass feature+logit inference, scoped Grad-CAM hooks with counterfactual `target_class` and raw heatmap support, startup MPS warmup, and live memory telemetry.
   - Add `/diagnostics` and enriched `/benchmark` endpoints with strict Pydantic schemas and zero local path leakage.
   - Profile post-optimization latency and memory on M4 MPS.
4. **Phase 5 & 6 — Stitch Design Exploration & Next.js 14 Workstation Redesign:**
   - Use `StitchMCP` tools to explore the 3 required design directions, document the comparative evaluation, select the winning direction, and implement the full Next.js 14 interface with real backend data across all 6 core research modules.
5. **Phase 7 & 8 — Full-System QA, Live Browser Verification & Final Deliverables:**
   - Expand `tests/` to cover architecture versioning, calibrated OOD detection, Grad-CAM invariance, and all API status codes.
   - Launch backend and frontend, verify every route and state in the browser, capture screenshots, and write all required documentation deliverables in `docs/`.

---

## 4. StitchMCP Three-Direction Visual & Architectural Exploration (Phase 5)

Using the integrated `StitchMCP` design server (Project ID `13244743375257226262` — *DermAI Research Workstation*), we explored and compared three distinct scientific interface directions before implementing the V2 Next.js 14 workstation:

| Dimension | Direction A: *Research Instrument* | Direction B: *Clinical Research Observatory* | Direction C: *Computational Biology Lab* |
|---|---|---|---|
| **Stitch Screen Reference** | Screens `fe9f0ed3392049968eb79a687928da16`, `47729a91d19848208093c54cc0161703`, `d7a0b6c9fe564508bfc6dfcdac05b623` | Screen `f82ed26d31024f3d90dce6349875b817` (*DermAI \| Clinical Research Observatory*) | Screen `3fcfb47b6368445b84a147e58df5ec7c` (*DermAI \| Computational Biology Lab*) |
| **Core Metaphor** | Optical Microscopy & Precision Imaging Instrument | Warm-Dark Editorial Clinical Dermoscopy Observatory | Zero-Chrome High-Density Computational Pathology & ML Workbench |
| **Surface Palette** | Deep Space Graphite (`#0A0C10` $\to$ `#111318` $\to$ `#1E2024`), 1px inner-glow cyan/slate borders | Obsidian Slate (`#0D1117`), Warm Ivory text (`#F0ECE1`), editorial serif annotations + clinical amber/emerald | Pure Carbon (`#07090C` $\to$ `#0D1017`), 1px solid hairlines (`#1A1F29`), zero backdrop blur |
| **Typography** | `Geist` (Display/Headers) + `Inter` (Body) + `JetBrains Mono` (Tabular Telemetry) | `Geist` + `Newsreader` (Editorial Italic) + `JetBrains Mono` | `Geist` + `JetBrains Mono` dominant across all tables and matrices |
| **Strengths** | Exceptional optical framing for $224 \times 224$ dermoscopic viewports, Grad-CAM crosshair overlays, and live stage-timing telemetry. | Strong split-screen paired comparator (Input Tensor A vs Counterfactual Grad-CAM Viewport B) and counterfactual class selector (`MEL`/`NV`/`BCC`/`BKL`/`AKIEC`/`VASC`/`DF`). | Unmatched empirical density for the 6-model benchmark table, interactive $7 \times 7$ confusion matrix, per-class Precision/Recall/F1 bars, and OOD v2 score distributions. |
| **Weaknesses if Used Alone** | Can under-serve dense statistical matrices (confusion matrices, bootstrap CIs) if cards are too spacious. | Serif editorial touches can feel slightly magazine-like on pure hardware/MPS diagnostics views. | Pure tabular density without optical viewport framing makes interactive single-image dermoscopy inspection feel austere. |

### Selected Synthesis Direction for DermAI V2
We selected **Direction A (*Research Instrument*) as the structural and chromatic foundation** (`#0A0C10` graphite canvas, `Geist` + `Inter` + `JetBrains Mono` typography, precision 4px/8px technical radii, and optical viewport framing), enhanced with:
1. **From Direction B (*Clinical Research Observatory*):** The **Paired Optical Dermoscopy & Counterfactual Grad-CAM Comparator** on `/classify`, allowing researchers to switch target classes (`AKIEC`, `BCC`, `BKL`, `DF`, `MEL`, `NV`, `VASC`) and adjust real-time RGBA heatmap opacity ($0\%–100\%$) alongside the calibrated 4-signal OOD Domain Gate readout.
2. **From Direction C (*Computational Biology Lab*):** The **High-Density Empirical Verification Matrices** on `/dashboard` and `/research`, including the interactive $7 \times 7$ confusion matrix across all 6 checkpoints, per-class Precision/Recall/F1 disaggregation, historical-vs-verified claim reconciliation, and the OOD v2 validation/test distribution breakdown ($N = 1,510$ ID vs $N = 200$ OOD).
