"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Image from "next/image";
import {
  UploadCloud, RefreshCw, ShieldAlert, CheckCircle2, XCircle,
  AlertTriangle, Eye, Loader2, Zap, Info, Sliders, Columns2, Layers, Crosshair,
} from "lucide-react";
import { StatusBadge, type StatusVariant } from "@/components/StatusBadge";
import { API_BASE_URL, resolveApiEndpoints, CLASS_METADATA, CLASS_ORDER, OOD_NOTE } from "@/lib/constants";

// ─── Types ────────────────────────────────────────────────────────────────────

type WorkstationState =
  | "idle"
  | "validating"
  | "success"
  | "rejected"
  | "model_unavailable"
  | "api_error"
  | "network_error";

interface OodSignals {
  prototype_cosine_similarity: number;
  mahalanobis_density_similarity: number;
  logit_energy_score: number;
  photometric_penalty: number;
}

interface OodValidation {
  valid: boolean;
  domain: string;
  score: number;
  threshold?: number;
  ood_version?: string;
  calibration_status?: string;
  signals?: OodSignals;
  reason: string;
}

interface Top3 {
  rank: number;
  class_code: string;
  class_name?: string;
  short_name?: string;
  full_name?: string;
  category: string;
  probability: number;
  probability_percentage?: string;
  urgency?: string;
  description?: string;
}

interface Explainability {
  method?: string;
  target_layer?: string;
  target_class_code?: string;
  target_class?: string;
  target_class_probability?: number;
  is_counterfactual?: boolean;
  overlay_alpha?: number;
  overlay_base64?: string;
  heatmap_base64?: string;
  peak_activation_coords?: { x: number; y: number };
  high_activation_area_fraction?: number;
  attribution_note?: string;
}

interface StageTimings {
  preprocessing_ms?: number;
  forward_pass_ms?: number;
  ood_gate_ms?: number;
  gradcam_backward_and_render_ms?: number;
}

interface APIResponse {
  status?: string;
  error_code?: string;
  detail?: string;
  input_validation?: OodValidation;
  predicted_class?: string;
  predicted_name?: string;
  full_name?: string;
  predicted_category?: string;
  category?: string;
  confidence?: number;
  confidence_percentage?: string;
  normalized_entropy?: number;
  probabilities?: Record<string, number>;
  top3_predictions?: Top3[];
  inference_time_ms?: number;
  stage_timings_ms?: StageTimings;
  device?: string;
  hardware_device?: string;
  model_name?: string;
  model_version?: string;
  checkpoint_sha256_short?: string;
  preprocessing_version?: string;
  class_mapping_version?: string;
  ood_version?: string;
  explainability?: Explainability;
}

// ─── Sample Presets ───────────────────────────────────────────────────────────

const PRESETS = [
  { code: "MEL",   label: "Melanoma (ISIC_0025751)",         file: "/samples/mel_sample.jpg"   },
  { code: "NV",    label: "Melanocytic Nevi (ISIC_0027411)",  file: "/samples/nv_sample.jpg"    },
  { code: "BKL",   label: "Benign Keratosis (ISIC_0032417)",  file: "/samples/bkl_sample.jpg"   },
  { code: "BCC",   label: "Basal Cell Carc. (ISIC_0024590)",  file: "/samples/bcc_sample.jpg"   },
  { code: "AKIEC", label: "Actinic Keratosis (ISIC_0026149)", file: "/samples/akiec_sample.jpg" },
  { code: "DF",    label: "Dermatofibroma (ISIC_0033695)",    file: "/samples/df_sample.jpg"    },
  { code: "VASC",  label: "Vascular Lesion (ISIC_0031215)",   file: "/samples/vasc_sample.jpg"  },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function categoryColor(cat: string) {
  const c = cat?.toLowerCase();
  if (c.includes("malignant") && !c.includes("pre"))
    return "text-status-critical bg-status-critical/10 border-status-critical/30";
  if (c.includes("pre-malignant"))
    return "text-status-warning bg-status-warning/10 border-status-warning/30";
  return "text-status-benign bg-status-benign/10 border-status-benign/30";
}

function toStatusVariant(state: WorkstationState): StatusVariant {
  if (state === "idle")              return "ready";
  if (state === "validating")        return "validating";
  if (state === "success")           return "success";
  if (state === "rejected")          return "rejected";
  if (state === "model_unavailable") return "unavailable";
  return "error";
}

export default function ClassifyPage() {
  const [state, setState]               = useState<WorkstationState>("idle");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [activeBlob, setActiveBlob]     = useState<{ blob: Blob; filename: string } | null>(null);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [result, setResult]             = useState<APIResponse | null>(null);
  const [errorMsg, setErrorMsg]         = useState<string>("");
  const [viewMode, setViewMode]         = useState<"split" | "overlay" | "original">("split");
  const [heatmapAlpha, setHeatmapAlpha] = useState<number>(0.55);
  const [targetClass, setTargetClass]   = useState<string | null>(null);
  const [explainingTarget, setExplainingTarget] = useState<boolean>(false);
  const [showOodNote, setShowOodNote]   = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const objectUrlRef = useRef<string | null>(null);

  // Clean up any created blob object URLs to prevent browser memory leaks
  const revokeCurrentObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      revokeCurrentObjectUrl();
    };
  }, [revokeCurrentObjectUrl]);

  // ── Core inference call ───────────────────────────────────────────────────

  const runInference = useCallback(
    async (blob: Blob, filename: string, preview: string, requestedTargetClass: string | null = null) => {
      setImagePreview(preview);
      setActiveBlob({ blob, filename });
      if (!requestedTargetClass) {
        setState("validating");
        setResult(null);
        setTargetClass(null);
      } else {
        setExplainingTarget(true);
      }
      setErrorMsg("");

      const qs = new URLSearchParams();
      if (requestedTargetClass) {
        qs.set("target_class", requestedTargetClass.toLowerCase());
      }
      qs.set("alpha", "0.45");

      try {
        const candidates = resolveApiEndpoints(`/predict/explain?${qs.toString()}`);
        let res: Response | null = null;
        let lastErr: unknown = null;

        for (const url of candidates) {
          try {
            const fdAttempt = new FormData();
            fdAttempt.append("file", blob, filename);
            res = await fetch(url, {
              method: "POST",
              body: fdAttempt,
            });
            break;
          } catch (e) {
            lastErr = e;
          }
        }

        if (!res) {
          throw lastErr ?? new Error("All inference endpoints unreachable");
        }

        const data: APIResponse = await res.json();

        if (!res.ok) {
          const detail = data.detail ?? res.statusText;
          const msgs: Record<number, string> = {
            400: `Invalid file: ${detail}`,
            413: "File too large (max 15 MB).",
            415: "Unsupported file format. Use JPEG, PNG, WebP, or BMP.",
            422: `Unprocessable image or parameter: ${detail}`,
            429: "Rate limit reached. Please wait a moment.",
            500: `Server error: ${detail}`,
            502: "Bad gateway — inference server may be restarting.",
            503: "Model unavailable — trained checkpoint required.",
            504: "Request timed out waiting for inference server.",
          };
          setErrorMsg(msgs[res.status] ?? `HTTP ${res.status}: ${detail}`);
          setState(res.status === 503 ? "model_unavailable" : "api_error");
          setExplainingTarget(false);
          return;
        }

        setResult(data);
        setExplainingTarget(false);

        if (data.error_code === "MODEL_UNAVAILABLE") {
          setState("model_unavailable");
        } else if (data.status === "rejected") {
          setState("rejected");
        } else if (data.status === "success") {
          setState("success");
          setTargetClass(
            data.explainability?.target_class_code?.toLowerCase() ??
              data.predicted_class?.toLowerCase() ??
              null
          );
        } else {
          setErrorMsg(data.detail ?? "Unexpected response from inference server.");
          setState("api_error");
        }
      } catch {
        setExplainingTarget(false);
        setState("network_error");
        setErrorMsg(
          `Cannot reach inference backend at ${API_BASE_URL}. Ensure the FastAPI server is running.`
        );
      }
    },
    []
  );

  // ── Counterfactual class switcher ─────────────────────────────────────────

  const handleSelectCounterfactualClass = useCallback(
    (clsCodeLower: string) => {
      if (!activeBlob || !imagePreview || state !== "success" || explainingTarget) return;
      setTargetClass(clsCodeLower);
      runInference(activeBlob.blob, activeBlob.filename, imagePreview, clsCodeLower);
    },
    [activeBlob, imagePreview, state, explainingTarget, runInference]
  );

  // ── File input ────────────────────────────────────────────────────────────

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setErrorMsg("Please select a valid image file (JPEG, PNG, WebP, BMP).");
      setState("api_error");
      return;
    }
    revokeCurrentObjectUrl();
    setActivePreset(null);
    const preview = URL.createObjectURL(file);
    objectUrlRef.current = preview;
    runInference(file, file.name, preview, null);
  };

  // ── Drag and drop ─────────────────────────────────────────────────────────

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    revokeCurrentObjectUrl();
    setActivePreset(null);
    const preview = URL.createObjectURL(file);
    objectUrlRef.current = preview;
    runInference(file, file.name, preview, null);
  };

  // ── Preset selection ──────────────────────────────────────────────────────

  const handlePreset = async (preset: (typeof PRESETS)[0]) => {
    if (state === "validating") return;
    revokeCurrentObjectUrl();
    setActivePreset(preset.code);
    setState("validating");
    setImagePreview(preset.file);
    setResult(null);
    setErrorMsg("");
    try {
      const r = await fetch(preset.file);
      const blob = await r.blob();
      runInference(blob, `${preset.code.toLowerCase()}_sample.jpg`, preset.file, null);
    } catch {
      setErrorMsg("Could not load preset image file.");
      setState("api_error");
    }
  };

  // ── Reset ─────────────────────────────────────────────────────────────────

  const handleReset = useCallback(() => {
    revokeCurrentObjectUrl();
    setState("idle");
    setImagePreview(null);
    setActiveBlob(null);
    setActivePreset(null);
    setResult(null);
    setErrorMsg("");
    setTargetClass(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [revokeCurrentObjectUrl]);

  // ── Derived ───────────────────────────────────────────────────────────────

  const probs = result?.probabilities;
  const hasGradCam = Boolean(
    result?.explainability?.heatmap_base64 || result?.explainability?.overlay_base64
  );
  const oodThreshold = result?.input_validation?.threshold ?? 0.25;
  const oodScore = result?.input_validation?.score ?? 0;

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-[1440px] mx-auto space-y-6">

      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4 flex-wrap border-b border-outline-variant/15 pb-4">
        <div>
          <div className="font-technical-data text-[10px] text-primary uppercase tracking-widest mb-1">
            Optical Dermoscopy & Counterfactual Attribution Observatory
          </div>
          <h1 className="font-headline-md text-xl sm:text-2xl font-bold text-on-surface">
            DermAI Analysis Workstation
          </h1>
          <p className="font-body-sm text-sm text-on-surface-variant mt-0.5">
            Single-pass EfficientNet-B4 classification, calibrated OOD domain gating (v2.0.0), and counterfactual Grad-CAM explainability.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge variant={toStatusVariant(state)} />
        </div>
      </div>

      {/* ── Main Grid (7-col Optical Comparator + 5-col Calibration & Probability Readout) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

        {/* ── LEFT 7 COLUMNS: Optical Viewport & Explainability Lab ──────── */}
        <div className="lg:col-span-7 space-y-4">

          {/* View Mode & Opacity Toolbar */}
          {state === "success" && hasGradCam && (
            <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-3 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-1.5" role="tablist" aria-label="Viewport display mode">
                {[
                  { id: "split" as const,    label: "SPLIT COMPARATOR", Icon: Columns2 },
                  { id: "overlay" as const,  label: "LIVE BLEND",       Icon: Layers   },
                  { id: "original" as const, label: "RAW INPUT",        Icon: Eye      },
                ].map(({ id, label, Icon }) => (
                  <button
                    key={id}
                    role="tab"
                    aria-selected={viewMode === id}
                    onClick={() => setViewMode(id)}
                    className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded border font-technical-label text-[11px] transition-colors ${
                      viewMode === id
                        ? "bg-primary text-on-primary border-primary font-bold"
                        : "border-outline-variant/30 text-on-surface-variant hover:text-primary hover:border-primary/40"
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" aria-hidden="true" />
                    {label}
                  </button>
                ))}
              </div>

              {/* Real-Time Heatmap Opacity Slider */}
              {viewMode !== "original" && (
                <div className="flex items-center gap-2.5 ml-auto">
                  <Sliders className="w-3.5 h-3.5 text-primary" aria-hidden="true" />
                  <label
                    htmlFor="gradcam-opacity-slider"
                    className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider"
                  >
                    Grad-CAM Opacity
                  </label>
                  <input
                    id="gradcam-opacity-slider"
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={Math.round(heatmapAlpha * 100)}
                    onChange={(e) => setHeatmapAlpha(Number(e.target.value) / 100)}
                    className="w-24 accent-cyan-400 cursor-pointer"
                  />
                  <span className="font-technical-data text-[11px] text-primary w-9 text-right">
                    {Math.round(heatmapAlpha * 100)}%
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Optical Viewport Container */}
          <div
            className="relative bg-surface-container-lowest border border-outline-variant/25 rounded-xl overflow-hidden"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            role="region"
            aria-label="Dermoscopic optical viewport"
          >
            {/* Scanner line during inference */}
            {(state === "validating" || explainingTarget) && (
              <div className="scanner-marquee z-20" aria-hidden="true" />
            )}

            {imagePreview && state !== "idle" ? (
              viewMode === "split" && state === "success" && hasGradCam ? (
                /* ── Paired Split-Screen Comparator (Direction B Synthesis) ── */
                <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-outline-variant/20">
                  {/* Viewport A: Raw Input Tensor */}
                  <div className="relative aspect-square bg-black/40 p-3 flex flex-col justify-between">
                    <div className="z-10 flex items-center justify-between">
                      <span className="px-2 py-0.5 rounded bg-surface-container/90 border border-outline-variant/30 font-technical-data text-[10px] text-on-surface">
                        VIEWPORT A · RAW DERMOSCOPY
                      </span>
                      <span className="font-technical-data text-[10px] text-on-surface-variant">
                        224×224 CROP
                      </span>
                    </div>
                    <div className="relative flex-1 my-2 rounded overflow-hidden">
                      <Image
                        src={imagePreview}
                        alt="Original dermoscopic input image"
                        fill
                        className="object-contain"
                        sizes="(max-width: 1024px) 50vw, 35vw"
                        priority
                      />
                    </div>
                    <div className="z-10 flex items-center justify-between font-technical-data text-[10px] text-on-surface-variant">
                      <span>NORM: IMAGENET μ/σ</span>
                      <span>DOMAIN: {result?.input_validation?.score?.toFixed(3)}</span>
                    </div>
                  </div>

                  {/* Viewport B: Interactive RGBA Grad-CAM Attribution */}
                  <div className="relative aspect-square bg-black/40 p-3 flex flex-col justify-between">
                    <div className="z-10 flex items-center justify-between">
                      <span className="px-2 py-0.5 rounded bg-primary/15 border border-primary/40 font-technical-data text-[10px] text-primary font-bold">
                        VIEWPORT B · GRAD-CAM ({result?.explainability?.target_class_code ?? result?.predicted_class})
                      </span>
                      {result?.explainability?.is_counterfactual && (
                        <span className="px-1.5 py-0.5 rounded bg-research-violet/20 border border-research-violet/40 font-technical-data text-[9px] text-research-violet font-bold">
                          COUNTERFACTUAL
                        </span>
                      )}
                    </div>
                    <div className="relative flex-1 my-2 rounded overflow-hidden">
                      {/* Base image */}
                      <Image
                        src={imagePreview}
                        alt="Base dermoscopic image for Grad-CAM overlay"
                        fill
                        className="object-contain"
                        sizes="(max-width: 1024px) 50vw, 35vw"
                      />
                      {/* Transparent RGBA Heatmap layer controlled by real-time opacity slider */}
                      {result?.explainability?.heatmap_base64 ? (
                        <Image
                          src={result.explainability.heatmap_base64}
                          alt={`Grad-CAM attribution map for ${result.explainability.target_class}`}
                          fill
                          className="object-contain transition-opacity duration-150"
                          style={{ opacity: heatmapAlpha }}
                          sizes="(max-width: 1024px) 50vw, 35vw"
                        />
                      ) : (
                        result?.explainability?.overlay_base64 && (
                          <Image
                            src={result.explainability.overlay_base64}
                            alt="Grad-CAM attribution overlay"
                            fill
                            className="object-contain"
                            style={{ opacity: heatmapAlpha }}
                            sizes="(max-width: 1024px) 50vw, 35vw"
                          />
                        )
                      )}
                    </div>
                    <div className="z-10 flex items-center justify-between font-technical-data text-[10px] text-on-surface-variant">
                      <span className="inline-flex items-center gap-1">
                        <Crosshair className="w-3 h-3 text-primary" aria-hidden="true" />
                        PEAK: ({result?.explainability?.peak_activation_coords?.x ?? "—"},{" "}
                        {result?.explainability?.peak_activation_coords?.y ?? "—"})
                      </span>
                      <span>
                        ROI ≥0.5:{" "}
                        {result?.explainability?.high_activation_area_fraction !== undefined
                          ? `${(result.explainability.high_activation_area_fraction * 100).toFixed(1)}%`
                          : "—"}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                /* ── Single Viewport Mode (Live Blend or Raw Input) ── */
                <div className="relative w-full" style={{ aspectRatio: "4 / 3", maxHeight: 460 }}>
                  <Image
                    src={imagePreview}
                    alt="Uploaded dermoscopic image for analysis"
                    fill
                    className="object-contain"
                    sizes="(max-width:1024px) 100vw, 60vw"
                    priority
                  />
                  {viewMode === "overlay" &&
                    state === "success" &&
                    (result?.explainability?.heatmap_base64 ||
                      result?.explainability?.overlay_base64) && (
                      <Image
                        src={
                          result.explainability.heatmap_base64 ??
                          result.explainability.overlay_base64!
                        }
                        alt="Grad-CAM heatmap overlay"
                        fill
                        className="object-contain transition-opacity duration-150"
                        style={{ opacity: heatmapAlpha }}
                        sizes="(max-width:1024px) 100vw, 60vw"
                      />
                    )}
                </div>
              )
            ) : (
              /* Upload zone */
              <button
                onClick={() => fileInputRef.current?.click()}
                style={{ minHeight: 380 }}
                className="w-full flex flex-col items-center justify-center gap-4 p-8 text-on-surface-variant hover:text-primary hover:bg-primary/5 transition-colors cursor-pointer"
                aria-label="Click to upload an image or drag and drop here"
              >
                <UploadCloud className="w-12 h-12 opacity-40" aria-hidden="true" />
                <div className="text-center space-y-1">
                  <p className="font-technical-label text-sm font-semibold text-on-surface">
                    READY FOR DERMOSCOPIC ACQUISITION
                  </p>
                  <p className="font-body-sm text-xs text-on-surface-variant">
                    Drag & drop a dermoscopic image here, or select a verified HAM10000 test sample below
                  </p>
                  <p className="font-technical-data text-[11px] text-on-surface-variant/60">
                    JPEG · PNG · WebP · BMP · max 15 MB
                  </p>
                </div>
                <span className="px-4 py-2 border border-primary/40 rounded text-primary font-technical-label text-xs font-bold">
                  SELECT IMAGE FILE
                </span>
              </button>
            )}

            {/* Processing overlay */}
            {(state === "validating" || explainingTarget) && (
              <div
                className="absolute inset-0 bg-surface-container/70 flex flex-col items-center justify-center gap-3 backdrop-blur-sm z-30"
                role="status"
                aria-live="polite"
              >
                <Loader2 className="w-8 h-8 text-primary animate-spin" aria-hidden="true" />
                <p className="font-technical-label text-sm text-primary">
                  {explainingTarget
                    ? `COMPUTING COUNTERFACTUAL GRAD-CAM (${targetClass?.toUpperCase()})…`
                    : "RUNNING SINGLE-PASS INFERENCE & OOD GATE…"}
                </p>
                <p className="font-technical-data text-[11px] text-on-surface-variant">
                  Resize(256) → CenterCrop(224) → EfficientNet-B4 → OOD v2 Gate → Scoped Grad-CAM
                </p>
              </div>
            )}
          </div>

          {/* ── Counterfactual Class Attribution Selector (Explainability Lab) ── */}
          {state === "success" && result && (
            <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-3">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-primary" aria-hidden="true" />
                  <span className="font-technical-label text-xs font-bold text-on-surface uppercase tracking-wider">
                    Counterfactual Class Attribution Selector (Grad-CAM `conv_head`)
                  </span>
                </div>
                <span className="font-technical-data text-[10px] text-on-surface-variant">
                  Click any class to inspect spatial evidence for that hypothesis
                </span>
              </div>

              <div className="grid grid-cols-7 gap-1.5" role="group" aria-label="Target class for Grad-CAM">
                {CLASS_ORDER.map((clsKey) => {
                  const meta = CLASS_METADATA[clsKey];
                  const probPct = ((probs?.[clsKey] ?? 0) * 100).toFixed(1);
                  const isSelected = (targetClass ?? result.predicted_class?.toLowerCase()) === clsKey;
                  const isPredicted = result.predicted_class?.toLowerCase() === clsKey;
                  return (
                    <button
                      key={clsKey}
                      onClick={() => handleSelectCounterfactualClass(clsKey)}
                      disabled={explainingTarget}
                      aria-pressed={isSelected}
                      className={`p-2 rounded border text-center transition-all disabled:opacity-50 ${
                        isSelected
                          ? "border-primary bg-primary/15 text-primary font-bold"
                          : "border-outline-variant/20 bg-surface-container-low text-on-surface-variant hover:border-primary/40 hover:text-on-surface"
                      }`}
                    >
                      <div className="font-technical-data text-[11px] font-bold flex items-center justify-center gap-1">
                        {meta.code}
                        {isPredicted && (
                          <span className="w-1.5 h-1.5 rounded-full bg-status-benign" title="Top-1 Predicted Class" />
                        )}
                      </div>
                      <div className="font-technical-data text-[10px] opacity-80 mt-0.5">
                        {probPct}%
                      </div>
                    </button>
                  );
                })}
              </div>

              <p className="font-body-sm text-[11px] text-on-surface-variant leading-relaxed border-t border-outline-variant/10 pt-2">
                <strong className="text-on-surface">Methodological Note:</strong>{" "}
                {result.explainability?.attribution_note}
              </p>
            </div>
          )}

          {/* ── Verified HAM10000 Demo Samples ── */}
          <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="font-technical-label text-[11px] text-on-surface uppercase tracking-wider">
                Verified HAM10000 Held-Out Test Samples (1 per Class)
              </span>
              <span className="font-technical-data text-[10px] text-on-surface-variant">
                Genuine 600×450 ISIC Dermoscopy
              </span>
            </div>
            <div className="grid grid-cols-7 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.code}
                  onClick={() => handlePreset(p)}
                  disabled={state === "validating" || explainingTarget}
                  aria-label={`Load ${p.label}`}
                  className={`flex flex-col items-center gap-1.5 p-1.5 rounded border transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                    activePreset === p.code
                      ? "border-primary bg-primary/15"
                      : "border-outline-variant/20 hover:border-primary/40 bg-surface-container-low"
                  }`}
                >
                  <div className="relative w-full" style={{ aspectRatio: "1 / 1" }}>
                    <Image
                      src={p.file}
                      alt={p.label}
                      fill
                      className="object-cover rounded"
                      sizes="80px"
                    />
                  </div>
                  <span
                    className={`font-technical-data text-[10px] font-bold ${
                      activePreset === p.code ? "text-primary" : "text-on-surface-variant"
                    }`}
                  >
                    {p.code}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
            aria-hidden="true"
          />
        </div>

        {/* ── RIGHT 5 COLUMNS: Calibrated OOD Gate, Probabilities & Telemetry ── */}
        <div className="lg:col-span-5 space-y-4">

          {/* IDLE */}
          {state === "idle" && (
            <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-6 text-center space-y-3">
              <p className="font-technical-label text-sm text-on-surface">
                Awaiting Dermoscopic Input
              </p>
              <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                Select one of the 7 verified HAM10000 test lesions on the left or upload an image to inspect
                the calibrated OOD domain gate, 7-class probability distribution, and counterfactual Grad-CAM heatmaps.
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="mt-2 px-4 py-2 border border-primary/40 rounded text-primary font-technical-label text-xs font-bold hover:bg-primary/10 transition-colors"
              >
                UPLOAD DERMOSCOPIC IMAGE
              </button>
            </div>
          )}

          {/* VALIDATING */}
          {state === "validating" && (
            <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-4" role="status" aria-live="polite">
              <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
                Single-Pass Execution Pipeline
              </div>
              {[
                { label: "File & MIME Verification",  note: "Size ≤ 15 MB · PIL integrity check",             done: true  },
                { label: "Deterministic Transform",   note: "Resize(256) → CenterCrop(224) → ImageNet norm",  done: true  },
                { label: "EfficientNet-B4 Forward",   note: "1,792-dim penultimate embedding + 7-class logits", done: false },
                { label: "Calibrated OOD Gate v2",    note: "49-prototype cos + Mahalanobis + Energy + Photo", done: false },
                { label: "Scoped Grad-CAM",           note: "conv_head backward pass (retain_graph=False)",   done: false },
              ].map((step, i) => (
                <div key={i} className="flex items-center gap-3">
                  {step.done ? (
                    <CheckCircle2 className="w-4 h-4 text-status-benign shrink-0" aria-hidden="true" />
                  ) : (
                    <Loader2 className="w-4 h-4 text-primary shrink-0 animate-spin" aria-hidden="true" />
                  )}
                  <div>
                    <p className="font-technical-label text-xs text-on-surface">{step.label}</p>
                    <p className="font-technical-data text-[10px] text-on-surface-variant">{step.note}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* REJECTED (OOD Gate v2 Triggered) */}
          {state === "rejected" && result && (
            <div className="bg-status-critical/5 rounded-xl border border-status-critical/30 p-5 space-y-4" role="alert">
              <div className="flex items-start gap-3">
                <ShieldAlert className="w-5 h-5 text-status-critical shrink-0 mt-0.5" aria-hidden="true" />
                <div>
                  <h2 className="font-technical-label text-sm font-bold text-status-critical">
                    OUT-OF-DOMAIN INPUT REJECTED (OOD GATE v2.0.0)
                  </h2>
                  <p className="font-body-sm text-xs text-on-surface-variant mt-1 leading-relaxed">
                    {result.input_validation?.reason ??
                      "Input image falls outside the calibrated HAM10000 dermoscopic distribution."}
                  </p>
                </div>
              </div>

              <div className="bg-surface-container rounded border border-outline-variant/15 p-3 font-technical-data text-[11px] space-y-2">
                <div className="flex justify-between gap-2">
                  <span className="text-on-surface-variant">Detected Domain</span>
                  <span className="text-status-critical font-bold uppercase">
                    {result.input_validation?.domain ?? "NON_DERMOSCOPIC"}
                  </span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-on-surface-variant">Composite Domain Score</span>
                  <span className="text-on-surface font-bold">
                    {oodScore.toFixed(4)} (Threshold ≥ {oodThreshold.toFixed(4)})
                  </span>
                </div>
                {result.input_validation?.signals && (
                  <>
                    <div className="flex justify-between gap-2 text-[10px]">
                      <span className="text-on-surface-variant">Prototype Cosine Similarity</span>
                      <span className="text-on-surface">
                        {result.input_validation.signals.prototype_cosine_similarity.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between gap-2 text-[10px]">
                      <span className="text-on-surface-variant">Mahalanobis Density Sim</span>
                      <span className="text-on-surface">
                        {result.input_validation.signals.mahalanobis_density_similarity.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between gap-2 text-[10px]">
                      <span className="text-on-surface-variant">Photometric Multiplier</span>
                      <span className="text-on-surface">
                        {result.input_validation.signals.photometric_penalty.toFixed(2)}×
                      </span>
                    </div>
                  </>
                )}
                <div className="border-t border-outline-variant/15 pt-2 flex justify-between gap-2">
                  <span className="text-on-surface-variant">Diagnostic Classification</span>
                  <span className="text-status-critical font-bold">SUPPRESSED (SAFE REJECT)</span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-on-surface-variant">Grad-CAM Attribution</span>
                  <span className="text-status-critical font-bold">NOT GENERATED</span>
                </div>
              </div>

              <button
                onClick={() => setShowOodNote(!showOodNote)}
                className="flex items-center gap-1.5 font-technical-data text-[10px] text-on-surface-variant hover:text-primary transition-colors"
              >
                <Info className="w-3 h-3" aria-hidden="true" /> About OODDomainGate v2.0.0 Calibration
              </button>
              {showOodNote && (
                <p className="font-technical-data text-[10px] text-on-surface-variant leading-relaxed bg-surface-container p-3 rounded border border-outline-variant/10">
                  {OOD_NOTE}
                </p>
              )}

              <button
                onClick={handleReset}
                className="w-full py-2.5 rounded border border-outline-variant/30 text-on-surface font-technical-label text-xs hover:border-primary/40 hover:text-primary transition-colors"
              >
                ↑ UPLOAD ANOTHER IMAGE
              </button>
            </div>
          )}

          {/* MODEL UNAVAILABLE */}
          {state === "model_unavailable" && (
            <div className="bg-status-warning/5 rounded-xl border border-status-warning/30 p-5 space-y-4" role="alert">
              <div className="flex items-start gap-3">
                <XCircle className="w-5 h-5 text-status-warning shrink-0 mt-0.5" aria-hidden="true" />
                <div>
                  <h2 className="font-technical-label text-sm font-bold text-status-warning">
                    MODEL UNAVAILABLE
                  </h2>
                  <p className="font-body-sm text-xs text-on-surface-variant mt-1">
                    The trained EfficientNet-B4 checkpoint is not loaded. Inference refuses to run on random weights.
                  </p>
                </div>
              </div>
              <button
                onClick={handleReset}
                className="w-full py-2.5 rounded border border-status-warning/30 text-status-warning font-technical-label text-xs hover:bg-status-warning/10 transition-colors"
              >
                ↺ RETRY
              </button>
            </div>
          )}

          {/* API ERROR / NETWORK ERROR */}
          {(state === "api_error" || state === "network_error") && (
            <div className="bg-status-warning/5 rounded-xl border border-status-warning/30 p-5 space-y-4" role="alert">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-status-warning shrink-0 mt-0.5" aria-hidden="true" />
                <div>
                  <h2 className="font-technical-label text-sm font-bold text-status-warning">
                    {state === "network_error" ? "BACKEND CONNECTION ERROR" : "REQUEST VALIDATION ERROR"}
                  </h2>
                  <p className="font-technical-data text-[11px] text-on-surface-variant mt-1">
                    {errorMsg}
                  </p>
                </div>
              </div>
              <button
                onClick={handleReset}
                className="w-full py-2.5 rounded border border-status-warning/30 text-status-warning font-technical-label text-xs hover:bg-status-warning/10 transition-colors"
              >
                ↺ RETRY
              </button>
            </div>
          )}

          {/* SUCCESS */}
          {state === "success" && result && (
            <div className="space-y-4">

              {/* Primary Prediction Card */}
              <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
                    Primary Classification (Top-1)
                  </span>
                  <span className="font-technical-data text-[11px] text-primary flex items-center gap-1">
                    <Zap className="w-3 h-3" aria-hidden="true" />
                    {result.inference_time_ms?.toFixed(1)} ms ({result.device?.toUpperCase()})
                  </span>
                </div>

                <div className="flex items-start justify-between gap-2">
                  <h2 className="font-headline-md text-xl font-bold text-on-surface leading-tight">
                    {result.predicted_name ?? result.full_name ?? result.predicted_class}
                  </h2>
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded border font-technical-label text-xs font-bold whitespace-nowrap ${categoryColor(
                      result.predicted_category ?? result.category ?? ""
                    )}`}
                  >
                    {result.predicted_class}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 pt-1">
                  <div className="p-2 rounded bg-surface-container-low border border-outline-variant/15">
                    <div className="font-technical-label text-[9px] text-on-surface-variant uppercase">
                      Top-1 Probability
                    </div>
                    <div className="font-technical-data text-sm font-bold text-primary mt-0.5">
                      {result.confidence_percentage}
                    </div>
                  </div>
                  <div className="p-2 rounded bg-surface-container-low border border-outline-variant/15">
                    <div className="font-technical-label text-[9px] text-on-surface-variant uppercase">
                      Risk Taxonomy
                    </div>
                    <div className="font-technical-data text-xs font-bold text-on-surface mt-0.5 truncate">
                      {result.predicted_category ?? "—"}
                    </div>
                  </div>
                  <div className="p-2 rounded bg-surface-container-low border border-outline-variant/15">
                    <div className="font-technical-label text-[9px] text-on-surface-variant uppercase">
                      Norm. Entropy
                    </div>
                    <div className="font-technical-data text-sm font-bold text-on-surface mt-0.5">
                      {result.normalized_entropy !== undefined
                        ? result.normalized_entropy.toFixed(4)
                        : "—"}
                    </div>
                  </div>
                </div>
              </div>

              {/* Calibrated OOD Domain Gate v2 Readout */}
              <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
                    Calibrated OOD Domain Gate (v{result.ood_version ?? "2.0.0"})
                  </span>
                  <span className="inline-flex items-center gap-1 font-technical-data text-[10px] text-status-benign font-bold">
                    <CheckCircle2 className="w-3.5 h-3.5" aria-hidden="true" />
                    IN-DISTRIBUTION
                  </span>
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between font-technical-data text-[11px]">
                    <span className="text-on-surface-variant">
                      Domain Score (Threshold ≥ {oodThreshold.toFixed(2)})
                    </span>
                    <span className="text-primary font-bold">{oodScore.toFixed(4)}</span>
                  </div>
                  <div className="relative h-2 w-full bg-surface-container-lowest rounded-full overflow-hidden border border-outline-variant/15">
                    {/* Threshold marker at 25% */}
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-status-warning z-10"
                      style={{ left: `${oodThreshold * 100}%` }}
                      title={`Threshold: ${oodThreshold}`}
                    />
                    <div
                      className="h-full bg-status-benign rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(oodScore * 100, 100)}%` }}
                    />
                  </div>
                </div>

                {result.input_validation?.signals && (
                  <div className="grid grid-cols-3 gap-2 pt-1 font-technical-data text-[10px]">
                    <div className="p-1.5 rounded bg-surface-container-low border border-outline-variant/10">
                      <span className="text-on-surface-variant block text-[9px]">PROTOTYPE COS</span>
                      <span className="text-on-surface font-semibold">
                        {result.input_validation.signals.prototype_cosine_similarity.toFixed(3)}
                      </span>
                    </div>
                    <div className="p-1.5 rounded bg-surface-container-low border border-outline-variant/10">
                      <span className="text-on-surface-variant block text-[9px]">MAHALANOBIS</span>
                      <span className="text-on-surface font-semibold">
                        {result.input_validation.signals.mahalanobis_density_similarity.toFixed(3)}
                      </span>
                    </div>
                    <div className="p-1.5 rounded bg-surface-container-low border border-outline-variant/10">
                      <span className="text-on-surface-variant block text-[9px]">LOGIT ENERGY</span>
                      <span className="text-on-surface font-semibold">
                        {result.input_validation.signals.logit_energy_score.toFixed(3)}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Full 7-Class Probability Distribution */}
              {probs && (
                <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
                      7-Class Softmax Probability Distribution
                    </span>
                    <span className="font-technical-data text-[10px] text-on-surface-variant">
                      Σp ={" "}
                      {Object.values(probs)
                        .reduce((a, b) => a + b, 0)
                        .toFixed(4)}
                    </span>
                  </div>
                  <div className="space-y-2">
                    {CLASS_ORDER.map((cls) => {
                      const pct = (probs[cls] ?? 0) * 100;
                      const meta = CLASS_METADATA[cls];
                      const isPred = cls === result.predicted_class?.toLowerCase();
                      const isExplained = cls === targetClass;
                      return (
                        <div key={cls} className="space-y-1">
                          <div className="flex items-center justify-between font-technical-data text-[11px]">
                            <div className="flex items-center gap-1.5">
                              <span
                                className={`font-bold w-12 ${
                                  isPred ? "text-primary" : "text-on-surface"
                                }`}
                              >
                                {meta?.code ?? cls.toUpperCase()}
                              </span>
                              <span className="text-on-surface-variant text-[10px] truncate max-w-[160px]">
                                {meta?.name}
                              </span>
                              {isExplained && (
                                <span className="px-1 py-0.2 rounded bg-primary/15 text-primary text-[9px]">
                                  CAM
                                </span>
                              )}
                            </div>
                            <span
                              className={
                                isPred ? "text-primary font-bold" : "text-on-surface-variant"
                              }
                            >
                              {pct.toFixed(2)}%
                            </span>
                          </div>
                          <div className="h-1.5 w-full bg-surface-container-lowest rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-300 ${
                                isPred ? "bg-primary" : "bg-secondary/50"
                              }`}
                              style={{ width: `${Math.max(pct, 0.4)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Stage-by-Stage Runtime Telemetry & Provenance */}
              <div className="bg-surface-container-low rounded-xl border border-outline-variant/15 p-3.5 font-technical-data text-[10px] text-on-surface-variant space-y-1.5">
                <div className="text-[9px] text-on-surface uppercase tracking-wider font-bold mb-1">
                  Live Execution Telemetry & Provenance
                </div>
                <div className="flex justify-between gap-2">
                  <span>Model / Checkpoint SHA-256</span>
                  <span className="text-on-surface">
                    EfficientNet-B4 · {result.checkpoint_sha256_short ?? "125340ffa59ca4a1"}
                  </span>
                </div>
                <div className="flex justify-between gap-2">
                  <span>Preprocessing Pipeline</span>
                  <span className="text-on-surface">Resize(256) → CenterCrop(224)</span>
                </div>
                {result.stage_timings_ms && (
                  <div className="grid grid-cols-2 gap-1.5 pt-1 border-t border-outline-variant/10">
                    <div>
                      Preprocess:{" "}
                      <span className="text-on-surface">
                        {result.stage_timings_ms.preprocessing_ms?.toFixed(2)} ms
                      </span>
                    </div>
                    <div>
                      Forward Pass:{" "}
                      <span className="text-on-surface">
                        {result.stage_timings_ms.forward_pass_ms?.toFixed(2)} ms
                      </span>
                    </div>
                    <div>
                      OOD v2 Gate:{" "}
                      <span className="text-on-surface">
                        {result.stage_timings_ms.ood_gate_ms?.toFixed(2)} ms
                      </span>
                    </div>
                    <div>
                      Grad-CAM:{" "}
                      <span className="text-on-surface">
                        {result.stage_timings_ms.gradcam_backward_and_render_ms?.toFixed(2) ?? "—"} ms
                      </span>
                    </div>
                  </div>
                )}
                <div className="flex justify-between gap-2 pt-1 border-t border-outline-variant/10">
                  <span>Total Live Latency ({result.device?.toUpperCase()})</span>
                  <span className="text-primary font-bold">
                    {result.inference_time_ms?.toFixed(2)} ms
                  </span>
                </div>
              </div>

              {/* Reset */}
              <button
                onClick={handleReset}
                className="w-full py-2.5 rounded border border-outline-variant/20 text-on-surface-variant font-technical-label text-xs hover:border-primary/40 hover:text-primary transition-colors flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" aria-hidden="true" /> RESET WORKSTATION
              </button>
            </div>
          )}

          {/* Always-visible Non-Clinical Disclaimer */}
          <div className="p-3 rounded-xl border border-status-warning/20 bg-surface-container-lowest font-technical-data text-[10px] text-on-surface-variant leading-relaxed">
            <strong className="text-status-warning">NON-CLINICAL RESEARCH PROTOTYPE:</strong>{" "}
            Predictions, OOD scores, and Grad-CAM heatmaps are strictly for academic deep learning evaluation on HAM10000. Never use for patient diagnosis or treatment decisions.
          </div>
        </div>
      </div>
    </div>
  );
}
