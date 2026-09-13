"use client";

import { useState, useRef, useCallback } from "react";
import Image from "next/image";
import {
  UploadCloud, RefreshCw, ShieldAlert, CheckCircle2, XCircle,
  AlertTriangle, Cpu, Eye, Loader2, Zap, Info,
} from "lucide-react";
import { StatusBadge, type StatusVariant } from "@/components/StatusBadge";
import { CLASS_METADATA, CLASS_ORDER, OOD_NOTE } from "@/lib/constants";

// ─── Types ────────────────────────────────────────────────────────────────────

type WorkstationState =
  | "idle"
  | "validating"
  | "success"
  | "rejected"
  | "model_unavailable"
  | "api_error"
  | "network_error";

interface OodValidation {
  valid: boolean;
  domain: string;
  score: number;
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
  target_class?: string;
  overlay_base64?: string;
  attribution_note?: string;
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
  probabilities?: Record<string, number>;
  top3_predictions?: Top3[];
  inference_time_ms?: number;
  device?: string;
  hardware_device?: string;
  explainability?: Explainability;
}

// ─── Sample Presets ───────────────────────────────────────────────────────────

const PRESETS = [
  { code: "MEL",   label: "Melanoma",          file: "/samples/mel_sample.jpg"   },
  { code: "NV",    label: "Melanocytic Nevi",   file: "/samples/nv_sample.jpg"    },
  { code: "BKL",   label: "Benign Keratosis",   file: "/samples/bkl_sample.jpg"   },
  { code: "BCC",   label: "Basal Cell Carc.",   file: "/samples/bcc_sample.jpg"   },
  { code: "AKIEC", label: "Actinic Keratosis",  file: "/samples/akiec_sample.jpg" },
  { code: "DF",    label: "Dermatofibroma",     file: "/samples/df_sample.jpg"    },
  { code: "VASC",  label: "Vascular Lesion",    file: "/samples/vasc_sample.jpg"  },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function categoryColor(cat: string) {
  const c = cat?.toLowerCase();
  if (c === "malignant")    return "text-status-critical bg-status-critical/10 border-status-critical/30";
  if (c === "pre-malignant") return "text-status-warning bg-status-warning/10 border-status-warning/30";
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

// ─── Component ────────────────────────────────────────────────────────────────

export default function ClassifyPage() {
  const [state, setState]             = useState<WorkstationState>("idle");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [result, setResult]           = useState<APIResponse | null>(null);
  const [errorMsg, setErrorMsg]       = useState<string>("");
  const [viewMode, setViewMode]       = useState<"original" | "overlay">("overlay");
  const [showOodNote, setShowOodNote] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Core inference call ───────────────────────────────────────────────────

  const runInference = useCallback(async (fd: FormData, preview: string) => {
    setImagePreview(preview);
    setState("validating");
    setResult(null);
    setErrorMsg("");

    try {
      const res = await fetch("http://127.0.0.1:8000/predict/explain", {
        method: "POST",
        body: fd,
      });

      const data: APIResponse = await res.json();

      if (!res.ok) {
        const detail = data.detail ?? res.statusText;
        const msgs: Record<number, string> = {
          400: `Invalid file: ${detail}`,
          413: "File too large (max 10 MB).",
          415: "Unsupported file format. Use JPEG, PNG or WebP.",
          422: "Image decoding failed — file may be corrupted or truncated.",
          429: "Rate limit reached. Please wait a moment.",
          500: `Server error: ${detail}`,
          502: "Bad gateway — inference server may be restarting.",
          503: "Service temporarily unavailable.",
          504: "Request timed out waiting for inference server.",
        };
        setErrorMsg(msgs[res.status] ?? `HTTP ${res.status}: ${detail}`);
        setState("api_error");
        return;
      }

      setResult(data);

      if (data.error_code === "MODEL_UNAVAILABLE") {
        setState("model_unavailable");
      } else if (data.status === "rejected") {
        setState("rejected");
      } else if (data.status === "success") {
        setState("success");
      } else {
        setErrorMsg(data.detail ?? "Unexpected response from inference server.");
        setState("api_error");
      }
    } catch {
      setState("network_error");
      setErrorMsg(
        "Cannot reach inference backend. Is the FastAPI server running on port 8000?"
      );
    }
  }, []);

  // ── File input ────────────────────────────────────────────────────────────

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setErrorMsg("Please select a valid image file (JPEG, PNG, WebP).");
      setState("api_error");
      return;
    }
    setActivePreset(null);
    const preview = URL.createObjectURL(file);
    const fd = new FormData();
    fd.append("file", file);
    runInference(fd, preview);
  };

  // ── Drag and drop ─────────────────────────────────────────────────────────

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    setActivePreset(null);
    const preview = URL.createObjectURL(file);
    const fd = new FormData();
    fd.append("file", file);
    runInference(fd, preview);
  };

  // ── Preset selection ──────────────────────────────────────────────────────

  const handlePreset = async (preset: typeof PRESETS[0]) => {
    if (state === "validating") return;
    setActivePreset(preset.code);
    setState("validating");
    setImagePreview(preset.file);
    setResult(null);
    setErrorMsg("");
    try {
      const r = await fetch(preset.file);
      const blob = await r.blob();
      const fd = new FormData();
      fd.append("file", blob, `${preset.code.toLowerCase()}_sample.jpg`);
      runInference(fd, preset.file);
    } catch {
      setErrorMsg("Could not load preset image file.");
      setState("api_error");
    }
  };

  // ── Reset ─────────────────────────────────────────────────────────────────

  const handleReset = useCallback(() => {
    setState("idle");
    setImagePreview(null);
    setActivePreset(null);
    setResult(null);
    setErrorMsg("");
    setViewMode("overlay");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  // ── Derived ───────────────────────────────────────────────────────────────

  const probs       = result?.probabilities;
  const hasGradCam  = Boolean(result?.explainability?.overlay_base64);
  const showOverlay = viewMode === "overlay" && hasGradCam && state === "success";

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-[1440px] mx-auto space-y-6">

      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="font-headline-md text-xl font-bold text-on-surface">
            Analysis Workstation
          </h1>
          <p className="font-body-sm text-sm text-on-surface-variant mt-0.5">
            Upload a dermoscopic image to classify with EfficientNet-B4 and Grad-CAM visual attribution.
          </p>
        </div>
        <StatusBadge variant={toStatusVariant(state)} />
      </div>

      {/* ── Main Grid ───────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-6 items-start">

        {/* ── LEFT: Image Viewport ──────────────────────────────────────── */}
        <div className="space-y-4">

          {/* Image Area */}
          <div
            className="relative bg-surface-container-low border border-outline-variant/20 rounded-xl overflow-hidden"
            style={{ aspectRatio: "1 / 1", maxHeight: 500 }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            role="region"
            aria-label="Image viewport"
          >
            {/* Scanner line during inference */}
            {state === "validating" && (
              <div className="scanner-marquee" aria-hidden="true" />
            )}

            {imagePreview && state !== "idle" ? (
              <>
                {/* Original image */}
                {(!showOverlay) && (
                  <Image
                    src={imagePreview}
                    alt="Uploaded dermoscopic image for analysis"
                    fill
                    className="object-contain"
                    sizes="(max-width:1024px) 100vw, 60vw"
                    priority
                  />
                )}
                {/* Grad-CAM overlay */}
                {showOverlay && result?.explainability?.overlay_base64 && (
                  <Image
                    src={result.explainability.overlay_base64}
                    alt="Grad-CAM heatmap overlay showing model attribution regions"
                    fill
                    className="object-contain"
                    sizes="(max-width:1024px) 100vw, 60vw"
                  />
                )}
              </>
            ) : (
              /* Upload zone */
              <button
                onClick={() => fileInputRef.current?.click()}
                className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-on-surface-variant hover:text-primary hover:bg-primary/5 transition-colors cursor-pointer w-full"
                aria-label="Click to upload an image or drag and drop here"
              >
                <UploadCloud className="w-12 h-12 opacity-40" aria-hidden="true" />
                <div className="text-center space-y-1">
                  <p className="font-technical-label text-sm font-semibold">READY FOR ANALYSIS</p>
                  <p className="font-body-sm text-xs text-on-surface-variant">
                    Upload a dermoscopic image or select a sample below
                  </p>
                  <p className="font-technical-data text-[11px] text-on-surface-variant/60">
                    JPEG · PNG · WebP · max 10 MB
                  </p>
                </div>
                <span className="px-4 py-2 border border-primary/40 rounded text-primary font-technical-label text-xs font-bold">
                  SELECT IMAGE
                </span>
              </button>
            )}

            {/* Processing overlay */}
            {state === "validating" && (
              <div
                className="absolute inset-0 bg-surface-container/60 flex flex-col items-center justify-center gap-3 backdrop-blur-sm"
                role="status"
                aria-live="polite"
              >
                <Loader2 className="w-8 h-8 text-primary animate-spin" aria-hidden="true" />
                <p className="font-technical-label text-sm text-primary">RUNNING INFERENCE…</p>
                <p className="font-technical-data text-[11px] text-on-surface-variant">
                  File validation → OOD gate → EfficientNet-B4 → Grad-CAM
                </p>
              </div>
            )}
          </div>

          {/* View Mode Tabs */}
          {state === "success" && hasGradCam && (
            <div className="flex gap-2" role="tablist" aria-label="Image view mode">
              {(["original", "overlay"] as const).map((mode) => (
                <button
                  key={mode}
                  role="tab"
                  aria-selected={viewMode === mode}
                  onClick={() => setViewMode(mode)}
                  className={`flex-1 py-2 font-technical-label text-xs rounded border transition-colors ${
                    viewMode === mode
                      ? "bg-primary text-on-primary border-primary"
                      : "border-outline-variant/30 text-on-surface-variant hover:text-primary hover:border-primary/30"
                  }`}
                >
                  {mode === "original" ? "ORIGINAL" : "GRAD-CAM OVERLAY"}
                </button>
              ))}
            </div>
          )}

          {/* Sample Presets */}
          <div className="space-y-2">
            <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-widest">
              HAM10000 Demo Samples
            </div>
            <div className="grid grid-cols-7 gap-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p.code}
                  onClick={() => handlePreset(p)}
                  disabled={state === "validating"}
                  aria-label={`Load ${p.label} sample`}
                  className={`flex flex-col items-center gap-1 p-1.5 rounded border transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                    activePreset === p.code
                      ? "border-primary bg-primary/10"
                      : "border-outline-variant/20 hover:border-primary/40 bg-surface-container"
                  }`}
                >
                  <div className="relative w-full" style={{ aspectRatio: "1 / 1" }}>
                    <Image
                      src={p.file}
                      alt={p.label}
                      fill
                      className="object-cover rounded"
                      sizes="60px"
                    />
                  </div>
                  <span
                    className={`font-technical-data text-[9px] font-bold ${
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

        {/* ── RIGHT: Analysis Panel ──────────────────────────────────────── */}
        <div className="space-y-4">

          {/* IDLE */}
          {state === "idle" && (
            <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-6 text-center space-y-3">
              <p className="font-technical-label text-sm text-on-surface-variant">
                No image selected.
              </p>
              <p className="font-body-sm text-xs text-on-surface-variant">
                Upload a dermoscopic image or select a HAM10000 sample to begin analysis.
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="mt-2 px-4 py-2 border border-primary/40 rounded text-primary font-technical-label text-xs font-bold hover:bg-primary/10 transition-colors"
              >
                SELECT IMAGE
              </button>
            </div>
          )}

          {/* VALIDATING */}
          {state === "validating" && (
            <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-4" role="status" aria-live="polite">
              <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
                Analysis Pipeline
              </div>
              {[
                { label: "File Validation",   note: "MIME · Size · Integrity check",           done: true  },
                { label: "Image Decoding",    note: "PIL decode · RGB conversion",              done: true  },
                { label: "OOD Domain Gate",   note: "MobileNetV2 · Prototype cosine similarity",done: false },
                { label: "EfficientNet-B4",   note: "Classification · Softmax → 7 classes",     done: false },
                { label: "Grad-CAM",          note: "Gradient attribution · Overlay generation", done: false },
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

          {/* REJECTED */}
          {state === "rejected" && result && (
            <div className="bg-status-critical/5 rounded-xl border border-status-critical/30 p-5 space-y-4" role="alert">
              <div className="flex items-start gap-3">
                <ShieldAlert className="w-5 h-5 text-status-critical shrink-0 mt-0.5" aria-hidden="true" />
                <div>
                  <h2 className="font-technical-label text-sm font-bold text-status-critical">
                    ANALYSIS REJECTED
                  </h2>
                  <p className="font-body-sm text-xs text-on-surface-variant mt-1">
                    Input does not appear to be a valid dermoscopic skin-lesion image.
                    Classification was <strong>not performed</strong>.
                  </p>
                </div>
              </div>

              <div className="bg-surface-container rounded border border-outline-variant/15 p-3 font-technical-data text-[11px] space-y-2">
                <div className="flex justify-between gap-2">
                  <span className="text-on-surface-variant">Detected Domain</span>
                  <span className="text-on-surface uppercase">{result.input_validation?.domain ?? "NON_DERMOSCOPIC"}</span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-on-surface-variant">Similarity Score</span>
                  <span className="text-on-surface">
                    {result.input_validation?.score?.toFixed(4) ?? "—"} / 1.0000
                  </span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-on-surface-variant">Gate Threshold</span>
                  <span className="text-on-surface">0.5500</span>
                </div>
                <div className="border-t border-outline-variant/15 pt-2 flex justify-between gap-2">
                  <span className="text-on-surface-variant">EfficientNet-B4</span>
                  <span className="text-status-critical font-bold">NOT RUN</span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-on-surface-variant">Grad-CAM</span>
                  <span className="text-status-critical font-bold">NOT GENERATED</span>
                </div>
              </div>

              <button
                onClick={() => setShowOodNote(!showOodNote)}
                className="flex items-center gap-1.5 font-technical-data text-[10px] text-on-surface-variant hover:text-primary transition-colors"
              >
                <Info className="w-3 h-3" aria-hidden="true" /> About OOD validation
              </button>
              {showOodNote && (
                <p className="font-technical-data text-[10px] text-on-surface-variant leading-relaxed bg-surface-container p-3 rounded border border-outline-variant/10">
                  {OOD_NOTE}
                </p>
              )}

              <button
                onClick={handleReset}
                className="w-full py-2.5 rounded border border-outline-variant/30 text-on-surface-variant font-technical-label text-xs hover:border-primary/40 hover:text-primary transition-colors"
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
                    The trained EfficientNet-B4 checkpoint is not loaded.
                    Inference will not run on random weights.
                  </p>
                </div>
              </div>
              <p className="font-technical-data text-[10px] text-on-surface-variant">
                Ensure{" "}
                <code className="bg-surface-container px-1 rounded">
                  checkpoints/efficientnet_b4/best_checkpoint.pth
                </code>{" "}
                is present and the FastAPI server has been restarted.
              </p>
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
                    {state === "network_error" ? "NETWORK ERROR" : "REQUEST ERROR"}
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

              {/* Validation status */}
              <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-2">
                <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
                  Input Validation
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <CheckCircle2 className="w-4 h-4 text-status-benign shrink-0" aria-hidden="true" />
                  <span className="font-technical-data text-xs text-status-benign">
                    DERMOSCOPIC DOMAIN CONFIRMED
                  </span>
                  <span className="ml-auto font-technical-data text-[11px] text-on-surface-variant">
                    sim = {result.input_validation?.score?.toFixed(4) ?? "—"}
                  </span>
                </div>
              </div>

              {/* Primary Prediction */}
              <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-3">
                <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
                  Primary Prediction
                </div>
                <div className="flex items-start justify-between gap-2">
                  <h2 className="font-headline-md text-xl font-bold text-on-surface leading-tight">
                    {result.predicted_name ?? result.full_name ?? result.predicted_class}
                  </h2>
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded border font-technical-label text-[10px] font-bold whitespace-nowrap ${categoryColor(result.predicted_category ?? result.category ?? "")}`}
                  >
                    {result.predicted_class}
                  </span>
                </div>
                <div className="flex items-center gap-3 flex-wrap">
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded border font-technical-label text-[11px] ${categoryColor(result.predicted_category ?? result.category ?? "")}`}
                  >
                    {(result.predicted_category ?? result.category ?? "").toUpperCase()}
                  </span>
                  <span className="font-technical-data text-xs text-on-surface-variant">
                    CONFIDENCE:{" "}
                    <strong className="text-primary text-sm">
                      {result.confidence_percentage}
                    </strong>
                  </span>
                  <span className="font-technical-data text-[11px] text-on-surface-variant ml-auto flex items-center gap-1">
                    <Zap className="w-3 h-3 text-primary" aria-hidden="true" />
                    {result.inference_time_ms?.toFixed(1)} ms
                  </span>
                </div>
              </div>

              {/* Top-3 Differential */}
              {result.top3_predictions && result.top3_predictions.length > 0 && (
                <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-3">
                  <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
                    Top-3 Differential
                  </div>
                  <div className="space-y-2.5">
                    {result.top3_predictions.map((p, i) => {
                      const pctVal = p.probability * 100;
                      return (
                        <div key={p.class_code} className="space-y-1">
                          <div className="flex items-center justify-between font-technical-data text-xs">
                            <span className="text-on-surface">
                              <span className="text-on-surface-variant mr-1">#{i + 1}</span>
                              <strong>{p.class_code}</strong>
                              {" — "}
                              {p.short_name ?? p.class_name}
                            </span>
                            <span
                              className={i === 0 ? "text-primary font-bold" : "text-on-surface-variant"}
                            >
                              {p.probability_percentage ?? `${pctVal.toFixed(2)}%`}
                            </span>
                          </div>
                          <div className="h-1.5 w-full bg-surface-container-lowest rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                i === 0 ? "bg-primary" : "bg-secondary/50"
                              }`}
                              style={{ width: `${Math.max(pctVal, 1)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Full 7-class probability distribution */}
              {probs && (
                <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-3">
                  <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
                    7-Class Probability Distribution
                  </div>
                  <div className="space-y-1.5">
                    {CLASS_ORDER.map((cls) => {
                      const pct      = (probs[cls] ?? 0) * 100;
                      const meta     = CLASS_METADATA[cls];
                      const isPred   = cls === result.predicted_class?.toLowerCase();
                      return (
                        <div key={cls} className="flex items-center gap-2 font-technical-data text-[11px]">
                          <span
                            className={`w-14 text-right font-bold ${
                              isPred ? "text-primary" : "text-on-surface-variant"
                            }`}
                          >
                            {meta?.code ?? cls.toUpperCase()}
                          </span>
                          <div className="flex-1 h-1.5 bg-surface-container-lowest rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                isPred ? "bg-primary" : "bg-outline/50"
                              }`}
                              style={{ width: `${Math.max(pct, 0.3)}%` }}
                            />
                          </div>
                          <span
                            className={`w-14 ${isPred ? "text-primary font-bold" : "text-on-surface-variant"}`}
                          >
                            {pct.toFixed(2)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Grad-CAM info panel */}
              {hasGradCam && (
                <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <Eye className="w-4 h-4 text-primary" aria-hidden="true" />
                    <span className="font-technical-label text-xs text-on-surface">
                      Grad-CAM Attribution Active
                    </span>
                  </div>
                  <div className="font-technical-data text-[10px] text-on-surface-variant space-y-0.5">
                    <div>Method: {result.explainability?.method}</div>
                    <div>Target class: {result.explainability?.target_class}</div>
                    <div>Layer: {result.explainability?.target_layer}</div>
                  </div>
                  <p className="font-body-sm text-[10px] text-on-surface-variant/70 italic">
                    Model attribution visualisation — highlights image regions associated with
                    the prediction. Not a clinical diagnosis.
                  </p>
                </div>
              )}

              {/* Technical Telemetry */}
              <div className="bg-surface-container-low rounded border border-outline-variant/10 p-3 font-technical-data text-[10px] text-on-surface-variant space-y-1">
                <div className="text-[9px] uppercase tracking-wider mb-1.5">Technical Telemetry</div>
                <div className="flex justify-between gap-2"><span>Model</span><span className="text-on-surface">EfficientNet-B4</span></div>
                <div className="flex justify-between gap-2"><span>Checkpoint</span><span className="text-status-benign">LOADED ✓</span></div>
                <div className="flex justify-between gap-2"><span>Preprocessing</span><span className="text-on-surface">Resize 256 → CenterCrop 224</span></div>
                <div className="flex justify-between gap-2">
                  <span>Live Latency (MPS)</span>
                  <span className="text-primary">{result.inference_time_ms?.toFixed(2)} ms</span>
                </div>
                <div className="flex justify-between gap-2">
                  <span>Benchmark Latency (LOQ CUDA)</span>
                  <span className="text-on-surface-variant">8.83 ms</span>
                </div>
                <div className="flex justify-between gap-2"><span>Device</span><span className="text-on-surface">{(result.device ?? result.hardware_device ?? "").toUpperCase()}</span></div>
                <div className="flex justify-between gap-2"><span>Classes</span><span className="text-on-surface">7 (HAM10000)</span></div>
              </div>

              {/* Grad-CAM note (for Benign classes to discourage over-reliance) */}
              <div className="bg-surface-container-lowest rounded border border-outline-variant/10 p-3 font-technical-data text-[10px] text-on-surface-variant/70 leading-relaxed">
                <strong className="text-on-surface-variant">ACADEMIC DISCLAIMER:</strong>{" "}
                Model outputs are for research/educational validation on HAM10000 only.
                Not for clinical use. Consult a licensed dermatologist for any skin concern.
              </div>

              {/* Reset */}
              <button
                onClick={handleReset}
                className="w-full py-2.5 rounded border border-outline-variant/20 text-on-surface-variant font-technical-label text-xs hover:border-primary/40 hover:text-primary transition-colors flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" aria-hidden="true" /> NEW ANALYSIS
              </button>
            </div>
          )}

          {/* Always-visible disclaimer when not in idle */}
          {state === "idle" && (
            <div className="p-3 rounded border border-outline-variant/10 bg-surface-container-lowest font-technical-data text-[10px] text-on-surface-variant/70 leading-relaxed">
              <strong className="text-on-surface-variant">ACADEMIC DISCLAIMER:</strong>{" "}
              Model outputs are for research/educational validation on HAM10000 only.
              Not for clinical use. Consult a licensed dermatologist for any skin concern.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
