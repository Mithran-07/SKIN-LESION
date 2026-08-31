"use client";

import { useState, useRef } from "react";
import { 
  Sparkles, UploadCloud, RefreshCw, AlertCircle, CheckCircle2, 
  Cpu, Layers, Eye, ShieldAlert, Zap, Activity, Info, ZoomIn, 
  Sliders, ArrowRight, CornerDownRight, Microscope 
} from "lucide-react";

interface Top3Prediction {
  rank?: number;
  class_code: string;
  class_name?: string;
  short_name?: string;
  full_name?: string;
  category: string;
  probability: number;
  probability_percentage?: string;
  percentage?: string;
  urgency?: string;
  description?: string;
}

interface PredictionResponse {
  predicted_class: string;
  predicted_name?: string;
  full_name?: string;
  predicted_category?: string;
  category?: string;
  urgency?: string;
  confidence?: number;
  confidence_score?: number;
  confidence_percentage: string;
  probabilities: Record<string, number>;
  top3_predictions: Top3Prediction[];
  inference_time_ms: number;
  device?: string;
  hardware_device?: string;
  model_name?: string;
  checkpoint_loaded?: boolean;
  checkpoint_status?: string;
  explainability?: {
    method?: string;
    target_layer?: string;
    target_class?: string;
    overlay_base64?: string;
    gradcam_overlay_base64?: string;
    attribution_note?: string;
    dimensions?: number[];
  };
  disclaimer?: string;
  academic_disclaimer?: string;
}

const PRESET_SAMPLES = [
  { code: "MEL", name: "Melanoma", file: "/samples/mel_sample.jpg", category: "Malignant" },
  { code: "BCC", name: "Basal Cell Carcinoma", file: "/samples/bcc_sample.jpg", category: "Malignant" },
  { code: "AKIEC", name: "Actinic Keratosis", file: "/samples/akiec_sample.jpg", category: "Pre-malignant" },
  { code: "NV", name: "Melanocytic Nevus", file: "/samples/nv_sample.jpg", category: "Benign" },
  { code: "BKL", name: "Benign Keratosis", file: "/samples/bkl_sample.jpg", category: "Benign" },
  { code: "DF", name: "Dermatofibroma", file: "/samples/df_sample.jpg", category: "Benign" },
  { code: "VASC", name: "Vascular Lesion", file: "/samples/vasc_sample.jpg", category: "Benign" },
];

export default function ClassifyPage() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"overlay" | "original" | "split">("overlay");
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(0.85);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (!file.type.startsWith("image/")) {
        setError("Invalid file format. Please upload a standard dermoscopic image (JPEG, PNG, WebP).");
        return;
      }
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setActivePreset(null);
      setResult(null);
      setError(null);
    }
  };

  const handlePresetSelect = async (preset: typeof PRESET_SAMPLES[0]) => {
    try {
      setActivePreset(preset.code);
      setError(null);
      setResult(null);
      setImagePreview(preset.file);

      const res = await fetch(preset.file);
      const blob = await res.blob();
      const file = new File([blob], `${preset.code.toLowerCase()}_sample.jpg`, { type: "image/jpeg" });
      setSelectedImage(file);
    } catch (err) {
      setError("Failed to load preset sample image.");
    }
  };

  const handleClassify = async () => {
    if (!selectedImage) {
      setError("Please select or upload a dermoscopic image first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedImage);

    const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

    try {
      const response = await fetch(`${API_BASE}/predict/explain`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Inference request failed" }));
        throw new Error(errData.detail || `Server returned HTTP ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to communicate with the FastAPI inference engine. Ensure the backend is active on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setActivePreset(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const getCategoryColor = (category?: string) => {
    if (!category) return "text-primary border-primary/30 bg-primary/10";
    const lower = category.toLowerCase();
    if (lower.includes("malignant") && !lower.includes("pre")) {
      return "text-status-critical border-status-critical/30 bg-status-critical/10";
    }
    if (lower.includes("pre")) {
      return "text-status-warning border-status-warning/30 bg-status-warning/10";
    }
    return "text-status-benign border-status-benign/30 bg-status-benign/10";
  };

  const gradcamSrc = result?.explainability?.overlay_base64 || result?.explainability?.gradcam_overlay_base64;
  const categoryLabel = result?.predicted_category || result?.category || "Lesion Evaluation";
  const modelTitle = result?.predicted_name || result?.full_name || result?.predicted_class || "Classification Result";

  return (
    <div className="py-6 px-4 sm:px-6 lg:px-8 space-y-6">
      
      {/* Workstation Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-outline-variant/15 pb-4">
        <div>
          <div className="flex items-center gap-2 font-technical-label text-xs text-primary uppercase tracking-widest">
            <span className="w-2 h-2 rounded-full bg-primary pulse-dot-cyan"></span>
            <span>CLINICAL DIAGNOSTIC WORKSTATION • V4.2</span>
          </div>
          <h1 className="font-headline-md text-2xl font-bold text-on-surface mt-1">
            Dermoscopic Analysis & Attribution
          </h1>
        </div>

        <div className="flex items-center gap-3 font-technical-data text-xs text-on-surface-variant">
          <span className="px-2.5 py-1 rounded bg-surface-container border border-outline-variant/20">
            MODEL: EfficientNet-B4
          </span>
          <span className="px-2.5 py-1 rounded bg-surface-container border border-outline-variant/20">
            HW: Apple Silicon MPS
          </span>
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-1 rounded bg-surface-container-high hover:bg-surface-variant text-on-surface border border-outline-variant/30 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-lg bg-status-critical/10 border border-status-critical/30 text-status-critical flex items-start gap-3 text-xs font-technical-label">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <strong>INFERENCE ENGINE NOTICE:</strong> {error}
          </div>
        </div>
      )}

      {/* Main Workstation Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[620px]">
        
        {/* Central Viewport Panel (Left: 7 cols) */}
        <div className="lg:col-span-7 bg-surface-container rounded-xl border border-outline-variant/20 flex flex-col overflow-hidden relative tech-border min-h-[480px]">
          
          {/* Viewport Top Bar */}
          <div className="h-10 border-b border-outline-variant/15 bg-surface-container-high px-4 flex items-center justify-between z-10">
            <div className="flex items-center gap-2 font-technical-data text-xs text-on-surface-variant">
              <span className={`w-2 h-2 rounded-full ${loading ? "bg-status-warning pulse-dot-cyan" : result ? "bg-status-benign" : "bg-outline"}`}></span>
              <span>{loading ? "SYS_PROCESSING_INFERENCE" : result ? "SYS_ATTRIBUTION_RENDERED" : "SYS_AWAITING_INPUT"}</span>
            </div>

            {/* View Toggles when Grad-CAM is ready */}
            {gradcamSrc && (
              <div className="flex items-center gap-1 font-technical-label text-[11px]">
                <button
                  onClick={() => setViewMode("overlay")}
                  className={`px-2.5 py-0.5 rounded ${viewMode === "overlay" ? "bg-primary/20 text-primary border border-primary/40" : "text-on-surface-variant hover:text-on-surface"}`}
                >
                  OVERLAY
                </button>
                <button
                  onClick={() => setViewMode("original")}
                  className={`px-2.5 py-0.5 rounded ${viewMode === "original" ? "bg-primary/20 text-primary border border-primary/40" : "text-on-surface-variant hover:text-on-surface"}`}
                >
                  ORIGINAL
                </button>
                <button
                  onClick={() => setViewMode("split")}
                  className={`px-2.5 py-0.5 rounded ${viewMode === "split" ? "bg-primary/20 text-primary border border-primary/40" : "text-on-surface-variant hover:text-on-surface"}`}
                >
                  SIDE-BY-SIDE
                </button>
              </div>
            )}
          </div>

          {/* Viewport Canvas */}
          <div className="flex-1 bg-surface-deep relative flex items-center justify-center p-6 lab-grid overflow-hidden min-h-[380px]">
            
            {/* Crosshair Guides */}
            <div className="absolute inset-0 pointer-events-none opacity-20">
              <div className="absolute top-1/2 left-0 w-full h-[1px] bg-primary"></div>
              <div className="absolute top-0 left-1/2 w-[1px] h-full bg-primary"></div>
              <div className="absolute top-6 left-6 w-6 h-6 border-t border-l border-primary"></div>
              <div className="absolute top-6 right-6 w-6 h-6 border-t border-r border-primary"></div>
              <div className="absolute bottom-6 left-6 w-6 h-6 border-b border-l border-primary"></div>
              <div className="absolute bottom-6 right-6 w-6 h-6 border-b border-r border-primary"></div>
            </div>

            {/* Empty State */}
            {!imagePreview && (
              <div className="z-10 text-center max-w-sm p-6 flex flex-col items-center">
                <div className="w-16 h-16 rounded-xl bg-surface-container border border-outline-variant/30 flex items-center justify-center text-outline mb-4">
                  <Microscope className="w-8 h-8 text-on-surface-variant" />
                </div>
                <h3 className="font-headline-md text-base font-semibold text-on-surface mb-1">
                  Awaiting Dermoscopic Capture
                </h3>
                <p className="font-body-sm text-xs text-on-surface-variant mb-4">
                  Upload an image or choose one of the verified 7-class demo presets below to run live inference.
                </p>
                <div className="font-technical-data text-[10px] text-primary/80 bg-primary/10 border border-primary/20 px-3 py-1 rounded">
                  SUPPORTED: JPEG, PNG, 224×224 RGB
                </div>
              </div>
            )}

            {/* Render Image / Grad-CAM */}
            {imagePreview && (
              <div className="z-10 w-full h-full flex items-center justify-center relative">
                
                {/* Single Overlay View */}
                {viewMode === "overlay" && (
                  <div className="relative max-w-[420px] w-full aspect-square rounded-lg border border-primary/30 overflow-hidden shadow-2xl bg-black">
                    {/* Base Image */}
                    <img
                      src={imagePreview}
                      alt="Dermoscopic Lesion"
                      className="w-full h-full object-cover"
                    />

                    {/* Grad-CAM Heatmap Overlay */}
                    {gradcamSrc && (
                      <img
                        src={gradcamSrc}
                        alt="Grad-CAM Spatial Attribution"
                        className="absolute inset-0 w-full h-full object-cover transition-opacity duration-300 pointer-events-none"
                        style={{ opacity: heatmapOpacity }}
                      />
                    )}

                    {/* Scanning Laser Line during Loading */}
                    {loading && <div className="scanner-marquee"></div>}
                  </div>
                )}

                {/* Original Only View */}
                {viewMode === "original" && (
                  <div className="relative max-w-[420px] w-full aspect-square rounded-lg border border-primary/30 overflow-hidden shadow-2xl bg-black">
                    <img
                      src={imagePreview}
                      alt="Dermoscopic Lesion"
                      className="w-full h-full object-cover"
                    />
                    {loading && <div className="scanner-marquee"></div>}
                  </div>
                )}

                {/* Side-by-Side View */}
                {viewMode === "split" && gradcamSrc && (
                  <div className="grid grid-cols-2 gap-4 max-w-2xl w-full">
                    <div className="rounded-lg border border-outline-variant/30 overflow-hidden bg-black">
                      <div className="p-1.5 bg-surface-container font-technical-data text-[10px] text-center text-on-surface-variant">
                        ORIGINAL INPUT (224×224)
                      </div>
                      <img src={imagePreview} alt="Original Lesion" className="w-full aspect-square object-cover" />
                    </div>
                    <div className="rounded-lg border border-primary/40 overflow-hidden bg-black">
                      <div className="p-1.5 bg-surface-container font-technical-data text-[10px] text-center text-primary">
                        GRAD-CAM ATTRIBUTION
                      </div>
                      <img src={gradcamSrc} alt="Grad-CAM" className="w-full aspect-square object-cover" />
                    </div>
                  </div>
                )}

              </div>
            )}

          </div>

          {/* Viewport Bottom Controls / Legend */}
          <div className="p-3 bg-surface-container-high border-t border-outline-variant/15 flex flex-wrap items-center justify-between gap-4 font-technical-label text-xs">
            <div className="flex items-center gap-3">
              <span className="text-on-surface-variant text-[11px]">CONTRIBUTION:</span>
              <div className="w-36 h-2 rounded bg-gradient-to-r from-surface-deep via-secondary-container to-primary relative">
                <div className="absolute -top-3.5 left-0 font-technical-data text-[9px] text-on-surface-variant">LOW</div>
                <div className="absolute -top-3.5 right-0 font-technical-data text-[9px] text-primary">HIGH</div>
              </div>
            </div>

            {gradcamSrc && (
              <div className="flex items-center gap-2">
                <span className="text-on-surface-variant text-[11px]">OPACITY:</span>
                <input
                  type="range"
                  min="0.2"
                  max="1.0"
                  step="0.05"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                  className="w-24 accent-primary cursor-pointer"
                />
                <span className="font-technical-data text-primary text-[11px] w-8">
                  {Math.round(heatmapOpacity * 100)}%
                </span>
              </div>
            )}
          </div>

        </div>

        {/* Control & Results Side Panel (Right: 5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          
          {/* Upload & Preset Source Card */}
          <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-4">
            <div className="font-technical-label text-xs text-on-surface border-b border-outline-variant/15 pb-2 flex items-center justify-between">
              <span>DATA SOURCE</span>
              <span className="font-technical-data text-[10px] text-on-surface-variant">HAM10000 MATRIX</span>
            </div>

            {/* Drag & Drop / Click Upload Button */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full py-4 px-4 border-2 border-dashed border-outline-variant/40 rounded-lg flex flex-col items-center justify-center text-on-surface-variant hover:border-primary/60 hover:bg-primary/5 transition-all group"
            >
              <UploadCloud className="w-6 h-6 text-on-surface-variant group-hover:text-primary mb-1 transition-colors" />
              <span className="font-technical-label text-xs text-on-surface group-hover:text-primary">
                SELECT CUSTOM DERMOSCOPIC IMAGE
              </span>
              <span className="font-technical-data text-[10px] text-on-surface-variant mt-0.5">
                PNG, JPG, DICOM (Max 15MB)
              </span>
            </button>

            {/* 7 Ground Truth Demo Presets */}
            <div>
              <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider mb-2">
                Verified Diagnostic Presets
              </div>
              <div className="grid grid-cols-4 gap-1.5">
                {PRESET_SAMPLES.map((preset) => (
                  <button
                    key={preset.code}
                    onClick={() => handlePresetSelect(preset)}
                    className={`py-1.5 px-2 rounded border font-technical-data text-xs text-center transition-all ${
                      activePreset === preset.code
                        ? "bg-primary/20 border-primary text-primary font-bold shadow-[0_0_8px_rgba(136,245,255,0.2)]"
                        : "bg-surface-container-low border-outline-variant/20 text-on-surface hover:border-primary/50"
                    }`}
                  >
                    {preset.code}
                  </button>
                ))}
              </div>
            </div>

            {/* Primary Action Button */}
            <button
              onClick={handleClassify}
              disabled={loading || !selectedImage}
              className={`w-full py-3 rounded font-technical-label text-xs tracking-wider uppercase font-bold flex items-center justify-center gap-2 transition-all ${
                loading
                  ? "bg-surface-container-highest text-on-surface-variant cursor-wait"
                  : selectedImage
                  ? "bg-primary text-on-primary hover:bg-primary-fixed-dim shadow-[0_0_15px_rgba(136,245,255,0.25)]"
                  : "bg-surface-container-high text-on-surface-variant/50 cursor-not-allowed border border-outline-variant/10"
              }`}
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-primary" />
                  <span>Processing MPS Inference...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Execute Neural Classification</span>
                </>
              )}
            </button>
          </div>

          {/* Real Inference Results Card */}
          {result && (
            <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-5 animate-in fade-in duration-300">
              
              {/* Primary Prediction Block */}
              <div className="border-b border-outline-variant/15 pb-4">
                <div className="font-technical-label text-[11px] text-on-surface-variant uppercase tracking-wider mb-1">
                  Primary Model Prediction
                </div>
                
                <div className="flex items-baseline justify-between gap-2">
                  <div className="font-headline-md text-2xl font-bold text-primary">
                    {modelTitle} ({result.predicted_class})
                  </div>
                </div>

                <div className="flex items-center gap-3 mt-2">
                  <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full border font-technical-label text-[11px] font-semibold ${getCategoryColor(categoryLabel)}`}>
                    <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
                    {categoryLabel.toUpperCase()}
                  </span>
                  
                  <span className="font-technical-data text-xs text-on-surface-variant">
                    CONFIDENCE: <strong className="text-on-surface font-bold text-sm">{result.confidence_percentage}</strong>
                  </span>

                  <span className="font-technical-data text-[11px] text-on-surface-variant ml-auto">
                    {result.inference_time_ms.toFixed(1)} ms
                  </span>
                </div>
              </div>

              {/* Top-3 Differential Ranking */}
              <div>
                <div className="font-technical-label text-[11px] text-on-surface-variant uppercase tracking-wider mb-2.5">
                  Top-3 Differential Ranking
                </div>
                
                <div className="space-y-2">
                  {result.top3_predictions.map((cand, idx) => {
                    const candPercentage = cand.probability_percentage || cand.percentage || (cand.probability * 100).toFixed(2) + "%";
                    const candName = cand.short_name || cand.class_name || cand.full_name || cand.class_code;
                    return (
                      <div key={cand.class_code} className="bg-surface-container-low p-2.5 rounded border border-outline-variant/15 space-y-1">
                        <div className="flex items-center justify-between font-technical-data text-xs">
                          <span className="text-on-surface font-medium">
                            #{idx + 1} {cand.class_code} — {candName}
                          </span>
                          <span className={idx === 0 ? "text-primary font-bold" : "text-on-surface-variant"}>
                            {candPercentage}
                          </span>
                        </div>
                        
                        {/* Bar */}
                        <div className="h-1.5 w-full bg-surface-variant rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${idx === 0 ? "bg-primary" : "bg-secondary/60"}`}
                            style={{ width: `${Math.max(cand.probability * 100, 2)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Full 7-Class Distribution */}
              {result.probabilities && (
                <div className="border-t border-outline-variant/15 pt-3">
                  <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-widest mb-2">
                    Complete 7-Class Probability Distribution
                  </div>
                  
                  <div className="grid grid-cols-7 gap-1 text-center font-technical-data text-[10px]">
                    {Object.entries(result.probabilities).map(([code, prob]) => (
                      <div key={code} className="bg-surface-container-lowest p-1.5 rounded border border-outline-variant/10">
                        <div className="text-on-surface-variant font-bold uppercase">{code}</div>
                        <div className="text-primary mt-0.5">{(prob * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Academic Disclaimer Note */}
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/10 font-technical-label text-[10px] text-on-surface-variant/80 leading-tight">
                <strong>ACADEMIC DISCLAIMER:</strong> Automated prediction output is generated for experimental validation on HAM10000 and must not be used as clinical diagnostic proof.
              </div>

            </div>
          )}

        </div>

      </div>

    </div>
  );
}
