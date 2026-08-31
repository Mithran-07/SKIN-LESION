"use client";

import { useState, useRef } from "react";
import Image from "next/image";
import { 
  UploadCloud, 
  Sparkles, 
  Eye, 
  RefreshCw, 
  AlertCircle, 
  ShieldCheck, 
  Clock, 
  Cpu, 
  CheckCircle2, 
  ChevronRight,
  Info,
  Layers,
  Flame
} from "lucide-react";

interface TopPrediction {
  rank: number;
  class_code: string;
  class_name: string;
  short_name: string;
  category: string;
  probability: number;
  probability_percentage: string;
  urgency: string;
  description: string;
}

interface PredictionResult {
  predicted_class: string;
  predicted_name: string;
  predicted_category: string;
  confidence: number;
  confidence_percentage: string;
  probabilities: Record<string, number>;
  top3_predictions: TopPrediction[];
  inference_time_ms: number;
  device: string;
  model_name: string;
  disclaimer: string;
  explainability?: {
    method: string;
    target_layer: string;
    target_class: string;
    overlay_base64: string;
    attribution_note: string;
  };
}

const SAMPLE_PRESETS = [
  { code: "MEL", name: "Melanoma", file: "mel_sample.jpg", category: "Malignant" },
  { code: "BCC", name: "Basal Cell Carcinoma", file: "bcc_sample.jpg", category: "Malignant" },
  { code: "AKIEC", name: "Actinic Keratosis", file: "akiec_sample.jpg", category: "Pre-malignant" },
  { code: "NV", name: "Melanocytic Nevus", file: "nv_sample.jpg", category: "Benign" },
  { code: "BKL", name: "Benign Keratosis", file: "bkl_sample.jpg", category: "Benign" },
  { code: "DF", name: "Dermatofibroma", file: "df_sample.jpg", category: "Benign" },
  { code: "VASC", name: "Vascular Lesion", file: "vasc_sample.jpg", category: "Benign" },
];

export default function ClassifyPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [showGradCam, setShowGradCam] = useState(true);
  const [activeTab, setActiveTab] = useState<"top3" | "all">("top3");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePresetSelect = async (presetFile: string) => {
    try {
      setLoading(true);
      setError(null);
      setResult(null);
      
      const res = await fetch(`/samples/${presetFile}`);
      const blob = await res.blob();
      const file = new File([blob], presetFile, { type: "image/jpeg" });
      
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(blob));
      
      // Auto classify preset
      await classifyImage(file);
    } catch (err: any) {
      setError(`Failed to load preset sample: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const classifyImage = async (fileToClassify?: File) => {
    const file = fileToClassify || selectedFile;
    if (!file) {
      setError("Please select or upload a dermoscopic image first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Direct call to FastAPI backend with Grad-CAM explainability
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const response = await fetch(`${apiUrl}/predict/explain`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Network error occurred." }));
        throw new Error(errData.detail || `Server responded with status ${response.status}`);
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(`Classification failed: ${err.message}. Ensure the FastAPI backend is running on port 8000.`);
    } finally {
      setLoading(false);
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency?.toLowerCase()) {
      case "critical":
        return "bg-rose-500/10 text-rose-400 border-rose-500/30";
      case "high":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      default:
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
    }
  };

  const getCategoryColor = (category: string) => {
    if (category?.includes("Malignant")) return "text-rose-400 bg-rose-500/10 border-rose-500/20";
    if (category?.includes("Pre-malignant")) return "text-amber-400 bg-amber-500/10 border-amber-500/20";
    return "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Sparkles className="w-7 h-7 text-cyan-400" />
            <span>Dermoscopic Image Classifier</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Powered by pre-trained <strong>EfficientNet-B4</strong> with integrated Grad-CAM model attribution.
          </p>
        </div>

        {/* Demo Preset Selector */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mr-1">Demo Presets:</span>
          {SAMPLE_PRESETS.map((p) => (
            <button
              key={p.code}
              onClick={() => handlePresetSelect(p.file)}
              className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-slate-900 border border-slate-700/80 text-slate-300 hover:text-cyan-300 hover:border-cyan-500/40 hover:bg-slate-800 transition-colors"
            >
              {p.code}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Upload on Left, Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Image Upload & Preview */}
        <div className="lg:col-span-5 space-y-6">
          <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300">
                1. Select Dermoscopic Image
              </h2>
              {previewUrl && (
                <button
                  onClick={() => {
                    setSelectedFile(null);
                    setPreviewUrl(null);
                    setResult(null);
                  }}
                  className="text-xs text-slate-400 hover:text-rose-400 transition-colors flex items-center gap-1"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Reset</span>
                </button>
              )}
            </div>

            {/* Dropzone */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
            />

            {!previewUrl ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-700 hover:border-cyan-500/50 bg-slate-900/40 hover:bg-slate-900/80 rounded-xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center space-y-3"
              >
                <div className="w-12 h-12 rounded-xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center">
                  <UploadCloud className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-200">Click to upload or drag & drop</p>
                  <p className="text-xs text-slate-500 mt-1">JPEG, PNG, or WebP (HAM10000 format, max 15MB)</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative aspect-square w-full rounded-xl overflow-hidden border border-slate-700 bg-slate-950 flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="Dermoscopic Preview"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-2 right-2 bg-slate-950/80 backdrop-blur-md px-2 py-1 rounded text-[10px] font-mono text-slate-300 border border-slate-700">
                    224×224 RGB
                  </div>
                </div>

                <button
                  onClick={() => classifyImage()}
                  disabled={loading}
                  className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold text-sm shadow-lg shadow-cyan-500/20 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Computing Model Probabilities...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Run AI Classification</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Guidelines */}
          <div className="glass-card rounded-2xl p-5 border border-slate-800 text-xs text-slate-400 space-y-2">
            <div className="flex items-center gap-2 text-slate-300 font-semibold">
              <Info className="w-4 h-4 text-cyan-400" />
              <span>Recommended Image Criteria</span>
            </div>
            <ul className="list-disc list-inside space-y-1 text-slate-400 pl-1">
              <li>High-resolution polarized or non-polarized dermoscopy</li>
              <li>Lesion centered with minimal dark border vignetting</li>
              <li>Free of extensive ink markings or heavy air bubbles</li>
            </ul>
          </div>
        </div>

        {/* Right Column: Prediction Results & Grad-CAM */}
        <div className="lg:col-span-7 space-y-6">
          {!result ? (
            <div className="glass-card rounded-2xl p-12 border border-slate-800 text-center flex flex-col items-center justify-center min-h-[420px] text-slate-500">
              <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mb-4 text-slate-600">
                <Eye className="w-8 h-8" />
              </div>
              <h3 className="text-base font-bold text-slate-400">Awaiting Image Submission</h3>
              <p className="text-xs max-w-sm mt-1 text-slate-500 leading-relaxed">
                Upload a dermoscopic image on the left or select a demo preset above to inspect model probabilities and Grad-CAM attribution.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              
              {/* Primary Prediction Result Banner */}
              <div className="glass-card rounded-2xl p-6 border border-slate-800 relative overflow-hidden">
                <div className="flex items-start justify-between flex-wrap gap-4 border-b border-slate-800/80 pb-4 mb-4">
                  <div>
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Predicted Primary Category</span>
                    <h2 className="text-2xl font-extrabold text-white flex items-center gap-3 mt-1">
                      <span>{result.predicted_name}</span>
                      <span className="text-sm px-2.5 py-0.5 rounded-full font-mono font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                        {result.predicted_class}
                      </span>
                    </h2>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getCategoryColor(result.predicted_category)}`}>
                      {result.predicted_category}
                    </span>
                  </div>
                </div>

                {/* Metrics ribbon */}
                <div className="grid grid-cols-3 gap-3 text-center mb-6">
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                    <div className="text-xl font-extrabold text-cyan-400">{result.confidence_percentage}</div>
                    <div className="text-[10px] text-slate-400 uppercase font-medium mt-0.5">Top-1 Probability</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col items-center justify-center">
                    <div className="text-xs font-bold text-slate-200 flex items-center gap-1">
                      <Clock className="w-3 h-3 text-blue-400" />
                      <span>{result.inference_time_ms} ms</span>
                    </div>
                    <div className="text-[10px] text-slate-400 uppercase font-medium mt-0.5">Latency</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col items-center justify-center">
                    <div className="text-xs font-bold text-emerald-400 flex items-center gap-1 uppercase font-mono">
                      <Cpu className="w-3 h-3 text-emerald-400" />
                      <span>{result.device}</span>
                    </div>
                    <div className="text-[10px] text-slate-400 uppercase font-medium mt-0.5">Hardware</div>
                  </div>
                </div>

                {/* Top-3 vs All Probabilities Tabs */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs border-b border-slate-800 pb-2">
                    <span className="font-bold text-slate-300 uppercase tracking-wider">Diagnostic Ranking</span>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setActiveTab("top3")}
                        className={`px-2 py-0.5 rounded font-medium ${activeTab === "top3" ? "bg-cyan-500/20 text-cyan-300" : "text-slate-500 hover:text-slate-300"}`}
                      >
                        Top-3
                      </button>
                      <button
                        onClick={() => setActiveTab("all")}
                        className={`px-2 py-0.5 rounded font-medium ${activeTab === "all" ? "bg-cyan-500/20 text-cyan-300" : "text-slate-500 hover:text-slate-300"}`}
                      >
                        All 7 Classes
                      </button>
                    </div>
                  </div>

                  {activeTab === "top3" ? (
                    <div className="space-y-3">
                      {result.top3_predictions.map((p) => (
                        <div key={p.class_code} className="space-y-1.5 p-3 rounded-xl bg-slate-900/40 border border-slate-800/80">
                          <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-2">
                              <span className="w-4 h-4 rounded-full bg-slate-800 text-[10px] font-bold flex items-center justify-center text-slate-400">
                                {p.rank}
                              </span>
                              <span className="font-semibold text-slate-200">{p.class_name} ({p.class_code})</span>
                            </div>
                            <span className="font-mono font-bold text-cyan-400">{p.probability_percentage}</span>
                          </div>
                          
                          {/* Progress Bar */}
                          <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${p.rank === 1 ? "bg-cyan-400" : p.rank === 2 ? "bg-blue-500" : "bg-slate-500"}`}
                              style={{ width: `${Math.max(p.probability * 100, 2)}%` }}
                            />
                          </div>

                          <p className="text-[11px] text-slate-400 leading-snug pt-1">{p.description}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {Object.entries(result.probabilities)
                        .sort(([, a], [, b]) => b - a)
                        .map(([cls, prob], idx) => (
                          <div key={cls} className="flex items-center justify-between text-xs py-1.5 px-3 rounded-lg bg-slate-900/30 border border-slate-800/50">
                            <span className="font-semibold uppercase text-slate-300">{cls}</span>
                            <span className="font-mono text-cyan-400 font-medium">{(prob * 100).toFixed(2)}%</span>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Grad-CAM Model Attribution Section */}
              {result.explainability && (
                <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <div className="flex items-center gap-2">
                      <Flame className="w-5 h-5 text-amber-400" />
                      <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                        Model Attribution Visualization (Grad-CAM)
                      </h3>
                    </div>
                    <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                      Target: {result.explainability.target_layer}
                    </span>
                  </div>

                  <p className="text-xs text-slate-400 leading-relaxed">
                    {result.explainability.attribution_note} Warmer colors (red/yellow) indicate spatial regions that contributed most positively to the classification score for <strong>{result.predicted_class}</strong>.
                  </p>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2 text-center">
                      <div className="aspect-square rounded-xl overflow-hidden border border-slate-700 bg-slate-950">
                        {previewUrl && <img src={previewUrl} alt="Original Lesion" className="w-full h-full object-cover" />}
                      </div>
                      <span className="text-[11px] font-semibold text-slate-400">Original Lesion (224×224)</span>
                    </div>

                    <div className="space-y-2 text-center">
                      <div className="aspect-square rounded-xl overflow-hidden border border-amber-500/40 bg-slate-950 shadow-lg shadow-amber-500/5">
                        <img src={result.explainability.overlay_base64} alt="Grad-CAM Overlay" className="w-full h-full object-cover" />
                      </div>
                      <span className="text-[11px] font-semibold text-amber-300">Grad-CAM Spatial Attribution</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Medical Disclaimer Callout */}
              <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/30 text-[11px] text-amber-300/90 leading-relaxed flex items-start gap-3">
                <ShieldCheck className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <strong>Medical Disclaimer:</strong> {result.disclaimer}
                </div>
              </div>

            </div>
          )}
        </div>

      </div>
    </div>
  );
}
