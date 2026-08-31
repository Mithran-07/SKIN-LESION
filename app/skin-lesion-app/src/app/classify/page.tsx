"use client";
import { useState, useCallback, useRef } from "react";
import { Upload, X, AlertTriangle, Loader2, Eye, ChevronDown } from "lucide-react";
import Image from "next/image";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CLASS_COLORS: Record<string, string> = {
  nv:    "bg-sky-500",
  mel:   "bg-rose-500",
  bkl:   "bg-emerald-500",
  bcc:   "bg-amber-500",
  akiec: "bg-purple-500",
  df:    "bg-indigo-500",
  vasc:  "bg-pink-500",
};

type PredictionResult = {
  predicted_label: string;
  predicted_display_name: string;
  predicted_description: string;
  probability: number;
  top3: Array<{ rank: number; label: string; display_name: string; probability: number }>;
  all_probabilities: Record<string, number>;
  gradcam?: { overlay_b64: string; heatmap_b64: string; original_b64: string };
  gradcam_note?: string;
  inference_time_ms: number;
  model: string;
  disclaimer: string;
};

export default function ClassifyPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [withExplain, setWithExplain] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    const allowed = ["image/jpeg", "image/png", "image/jpg"];
    if (!allowed.includes(f.type)) {
      setError("Invalid file type. Please upload a JPG or PNG image.");
      return;
    }
    if (f.size > 20 * 1024 * 1024) {
      setError("File too large. Maximum size is 20MB.");
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    const endpoint = withExplain ? `${API_BASE}/predict/explain` : `${API_BASE}/predict`;

    try {
      const resp = await fetch(endpoint, { method: "POST", body: formData });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      if (msg.includes("fetch") || msg.includes("NetworkError") || msg.includes("Failed to fetch")) {
        setError("Cannot connect to the API server. Make sure the backend is running on port 8000.");
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setShowAll(false);
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-white mb-2">Image Classification</h1>
        <p className="text-slate-400">
          Upload a dermoscopic image to receive an AI-generated classification from EfficientNet-B4.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Upload Panel */}
        <div>
          {!preview ? (
            <div
              className={`drop-zone rounded-2xl p-12 text-center cursor-pointer ${dragging ? "active" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
            >
              <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center mx-auto mb-4">
                <Upload className="text-sky-400" size={28} />
              </div>
              <p className="text-slate-300 font-medium mb-2">Drag & drop or click to upload</p>
              <p className="text-slate-500 text-sm">Supported: JPG, JPEG, PNG · Max 20MB</p>
              <input
                ref={inputRef}
                type="file"
                accept=".jpg,.jpeg,.png"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
            </div>
          ) : (
            <div className="glass rounded-2xl overflow-hidden">
              <div className="relative aspect-square">
                <img src={preview} alt="uploaded" className="w-full h-full object-cover" />
                <button
                  onClick={reset}
                  className="absolute top-3 right-3 w-8 h-8 bg-slate-900/80 rounded-full flex items-center justify-center hover:bg-red-500/80 transition-colors"
                >
                  <X size={14} />
                </button>
              </div>
              <div className="p-4 space-y-3">
                <label className="flex items-center gap-3 cursor-pointer">
                  <div
                    className={`w-10 h-6 rounded-full transition-colors relative ${withExplain ? "bg-sky-500" : "bg-slate-700"}`}
                    onClick={() => setWithExplain((v) => !v)}
                  >
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${withExplain ? "translate-x-5" : "translate-x-1"}`} />
                  </div>
                  <span className="text-sm text-slate-300 select-none">Include Grad-CAM Explainability</span>
                </label>
                <button
                  onClick={handleSubmit}
                  disabled={loading}
                  className="w-full bg-sky-500 hover:bg-sky-400 disabled:opacity-50 disabled:cursor-not-allowed text-white py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
                >
                  {loading ? <Loader2 size={18} className="animate-spin" /> : <Eye size={18} />}
                  {loading ? "Classifying..." : "Classify Image"}
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex gap-3">
              <AlertTriangle className="text-red-400 flex-shrink-0" size={18} />
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div>
          {!result && !loading && (
            <div className="glass rounded-2xl p-12 text-center h-full flex flex-col items-center justify-center">
              <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center mx-auto mb-4 opacity-50">
                <Eye size={28} className="text-slate-400" />
              </div>
              <p className="text-slate-500">Upload an image and click Classify to see predictions.</p>
            </div>
          )}

          {loading && (
            <div className="glass rounded-2xl p-12 flex flex-col items-center justify-center h-full">
              <Loader2 size={40} className="text-sky-400 animate-spin mb-4" />
              <p className="text-slate-400">Running inference...</p>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {/* Primary Prediction */}
              <div className="glass rounded-2xl p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-slate-400 text-sm mb-1">Predicted Class</p>
                    <h2 className="text-xl font-bold text-white">{result.predicted_display_name}</h2>
                    <p className="text-slate-400 text-sm mt-1 uppercase tracking-wider">{result.predicted_label}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-slate-400 text-sm mb-1">Probability</p>
                    <p className="text-3xl font-bold text-sky-400">
                      {(result.probability * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
                <p className="text-slate-400 text-sm border-t border-slate-800 pt-4 leading-relaxed">
                  {result.predicted_description}
                </p>
              </div>

              {/* Top 3 */}
              <div className="glass rounded-2xl p-6">
                <h3 className="font-semibold text-white mb-4">Top Predictions</h3>
                <div className="space-y-3">
                  {result.top3.map((item) => (
                    <div key={item.label}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-300">{item.display_name}</span>
                        <span className="text-slate-400 font-medium">{(item.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full prob-bar ${CLASS_COLORS[item.label] || "bg-sky-500"}`}
                          style={{ width: `${item.probability * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  className="text-sky-400 text-sm mt-4 flex items-center gap-1 hover:text-sky-300"
                  onClick={() => setShowAll((v) => !v)}
                >
                  <ChevronDown size={14} className={`transition-transform ${showAll ? "rotate-180" : ""}`} />
                  {showAll ? "Hide" : "Show all 7 classes"}
                </button>
                {showAll && (
                  <div className="mt-4 space-y-2 border-t border-slate-800 pt-4">
                    {Object.entries(result.all_probabilities).map(([cls, prob]) => (
                      <div key={cls} className="flex justify-between text-xs text-slate-400">
                        <span className="uppercase">{cls}</span>
                        <span>{(prob * 100).toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Grad-CAM */}
              {result.gradcam && (
                <div className="glass rounded-2xl p-6">
                  <h3 className="font-semibold text-white mb-2">Grad-CAM Explanation</h3>
                  <p className="text-slate-500 text-xs mb-4">{result.gradcam_note}</p>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-slate-500 text-xs mb-1">Original</p>
                      <img src={`data:image/png;base64,${result.gradcam.original_b64}`} className="rounded-lg w-full" alt="original" />
                    </div>
                    <div>
                      <p className="text-slate-500 text-xs mb-1">Grad-CAM Overlay</p>
                      <img src={`data:image/png;base64,${result.gradcam.overlay_b64}`} className="rounded-lg w-full" alt="gradcam" />
                    </div>
                  </div>
                </div>
              )}

              {/* Meta */}
              <div className="text-xs text-slate-600 flex justify-between px-1">
                <span>Model: {result.model}</span>
                <span>Inference: {result.inference_time_ms}ms</span>
              </div>

              {/* Disclaimer */}
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-3">
                <p className="text-amber-400 text-xs leading-relaxed">{result.disclaimer}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
