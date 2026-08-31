export default function ArchitecturePage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-white mb-2">Project Architecture</h1>
        <p className="text-slate-400">System-level view of the complete research and application pipeline.</p>
      </div>

      {/* Pipeline */}
      <div className="glass rounded-2xl p-8 mb-8">
        <h2 className="text-lg font-semibold text-white mb-6">End-to-End Pipeline</h2>
        <div className="flex flex-col items-center gap-0">
          {[
            { label: "HAM10000 Dataset", sub: "10,015 dermoscopic images, 7 classes", color: "bg-emerald-500/20 border-emerald-500/40" },
            { label: "Patient-Level Split", sub: "70/15/15 — no patient leakage", color: "bg-sky-500/20 border-sky-500/40" },
            { label: "Preprocessing", sub: "224×224 resize · ImageNet normalization · Augmentation", color: "bg-sky-500/20 border-sky-500/40" },
            { label: "Baseline Models", sub: "ResNet50 · DenseNet121 · EfficientNet-B4", color: "bg-indigo-500/20 border-indigo-500/40" },
            { label: "Dual-Branch CNN", sub: "WideResNet-50-2 (texture) + DenseNet-121 (structure) + Attention Gate", color: "bg-purple-500/20 border-purple-500/40" },
            { label: "Training", sub: "AdamW · Cosine LR · AMP · Class Weights · Early Stopping", color: "bg-purple-500/20 border-purple-500/40" },
            { label: "Evaluation", sub: "Accuracy · Balanced Acc · Macro F1 · ROC-AUC · Fusion Diagnostics", color: "bg-amber-500/20 border-amber-500/40" },
            { label: "Model Selection → EfficientNet-B4", sub: "Best performance across all metrics", color: "bg-sky-500/30 border-sky-500/60" },
            { label: "Inference API (FastAPI)", sub: "POST /predict · POST /predict/explain · GET /health", color: "bg-slate-700/50 border-slate-600" },
            { label: "Web Application (Next.js)", sub: "Upload · Classify · Grad-CAM · Dashboard · Research Story", color: "bg-slate-700/50 border-slate-600" },
            { label: "Prediction + Explainability", sub: "Class · Probability · Top-3 · Grad-CAM Overlay", color: "bg-emerald-500/20 border-emerald-500/40" },
          ].map((step, i) => (
            <div key={i} className="flex flex-col items-center w-full max-w-lg">
              <div className={`w-full border rounded-xl px-5 py-3 text-center ${step.color}`}>
                <p className="font-medium text-white text-sm">{step.label}</p>
                <p className="text-slate-400 text-xs mt-0.5">{step.sub}</p>
              </div>
              {i < 10 && (
                <div className="w-0.5 h-6 bg-slate-700 my-0.5" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Dual-Branch Architecture Detail */}
      <div className="glass rounded-2xl p-8 mb-8">
        <h2 className="text-lg font-semibold text-white mb-6">Dual-Branch CNN — Internal Architecture</h2>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="font-medium text-sky-400 mb-3">Texture Branch (Shallow-Wide)</h3>
            <div className="space-y-2 text-sm text-slate-400">
              <div className="bg-slate-800/50 rounded-lg px-4 py-2">Input: 224×224×3</div>
              <div className="bg-slate-800/50 rounded-lg px-4 py-2">WideResNet-50-2 blocks</div>
              <div className="bg-slate-800/50 rounded-lg px-4 py-2">Global Average Pooling</div>
              <div className="bg-sky-500/20 border border-sky-500/30 rounded-lg px-4 py-2 font-medium text-sky-300">Output: 1024-d feature vector</div>
            </div>
          </div>
          <div>
            <h3 className="font-medium text-indigo-400 mb-3">Structure Branch (Deep-Narrow)</h3>
            <div className="space-y-2 text-sm text-slate-400">
              <div className="bg-slate-800/50 rounded-lg px-4 py-2">Input: 224×224×3</div>
              <div className="bg-slate-800/50 rounded-lg px-4 py-2">DenseNet-121 blocks</div>
              <div className="bg-slate-800/50 rounded-lg px-4 py-2">Global Average Pooling</div>
              <div className="bg-indigo-500/20 border border-indigo-500/30 rounded-lg px-4 py-2 font-medium text-indigo-300">Output: 256-d feature vector</div>
            </div>
          </div>
        </div>
        <div className="mt-6 border-t border-slate-800 pt-6">
          <h3 className="font-medium text-purple-400 mb-3">Fusion Gate (V2 — Independent Scalar Gates)</h3>
          <div className="text-sm text-slate-400 space-y-2">
            <div className="bg-slate-800/50 rounded-lg px-4 py-2">gate_t = sigmoid(MLP(texture_feat)) — scalar in [0,1]</div>
            <div className="bg-slate-800/50 rounded-lg px-4 py-2">gate_s = sigmoid(MLP(structure_feat)) — scalar in [0,1]</div>
            <div className="bg-slate-800/50 rounded-lg px-4 py-2">fused = concat(gate_t × texture, gate_s × structure)</div>
            <div className="bg-purple-500/20 border border-purple-500/30 rounded-lg px-4 py-2 font-medium text-purple-300">Output: 7-class logits via MLP + Classifier</div>
          </div>
          <div className="mt-4 bg-amber-500/10 border border-amber-500/20 rounded-lg px-4 py-3">
            <p className="text-amber-300 text-xs">
              ⚠️ <strong>Finding:</strong> Despite balanced initialization (0.51/0.48), gate_s converged to 0.91 and gate_t to 0.35
              after training — confirming that fusion collapse is optimization-driven rather than a dimensional-bias artefact.
            </p>
          </div>
        </div>
      </div>

      {/* Model Specs */}
      <div className="glass rounded-2xl p-8">
        <h2 className="text-lg font-semibold text-white mb-4">EfficientNet-B4 — Deployment Model</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {[
            { label: "Architecture", value: "EfficientNet-B4" },
            { label: "Dataset", value: "HAM10000" },
            { label: "Classes", value: "7" },
            { label: "Input Resolution", value: "224 × 224" },
            { label: "Framework", value: "PyTorch 2.7" },
            { label: "Parameters", value: "17.56M" },
            { label: "Test Accuracy", value: "73.64%" },
            { label: "ROC-AUC", value: "95.92%" },
            { label: "Purpose", value: "Academic Research" },
          ].map((s) => (
            <div key={s.label} className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-slate-500 text-xs mb-1">{s.label}</p>
              <p className="text-white font-medium">{s.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
