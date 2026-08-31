import Link from "next/link";
import { Sparkles, LayoutDashboard, BookOpen, Layers, ShieldCheck, CheckCircle2, AlertTriangle, ArrowRight, Zap, Target } from "lucide-react";

export default function HomePage() {
  return (
    <div className="space-y-12 pb-8">
      
      {/* Hero Section */}
      <section className="text-center space-y-5 max-w-3xl mx-auto pt-6">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold">
          <Zap className="w-3.5 h-3.5" />
          <span>Verified Final College Project • EfficientNet-B4 + Grad-CAM</span>
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
          Non-Melanoma & Melanoma <br />
          <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent">
            Dermoscopic Classification
          </span>
        </h1>

        <p className="text-slate-400 text-base sm:text-lg leading-relaxed">
          An empirical deep learning investigation evaluating a novel Decoupled Dual-Branch CNN 
          against established compound-scaled architectures on the 7-class HAM10000 dermoscopic dataset.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
          <Link
            href="/classify"
            className="inline-flex items-center gap-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-medium px-6 py-3 rounded-xl shadow-lg shadow-cyan-500/20 transition-all hover:scale-105 text-sm"
          >
            <Sparkles className="w-4 h-4" />
            <span>Launch Image Classifier</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
          
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 bg-slate-900/80 hover:bg-slate-800 text-slate-300 font-medium px-6 py-3 rounded-xl border border-slate-700/80 transition-colors text-sm"
          >
            <LayoutDashboard className="w-4 h-4 text-blue-400" />
            <span>View Benchmark Results</span>
          </Link>
        </div>
      </section>

      {/* Established Results Banner */}
      <section className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl -z-10 pointer-events-none" />
        
        <div className="flex items-center justify-between flex-wrap gap-4 border-b border-slate-800 pb-6 mb-6">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400">Archived Benchmark Findings</span>
            <h2 className="text-2xl font-bold text-white mt-1">Deployment Model: EfficientNet-B4</h2>
          </div>
          <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-500/10 px-3 py-1.5 rounded-full border border-emerald-500/20 font-medium">
            <CheckCircle2 className="w-4 h-4" />
            <span>LOQ Experimental Ground Truth</span>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-6 text-center">
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800/80">
            <div className="text-3xl font-extrabold text-cyan-400">95.92%</div>
            <div className="text-xs text-slate-400 font-medium mt-1">ROC-AUC (Macro)</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800/80">
            <div className="text-3xl font-extrabold text-blue-400">79.16%</div>
            <div className="text-xs text-slate-400 font-medium mt-1">Balanced Accuracy</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800/80">
            <div className="text-3xl font-extrabold text-indigo-400">73.64%</div>
            <div className="text-xs text-slate-400 font-medium mt-1">Overall Test Accuracy</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800/80">
            <div className="text-3xl font-extrabold text-emerald-400">69.19%</div>
            <div className="text-xs text-slate-400 font-medium mt-1">Macro F1 Score</div>
          </div>
        </div>

        <div className="mt-6 p-4 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-300 leading-relaxed flex items-start gap-3">
          <Target className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
          <div>
            <strong>Scientific Integrity Conclusion:</strong> While the custom Decoupled Dual-Branch CNN was designed to separate textural features (Shallow-Wide branch) from structural morphology (Deep-Narrow branch) and attained <strong>90.98% ROC-AUC</strong>, empirical evaluation proved that compound scaling in <strong>EfficientNet-B4</strong> provided superior discrimination across highly imbalanced classes. Hence, EfficientNet-B4 was chosen as the verified deployment model.
          </div>
        </div>
      </section>

      {/* Feature Exploration Grid */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        
        <Link href="/classify" className="glass-card glass-card-hover rounded-2xl p-6 flex flex-col justify-between group">
          <div>
            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-4 group-hover:scale-110 transition-transform">
              <Sparkles className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2 group-hover:text-cyan-300 transition-colors">
              Interactive Classification
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Upload dermoscopic imagery to run live inference on EfficientNet-B4, inspect top-3 class probabilities, and generate Grad-CAM attribution heatmaps.
            </p>
          </div>
          <div className="mt-6 flex items-center gap-1 text-xs font-semibold text-cyan-400">
            <span>Try Classifier</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>

        <Link href="/dashboard" className="glass-card glass-card-hover rounded-2xl p-6 flex flex-col justify-between group">
          <div>
            <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mb-4 group-hover:scale-110 transition-transform">
              <LayoutDashboard className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2 group-hover:text-blue-300 transition-colors">
              Research Dashboard
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Compare complete benchmarks across EfficientNet-B4, DenseNet-121, ResNet-50, and 3 distinct random seeds of the Dual-Branch CNN.
            </p>
          </div>
          <div className="mt-6 flex items-center gap-1 text-xs font-semibold text-blue-400">
            <span>View Comparison</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>

        <Link href="/research" className="glass-card glass-card-hover rounded-2xl p-6 flex flex-col justify-between group">
          <div>
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 mb-4 group-hover:scale-110 transition-transform">
              <BookOpen className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2 group-hover:text-emerald-300 transition-colors">
              Dataset & Methodology
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Explore the 10,015-image HAM10000 dataset, patient-aware stratified splitting, extreme class imbalance, and 7-class diagnostic taxonomy.
            </p>
          </div>
          <div className="mt-6 flex items-center gap-1 text-xs font-semibold text-emerald-400">
            <span>Explore Dataset</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>

        <Link href="/architecture" className="glass-card glass-card-hover rounded-2xl p-6 flex flex-col justify-between group">
          <div>
            <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 mb-4 group-hover:scale-110 transition-transform">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2 group-hover:text-purple-300 transition-colors">
              Dual-Branch Design
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Deep dive into the Shallow-Wide (texture) and Deep-Narrow (structure) branch mechanisms, attention-gated fusion, and gate collapse diagnostics.
            </p>
          </div>
          <div className="mt-6 flex items-center gap-1 text-xs font-semibold text-purple-400">
            <span>Study Architecture</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>

      </section>

      {/* Medical Safety Disclaimer Callout */}
      <section className="rounded-2xl bg-amber-950/20 border border-amber-500/30 p-6 flex items-start gap-4">
        <AlertTriangle className="w-6 h-6 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="text-sm font-bold text-amber-200 uppercase tracking-wide">
            Mandatory Academic Research Disclaimer
          </h4>
          <p className="text-xs text-amber-300/80 leading-relaxed">
            This application is strictly an academic research artifact developed for educational purposes and college project demonstration. It does not provide medical diagnosis, clinical certainty, or replace professional dermatological inspection. All classifications reflect statistical model probabilities and attribution visualizations only.
          </p>
        </div>
      </section>

    </div>
  );
}
