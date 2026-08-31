import { Layers, Cpu, Sparkles, AlertTriangle, CheckCircle2, ArrowDown, Network } from "lucide-react";

export default function ArchitecturePage() {
  return (
    <div className="space-y-10 max-w-5xl mx-auto">
      
      {/* Header */}
      <div className="border-b border-slate-800 pb-6">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs font-semibold mb-2">
          <Layers className="w-3.5 h-3.5" />
          <span>Research Architecture Deep Dive</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Decoupled Dual-Branch CNN Architecture
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Theoretical motivation, architectural topology, and empirical diagnostic analysis.
        </p>
      </div>

      {/* Architecture Topology Breakdown */}
      <div className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-6">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Network className="w-5 h-5 text-purple-400" />
          <span>Decoupled Branch Design</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Shallow-Wide Branch */}
          <div className="p-5 rounded-xl bg-slate-900/60 border border-cyan-500/30 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wide">Branch 1</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300">1024-DIM TEXTURE</span>
            </div>
            <h3 className="text-base font-bold text-white">Shallow-Wide Texture Branch</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Designed with high channel width (3 → 256 → 512 → 1024) but minimal spatial downsampling. 
              Maintains high-resolution feature maps to capture fine micro-textures, arborizing telangiectasia (BCC), and surface keratinization (SCC).
            </p>
            <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] font-mono text-slate-300 space-y-1">
              <div>• Block 1: Conv(3→256) + BN + ReLU</div>
              <div>• Block 2: Conv(256→512) + MaxPool</div>
              <div>• Block 3: Conv(512→1024) + BN + ReLU</div>
              <div>• AdaptiveAvgPool(1,1) → 1024-dim</div>
            </div>
          </div>

          {/* Deep-Narrow Branch */}
          <div className="p-5 rounded-xl bg-slate-900/60 border border-blue-500/30 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-blue-400 uppercase tracking-wide">Branch 2</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-300">256-DIM STRUCTURE</span>
            </div>
            <h3 className="text-base font-bold text-white">Deep-Narrow Structural Branch</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Restricts channel width (base 64 channels) while expanding depth across 4 residual stages. 
              The large receptive field extracts macroscopic lesion morphology, asymmetry, border irregularity, and global pigment contrast.
            </p>
            <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] font-mono text-slate-300 space-y-1">
              <div>• Stem: Conv(3→64, k=7, s=2) + MaxPool</div>
              <div>• Stages 1-4: Residual Blocks (64→128)</div>
              <div>• Bottleneck: Conv(128→256, k=1)</div>
              <div>• AdaptiveAvgPool(1,1) → 256-dim</div>
            </div>
          </div>

        </div>

        {/* Attention Gated Fusion Box */}
        <div className="p-5 rounded-xl bg-slate-950/80 border border-purple-500/30 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-purple-400 uppercase tracking-wide">Fusion Module</span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-300">1280 → 256-DIM</span>
          </div>
          <h3 className="text-base font-bold text-white">Attention-Gated Feature Fusion</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Concatenates the texture (1024) and structure (256) vectors. A learned sigmoid attention gate dynamically modulates channel weights before two-stage MLP projection with dropout.
          </p>
        </div>
      </div>

      {/* Empirical Finding: Why EfficientNet Won */}
      <div className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-amber-400" />
          <span>Empirical Diagnostic Findings: Why Single-Branch Won</span>
        </h2>

        <div className="space-y-3 text-xs text-slate-400 leading-relaxed">
          <p>
            Our systematic benchmarking revealed that while the Dual-Branch CNN reached <strong>90.98% ROC-AUC</strong>, single-branch compound scaling (<strong>EfficientNet-B4 at 95.92% ROC-AUC</strong>) consistently outperformed it across all clinical metrics.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
              <h4 className="font-bold text-slate-200">1. Gate Saliency Imbalance</h4>
              <p className="text-slate-400 text-[11px]">
                Fusion gate weights heavily skewed toward the deep structural branch (&gt;78%), under-utilizing high-dimensional texture channels.
              </p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
              <h4 className="font-bold text-slate-200">2. Compound Scaling Balance</h4>
              <p className="text-slate-400 text-[11px]">
                EfficientNet-B4 uniformly scales depth, width, and resolution using neural architecture search, preventing branch optimization bottlenecks.
              </p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
              <h4 className="font-bold text-slate-200">3. Parameter Efficiency</h4>
              <p className="text-slate-400 text-[11px]">
                EfficientNet-B4 achieves higher accuracy with lower latency (8.83 ms) compared to the Dual-Branch pipeline (27.23 ms).
              </p>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
