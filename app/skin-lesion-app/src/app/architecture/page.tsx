import { 
  Layers, Cpu, Network, GitMerge, ArrowDown, 
  CheckCircle2, AlertCircle, Sparkles, Box, ShieldCheck, Microscope 
} from "lucide-react";

export default function ArchitecturePage() {
  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8 space-y-10">
      
      {/* Header */}
      <header className="border-b border-outline-variant/15 pb-6">
        <div className="flex items-center gap-2 font-technical-label text-xs text-research-violet uppercase tracking-widest mb-1">
          <Network className="w-4 h-4 text-research-violet" />
          <span>NEURAL NETWORK BLUEPRINT & TOPOLOGY DIAGNOSTICS</span>
        </div>
        <h1 className="font-headline-lg text-2xl sm:text-4xl font-bold text-on-surface">
          Decoupled Dual-Branch CNN Topology
        </h1>
        <p className="font-body-lg text-sm sm:text-base text-on-surface-variant max-w-3xl mt-2 leading-relaxed">
          Detailed architectural breakdown of the experimental Decoupled Dual-Branch network, outlining the feature decoupling hypothesis, mathematical attention-gated fusion, and comparison with compound scaling in EfficientNet-B4.
        </p>
      </header>

      {/* Production vs Research Role Distinction Banner */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 grid grid-cols-1 md:grid-cols-2 gap-6 tech-border">
        <div className="space-y-2 border-b md:border-b-0 md:border-r border-outline-variant/15 pb-4 md:pb-0 md:pr-6">
          <div className="flex items-center gap-2 font-technical-label text-xs text-primary uppercase">
            <span className="w-2 h-2 rounded-full bg-primary pulse-dot-cyan"></span>
            <span>DEPLOYED CHAMPION ARCHITECTURE</span>
          </div>
          <div className="font-headline-md text-xl font-bold text-on-surface">
            EfficientNet-B4 (Single-Branch Compound Scaling)
          </div>
          <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
            Unifies resolution (380×380), depth (d=1.8), and width (w=1.4) scaling. Achieved <strong>95.92% ROC-AUC</strong>, <strong>79.16% Balanced Accuracy</strong>, and <strong>8.83 ms latency</strong> on Apple MPS. Selected as the verified production inference engine.
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2 font-technical-label text-xs text-research-violet uppercase">
            <span className="w-2 h-2 rounded-full bg-research-violet"></span>
            <span>RESEARCH EXPERIMENTAL HYPOTHESIS</span>
          </div>
          <div className="font-headline-md text-xl font-bold text-on-surface">
            Decoupled Dual-Branch CNN (Texture + Morphology)
          </div>
          <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
            Designed to prevent spatial downsampling destruction of microvascular features. Attained <strong>90.98% ROC-AUC</strong>, establishing an important empirical negative finding regarding attention gate gradient saturation.
          </p>
        </div>
      </div>

      {/* Topology Blueprint Visualizer */}
      <section className="bg-surface-container rounded-xl border border-outline-variant/20 p-6 sm:p-8 space-y-8 lab-grid">
        
        <div className="flex items-center justify-between border-b border-outline-variant/15 pb-3">
          <h2 className="font-headline-md text-lg font-bold text-on-surface flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary" />
            <span>Dual-Branch Parallel Pipeline & Feature Dimensions</span>
          </h2>
          <span className="font-technical-data text-xs text-on-surface-variant">INPUT: (B, 3, 224, 224)</span>
        </div>

        {/* Parallel Branches Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Branch 1: Shallow-Wide */}
          <div className="bg-surface-container-low rounded-xl border border-primary/30 p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-outline-variant/15 pb-2">
              <span className="font-technical-label text-xs font-bold text-primary uppercase">
                BRANCH 1: SHALLOW-WIDE (TEXTURE)
              </span>
              <span className="font-technical-data text-[10px] text-primary bg-primary/10 px-2 py-0.5 rounded border border-primary/20">
                1024-DIM VECTOR
              </span>
            </div>

            <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
              Acts as a dense unpooled filter bank. Preserves high-frequency spatial gradients, micro-keratin textures, and arborizing telangiectasia in BCC.
            </p>

            <div className="space-y-2 font-technical-data text-xs">
              <div className="p-2.5 rounded bg-surface-container border border-outline-variant/15 flex justify-between">
                <span className="text-on-surface">Block 1: Conv(3→256, k=3, s=1)</span>
                <span className="text-primary">(256, 224, 224)</span>
              </div>
              <div className="p-2.5 rounded bg-surface-container border border-outline-variant/15 flex justify-between">
                <span className="text-on-surface">Block 2: Conv(256→512, k=3) + MaxPool</span>
                <span className="text-primary">(512, 112, 112)</span>
              </div>
              <div className="p-2.5 rounded bg-surface-container border border-outline-variant/15 flex justify-between">
                <span className="text-on-surface">Block 3: Conv(512→1024, k=3, unpooled)</span>
                <span className="text-primary">(1024, 112, 112)</span>
              </div>
              <div className="p-2.5 rounded bg-surface-container-high border border-primary/30 flex justify-between font-bold">
                <span className="text-primary">AdaptiveAvgPool2d(1, 1) + Flatten</span>
                <span className="text-primary">(B, 1024)</span>
              </div>
            </div>
          </div>

          {/* Branch 2: Deep-Narrow */}
          <div className="bg-surface-container-low rounded-xl border border-secondary/30 p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-outline-variant/15 pb-2">
              <span className="font-technical-label text-xs font-bold text-secondary uppercase">
                BRANCH 2: DEEP-NARROW (STRUCTURE)
              </span>
              <span className="font-technical-data text-[10px] text-secondary bg-secondary/10 px-2 py-0.5 rounded border border-secondary/20">
                256-DIM VECTOR
              </span>
            </div>

            <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
              Acts as a low-pass morphological filter with large receptive fields. Distills global pigment geometry, border asymmetry, and macroscopic lesion contrast.
            </p>

            <div className="space-y-2 font-technical-data text-xs">
              <div className="p-2.5 rounded bg-surface-container border border-outline-variant/15 flex justify-between">
                <span className="text-on-surface">Stage 1: Conv(3→64) + MaxPool</span>
                <span className="text-secondary">(64, 112, 112)</span>
              </div>
              <div className="p-2.5 rounded bg-surface-container border border-outline-variant/15 flex justify-between">
                <span className="text-on-surface">Stage 2: 2× Residual Blocks (64→128, s=2)</span>
                <span className="text-secondary">(128, 56, 56)</span>
              </div>
              <div className="p-2.5 rounded bg-surface-container border border-outline-variant/15 flex justify-between">
                <span className="text-on-surface">Stage 3-4: 6× Residual Blocks (128→256)</span>
                <span className="text-secondary">(256, 7, 7)</span>
              </div>
              <div className="p-2.5 rounded bg-surface-container-high border border-secondary/30 flex justify-between font-bold">
                <span className="text-secondary">AdaptiveAvgPool2d(1, 1) + Flatten</span>
                <span className="text-secondary">(B, 256)</span>
              </div>
            </div>
          </div>

        </div>

        {/* Fusion Mechanism */}
        <div className="bg-surface-container-low rounded-xl border border-research-violet/30 p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-outline-variant/15 pb-2">
            <span className="font-technical-label text-xs font-bold text-research-violet uppercase flex items-center gap-2">
              <GitMerge className="w-4 h-4 text-research-violet" />
              <span>Attention-Gated Feature Fusion Mechanism</span>
            </span>
            <span className="font-technical-data text-[10px] text-research-violet bg-research-violet/10 px-2 py-0.5 rounded border border-research-violet/20">
              OUTPUT: (B, 7)
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-technical-data text-xs">
            <div className="p-3 rounded bg-surface-container border border-outline-variant/15">
              <div className="text-on-surface-variant text-[10px] uppercase mb-1">Step 1: Concatenation</div>
              <div className="text-on-surface font-semibold">[f_tex, f_struct]</div>
              <div className="text-on-surface-variant text-[11px] mt-1">1024 + 256 = 1280-dim</div>
            </div>

            <div className="p-3 rounded bg-surface-container border border-outline-variant/15">
              <div className="text-on-surface-variant text-[10px] uppercase mb-1">Step 2: Gate Vector</div>
              <div className="text-research-violet font-semibold">z = σ(W_a · [f_tex, f_struct])</div>
              <div className="text-on-surface-variant text-[11px] mt-1">Learned scalar gating in (0, 1)</div>
            </div>

            <div className="p-3 rounded bg-surface-container border border-outline-variant/15">
              <div className="text-on-surface-variant text-[10px] uppercase mb-1">Step 3: Gated Projection</div>
              <div className="text-primary font-semibold">f_fused = Linear(f_gated, 256)</div>
              <div className="text-on-surface-variant text-[11px] mt-1">Linear(256→7) + Softmax</div>
            </div>
          </div>
        </div>

      </section>

    </div>
  );
}
