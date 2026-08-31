import Link from "next/link";
import { Sparkles, LayoutDashboard, BookOpen, Layers, CheckCircle2, ArrowRight, Zap, Target, Cpu, Activity, ShieldCheck, Microscope } from "lucide-react";

export default function HomePage() {
  const workflowSteps = [
    {
      step: "01",
      title: "Dermoscopic Capture",
      desc: "224×224 RGB standardized cutaneous lesion imaging",
      tag: "INPUT"
    },
    {
      step: "02",
      title: "Decoupled Extraction",
      desc: "Texture branch (1024-dim) & Structure branch (256-dim)",
      tag: "HYPOTHESIS"
    },
    {
      step: "03",
      title: "Focal Loss Balancing",
      desc: "γ=2.0 modulation for severe 58.3:1 class imbalance",
      tag: "OPTIMIZATION"
    },
    {
      step: "04",
      title: "Grad-CAM Attribution",
      desc: "Conv-head gradient backpropagation for spatial heatmaps",
      tag: "EXPLAINABILITY"
    },
    {
      step: "05",
      title: "Deployment Classifier",
      desc: "EfficientNet-B4 compound scaling (95.92% ROC-AUC)",
      tag: "PRODUCTION"
    }
  ];

  return (
    <div className="space-y-12 py-8 px-4 sm:px-6 lg:px-8">
      
      {/* Hero Section */}
      <section className="relative pt-6 pb-4 max-w-4xl mx-auto text-center space-y-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded bg-surface-container-high border border-primary/30 font-technical-label text-[11px] text-primary tracking-widest uppercase">
          <span className="w-1.5 h-1.5 rounded-full bg-primary pulse-dot-cyan"></span>
          <span>Dermal Intelligence Lab • Research Workstation</span>
        </div>

        <h1 className="font-headline-lg text-3xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-on-surface leading-[1.15]">
          Non-Melanoma & Melanoma <br />
          <span className="bg-gradient-to-r from-primary via-secondary to-research-violet bg-clip-text text-transparent">
            Dermoscopic Classification
          </span>
        </h1>

        <p className="font-body-lg text-base sm:text-lg text-on-surface-variant max-w-2xl mx-auto leading-relaxed">
          An empirical deep learning investigation evaluating a novel Decoupled Dual-Branch CNN 
          against established compound-scaled architectures on the 7-class HAM10000 dermoscopic dataset.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
          <Link
            href="/classify"
            className="inline-flex items-center gap-2 bg-primary text-on-primary font-technical-label text-xs tracking-wider uppercase font-bold px-6 py-3 rounded hover:bg-primary-fixed-dim transition-all shadow-[0_0_15px_rgba(136,245,255,0.2)]"
          >
            <Sparkles className="w-4 h-4" />
            <span>Launch Analysis Workstation</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
          
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 bg-surface-container-low text-on-surface font-technical-label text-xs tracking-wider uppercase font-medium px-6 py-3 rounded border border-outline-variant/30 hover:border-primary/40 hover:bg-surface-container transition-colors"
          >
            <LayoutDashboard className="w-4 h-4 text-secondary" />
            <span>Empirical Benchmark Archive</span>
          </Link>
        </div>
      </section>

      {/* Verified Empirical Benchmark Bento Grid */}
      <section className="bg-surface-container rounded-xl border border-outline-variant/20 p-6 sm:p-8 relative overflow-hidden tech-border">
        <div className="flex items-center justify-between flex-wrap gap-4 border-b border-outline-variant/15 pb-4 mb-6">
          <div>
            <div className="font-technical-label text-xs text-primary uppercase tracking-widest">
              Deployment Model Evaluation
            </div>
            <h2 className="font-headline-md text-xl sm:text-2xl font-bold text-on-surface mt-1">
              Champion Architecture: EfficientNet-B4
            </h2>
          </div>
          <div className="flex items-center gap-2 font-technical-data text-xs text-status-benign bg-status-benign/10 px-3 py-1.5 rounded border border-status-benign/20">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>LOQ Verified Ground Truth</span>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
          <div className="p-4 rounded bg-surface-container-low border border-outline-variant/15">
            <div className="font-technical-data text-2xl sm:text-3xl font-bold text-primary">95.92%</div>
            <div className="font-technical-label text-[11px] text-on-surface-variant mt-1 uppercase">ROC-AUC (Macro)</div>
          </div>
          <div className="p-4 rounded bg-surface-container-low border border-outline-variant/15">
            <div className="font-technical-data text-2xl sm:text-3xl font-bold text-secondary">79.16%</div>
            <div className="font-technical-label text-[11px] text-on-surface-variant mt-1 uppercase">Balanced Accuracy</div>
          </div>
          <div className="p-4 rounded bg-surface-container-low border border-outline-variant/15">
            <div className="font-technical-data text-2xl sm:text-3xl font-bold text-research-violet">73.64%</div>
            <div className="font-technical-label text-[11px] text-on-surface-variant mt-1 uppercase">Top-1 Accuracy</div>
          </div>
          <div className="p-4 rounded bg-surface-container-low border border-outline-variant/15">
            <div className="font-technical-data text-2xl sm:text-3xl font-bold text-status-benign">8.83 ms</div>
            <div className="font-technical-label text-[11px] text-on-surface-variant mt-1 uppercase">Inference Latency</div>
          </div>
        </div>

        <div className="mt-6 p-4 rounded bg-surface-container-lowest border border-outline-variant/15 font-body-sm text-xs text-on-surface-variant leading-relaxed flex items-start gap-3">
          <Target className="w-4 h-4 text-primary shrink-0 mt-0.5" />
          <div>
            <strong className="text-on-surface">Scientific Integrity Finding:</strong> While the novel Decoupled Dual-Branch CNN successfully isolated high-frequency texture (1024ch) from structural morphology (256ch) and reached <strong className="text-primary">90.98% ROC-AUC</strong>, empirical multi-seed evaluation confirmed that compound scaled depth/width/resolution in <strong className="text-primary">EfficientNet-B4</strong> offered superior class-balanced generalization on HAM10000.
          </div>
        </div>
      </section>

      {/* 5-Step Research Pipeline Diagram */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-outline-variant/15 pb-2">
          <h3 className="font-technical-label text-xs uppercase tracking-widest text-on-surface-variant">
            Scientific Investigation Pipeline
          </h3>
          <span className="font-technical-data text-[11px] text-primary">HAM10000 • 7-CLASS TAXONOMY</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {workflowSteps.map((item, idx) => (
            <div
              key={idx}
              className="bg-surface-container-low border border-outline-variant/15 rounded p-4 relative flex flex-col justify-between hover:border-primary/40 transition-colors"
            >
              <div>
                <div className="flex items-center justify-between font-technical-data text-[10px] text-primary mb-2">
                  <span>STAGE {item.step}</span>
                  <span className="bg-surface-variant px-1.5 py-0.5 rounded text-on-surface-variant">{item.tag}</span>
                </div>
                <div className="font-headline-md text-sm font-semibold text-on-surface mb-1">
                  {item.title}
                </div>
                <div className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                  {item.desc}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Deep Dive Navigation Grid */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <Link href="/classify" className="bg-surface-container-low border border-outline-variant/15 rounded-xl p-5 flex flex-col justify-between group hover:border-primary/40 hover:bg-surface-container transition-all">
          <div>
            <div className="w-10 h-10 rounded bg-primary/10 border border-primary/20 flex items-center justify-center text-primary mb-4 group-hover:scale-105 transition-transform">
              <Sparkles className="w-5 h-5" />
            </div>
            <h4 className="font-headline-md text-base font-semibold text-on-surface mb-1 group-hover:text-primary transition-colors">
              Analysis Workstation
            </h4>
            <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
              Upload dermoscopic lesions, execute live MPS inference, and view top-3 differential candidate sets.
            </p>
          </div>
          <div className="font-technical-label text-[11px] text-primary flex items-center gap-1 mt-4">
            <span>Launch UI</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>

        <Link href="/dashboard" className="bg-surface-container-low border border-outline-variant/15 rounded-xl p-5 flex flex-col justify-between group hover:border-secondary/40 hover:bg-surface-container transition-all">
          <div>
            <div className="w-10 h-10 rounded bg-secondary/10 border border-secondary/20 flex items-center justify-center text-secondary mb-4 group-hover:scale-105 transition-transform">
              <LayoutDashboard className="w-5 h-5" />
            </div>
            <h4 className="font-headline-md text-base font-semibold text-on-surface mb-1 group-hover:text-secondary transition-colors">
              Benchmark Archive
            </h4>
            <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
              Examine comparative empirical matrices across Dual-Branch variants, ResNet-50, and DenseNet-121.
            </p>
          </div>
          <div className="font-technical-label text-[11px] text-secondary flex items-center gap-1 mt-4">
            <span>View Tables</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>

        <Link href="/research" className="bg-surface-container-low border border-outline-variant/15 rounded-xl p-5 flex flex-col justify-between group hover:border-status-benign/40 hover:bg-surface-container transition-all">
          <div>
            <div className="w-10 h-10 rounded bg-status-benign/10 border border-status-benign/20 flex items-center justify-center text-status-benign mb-4 group-hover:scale-105 transition-transform">
              <BookOpen className="w-5 h-5" />
            </div>
            <h4 className="font-headline-md text-base font-semibold text-on-surface mb-1 group-hover:text-status-benign transition-colors">
              Dataset & Methodology
            </h4>
            <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
              Review HAM10000 pathology distributions, patient-aware splitting by lesion_id, and class weighting.
            </p>
          </div>
          <div className="font-technical-label text-[11px] text-status-benign flex items-center gap-1 mt-4">
            <span>Read Story</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>

        <Link href="/architecture" className="bg-surface-container-low border border-outline-variant/15 rounded-xl p-5 flex flex-col justify-between group hover:border-research-violet/40 hover:bg-surface-container transition-all">
          <div>
            <div className="w-10 h-10 rounded bg-research-violet/10 border border-research-violet/20 flex items-center justify-center text-research-violet mb-4 group-hover:scale-105 transition-transform">
              <Layers className="w-5 h-5" />
            </div>
            <h4 className="font-headline-md text-base font-semibold text-on-surface mb-1 group-hover:text-research-violet transition-colors">
              Architecture Deep Dive
            </h4>
            <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
              Explore the Decoupled Shallow-Wide vs Deep-Narrow topology, attention gate, and feature dimensions.
            </p>
          </div>
          <div className="font-technical-label text-[11px] text-research-violet flex items-center gap-1 mt-4">
            <span>Inspect Blueprint</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </Link>

      </section>

    </div>
  );
}
