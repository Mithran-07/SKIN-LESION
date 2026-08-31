import { 
  BookOpen, Layers, Target, AlertTriangle, CheckCircle2, 
  Split, Scale, Database, Activity, ArrowRight, ShieldCheck, Microscope 
} from "lucide-react";

export default function ResearchPage() {
  const datasetTaxonomy = [
    { code: "NV", name: "Melanocytic Nevi", count: 6705, pct: 67.0, category: "Benign", color: "bg-status-benign" },
    { code: "MEL", name: "Melanoma", count: 1113, pct: 11.1, category: "Malignant", color: "bg-status-critical" },
    { code: "BKL", name: "Benign Keratosis-like Lesions", count: 1099, pct: 11.0, category: "Benign", color: "bg-status-benign" },
    { code: "BCC", name: "Basal Cell Carcinoma", count: 514, pct: 5.1, category: "Malignant (NMSC)", color: "bg-status-critical" },
    { code: "AKIEC", name: "Actinic Keratoses & Carcinoma", count: 327, pct: 3.3, category: "Pre-malignant", color: "bg-status-warning" },
    { code: "VASC", name: "Vascular Lesions", count: 142, pct: 1.4, category: "Benign", color: "bg-status-benign" },
    { code: "DF", name: "Dermatofibroma", count: 115, pct: 1.1, category: "Benign", color: "bg-status-benign" },
  ];

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8 space-y-10">
      
      {/* Header Section */}
      <header className="border-b border-outline-variant/15 pb-6">
        <div className="flex items-center gap-2 font-technical-label text-xs text-primary uppercase tracking-widest mb-1">
          <Database className="w-4 h-4 text-primary" />
          <span>HAM10000 BENCHMARK & CLINICAL METHODOLOGY</span>
        </div>
        <h1 className="font-headline-lg text-2xl sm:text-4xl font-bold text-on-surface">
          Dataset Characteristics & Research Methodology
        </h1>
        <p className="font-body-lg text-sm sm:text-base text-on-surface-variant max-w-3xl mt-2 leading-relaxed">
          Comprehensive analysis of the Human Against Machine dermatoscopy dataset (HAM10000). This document outlines the severe intrinsic class imbalance, patient-aware splitting methodology, Focal Loss gradient modulation, and empirical failure analysis of the decoupled architecture.
        </p>
      </header>

      {/* 4-Card Bento Grid */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <div className="bg-surface-container rounded-xl border border-outline-variant/15 p-5 relative overflow-hidden">
          <div className="font-technical-label text-xs text-on-surface-variant uppercase tracking-wider flex items-center justify-between">
            <span>Total Captures</span>
            <Database className="w-4 h-4 text-primary/60" />
          </div>
          <div className="font-headline-md text-3xl font-bold text-primary mt-3">10,015</div>
          <div className="font-technical-data text-[11px] text-on-surface-variant mt-2 border-t border-outline-variant/10 pt-2">
            High-res dermoscopic dermatoscopy images
          </div>
        </div>

        <div className="bg-surface-container rounded-xl border border-outline-variant/15 p-5 relative overflow-hidden">
          <div className="font-technical-label text-xs text-on-surface-variant uppercase tracking-wider flex items-center justify-between">
            <span>Pathology Classes</span>
            <Target className="w-4 h-4 text-secondary/60" />
          </div>
          <div className="font-headline-md text-3xl font-bold text-secondary mt-3">7 Categories</div>
          <div className="font-technical-data text-[11px] text-on-surface-variant mt-2 border-t border-outline-variant/10 pt-2">
            Histopathology & consensus validated
          </div>
        </div>

        <div className="bg-surface-container rounded-xl border border-outline-variant/15 p-5 relative overflow-hidden">
          <div className="font-technical-label text-xs text-on-surface-variant uppercase tracking-wider flex items-center justify-between">
            <span>Max Imbalance</span>
            <Scale className="w-4 h-4 text-status-critical/60" />
          </div>
          <div className="font-headline-md text-3xl font-bold text-status-critical mt-3">58.3 : 1</div>
          <div className="font-technical-data text-[11px] text-on-surface-variant mt-2 border-t border-outline-variant/10 pt-2">
            NV (Majority: 6,705) vs DF (Minority: 115)
          </div>
        </div>

        <div className="bg-surface-container rounded-xl border border-outline-variant/15 p-5 relative overflow-hidden">
          <div className="font-technical-label text-xs text-on-surface-variant uppercase tracking-wider flex items-center justify-between">
            <span>Split Partitioning</span>
            <Split className="w-4 h-4 text-status-benign/60" />
          </div>
          <div className="font-headline-md text-3xl font-bold text-status-benign mt-3">70 / 15 / 15</div>
          <div className="font-technical-data text-[11px] text-on-surface-variant mt-2 border-t border-outline-variant/10 pt-2">
            Patient-level isolation via lesion_id
          </div>
        </div>

      </section>

      {/* Segmented Class Taxonomy Breakdown */}
      <section className="bg-surface-container rounded-xl border border-outline-variant/20 p-6 space-y-6 tech-border">
        <div className="flex items-center justify-between border-b border-outline-variant/15 pb-3">
          <h2 className="font-headline-md text-lg font-bold text-on-surface">
            Segmented Class Distribution & Imbalance Matrix
          </h2>
          <span className="font-technical-data text-xs text-primary">N = 10,015 TOTAL SAMPLES</span>
        </div>

        <div className="space-y-4">
          {datasetTaxonomy.map((item) => (
            <div key={item.code} className="space-y-1.5">
              <div className="flex items-center justify-between font-technical-data text-xs">
                <div className="flex items-center gap-2">
                  <span className="font-bold px-1.5 py-0.5 rounded bg-surface-container-high border border-outline-variant/20 text-primary">
                    {item.code}
                  </span>
                  <span className="text-on-surface font-medium">{item.name}</span>
                  <span className="text-on-surface-variant text-[11px]">({item.category})</span>
                </div>
                <div className="text-on-surface font-semibold">
                  {item.count.toLocaleString()} ({item.pct}%)
                </div>
              </div>

              <div className="h-2 w-full bg-surface-container-lowest rounded-full overflow-hidden border border-outline-variant/10">
                <div
                  className={`h-full rounded-full ${item.color}`}
                  style={{ width: `${item.pct}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Methodological Strategy & Fusion Collapse Analysis */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        <div className="bg-surface-container rounded-xl border border-outline-variant/15 p-6 space-y-4">
          <div className="font-technical-label text-xs text-primary uppercase tracking-wider flex items-center gap-2">
            <Activity className="w-4 h-4 text-primary" />
            <span>Loss & Data Balancing Strategy</span>
          </div>
          <h3 className="font-headline-md text-base font-semibold text-on-surface">
            Focal Loss & Patient-Aware Partitioning
          </h3>
          <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
            Standard Cross-Entropy loss suffers from gradient overwhelming caused by the dominant 67% NV class. We applied <strong>$\alpha$-balanced Focal Loss</strong> with focusing parameter $\gamma = 2.0$, dynamically down-weighting well-classified easy samples and concentrating gradient updates on difficult minority neoplasms (DF, VASC, AKIEC).
          </p>
          <div className="p-3 rounded bg-surface-container-low border border-outline-variant/15 font-technical-data text-xs text-on-surface">
            FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
          </div>
        </div>

        <div className="bg-surface-container rounded-xl border border-outline-variant/15 p-6 space-y-4">
          <div className="font-technical-label text-xs text-research-violet uppercase tracking-wider flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-research-violet" />
            <span>Failure Analysis & Fusion Collapse</span>
          </div>
          <h3 className="font-headline-md text-base font-semibold text-on-surface">
            The Dual-Branch Trade-Off
          </h3>
          <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
            The Decoupled Dual-Branch CNN physically isolated texture (1024-dim) from morphology (256-dim). However, multi-seed validation revealed <strong>attention gate fusion collapse</strong>: backpropagation gradients caused one branch to dominate the fused representation, preventing synergistic feature interaction. <strong>EfficientNet-B4</strong> compound scaling unified these feature scales into a single backbone, outperforming all decoupled variants.
          </p>
          <div className="p-3 rounded bg-surface-container-low border border-outline-variant/15 font-technical-data text-xs text-primary">
            Deployment Decision: EfficientNet-B4 (73.64% Top-1 Acc, 95.92% ROC-AUC)
          </div>
        </div>

      </section>

    </div>
  );
}
