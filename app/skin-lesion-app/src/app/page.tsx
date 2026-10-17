import Link from "next/link";
import {
  ArrowRight,
  Database,
  Layers,
  BarChart3,
  Eye,
  FlaskConical,
  Award,
  ShieldCheck,
  Activity,
  GitMerge,
  Cpu,
} from "lucide-react";
import { BENCHMARK_DATA, DATASET, OOD_EVALUATION_V2 } from "@/lib/constants";

const champion = BENCHMARK_DATA.find((m) => m.isChampion)!;

const PIPELINE_STEPS = [
  { Icon: Database, label: "HAM10000 Cohort", detail: "10,015 images · 7,470 lesions" },
  { Icon: Layers, label: "Lesion-Disjoint Split", detail: "0 cross-split leakage" },
  { Icon: ShieldCheck, label: "OOD Gate v2.0", detail: "Mahalanobis + Energy + Proto" },
  { Icon: FlaskConical, label: "Deterministic Crop", detail: "256 → 224 ImageNet Norm" },
  { Icon: Award, label: "EfficientNet-B4", detail: "79.23% Bal. Acc · 95.91% AUC" },
  { Icon: Eye, label: "Counterfactual CAM", detail: "7-Class Spatial Attribution" },
];

const KEY_STATS = [
  {
    value: DATASET.total.toLocaleString(),
    sub: `${DATASET.uniqueLesions.toLocaleString()} Unique Lesions`,
    label: "HAM10000 Dermoscopic Cohort",
  },
  {
    value: `${champion.balAcc.toFixed(2)}%`,
    sub: `${champion.auc.toFixed(2)}% Macro ROC-AUC`,
    label: "Best Balanced Accuracy (Test N=1,510)",
  },
  {
    value: OOD_EVALUATION_V2.auroc.toFixed(4),
    sub: `${OOD_EVALUATION_V2.idAcceptanceRate}% ID · ${OOD_EVALUATION_V2.oodRejectionRate}% OOD`,
    label: "Calibrated OOD Gate v2 AUROC",
  },
  {
    value: "27.64 ms",
    sub: "69.95 MB Active MPS Tensor RAM",
    label: "Apple M4 MPS Warm Inference",
  },
];

const WORKSTATION_MODULES = [
  {
    href: "/classify",
    title: "Analysis Workstation",
    badge: "Interactive MPS Engine",
    Icon: Eye,
    desc: "Split-screen optical comparator with real-time RGBA Grad-CAM opacity blending, 7-class counterfactual attribution, and 4-signal OOD gate telemetry.",
  },
  {
    href: "/dashboard",
    title: "Benchmark Lab",
    badge: "6 Checkpoints Verified",
    Icon: BarChart3,
    desc: "Interactive multi-model 7×7 confusion matrices, per-class recall/precision/F1 breakdown, SHA-256 checkpoint digests, and historical claim reconciliation.",
  },
  {
    href: "/research",
    title: "Dataset & OOD Integrity",
    badge: "Zero Lesion Leakage",
    Icon: Database,
    desc: "Patient/lesion-aware 70/15/15 split verification (7,002 / 1,503 / 1,510), histopathology ground-truth distribution, and 10-category OOD v2 evaluation.",
  },
  {
    href: "/architecture",
    title: "Architecture Comparator",
    badge: "v1_matrix vs v2_scalar",
    Icon: GitMerge,
    desc: "Structural comparison of compound-scaled EfficientNet-B4 (17.56M) against custom Dual-Branch CNN v1_matrix (10.67M) and v2_scalar (9.03M).",
  },
  {
    href: "/diagnostics",
    title: "System & MPS Diagnostics",
    badge: "Live Telemetry",
    Icon: Activity,
    desc: "Live Apple Silicon M4 unified memory profiling, Metal shader warmup status, checkpoint SHA-256 verification, and before/after latency benchmarks.",
  },
];

export default function LandingPage() {
  return (
    <div className="px-4 sm:px-6 lg:px-8 py-10 space-y-16 max-w-6xl mx-auto">
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="text-center space-y-5 pt-4">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-primary/30 bg-primary/5 font-technical-data text-[11px] text-primary">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          DERMAI V2 RESEARCH PLATFORM · HAM10000 (N=1,510 TEST) · APPLE M4 MPS
        </div>

        <h1 className="font-headline-lg text-4xl sm:text-5xl font-bold text-on-surface tracking-tight leading-tight">
          Calibrated Dermoscopic AI
          <br />
          <span className="text-primary">&amp; Empirical Architecture Lab</span>
        </h1>

        <p className="font-body-lg text-on-surface-variant text-base sm:text-lg max-w-2xl mx-auto leading-relaxed">
          An evidence-based computational dermatology workstation combining{" "}
          <strong className="text-on-surface">EfficientNet-B4</strong> inference,
          multi-signal out-of-distribution rejection, counterfactual Grad-CAM attribution,
          and rigorous reconciliation of custom{" "}
          <strong className="text-on-surface">Dual-Branch CNN</strong> experiments.
        </p>

        <div className="flex flex-col sm:flex-row gap-3 justify-center pt-2">
          <Link
            href="/classify"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-primary text-on-primary font-technical-label text-xs font-bold rounded hover:bg-primary/90 transition-colors"
          >
            LAUNCH ANALYSIS WORKSTATION <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            href="/dashboard"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 border border-outline-variant/40 text-on-surface font-technical-label text-xs rounded hover:border-primary/40 hover:text-primary transition-colors"
          >
            INSPECT VERIFIED BENCHMARKS
          </Link>
          <Link
            href="/diagnostics"
            className="inline-flex items-center justify-center gap-2 px-5 py-3 border border-outline-variant/30 text-on-surface-variant font-technical-label text-xs rounded hover:border-primary/40 hover:text-primary transition-colors"
          >
            <Cpu className="w-3.5 h-3.5" /> M4 MPS TELEMETRY
          </Link>
        </div>
      </section>

      {/* ── Key Verified Metrics ─────────────────────────────────────────── */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {KEY_STATS.map(({ value, sub, label }) => (
          <div
            key={label}
            className="p-5 rounded-xl bg-surface-container border border-outline-variant/20 space-y-1.5"
          >
            <div className="font-technical-data text-2xl font-bold text-primary">{value}</div>
            <div className="font-technical-data text-[11px] text-on-surface font-semibold">
              {sub}
            </div>
            <div className="font-technical-label text-[10px] text-on-surface-variant uppercase tracking-wider">
              {label}
            </div>
          </div>
        ))}
      </section>

      {/* ── End-to-End Verification Pipeline ─────────────────────────────── */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="font-technical-label text-xs text-on-surface-variant uppercase tracking-widest">
            Verified Execution Pipeline
          </span>
          <span className="font-technical-data text-[11px] text-on-surface-variant">
            SHA-256 Verified · Strict State-Dict Loading
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
          {PIPELINE_STEPS.map(({ Icon, label, detail }, i) => (
            <div
              key={label}
              className="flex flex-col justify-between p-3.5 rounded-xl bg-surface-container border border-outline-variant/15 space-y-2"
            >
              <div className="flex items-center justify-between">
                <Icon className="w-4 h-4 text-primary" />
                <span className="font-technical-data text-[10px] text-on-surface-variant">
                  0{i + 1}
                </span>
              </div>
              <div>
                <div className="font-technical-label text-[11px] font-bold text-on-surface leading-tight">
                  {label}
                </div>
                <div className="font-technical-data text-[10px] text-on-surface-variant mt-0.5">
                  {detail}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Workstation Modules Grid ─────────────────────────────────────── */}
      <section className="space-y-3">
        <div className="font-technical-label text-xs text-on-surface-variant uppercase tracking-widest">
          Research Platform Modules
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {WORKSTATION_MODULES.map(({ href, title, badge, Icon, desc }) => (
            <Link
              key={href}
              href={href}
              className="group p-5 rounded-xl bg-surface-container border border-outline-variant/20 hover:border-primary/40 transition-all flex flex-col justify-between space-y-4"
            >
              <div className="space-y-2.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="p-2 rounded-lg bg-primary/10 text-primary">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="font-technical-data text-[10px] px-2 py-0.5 rounded border border-outline-variant/25 text-on-surface-variant">
                    {badge}
                  </span>
                </div>
                <h2 className="font-headline-md text-base font-bold text-on-surface group-hover:text-primary transition-colors">
                  {title}
                </h2>
                <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                  {desc}
                </p>
              </div>
              <div className="inline-flex items-center gap-1.5 font-technical-label text-[11px] text-primary">
                OPEN MODULE <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* ── Scientific Audit & Integrity Statement ───────────────────────── */}
      <section className="p-6 rounded-xl bg-surface-container border border-outline-variant/20 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-outline-variant/15 pb-3">
          <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Scientific Audit &amp; Architecture Reconciliation
          </div>
          <span className="font-technical-data text-[10px] text-status-benign">
            All 6 Checkpoints Verified on Held-Out Test Split (N = 1,510)
          </span>
        </div>
        <div className="grid sm:grid-cols-3 gap-6 font-body-sm text-xs text-on-surface-variant leading-relaxed">
          <div className="space-y-1.5">
            <span className="text-primary font-semibold block font-technical-label text-[11px] uppercase">
              1. Champion Baseline — EfficientNet-B4
            </span>
            <p>
              Achieved <strong className="text-on-surface">79.23% Balanced Accuracy</strong>,{" "}
              <strong className="text-on-surface">95.91% Macro ROC-AUC</strong>, and{" "}
              <strong className="text-on-surface">69.37% Macro F1</strong> on the lesion-disjoint
              HAM10000 test split (`N = 1,510`). Deployed for live inference with `strict=True`
              checkpoint verification.
            </p>
          </div>
          <div className="space-y-1.5">
            <span className="text-research-violet font-semibold block font-technical-label text-[11px] uppercase">
              2. Dual-Branch v1_matrix vs v2_scalar
            </span>
            <p>
              Reconciled two distinct fusion topologies across historical checkpoints:{" "}
              <code className="text-on-surface">v1_matrix</code> (10.67M params, element-wise gate,
              90.98% ROC-AUC on seed 123) and{" "}
              <code className="text-on-surface">v2_scalar</code> (9.03M params, scalar gate,
              90.54% ROC-AUC on seed 42).
            </p>
          </div>
          <div className="space-y-1.5">
            <span className="text-status-benign font-semibold block font-technical-label text-[11px] uppercase">
              3. Calibrated OOD Gate v2.0
            </span>
            <p>
              Fitted strictly on training embeddings (`N = 2,123`), threshold-tuned on validation
              (`0.2500`), and evaluated once on held-out test (`N = 1,510` ID vs `N = 200` OOD):{" "}
              <strong className="text-on-surface">0.9985 AUROC</strong>, 98.08% ID acceptance, and
              100% OOD rejection.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
