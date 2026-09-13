import Link from "next/link";
import { ArrowRight, Database, Layers, BarChart3, Eye, FlaskConical, Award } from "lucide-react";
import { BENCHMARK_DATA, DATASET } from "@/lib/constants";

const champion = BENCHMARK_DATA.find((m) => m.isChampion)!;

const PIPELINE_STEPS = [
  { Icon: Database,     label: "HAM10000 Dataset" },
  { Icon: Layers,       label: "Patient-Level Split" },
  { Icon: FlaskConical, label: "Preprocessing" },
  { Icon: BarChart3,    label: "CNN Benchmarking" },
  { Icon: Award,        label: "Classification" },
  { Icon: Eye,          label: "Grad-CAM" },
];

const KEY_STATS = [
  { value: DATASET.total.toLocaleString(), label: "HAM10000 Images" },
  { value: "7",                            label: "Diagnostic Classes" },
  { value: `${champion.auc}%`,            label: "Best ROC-AUC (EfficientNet-B4)" },
  { value: `${champion.balAcc}%`,         label: "Best Balanced Accuracy" },
];

export default function LandingPage() {
  return (
    <div className="px-4 sm:px-6 lg:px-8 py-12 space-y-20 max-w-5xl mx-auto">

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="text-center space-y-6 pt-6">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-primary/30 bg-primary/5 font-technical-data text-[11px] text-primary">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          HAM10000 · EfficientNet-B4 · Grad-CAM Explainability
        </div>

        <h1 className="font-headline-lg text-4xl sm:text-5xl font-bold text-on-surface tracking-tight leading-tight">
          Understanding Skin Lesions
          <br />
          <span className="text-primary">Through Deep Learning</span>
        </h1>

        <p className="font-body-lg text-on-surface-variant text-lg max-w-2xl mx-auto leading-relaxed">
          A research workstation for dermoscopic image classification,
          multi-architecture benchmarking, and visual explainability via Grad-CAM.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center pt-2">
          <Link
            href="/classify"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-primary text-on-primary font-technical-label text-sm font-bold rounded hover:bg-primary/90 transition-colors"
          >
            LAUNCH WORKSTATION <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            href="/dashboard"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 border border-outline-variant/40 text-on-surface font-technical-label text-sm rounded hover:border-primary/40 hover:text-primary transition-colors"
          >
            EXPLORE BENCHMARKS
          </Link>
        </div>
      </section>

      {/* ── Research Pipeline ─────────────────────────────────────────────── */}
      <section className="space-y-4">
        <div className="font-technical-label text-xs text-on-surface-variant uppercase tracking-widest text-center">
          Research Pipeline
        </div>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
          {PIPELINE_STEPS.map(({ Icon, label }, i) => (
            <div
              key={i}
              className="flex flex-col items-center gap-2 p-3 rounded bg-surface-container border border-outline-variant/15 text-center"
            >
              <Icon className="w-5 h-5 text-primary" />
              <span className="font-technical-label text-[10px] text-on-surface-variant leading-tight">
                {label}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Key Stats ────────────────────────────────────────────────────── */}
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {KEY_STATS.map(({ value, label }) => (
          <div
            key={label}
            className="p-5 rounded-xl bg-surface-container border border-outline-variant/20 text-center space-y-1"
          >
            <div className="font-headline-md text-2xl font-bold text-primary">{value}</div>
            <div className="font-technical-label text-[11px] text-on-surface-variant">{label}</div>
          </div>
        ))}
      </section>

      {/* ── Research Integrity Statement ─────────────────────────────────── */}
      <section className="p-6 rounded-xl bg-surface-container border border-outline-variant/20 space-y-4">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
          Research Integrity Statement
        </div>
        <div className="grid sm:grid-cols-2 gap-6 font-body-sm text-sm text-on-surface-variant leading-relaxed">
          <div>
            <span className="text-status-benign font-semibold block mb-1">
              Best Evaluated Baseline — EfficientNet-B4
            </span>
            Achieved 95.92% Macro ROC-AUC and 79.16% Balanced Accuracy on the
            HAM10000 patient-stratified test set. Selected as the deployed inference
            model due to consistently superior performance.
          </div>
          <div>
            <span className="text-research-violet font-semibold block mb-1">
              Experimental Architecture — Decoupled Dual-Branch CNN
            </span>
            The research hypothesis: separate high-frequency texture from macroscopic
            structural representations. Did not surpass EfficientNet-B4. This is an
            honest, documented finding — not a failure, but a directional insight.
          </div>
        </div>
      </section>

      {/* ── CTA to Workstation ───────────────────────────────────────────── */}
      <section className="text-center space-y-3 pb-4">
        <p className="font-technical-label text-xs text-on-surface-variant uppercase tracking-widest">
          Ready to analyse dermoscopic images?
        </p>
        <Link
          href="/classify"
          className="inline-flex items-center gap-2 px-8 py-3 bg-surface-container border border-primary/30 text-primary font-technical-label text-sm font-bold rounded hover:bg-primary/10 transition-colors"
        >
          OPEN ANALYSIS WORKSTATION <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
