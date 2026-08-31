"use client";

import { useState } from "react";
import { 
  Trophy, CheckCircle2, TrendingUp, Cpu, Zap, 
  BarChart2, Layers, AlertCircle, ShieldAlert, Award 
} from "lucide-react";

interface ModelRecord {
  rank: number;
  name: string;
  family: string;
  params: string;
  auc: number;
  balAcc: number;
  acc: number;
  f1: number;
  latency: number;
  vram: string;
  status: "Champion (Deployed)" | "Baseline" | "Research Hypothesis";
  statusColor: string;
}

const BENCHMARK_DATA: ModelRecord[] = [
  {
    rank: 1,
    name: "EfficientNet-B4",
    family: "Compound Scaled CNN",
    params: "17.56M",
    auc: 95.92,
    balAcc: 79.16,
    acc: 73.64,
    f1: 69.19,
    latency: 8.83,
    vram: "3.8 GB",
    status: "Champion (Deployed)",
    statusColor: "text-primary border-primary/30 bg-primary/10",
  },
  {
    rank: 2,
    name: "DenseNet-121",
    family: "Dense Residual Network",
    params: "6.96M",
    auc: 95.31,
    balAcc: 79.14,
    acc: 66.36,
    f1: 62.42,
    latency: 20.37,
    vram: "2.4 GB",
    status: "Baseline",
    statusColor: "text-secondary border-secondary/30 bg-secondary/10",
  },
  {
    rank: 3,
    name: "Dual-Branch V1.1",
    family: "Decoupled CNN (Optimized Training)",
    params: "10.67M",
    auc: 90.06,
    balAcc: 62.18,
    acc: 65.76,
    f1: 48.14,
    latency: 14.50,
    vram: "4.1 GB",
    status: "Research Hypothesis",
    statusColor: "text-research-violet border-research-violet/30 bg-research-violet/10",
  },
  {
    rank: 4,
    name: "Dual-Branch V2",
    family: "Decoupled CNN (Refined Topology)",
    params: "9.03M",
    auc: 90.15,
    balAcc: 59.48,
    acc: 64.24,
    f1: 49.50,
    latency: 12.10,
    vram: "3.6 GB",
    status: "Research Hypothesis",
    statusColor: "text-research-violet border-research-violet/30 bg-research-violet/10",
  },
  {
    rank: 5,
    name: "Dual-Branch V1 (Seed 123)",
    family: "Decoupled CNN (Original Topology)",
    params: "10.67M",
    auc: 90.98,
    balAcc: 70.31,
    acc: 55.50,
    f1: 48.55,
    latency: 15.20,
    vram: "4.1 GB",
    status: "Research Hypothesis",
    statusColor: "text-research-violet border-research-violet/30 bg-research-violet/10",
  },
  {
    rank: 6,
    name: "ResNet-50",
    family: "Deep Residual Network",
    params: "23.52M",
    auc: 93.52,
    balAcc: 75.13,
    acc: 56.62,
    f1: 53.52,
    latency: 20.96,
    vram: "5.2 GB",
    status: "Baseline",
    statusColor: "text-on-surface-variant border-outline-variant/30 bg-surface-variant/30",
  },
];

export default function DashboardPage() {
  const [selectedMetric, setSelectedMetric] = useState<"auc" | "balAcc" | "acc" | "f1" | "latency">("auc");

  const metricMeta = {
    auc: { label: "Macro ROC-AUC", unit: "%", better: "Higher is better", max: 100 },
    balAcc: { label: "Balanced Accuracy", unit: "%", better: "Higher is better", max: 100 },
    acc: { label: "Overall Test Accuracy", unit: "%", better: "Higher is better", max: 100 },
    f1: { label: "Macro F1 Score", unit: "%", better: "Higher is better", max: 100 },
    latency: { label: "Inference Latency", unit: "ms", better: "Lower is better", max: 25 },
  };

  return (
    <div className="py-8 px-4 sm:px-6 lg:px-8 space-y-8">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-outline-variant/15 pb-4">
        <div>
          <div className="flex items-center gap-2 font-technical-label text-xs text-primary uppercase tracking-widest">
            <Trophy className="w-4 h-4 text-primary" />
            <span>EMPIRICAL BENCHMARK ARCHIVE • HAM10000 TEST SPLIT</span>
          </div>
          <h1 className="font-headline-md text-2xl sm:text-3xl font-bold text-on-surface mt-1">
            Comparative Model Performance
          </h1>
        </div>

        <div className="flex items-center gap-2 font-technical-data text-xs text-status-benign bg-status-benign/10 px-3 py-1.5 rounded border border-status-benign/20">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>Patient-Aware 70/15/15 Split Verified</span>
        </div>
      </div>

      {/* Metric Selector & Visualization Bento */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-6 space-y-6 tech-border">
        
        {/* Metric Selector Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-outline-variant/15 pb-4">
          <div className="flex flex-wrap items-center gap-2">
            {(["auc", "balAcc", "acc", "f1", "latency"] as const).map((key) => (
              <button
                key={key}
                onClick={() => setSelectedMetric(key)}
                className={`px-3.5 py-1.5 rounded font-technical-label text-xs tracking-wider uppercase transition-all ${
                  selectedMetric === key
                    ? "bg-primary text-on-primary font-bold shadow-[0_0_10px_rgba(136,245,255,0.25)]"
                    : "bg-surface-container-low border border-outline-variant/20 text-on-surface-variant hover:text-on-surface hover:border-primary/40"
                }`}
              >
                {metricMeta[key].label}
              </button>
            ))}
          </div>

          <div className="font-technical-data text-xs text-on-surface-variant">
            CRITERION: <span className="text-primary font-bold">{metricMeta[selectedMetric].better}</span>
          </div>
        </div>

        {/* Dynamic Comparison Bar Chart */}
        <div className="space-y-4">
          {BENCHMARK_DATA.map((item) => {
            const val = item[selectedMetric];
            const maxVal = metricMeta[selectedMetric].max;
            const pct = selectedMetric === "latency" ? ((maxVal - val) / maxVal) * 100 : (val / maxVal) * 100;
            const isChampion = item.name === "EfficientNet-B4";

            return (
              <div key={item.name} className="space-y-1.5">
                <div className="flex items-center justify-between font-technical-data text-xs">
                  <div className="flex items-center gap-2">
                    <span className="text-on-surface font-semibold">{item.name}</span>
                    <span className="text-on-surface-variant text-[11px]">({item.params})</span>
                    {isChampion && (
                      <span className="px-1.5 py-0.2 rounded bg-primary/20 text-primary border border-primary/40 text-[9px] font-bold uppercase">
                        CHAMPION
                      </span>
                    )}
                  </div>
                  <div className="font-bold text-on-surface">
                    {val.toFixed(2)} {metricMeta[selectedMetric].unit}
                  </div>
                </div>

                <div className="h-2 w-full bg-surface-container-lowest rounded-full overflow-hidden border border-outline-variant/10">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isChampion ? "bg-primary" : "bg-secondary/70"
                    }`}
                    style={{ width: `${Math.max(pct, 5)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

      </div>

      {/* Full Benchmark Table */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high flex items-center justify-between">
          <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Canonical Evaluation Matrix (HAM10000 Test Set, N=1,503)
          </div>
          <span className="font-technical-data text-[11px] text-on-surface-variant">
            6 Architectures Evaluated
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left font-technical-data text-xs">
            <thead className="bg-surface-container-lowest text-on-surface-variant uppercase text-[10px] tracking-wider border-b border-outline-variant/15">
              <tr>
                <th className="py-3 px-4">Rank</th>
                <th className="py-3 px-4">Model Architecture</th>
                <th className="py-3 px-4">Params</th>
                <th className="py-3 px-4">ROC-AUC</th>
                <th className="py-3 px-4">Bal. Acc</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Macro F1</th>
                <th className="py-3 px-4">Latency</th>
                <th className="py-3 px-4">VRAM</th>
                <th className="py-3 px-4">Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/10">
              {BENCHMARK_DATA.map((row) => (
                <tr key={row.name} className="hover:bg-surface-container-high/50 transition-colors">
                  <td className="py-3 px-4 font-bold text-on-surface-variant">#{row.rank}</td>
                  <td className="py-3 px-4 font-semibold text-on-surface flex items-center gap-1.5">
                    {row.name}
                    {row.rank === 1 && <Award className="w-3.5 h-3.5 text-primary inline" />}
                  </td>
                  <td className="py-3 px-4 text-on-surface-variant">{row.params}</td>
                  <td className="py-3 px-4 font-bold text-primary">{row.auc.toFixed(2)}%</td>
                  <td className="py-3 px-4 text-on-surface">{row.balAcc.toFixed(2)}%</td>
                  <td className="py-3 px-4 text-on-surface">{row.acc.toFixed(2)}%</td>
                  <td className="py-3 px-4 text-on-surface">{row.f1.toFixed(2)}%</td>
                  <td className="py-3 px-4 text-status-benign">{row.latency.toFixed(2)} ms</td>
                  <td className="py-3 px-4 text-on-surface-variant">{row.vram}</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-0.5 rounded border text-[10px] uppercase font-semibold ${row.statusColor}`}>
                      {row.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scientific Narrative Callout */}
      <div className="p-5 rounded-xl bg-surface-container border border-outline-variant/15 font-body-sm text-xs text-on-surface-variant leading-relaxed space-y-2">
        <div className="font-technical-label text-xs text-primary uppercase tracking-wider flex items-center gap-1.5">
          <TrendingUp className="w-4 h-4 text-primary" />
          <span>Empirical Analysis of Architectural Hypotheses</span>
        </div>
        <p>
          The Dual-Branch CNN framework was designed to address the spatial trade-off in dermoscopy by decoupling high-frequency texture representations (via an unpooled wide branch) from macroscopic lesion morphology (via a deep narrow branch). While the architecture attained competitive discriminative power (<strong className="text-primary">90.98% ROC-AUC</strong>), the compound scaling mechanism in <strong className="text-primary">EfficientNet-B4</strong> (balanced depth, width, and resolution scaling) demonstrated superior capability in capturing multi-scale dermoscopic patterns, leading to <strong className="text-primary">95.92% ROC-AUC</strong> and <strong className="text-primary">79.16% Balanced Accuracy</strong>.
        </p>
      </div>

    </div>
  );
}
