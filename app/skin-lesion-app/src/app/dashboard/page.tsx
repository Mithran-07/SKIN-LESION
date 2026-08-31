"use client";

import { useState } from "react";
import { 
  Trophy, 
  BarChart3, 
  Layers, 
  Cpu, 
  Clock, 
  CheckCircle2, 
  TrendingUp, 
  ShieldAlert, 
  ArrowUpRight, 
  SlidersHorizontal 
} from "lucide-react";

interface ModelMetric {
  name: string;
  type: string;
  params: string;
  accuracy: number;
  balancedAccuracy: number;
  macroF1: number;
  macroAuc: number;
  latencyMs: number;
  vramMb: number;
  rank: number;
  isBest?: boolean;
}

const BENCHMARK_MODELS: ModelMetric[] = [
  {
    name: "EfficientNet-B4 (Best Model)",
    type: "Single-Branch Compound Scale",
    params: "17.56M",
    accuracy: 73.64,
    balancedAccuracy: 79.16,
    macroF1: 69.19,
    macroAuc: 95.92,
    latencyMs: 8.83,
    vramMb: 677.7,
    rank: 1,
    isBest: true,
  },
  {
    name: "DenseNet-121 Baseline",
    type: "Single-Branch Feature Reuse",
    params: "6.96M",
    accuracy: 66.36,
    balancedAccuracy: 79.14,
    macroF1: 62.42,
    macroAuc: 95.31,
    latencyMs: 20.37,
    vramMb: 393.7,
    rank: 2,
  },
  {
    name: "Dual-Branch CNN (Seed 123)",
    type: "Decoupled Texture + Structure",
    params: "10.67M",
    accuracy: 55.50,
    balancedAccuracy: 70.31,
    macroF1: 48.55,
    macroAuc: 90.98,
    latencyMs: 27.23,
    vramMb: 1561.1,
    rank: 3,
  },
  {
    name: "Dual-Branch CNN (Seed 999)",
    type: "Decoupled Texture + Structure",
    params: "10.67M",
    accuracy: 54.97,
    balancedAccuracy: 66.39,
    macroF1: 45.77,
    macroAuc: 89.73,
    latencyMs: 26.32,
    vramMb: 1561.1,
    rank: 4,
  },
  {
    name: "Dual-Branch CNN (Seed 42)",
    type: "Decoupled Texture + Structure",
    params: "10.67M",
    accuracy: 53.91,
    balancedAccuracy: 68.62,
    macroF1: 44.90,
    macroAuc: 90.54,
    latencyMs: 28.47,
    vramMb: 1561.1,
    rank: 5,
  },
  {
    name: "ResNet-50 Baseline",
    type: "Single-Branch Residual Network",
    params: "23.52M",
    accuracy: 56.62,
    balancedAccuracy: 75.13,
    macroF1: 53.52,
    macroAuc: 93.52,
    latencyMs: 20.96,
    vramMb: 900.3,
    rank: 6,
  },
];

export default function DashboardPage() {
  const [selectedMetric, setSelectedMetric] = useState<"macroAuc" | "balancedAccuracy" | "accuracy" | "macroF1" | "latencyMs">("macroAuc");

  const getMetricLabel = (m: string) => {
    switch (m) {
      case "macroAuc": return "ROC-AUC (Macro %)";
      case "balancedAccuracy": return "Balanced Accuracy (%)";
      case "accuracy": return "Test Accuracy (%)";
      case "macroF1": return "Macro F1 Score (%)";
      case "latencyMs": return "Inference Latency (ms/img - lower is better)";
      default: return "";
    }
  };

  return (
    <div className="space-y-10 max-w-6xl mx-auto">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold mb-2">
            <Trophy className="w-3.5 h-3.5" />
            <span>Lenovo LOQ Experimental Benchmark Archive</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Research & Benchmark Dashboard
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical comparative analysis across compound-scaled baselines and the experimental Dual-Branch CNN on HAM10000.
          </p>
        </div>
      </div>

      {/* Champion Model Card */}
      <div className="glass-card rounded-2xl p-6 sm:p-8 border border-cyan-500/30 bg-gradient-to-br from-slate-900 via-slate-900/90 to-cyan-950/20 relative overflow-hidden">
        <div className="flex items-start justify-between flex-wrap gap-4 border-b border-slate-800 pb-6 mb-6">
          <div>
            <div className="text-xs font-bold text-cyan-400 uppercase tracking-widest flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" />
              <span>Highest Evaluated Diagnostic Performance</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-black text-white mt-1">
              EfficientNet-B4
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Compound Scaled CNN with Depthwise Separable Convolutions & Squeeze-and-Excitation
            </p>
          </div>
          <div className="px-4 py-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-bold font-mono">
            DEPLOYED APPLICATION MODEL
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="text-2xl sm:text-3xl font-black text-cyan-400">95.92%</div>
            <div className="text-[11px] text-slate-400 font-medium mt-1 uppercase">Macro ROC-AUC</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="text-2xl sm:text-3xl font-black text-blue-400">79.16%</div>
            <div className="text-[11px] text-slate-400 font-medium mt-1 uppercase">Balanced Accuracy</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="text-2xl sm:text-3xl font-black text-indigo-400">69.19%</div>
            <div className="text-[11px] text-slate-400 font-medium mt-1 uppercase">Macro F1 Score</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="text-2xl sm:text-3xl font-black text-emerald-400">8.83 ms</div>
            <div className="text-[11px] text-slate-400 font-medium mt-1 uppercase">Inference Speed</div>
          </div>
        </div>
      </div>

      {/* Interactive Metric Comparison Bar Chart */}
      <div className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              <span>Comparative Performance by Architecture</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">Toggle metrics to compare trade-offs across evaluated models.</p>
          </div>

          <div className="flex items-center gap-1.5 flex-wrap">
            {[
              { id: "macroAuc", label: "ROC-AUC" },
              { id: "balancedAccuracy", label: "Balanced Acc" },
              { id: "accuracy", label: "Accuracy" },
              { id: "macroF1", label: "Macro F1" },
              { id: "latencyMs", label: "Latency" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSelectedMetric(tab.id as any)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                  selectedMetric === tab.id
                    ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                    : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Visual Bar Representation */}
        <div className="space-y-4 pt-2">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            {getMetricLabel(selectedMetric)}
          </div>

          {BENCHMARK_MODELS.map((m) => {
            const val = m[selectedMetric];
            const maxVal = selectedMetric === "latencyMs" ? 30 : 100;
            const barWidth = selectedMetric === "latencyMs" 
              ? Math.max(10, 100 - (val / maxVal) * 80)
              : Math.max(10, val);

            return (
              <div key={m.name} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className={`w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center ${m.isBest ? "bg-cyan-500 text-slate-950" : "bg-slate-800 text-slate-400"}`}>
                      {m.rank}
                    </span>
                    <span className={`font-semibold ${m.isBest ? "text-cyan-300" : "text-slate-200"}`}>
                      {m.name}
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono">({m.params})</span>
                  </div>
                  <span className="font-mono font-bold text-white">
                    {selectedMetric === "latencyMs" ? `${val.toFixed(2)} ms` : `${val.toFixed(2)}%`}
                  </span>
                </div>

                <div className="h-3 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      m.isBest 
                        ? "bg-gradient-to-r from-cyan-400 to-blue-500" 
                        : m.name.includes("DenseNet") 
                        ? "bg-blue-500" 
                        : m.name.includes("Dual-Branch") 
                        ? "bg-purple-500" 
                        : "bg-slate-600"
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Comprehensive Benchmark Table */}
      <div className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <SlidersHorizontal className="w-5 h-5 text-blue-400" />
          <span>Full LOQ Experimental Benchmark Archive</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-3 px-3">Rank</th>
                <th className="py-3 px-3">Model Architecture</th>
                <th className="py-3 px-3">Params</th>
                <th className="py-3 px-3 text-cyan-400">ROC-AUC</th>
                <th className="py-3 px-3">Bal. Acc</th>
                <th className="py-3 px-3">Accuracy</th>
                <th className="py-3 px-3">Macro F1</th>
                <th className="py-3 px-3">Latency</th>
                <th className="py-3 px-3">VRAM</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {BENCHMARK_MODELS.map((m) => (
                <tr 
                  key={m.name} 
                  className={`hover:bg-slate-800/40 transition-colors ${m.isBest ? "bg-cyan-500/5 font-semibold" : ""}`}
                >
                  <td className="py-3.5 px-3">
                    <span className={`w-5 h-5 rounded-full inline-flex items-center justify-center text-[10px] ${m.isBest ? "bg-cyan-500 text-slate-950 font-bold" : "bg-slate-800 text-slate-400"}`}>
                      {m.rank}
                    </span>
                  </td>
                  <td className="py-3.5 px-3 font-sans text-slate-200">
                    {m.name}
                    {m.isBest && <span className="ml-2 text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">DEPLOYED</span>}
                  </td>
                  <td className="py-3.5 px-3 text-slate-400">{m.params}</td>
                  <td className="py-3.5 px-3 text-cyan-400 font-bold">{m.macroAuc.toFixed(2)}%</td>
                  <td className="py-3.5 px-3 text-slate-300">{m.balancedAccuracy.toFixed(2)}%</td>
                  <td className="py-3.5 px-3 text-slate-300">{m.accuracy.toFixed(2)}%</td>
                  <td className="py-3.5 px-3 text-slate-300">{m.macroF1.toFixed(2)}%</td>
                  <td className="py-3.5 px-3 text-slate-400">{m.latencyMs.toFixed(2)} ms</td>
                  <td className="py-3.5 px-3 text-slate-400">{m.vramMb.toFixed(1)} MB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
