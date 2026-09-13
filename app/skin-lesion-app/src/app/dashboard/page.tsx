"use client";

import { useState } from "react";
import { Award, TrendingUp, Cpu } from "lucide-react";
import { BENCHMARK_DATA } from "@/lib/constants";

type Metric = "auc" | "balAcc" | "acc" | "f1" | "latency";

const METRIC_META: Record<Metric, { label: string; unit: string; max: number; higherIsBetter: boolean }> = {
  auc:     { label: "Macro ROC-AUC",     unit: "%",  max: 100, higherIsBetter: true  },
  balAcc:  { label: "Balanced Accuracy", unit: "%",  max: 100, higherIsBetter: true  },
  acc:     { label: "Test Accuracy",     unit: "%",  max: 100, higherIsBetter: true  },
  f1:      { label: "Macro F1",          unit: "%",  max: 100, higherIsBetter: true  },
  latency: { label: "Inference Latency", unit: " ms", max: 35,  higherIsBetter: false },
};

export default function DashboardPage() {
  const [selected, setSelected] = useState<Metric>("auc");
  const meta = METRIC_META[selected];

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-5xl mx-auto space-y-6">

      {/* Header */}
      <div>
        <h1 className="font-headline-md text-xl font-bold text-on-surface">Benchmark Dashboard</h1>
        <p className="font-body-sm text-sm text-on-surface-variant mt-0.5">
          Verified empirical evaluation of 6 architectures on the HAM10000 patient-stratified test set (N ≈ 1,503).
        </p>
      </div>

      {/* Environment caveat — critical note */}
      <div className="flex items-start gap-2 p-3 rounded bg-surface-container border border-outline-variant/15 font-technical-data text-[11px] text-on-surface-variant">
        <Cpu className="w-4 h-4 text-primary shrink-0 mt-0.5" aria-hidden="true" />
        <span>
          <strong className="text-on-surface">Benchmark Environment:</strong>{" "}
          All metrics — including latency (ms/image) — were measured on the Lenovo LOQ with NVIDIA CUDA.
          Live inference latency on Apple M4 MPS is ~14–16 ms and is reported separately in the Workstation.
          These two latency values are <em>not comparable</em>.
        </span>
      </div>

      {/* Champion card */}
      <div className="bg-primary/5 border border-primary/30 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Award className="w-5 h-5 text-primary" aria-hidden="true" />
          <span className="font-technical-label text-sm font-bold text-primary uppercase tracking-wider">
            Best Evaluated Baseline
          </span>
        </div>
        <div className="font-headline-md text-2xl font-bold text-on-surface">EfficientNet-B4</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            ["ROC-AUC",      "95.92%"],
            ["Balanced Acc", "79.16%"],
            ["Test Acc",     "73.64%"],
            ["Macro F1",     "69.19%"],
          ].map(([k, v]) => (
            <div key={k} className="text-center">
              <div className="font-technical-data text-lg font-bold text-primary">{v}</div>
              <div className="font-technical-label text-[10px] text-on-surface-variant">{k}</div>
            </div>
          ))}
        </div>
        <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
          Compound scaling of depth, width, and resolution demonstrated superior multi-scale dermoscopic
          pattern capture over all tested architectures — including the Dual-Branch CNN research hypothesis.
        </p>
      </div>

      {/* Metric Selector */}
      <div className="flex flex-wrap gap-2" role="group" aria-label="Select metric">
        {(Object.keys(METRIC_META) as Metric[]).map((m) => (
          <button
            key={m}
            onClick={() => setSelected(m)}
            aria-pressed={selected === m}
            className={`px-3 py-1.5 rounded border font-technical-label text-xs transition-colors ${
              selected === m
                ? "bg-primary text-on-primary border-primary"
                : "border-outline-variant/30 text-on-surface-variant hover:text-primary hover:border-primary/40"
            }`}
          >
            {METRIC_META[m].label}
          </button>
        ))}
      </div>

      {/* Bar chart */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-4">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
          {meta.label}
        </div>
        <div className="space-y-4">
          {BENCHMARK_DATA.map((item) => {
            const val = item[selected];
            const pct = meta.higherIsBetter
              ? (val / meta.max) * 100
              : ((meta.max - val) / meta.max) * 100;
            return (
              <div key={item.name} className="space-y-1.5">
                <div className="flex items-center justify-between font-technical-data text-xs">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-on-surface font-semibold">{item.name}</span>
                    <span className="text-on-surface-variant text-[10px]">({item.params})</span>
                    {item.isChampion && (
                      <span className="px-1.5 py-0.5 rounded bg-primary/20 text-primary border border-primary/40 text-[9px] font-bold uppercase">
                        BEST
                      </span>
                    )}
                  </div>
                  <span className="font-bold text-on-surface whitespace-nowrap">
                    {val.toFixed(2)}{meta.unit}
                  </span>
                </div>
                <div className="h-2 w-full bg-surface-container-lowest rounded-full overflow-hidden border border-outline-variant/10">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      item.isChampion ? "bg-primary" : "bg-secondary/60"
                    }`}
                    style={{ width: `${Math.max(pct, 3)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Full table */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high">
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Canonical Evaluation Matrix — HAM10000 Test Set
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left font-technical-data text-xs" aria-label="Benchmark results table">
            <thead className="bg-surface-container-lowest text-on-surface-variant uppercase text-[10px] tracking-wider border-b border-outline-variant/15">
              <tr>
                <th className="py-2.5 px-4" scope="col">Rank</th>
                <th className="py-2.5 px-4" scope="col">Architecture</th>
                <th className="py-2.5 px-4" scope="col">Params</th>
                <th className="py-2.5 px-4" scope="col">ROC-AUC</th>
                <th className="py-2.5 px-4" scope="col">Bal. Acc</th>
                <th className="py-2.5 px-4" scope="col">Accuracy</th>
                <th className="py-2.5 px-4" scope="col">Macro F1</th>
                <th className="py-2.5 px-4" scope="col">Latency*</th>
                <th className="py-2.5 px-4" scope="col">Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/10">
              {BENCHMARK_DATA.map((row) => (
                <tr key={row.name} className="hover:bg-surface-container-high/50 transition-colors">
                  <td className="py-3 px-4 text-on-surface-variant font-bold">#{row.rank}</td>
                  <td className="py-3 px-4 font-semibold text-on-surface">
                    {row.name}
                    {row.isChampion && (
                      <Award className="w-3.5 h-3.5 text-primary inline ml-1.5" aria-label="Best evaluated" />
                    )}
                  </td>
                  <td className="py-3 px-4 text-on-surface-variant">{row.params}</td>
                  <td className="py-3 px-4 font-bold text-primary">{row.auc.toFixed(2)}%</td>
                  <td className="py-3 px-4">{row.balAcc.toFixed(2)}%</td>
                  <td className="py-3 px-4">{row.acc.toFixed(2)}%</td>
                  <td className="py-3 px-4">{row.f1.toFixed(2)}%</td>
                  <td className="py-3 px-4 text-status-benign">{row.latency.toFixed(2)} ms</td>
                  <td className="py-3 px-4">
                    <span
                      className={`px-2 py-0.5 rounded border text-[10px] font-semibold ${
                        row.role === "Best Evaluated Baseline"
                          ? "text-primary border-primary/30 bg-primary/10"
                          : row.role === "Experimental Architecture"
                          ? "text-research-violet border-research-violet/30 bg-research-violet/10"
                          : "text-on-surface-variant border-outline-variant/30"
                      }`}
                    >
                      {row.role}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="font-technical-data text-[10px] text-on-surface-variant px-4 py-2 border-t border-outline-variant/10">
            *Latency measured on Lenovo LOQ CUDA environment — not representative of MacBook MPS live inference.
          </p>
        </div>
      </div>

      {/* Dual-Branch research note */}
      <div className="p-5 rounded-xl bg-research-violet/5 border border-research-violet/20 space-y-2">
        <div className="font-technical-label text-xs text-research-violet uppercase tracking-wider flex items-center gap-1.5">
          <TrendingUp className="w-4 h-4" aria-hidden="true" /> Experimental Architecture Findings
        </div>
        <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
          The Decoupled Dual-Branch CNN was designed to separate high-frequency texture
          (shallow-wide branch) from macroscopic morphological structure (deep-narrow branch).
          While achieving respectable discrimination (~90.98% AUC), the pre-trained
          compound-scaled EfficientNet-B4 surpassed it across all metrics. This is an
          honest, documented finding — presented as an{" "}
          <strong className="text-research-violet">experimental architecture investigation</strong>,
          not a failure.
        </p>
      </div>
    </div>
  );
}
