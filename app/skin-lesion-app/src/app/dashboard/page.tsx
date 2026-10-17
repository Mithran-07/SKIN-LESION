"use client";

import { useState } from "react";
import { Award, Cpu, Grid3X3, FileCheck2, ShieldAlert } from "lucide-react";
import {
  BENCHMARK_DATA,
  CANONICAL_CLASSES,
  HISTORICAL_CLAIMS_RECONCILIATION,
} from "@/lib/constants";

type Metric = "balAcc" | "auc" | "prAuc" | "acc" | "f1" | "mpsLatencyMs" | "cudaLatencyMs";

const METRIC_META: Record<
  Metric,
  { label: string; unit: string; max: number; higherIsBetter: boolean }
> = {
  balAcc:        { label: "Balanced Accuracy", unit: "%",   max: 100, higherIsBetter: true },
  auc:           { label: "Macro ROC-AUC",     unit: "%",   max: 100, higherIsBetter: true },
  prAuc:         { label: "Macro PR-AUC",      unit: "%",   max: 100, higherIsBetter: true },
  acc:           { label: "Test Accuracy",     unit: "%",   max: 100, higherIsBetter: true },
  f1:            { label: "Macro F1",          unit: "%",   max: 100, higherIsBetter: true },
  mpsLatencyMs:  { label: "Apple M4 MPS Latency", unit: " ms", max: 50, higherIsBetter: false },
  cudaLatencyMs: { label: "Lenovo CUDA Latency",  unit: " ms", max: 35, higherIsBetter: false },
};

export default function DashboardPage() {
  const [selectedMetric, setSelectedMetric] = useState<Metric>("balAcc");
  const [selectedModelId, setSelectedModelId] = useState<string>("efficientnet_b4");
  const [normalizeMatrix, setNormalizeMatrix] = useState<boolean>(true);

  const meta = METRIC_META[selectedMetric];
  const activeModel =
    BENCHMARK_DATA.find((m) => m.id === selectedModelId) ?? BENCHMARK_DATA[0];

  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-headline-md text-xl font-bold text-on-surface">
            Benchmark &amp; Evaluation Lab
          </h1>
          <p className="font-body-sm text-sm text-on-surface-variant mt-0.5">
            Machine-verified evaluation of all 6 trained checkpoints on the lesion-disjoint HAM10000
            test split (N = 1,510 images across 1,121 unique lesions).
          </p>
        </div>
        <div className="font-technical-data text-[11px] px-3 py-1.5 rounded border border-primary/30 bg-primary/5 text-primary">
          Primary Ranking Criterion: Balanced Accuracy (ISIC Standard)
        </div>
      </div>

      {/* Hardware & Split Provenance Banner */}
      <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-surface-container border border-outline-variant/20 font-technical-data text-[11px] text-on-surface-variant">
        <Cpu className="w-4 h-4 text-primary shrink-0 mt-0.5" aria-hidden="true" />
        <div className="space-y-1">
          <div>
            <strong className="text-on-surface">Dual-Hardware Benchmark Provenance:</strong>{" "}
            Classification metrics (Balanced Accuracy, Macro ROC-AUC, PR-AUC, F1, Confusion Matrices)
            were verified directly from checkpoint weights on the canonical held-out test set{" "}
            <code className="text-on-surface">splits/test.csv (N=1,510)</code>.
          </div>
          <div>
            Latency is reported separately for{" "}
            <strong className="text-on-surface">Apple M4 MPS (16 GB unified RAM)</strong> and{" "}
            <strong className="text-on-surface">Lenovo LOQ (NVIDIA RTX 3050 6 GB CUDA)</strong>.
          </div>
        </div>
      </div>

      {/* Champion Summary Card */}
      <div className="bg-primary/5 border border-primary/30 rounded-xl p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-primary" aria-hidden="true" />
            <span className="font-technical-label text-xs font-bold text-primary uppercase tracking-wider">
              Deployed Champion Baseline · Rank #1
            </span>
          </div>
          <span className="font-technical-data text-[10px] text-on-surface-variant">
            SHA-256: 125340ffa59ca4a18fb0cd0bdca024eae9ca2fe3ce54d1f230b465bd9ad9658e
          </span>
        </div>
        <div className="font-headline-md text-2xl font-bold text-on-surface">
          EfficientNet-B4 (17.56M Parameters)
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-6 gap-3 pt-1">
          {[
            ["Balanced Acc", "79.23%"],
            ["Macro ROC-AUC", "95.91%"],
            ["Macro PR-AUC", "77.52%"],
            ["Test Accuracy", "73.51%"],
            ["Macro F1", "69.24%"],
            ["M4 MPS Warm", "27.64 ms"],
          ].map(([k, v]) => (
            <div
              key={k}
              className="p-2.5 rounded-lg bg-surface-container/90 border border-outline-variant/15 text-center"
            >
              <div className="font-technical-data text-base font-bold text-primary">{v}</div>
              <div className="font-technical-label text-[10px] text-on-surface-variant">{k}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Metric Selector & Comparative Bar Chart */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Comparative Architecture Ranking — {meta.label}
          </div>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Select benchmark metric">
            {(Object.keys(METRIC_META) as Metric[]).map((m) => (
              <button
                key={m}
                onClick={() => setSelectedMetric(m)}
                aria-pressed={selectedMetric === m}
                className={`px-2.5 py-1 rounded border font-technical-label text-[11px] transition-colors ${
                  selectedMetric === m
                    ? "bg-primary text-on-primary border-primary font-bold"
                    : "border-outline-variant/30 text-on-surface-variant hover:text-primary hover:border-primary/40"
                }`}
              >
                {METRIC_META[m].label}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-3.5 pt-1">
          {BENCHMARK_DATA.map((item) => {
            const val = item[selectedMetric];
            const pct = meta.higherIsBetter
              ? (val / meta.max) * 100
              : ((meta.max - val) / meta.max) * 100;
            return (
              <div key={item.id} className="space-y-1">
                <div className="flex items-center justify-between font-technical-data text-xs">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-on-surface font-semibold">{item.name}</span>
                    <span className="text-on-surface-variant text-[10px]">
                      ({item.params} · {item.fusionVersion ?? item.architectureVersion})
                    </span>
                    {item.isChampion && (
                      <span className="px-1.5 py-0.5 rounded bg-primary/20 text-primary border border-primary/40 text-[9px] font-bold uppercase">
                        CHAMPION
                      </span>
                    )}
                  </div>
                  <span className="font-bold text-on-surface whitespace-nowrap">
                    {val.toFixed(2)}
                    {meta.unit}
                  </span>
                </div>
                <div className="h-2 w-full bg-surface-container-lowest rounded-full overflow-hidden border border-outline-variant/10">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      item.isChampion
                        ? "bg-primary"
                        : item.role === "Experimental Dual-Branch"
                        ? "bg-research-violet/80"
                        : "bg-secondary/70"
                    }`}
                    style={{ width: `${Math.max(pct, 4)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Canonical 6-Checkpoint Evaluation Table */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high flex flex-wrap items-center justify-between gap-2">
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Canonical 6-Checkpoint Evaluation Matrix — HAM10000 Held-Out Test Set (N = 1,510)
          </span>
          <span className="font-technical-data text-[10px] text-on-surface-variant">
            Click any row to inspect its 7×7 Confusion Matrix &amp; Per-Class Recall below
          </span>
        </div>
        <div className="overflow-x-auto">
          <table
            className="w-full text-left font-technical-data text-xs"
            aria-label="Verified 6-checkpoint benchmark results"
          >
            <thead className="bg-surface-container-lowest text-on-surface-variant uppercase text-[10px] tracking-wider border-b border-outline-variant/15">
              <tr>
                <th className="py-2.5 px-3" scope="col">Rank</th>
                <th className="py-2.5 px-3" scope="col">Architecture</th>
                <th className="py-2.5 px-3" scope="col">Params</th>
                <th className="py-2.5 px-3" scope="col">Bal. Acc</th>
                <th className="py-2.5 px-3" scope="col">ROC-AUC</th>
                <th className="py-2.5 px-3" scope="col">PR-AUC</th>
                <th className="py-2.5 px-3" scope="col">Accuracy</th>
                <th className="py-2.5 px-3" scope="col">Macro F1</th>
                <th className="py-2.5 px-3" scope="col">M4 MPS</th>
                <th className="py-2.5 px-3" scope="col">CUDA</th>
                <th className="py-2.5 px-3" scope="col">SHA-256</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/10">
              {BENCHMARK_DATA.map((row) => {
                const isSelected = row.id === selectedModelId;
                return (
                  <tr
                    key={row.id}
                    onClick={() => setSelectedModelId(row.id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? "bg-primary/10"
                        : "hover:bg-surface-container-high/50"
                    }`}
                  >
                    <td className="py-3 px-3 text-on-surface-variant font-bold">#{row.rank}</td>
                    <td className="py-3 px-3 font-semibold text-on-surface">
                      <div className="flex items-center gap-1.5">
                        <span>{row.name}</span>
                        {row.isChampion && (
                          <Award className="w-3.5 h-3.5 text-primary shrink-0" aria-label="Champion" />
                        )}
                      </div>
                      <div className="text-[10px] text-on-surface-variant font-normal">
                        {row.fusionVersion ?? row.architectureVersion}
                      </div>
                    </td>
                    <td className="py-3 px-3 text-on-surface-variant">{row.params}</td>
                    <td className="py-3 px-3 font-bold text-primary">{row.balAcc.toFixed(2)}%</td>
                    <td className="py-3 px-3 font-semibold text-on-surface">{row.auc.toFixed(2)}%</td>
                    <td className="py-3 px-3">{row.prAuc.toFixed(2)}%</td>
                    <td className="py-3 px-3">{row.acc.toFixed(2)}%</td>
                    <td className="py-3 px-3">{row.f1.toFixed(2)}%</td>
                    <td className="py-3 px-3 text-status-benign">{row.mpsLatencyMs.toFixed(1)} ms</td>
                    <td className="py-3 px-3 text-on-surface-variant">{row.cudaLatencyMs.toFixed(2)} ms</td>
                    <td className="py-3 px-3 text-on-surface-variant font-mono text-[10px]">
                      {row.sha256Short}…
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive 7x7 Confusion Matrix & Per-Class Breakdown */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-outline-variant/15 pb-3">
          <div className="flex items-center gap-2">
            <Grid3X3 className="w-4 h-4 text-primary" aria-hidden="true" />
            <div>
              <h2 className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
                Interactive 7×7 Confusion Matrix &amp; Per-Class Diagnostics
              </h2>
              <p className="font-technical-data text-[11px] text-on-surface-variant">
                Inspecting:{" "}
                <strong className="text-primary">{activeModel.name}</strong> (
                {activeModel.exactParams.toLocaleString()} params · SHA-256{" "}
                {activeModel.sha256Short}…)
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <select
              aria-label="Select model for confusion matrix"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="bg-surface-container-low border border-outline-variant/30 rounded px-2.5 py-1.5 font-technical-data text-xs text-on-surface"
            >
              {BENCHMARK_DATA.map((m) => (
                <option key={m.id} value={m.id}>
                  #{m.rank} {m.name} (BalAcc {m.balAcc.toFixed(2)}%)
                </option>
              ))}
            </select>

            <button
              onClick={() => setNormalizeMatrix((prev) => !prev)}
              className="px-3 py-1.5 rounded border border-outline-variant/30 font-technical-label text-[11px] text-on-surface hover:border-primary/40 hover:text-primary transition-colors"
            >
              {normalizeMatrix ? "Showing: Row Recall (%)" : "Showing: Raw Sample Counts"}
            </button>
          </div>
        </div>

        <div className="grid lg:grid-cols-12 gap-6 items-start">
          {/* 7x7 Matrix */}
          <div className="lg:col-span-7 overflow-x-auto">
            <div className="font-technical-label text-[10px] text-on-surface-variant uppercase mb-2">
              Rows = Ground Truth Diagnosis · Columns = Model Prediction (Test Set N = 1,510)
            </div>
            <table
              className="w-full border-collapse font-technical-data text-[11px]"
              aria-label={`Confusion matrix for ${activeModel.name}`}
            >
              <thead>
                <tr>
                  <th className="p-2 text-left text-[10px] text-on-surface-variant border border-outline-variant/15 bg-surface-container-lowest">
                    True \ Pred
                  </th>
                  {CANONICAL_CLASSES.map((c) => (
                    <th
                      key={c}
                      className="p-2 text-center text-[10px] font-bold text-on-surface border border-outline-variant/15 bg-surface-container-lowest"
                    >
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {activeModel.confusionMatrix.map((row, rIdx) => {
                  const rowSum = row.reduce((a, b) => a + b, 0) || 1;
                  return (
                    <tr key={CANONICAL_CLASSES[rIdx]}>
                      <th
                        scope="row"
                        className="p-2 text-left font-bold text-on-surface border border-outline-variant/15 bg-surface-container-lowest whitespace-nowrap"
                      >
                        {CANONICAL_CLASSES[rIdx]}{" "}
                        <span className="text-[9px] text-on-surface-variant font-normal">
                          (n={rowSum})
                        </span>
                      </th>
                      {row.map((cell, cIdx) => {
                        const pct = (cell / rowSum) * 100;
                        const isDiag = rIdx === cIdx;
                        const bgStyle = isDiag
                          ? { backgroundColor: `rgba(56, 189, 248, ${Math.min(0.12 + (pct / 100) * 0.55, 0.7)})` }
                          : pct > 8
                          ? { backgroundColor: `rgba(244, 63, 94, ${Math.min((pct / 100) * 0.65, 0.5)})` }
                          : undefined;
                        return (
                          <td
                            key={cIdx}
                            style={bgStyle}
                            className={`p-2 text-center border border-outline-variant/15 ${
                              isDiag
                                ? "font-bold text-on-surface"
                                : cell > 0
                                ? "text-on-surface-variant"
                                : "text-on-surface-variant/30"
                            }`}
                          >
                            {normalizeMatrix ? `${pct.toFixed(1)}%` : cell}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Per-Class Precision / Recall / F1 Table */}
          <div className="lg:col-span-5 space-y-2">
            <div className="font-technical-label text-[10px] text-on-surface-variant uppercase">
              Per-Class Precision, Recall &amp; F1 ({activeModel.name})
            </div>
            <div className="overflow-x-auto rounded-lg border border-outline-variant/15">
              <table className="w-full text-left font-technical-data text-[11px]">
                <thead className="bg-surface-container-lowest text-on-surface-variant uppercase text-[10px] border-b border-outline-variant/15">
                  <tr>
                    <th className="py-2 px-2.5">Class</th>
                    <th className="py-2 px-2.5">Recall</th>
                    <th className="py-2 px-2.5">Precision</th>
                    <th className="py-2 px-2.5">F1</th>
                    <th className="py-2 px-2.5">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/10">
                  {CANONICAL_CLASSES.map((cls) => {
                    const stats = activeModel.perClass[cls];
                    return (
                      <tr key={cls}>
                        <td className="py-2 px-2.5 font-bold text-on-surface">{cls}</td>
                        <td className="py-2 px-2.5 text-primary font-semibold">
                          {stats.recall.toFixed(2)}%
                        </td>
                        <td className="py-2 px-2.5">{stats.precision.toFixed(2)}%</td>
                        <td className="py-2 px-2.5 font-semibold">{stats.f1.toFixed(2)}%</td>
                        <td className="py-2 px-2.5 text-on-surface-variant">{stats.support}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Historical Claim vs Verified Truth Reconciliation */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high flex items-center gap-2">
          <FileCheck2 className="w-4 h-4 text-primary" aria-hidden="true" />
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Scientific Audit: Historical Claims vs. Machine-Verified Ground Truth
          </span>
        </div>
        <div className="divide-y divide-outline-variant/10">
          {HISTORICAL_CLAIMS_RECONCILIATION.map((item) => (
            <div key={item.claim} className="p-4 space-y-1.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-technical-label text-xs font-bold text-on-surface flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-status-warning shrink-0" />
                  {item.claim}
                </span>
                <span className="font-technical-data text-[11px] text-primary font-semibold">
                  Verified: {item.verifiedTruth}
                </span>
              </div>
              <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                {item.verdict}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
