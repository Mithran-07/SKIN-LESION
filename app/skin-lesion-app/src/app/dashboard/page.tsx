"use client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";

const MODELS = [
  { name: "ResNet50", shortName: "ResNet50", type: "Baseline", accuracy: 56.62, balAcc: 75.13, f1: 53.52, auc: 93.52, params: 23.5, trainTime: 30.9, inference: 20.96, color: "#64748b" },
  { name: "DenseNet121", shortName: "Dense121", type: "Baseline", accuracy: 66.36, balAcc: 79.14, f1: 62.42, auc: 95.31, params: 7.0, trainTime: 24.6, inference: 20.37, color: "#6366f1" },
  { name: "EfficientNet-B4", shortName: "EffNet-B4", type: "Baseline (Best)", accuracy: 73.64, balAcc: 79.16, f1: 69.19, auc: 95.92, params: 17.6, trainTime: 38.9, inference: 8.83, color: "#38bdf8" },
  { name: "Dual-Branch V1", shortName: "DB-V1", type: "Experimental", accuracy: 54.79, balAcc: 68.44, f1: 46.41, auc: 90.41, params: 10.7, trainTime: 385.5, inference: 27.34, color: "#f59e0b" },
  { name: "Dual-Branch V1.1", shortName: "DB-V1.1", type: "Experimental", accuracy: 65.76, balAcc: 62.18, f1: 48.14, auc: 90.06, params: 10.7, trainTime: 171.2, inference: 24.56, color: "#f97316" },
  { name: "Dual-Branch V2", shortName: "DB-V2", type: "Experimental", accuracy: 64.24, balAcc: 59.48, f1: 49.50, auc: 90.15, params: 9.0, trainTime: 181.2, inference: 25.54, color: "#ef4444" },
];

const METRICS_DATA = MODELS.map((m) => ({
  name: m.shortName,
  Accuracy: m.accuracy,
  "Balanced Acc": m.balAcc,
  "Macro F1": m.f1,
  "ROC-AUC": m.auc,
  type: m.type,
}));

const RADAR_DATA = [
  { metric: "Accuracy", "EfficientNet-B4": 73.64, "Dual-Branch V2": 64.24 },
  { metric: "Balanced Acc", "EfficientNet-B4": 79.16, "Dual-Branch V2": 59.48 },
  { metric: "Macro F1", "EfficientNet-B4": 69.19, "Dual-Branch V2": 49.50 },
  { metric: "ROC-AUC", "EfficientNet-B4": 95.92, "Dual-Branch V2": 90.15 },
];

const CLASS_METRICS = [
  { cls: "NV (Nevi)", precision: 95.9, recall: 67.2, f1: 79.0, support: 999 },
  { cls: "MEL (Melanoma)", precision: 33.3, recall: 48.1, f1: 39.4, support: 181 },
  { cls: "BKL (Keratosis)", precision: 45.5, recall: 67.2, f1: 54.2, support: 186 },
  { cls: "BCC (Basal Cell)", precision: 39.2, recall: 67.8, f1: 49.7, support: 59 },
  { cls: "AKIEC", precision: 39.3, recall: 45.3, f1: 42.1, support: 53 },
  { cls: "DF (Dermatofibroma)", precision: 3.7, recall: 37.5, f1: 6.7, support: 8 },
  { cls: "VASC (Vascular)", precision: 69.0, recall: 83.3, f1: 75.5, support: 24 },
];

export default function DashboardPage() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-white mb-2">Research Dashboard</h1>
        <p className="text-slate-400">
          Complete performance comparison across all evaluated models. All values are from held-out test sets.
        </p>
      </div>

      {/* Top Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        {[
          { label: "Best Accuracy", value: "73.64%", model: "EfficientNet-B4", color: "sky" },
          { label: "Best Balanced Acc", value: "79.16%", model: "EfficientNet-B4", color: "indigo" },
          { label: "Best Macro F1", value: "69.19%", model: "EfficientNet-B4", color: "purple" },
          { label: "Best ROC-AUC", value: "95.92%", model: "EfficientNet-B4", color: "emerald" },
        ].map((s) => (
          <div key={s.label} className="glass rounded-2xl p-5">
            <p className="text-slate-400 text-xs mb-1">{s.label}</p>
            <p className="text-2xl font-bold text-white mb-1">{s.value}</p>
            <p className="text-sky-400 text-xs">{s.model}</p>
          </div>
        ))}
      </div>

      {/* Full Metrics Table */}
      <div className="glass rounded-2xl p-6 mb-8">
        <h2 className="text-lg font-semibold text-white mb-4">Complete Benchmark</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 text-left border-b border-slate-800">
                <th className="pb-3 pr-4">Model</th>
                <th className="pb-3 pr-4">Type</th>
                <th className="pb-3 pr-4">Accuracy</th>
                <th className="pb-3 pr-4">Balanced Acc</th>
                <th className="pb-3 pr-4">Macro F1</th>
                <th className="pb-3 pr-4">ROC-AUC</th>
                <th className="pb-3 pr-4">Params (M)</th>
                <th className="pb-3">Inference (ms)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {MODELS.map((m) => (
                <tr
                  key={m.name}
                  className={m.name === "EfficientNet-B4" ? "bg-sky-500/5" : ""}
                >
                  <td className="py-3 pr-4 font-medium text-white flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: m.color }}
                    />
                    {m.name}
                    {m.name === "EfficientNet-B4" && (
                      <span className="bg-sky-500/20 text-sky-400 text-xs px-2 py-0.5 rounded-full">★ Final</span>
                    )}
                  </td>
                  <td className="py-3 pr-4 text-slate-400">{m.type}</td>
                  <td className="py-3 pr-4 text-slate-300">{m.accuracy.toFixed(2)}%</td>
                  <td className="py-3 pr-4 text-slate-300">{m.balAcc.toFixed(2)}%</td>
                  <td className="py-3 pr-4 text-slate-300">{m.f1.toFixed(2)}%</td>
                  <td className="py-3 pr-4 text-slate-300">{m.auc.toFixed(2)}%</td>
                  <td className="py-3 pr-4 text-slate-400">{m.params.toFixed(1)}M</td>
                  <td className="py-3 text-slate-400">{m.inference.toFixed(1)}ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bar Chart */}
      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        <div className="glass rounded-2xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Accuracy Comparison</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={METRICS_DATA} margin={{ left: -15 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis domain={[40, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                labelStyle={{ color: "#e2e8f0" }}
                itemStyle={{ color: "#94a3b8" }}
              />
              <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
              <Bar dataKey="Accuracy" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Balanced Acc" fill="#818cf8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Macro F1" fill="#c084fc" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div className="glass rounded-2xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">EfficientNet-B4 vs Dual-Branch V2</h2>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={RADAR_DATA}>
              <PolarGrid stroke="#1e293b" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[40, 100]} tick={{ fill: "#64748b", fontSize: 9 }} />
              <Radar name="EfficientNet-B4" dataKey="EfficientNet-B4" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.2} />
              <Radar name="Dual-Branch V2" dataKey="Dual-Branch V2" stroke="#ef4444" fill="#ef4444" fillOpacity={0.15} />
              <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                labelStyle={{ color: "#e2e8f0" }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Class-Level Performance */}
      <div className="glass rounded-2xl p-6 mb-8">
        <h2 className="text-lg font-semibold text-white mb-1">Per-Class Performance (EfficientNet-B4 / V2)</h2>
        <p className="text-slate-500 text-sm mb-4">Classification report from Dual-Branch V2 on the test split</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 text-left border-b border-slate-800">
                <th className="pb-3 pr-4">Class</th>
                <th className="pb-3 pr-4">Precision</th>
                <th className="pb-3 pr-4">Recall</th>
                <th className="pb-3 pr-4">F1-Score</th>
                <th className="pb-3">Test Samples</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {CLASS_METRICS.map((c) => (
                <tr key={c.cls}>
                  <td className="py-3 pr-4 font-medium text-white">{c.cls}</td>
                  <td className="py-3 pr-4">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-slate-800 rounded-full">
                        <div className="h-full bg-sky-400 rounded-full" style={{ width: `${c.precision}%` }} />
                      </div>
                      <span className="text-slate-300">{c.precision}%</span>
                    </div>
                  </td>
                  <td className="py-3 pr-4">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-slate-800 rounded-full">
                        <div className="h-full bg-purple-400 rounded-full" style={{ width: `${c.recall}%` }} />
                      </div>
                      <span className="text-slate-300">{c.recall}%</span>
                    </div>
                  </td>
                  <td className="py-3 pr-4 text-slate-300">{c.f1}%</td>
                  <td className="py-3 text-slate-400">{c.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Fusion Collapse Note */}
      <div className="glass rounded-2xl p-6">
        <h2 className="text-lg font-semibold text-white mb-3">Key Research Finding: Fusion Collapse</h2>
        <p className="text-slate-400 text-sm leading-relaxed mb-4">
          During Dual-Branch CNN evaluation, the attention gate consistently collapsed toward the Structure branch
          (DenseNet-derived, 79% weight) and suppressed the Texture branch (WideResNet-derived, 21% weight).
          This occurred across all 3 seeds in V1 and persisted even after the fusion redesign in V2,
          indicating that the optimizer actively rejected the textural features as unhelpful.
        </p>
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: "Structure Gate Mean (V2)", value: "90.8%", color: "rose" },
            { label: "Texture Gate Mean (V2)", value: "34.6%", color: "sky" },
            { label: "Collapse Rate (V2)", value: "53.5%", color: "amber" },
          ].map((s) => (
            <div key={s.label} className="bg-slate-800/50 rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-white mb-1">{s.value}</p>
              <p className="text-slate-400 text-xs">{s.label}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
