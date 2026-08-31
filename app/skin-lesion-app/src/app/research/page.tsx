import { BookOpen, Database, PieChart, ShieldCheck, AlertCircle, Layers, CheckCircle2 } from "lucide-react";

const HAM10000_DISTRIBUTION = [
  { code: "NV", name: "Melanocytic Nevi", count: 6705, pct: "67.0%", type: "Benign", urgency: "Low", color: "bg-blue-500" },
  { code: "MEL", name: "Melanoma", count: 1113, pct: "11.1%", type: "Malignant", urgency: "Critical", color: "bg-rose-500" },
  { code: "BKL", name: "Benign Keratosis-like", count: 1099, pct: "11.0%", type: "Benign", urgency: "Low", color: "bg-amber-500" },
  { code: "BCC", name: "Basal Cell Carcinoma", count: 514, pct: "5.1%", type: "Malignant (NMSC)", urgency: "High", color: "bg-orange-500" },
  { code: "AKIEC", name: "Actinic Keratoses", count: 327, pct: "3.3%", type: "Pre-malignant", urgency: "High", color: "bg-purple-500" },
  { code: "VASC", name: "Vascular Lesions", count: 142, pct: "1.4%", type: "Benign", urgency: "Low", color: "bg-emerald-500" },
  { code: "DF", name: "Dermatofibroma", count: 115, pct: "1.1%", type: "Benign", urgency: "Low", color: "bg-teal-500" },
];

export default function ResearchPage() {
  return (
    <div className="space-y-10 max-w-5xl mx-auto">
      
      {/* Header */}
      <div className="border-b border-slate-800 pb-6">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold mb-2">
          <BookOpen className="w-3.5 h-3.5" />
          <span>Clinical & Experimental Methodology</span>
        </div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          HAM10000 Dataset & Clinical Context
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          A collection of 10,015 multi-source dermatoscopic images across 7 diagnostic categories.
        </p>
      </div>

      {/* Dataset Statistics Overview */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="glass-card rounded-2xl p-5 border border-slate-800 text-center">
          <div className="text-2xl sm:text-3xl font-extrabold text-emerald-400">10,015</div>
          <div className="text-xs text-slate-400 mt-1 font-medium">Total Images</div>
        </div>
        <div className="glass-card rounded-2xl p-5 border border-slate-800 text-center">
          <div className="text-2xl sm:text-3xl font-extrabold text-cyan-400">7</div>
          <div className="text-xs text-slate-400 mt-1 font-medium">Diagnostic Classes</div>
        </div>
        <div className="glass-card rounded-2xl p-5 border border-slate-800 text-center">
          <div className="text-2xl sm:text-3xl font-extrabold text-amber-400">58.3:1</div>
          <div className="text-xs text-slate-400 mt-1 font-medium">Imbalance Ratio (NV:DF)</div>
        </div>
        <div className="glass-card rounded-2xl p-5 border border-slate-800 text-center">
          <div className="text-2xl sm:text-3xl font-extrabold text-purple-400">70/15/15</div>
          <div className="text-xs text-slate-400 mt-1 font-medium">Patient Split (Tr/Val/Te)</div>
        </div>
      </div>

      {/* Class Distribution Breakdown */}
      <div className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <PieChart className="w-5 h-5 text-emerald-400" />
            <span>Class Distribution & Taxonomy</span>
          </h2>
          <span className="text-xs text-slate-400 font-mono">10,015 ground-truth samples</span>
        </div>

        {/* Stacked Percentage Bar */}
        <div className="h-4 w-full rounded-full overflow-hidden flex bg-slate-900 border border-slate-800">
          {HAM10000_DISTRIBUTION.map((c) => (
            <div
              key={c.code}
              className={`${c.color} h-full transition-all`}
              style={{ width: c.pct }}
              title={`${c.name} (${c.code}): ${c.pct}`}
            />
          ))}
        </div>

        {/* Breakdown Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-2.5 px-3">Code</th>
                <th className="py-2.5 px-3">Full Diagnosis</th>
                <th className="py-2.5 px-3">Category</th>
                <th className="py-2.5 px-3">Clinical Urgency</th>
                <th className="py-2.5 px-3 text-right">Sample Count</th>
                <th className="py-2.5 px-3 text-right">Proportion</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {HAM10000_DISTRIBUTION.map((c) => (
                <tr key={c.code} className="hover:bg-slate-800/30">
                  <td className="py-3 px-3">
                    <span className="font-bold text-white px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                      {c.code}
                    </span>
                  </td>
                  <td className="py-3 px-3 font-sans text-slate-200">{c.name}</td>
                  <td className="py-3 px-3 font-sans text-slate-300">{c.type}</td>
                  <td className="py-3 px-3 font-sans">
                    <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                      c.urgency === "Critical" ? "text-rose-400 bg-rose-500/10" :
                      c.urgency === "High" ? "text-amber-400 bg-amber-500/10" : "text-emerald-400 bg-emerald-500/10"
                    }`}>
                      {c.urgency}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">{c.count.toLocaleString()}</td>
                  <td className="py-3 px-3 text-right text-cyan-400 font-bold">{c.pct}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Methodological Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-cyan-400 font-bold text-sm">
            <ShieldCheck className="w-5 h-5" />
            <span>Patient-Aware Stratified Splitting</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Many patients in HAM10000 contribute multiple images of the same lesion or multiple lesions. 
            To prevent severe data leakage, our dataset pipeline partitions by <code>lesion_id</code> so all images belonging to the same lesion remain strictly within train, val, or test sets.
          </p>
        </div>

        <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
            <AlertCircle className="w-5 h-5" />
            <span>Focal Loss & Class Balancing</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Standard Cross-Entropy is dominated by the majority class (NV at 67%). We utilize a <strong>Focal Loss</strong> objective with inverse class frequency alpha-weighting and gamma focusing parameter (\(\gamma=2.0\)) to suppress easy negative loss and focus gradients on rare malignancies (DF, VASC, AKIEC).
          </p>
        </div>

      </div>

    </div>
  );
}
