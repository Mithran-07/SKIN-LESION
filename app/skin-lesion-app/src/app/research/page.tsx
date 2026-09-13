import { Database, GitBranch, Layers, Shuffle } from "lucide-react";
import { DATASET, PREPROCESSING_STEPS } from "@/lib/constants";

const CATEGORY_COLOR: Record<string, string> = {
  Malignant:    "text-status-critical border-status-critical/30 bg-status-critical/10",
  "Pre-malignant": "text-status-warning border-status-warning/30 bg-status-warning/10",
  Benign:       "text-status-benign border-status-benign/30 bg-status-benign/10",
};

const maxCount = Math.max(...DATASET.classes.map((c) => c.count));

export default function DatasetPage() {
  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-5xl mx-auto space-y-8">

      {/* Header */}
      <div>
        <h1 className="font-headline-md text-xl font-bold text-on-surface">Dataset & Methodology</h1>
        <p className="font-body-sm text-sm text-on-surface-variant mt-0.5">
          HAM10000 dataset structure, class distribution, patient-level splitting, and preprocessing pipeline.
        </p>
      </div>

      {/* Overview stats */}
      <div className="grid sm:grid-cols-3 gap-4">
        {[
          { Icon: Database,  label: "Total Images",    value: DATASET.total.toLocaleString() },
          { Icon: Layers,    label: "Classes",          value: "7" },
          { Icon: GitBranch, label: "Split Strategy",  value: "Patient-Aware" },
        ].map(({ Icon, label, value }) => (
          <div key={label} className="flex items-center gap-4 p-4 rounded-xl bg-surface-container border border-outline-variant/20">
            <Icon className="w-8 h-8 text-primary opacity-80 shrink-0" aria-hidden="true" />
            <div>
              <div className="font-headline-md text-xl font-bold text-primary">{value}</div>
              <div className="font-technical-label text-[11px] text-on-surface-variant">{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Class Distribution */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high flex items-center justify-between">
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Class Distribution
          </span>
          <span className="font-technical-data text-[11px] text-on-surface-variant">
            N = {DATASET.total.toLocaleString()} images
          </span>
        </div>
        <div className="p-4 space-y-3">
          {DATASET.classes.map((cls) => (
            <div key={cls.code} className="space-y-1">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className={`font-technical-label text-[11px] font-bold px-1.5 py-0.5 rounded border shrink-0 ${CATEGORY_COLOR[cls.category]}`}
                  >
                    {cls.code}
                  </span>
                  <span className="font-body-sm text-xs text-on-surface truncate">{cls.name}</span>
                </div>
                <div className="font-technical-data text-[11px] text-on-surface-variant whitespace-nowrap">
                  {cls.count.toLocaleString()} ({cls.pct}%)
                </div>
              </div>
              <div
                className="h-1.5 w-full bg-surface-container-lowest rounded-full overflow-hidden"
                role="progressbar"
                aria-valuenow={cls.pct}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`${cls.code} proportion`}
              >
                <div
                  className="h-full bg-primary/70 rounded-full"
                  style={{ width: `${(cls.count / maxCount) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
        <div className="px-4 pb-3 flex gap-4 text-[10px] font-technical-label">
          <span className="text-status-critical">■ Malignant</span>
          <span className="text-status-warning">■ Pre-malignant</span>
          <span className="text-status-benign">■ Benign</span>
        </div>
      </div>

      {/* Class imbalance note */}
      <div className="p-4 rounded-xl bg-surface-container border border-outline-variant/15 font-body-sm text-xs text-on-surface-variant leading-relaxed">
        <strong className="text-on-surface">Class Imbalance:</strong>{" "}
        NV accounts for 66.9% of the dataset (6,705 images) vs DF at 1.1% (115 images) — a{" "}
        <strong className="text-on-surface">58× ratio</strong>. WeightedRandomSampler and focal
        loss are used to mitigate this during training. Rare classes (BCC, DF, VASC) remain the
        most challenging for all evaluated architectures — this is expected and documented.
      </div>

      {/* Patient-level split */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-4">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
          <Shuffle className="w-4 h-4 text-primary" aria-hidden="true" /> Patient-Level Splitting
        </div>
        <p className="font-body-sm text-sm text-on-surface-variant leading-relaxed">
          HAM10000 contains multiple images per patient/lesion. A naive random split would allow
          the same patient&apos;s images to appear in both train and test sets, causing data leakage
          and artificially inflated test performance. We apply{" "}
          <strong className="text-on-surface">patient-aware stratified splitting</strong> to ensure
          no patient identity crosses partition boundaries.
        </p>
        <div className="grid grid-cols-3 gap-3">
          {[
            ["Train",      "70%", "text-primary      border-primary/30      bg-primary/10"],
            ["Validation", "15%", "text-secondary    border-secondary/30    bg-secondary/10"],
            ["Test",       "15%", "text-status-benign border-status-benign/30 bg-status-benign/10"],
          ].map(([label, val, cls]) => (
            <div key={label} className={`text-center p-4 rounded-xl border ${cls}`}>
              <div className="font-headline-md text-2xl font-bold">{val}</div>
              <div className="font-technical-label text-[11px] mt-1">{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Preprocessing pipeline */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high">
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Evaluation Preprocessing Pipeline
          </span>
        </div>
        <div className="p-4 space-y-0">
          {PREPROCESSING_STEPS.map((step, i) => (
            <div
              key={i}
              className="flex items-center gap-4 py-3 border-b border-outline-variant/10 last:border-0"
            >
              <span className="w-6 h-6 rounded-full bg-primary/15 text-primary text-[10px] font-bold font-technical-label flex items-center justify-center shrink-0">
                {i + 1}
              </span>
              <div className="min-w-0">
                <div className="font-technical-label text-xs text-on-surface">{step.label}</div>
                <div className="font-technical-data text-[11px] text-on-surface-variant break-all">{step.value}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="px-4 py-2.5 border-t border-outline-variant/10 bg-surface-container-low font-technical-data text-[10px] text-on-surface-variant">
          Training uses additional augmentations: RandomResizedCrop, HorizontalFlip, ColorJitter,
          ElasticTransform, HairAugmentation. Evaluation uses the deterministic pipeline above only.
        </div>
      </div>
    </div>
  );
}
