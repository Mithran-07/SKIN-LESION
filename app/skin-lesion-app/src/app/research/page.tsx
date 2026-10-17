import {
  Database,
  GitBranch,
  Layers,
  Shuffle,
  ShieldCheck,
  Microscope,
  CheckCircle2,
} from "lucide-react";
import {
  DATASET,
  PREPROCESSING_STEPS,
  OOD_EVALUATION_V2,
  SAMPLE_CASES,
} from "@/lib/constants";

const CATEGORY_COLOR: Record<string, string> = {
  Malignant: "text-status-critical border-status-critical/30 bg-status-critical/10",
  "Pre-malignant": "text-status-warning border-status-warning/30 bg-status-warning/10",
  Benign: "text-status-benign border-status-benign/30 bg-status-benign/10",
};

const maxCount = Math.max(...DATASET.classes.map((c) => c.count));

export default function DatasetPage() {
  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="font-headline-md text-xl font-bold text-on-surface">
          Dataset, Split Integrity &amp; OOD Calibration
        </h1>
        <p className="font-body-sm text-sm text-on-surface-variant mt-0.5">
          Machine-verified HAM10000 partition integrity, histopathology provenance, and Calibrated
          OOD Domain Gate v2 evaluation.
        </p>
      </div>

      {/* Overview Stats */}
      <div className="grid sm:grid-cols-4 gap-4">
        {[
          {
            Icon: Database,
            label: "Total Dermoscopic Images",
            value: DATASET.total.toLocaleString(),
            sub: "HAM10000 Canonical Archive",
          },
          {
            Icon: Layers,
            label: "Unique Tracked Lesions",
            value: DATASET.uniqueLesions.toLocaleString(),
            sub: "1.34 images / lesion mean",
          },
          {
            Icon: GitBranch,
            label: "Cross-Split Lesion Overlap",
            value: "0 Lesions",
            sub: "100% Lesion-Disjoint Splits",
          },
          {
            Icon: ShieldCheck,
            label: "OOD v2 Held-Out AUROC",
            value: OOD_EVALUATION_V2.auroc.toFixed(4),
            sub: "N=1,510 ID vs N=200 OOD",
          },
        ].map(({ Icon, label, value, sub }) => (
          <div
            key={label}
            className="flex items-start gap-3.5 p-4 rounded-xl bg-surface-container border border-outline-variant/20"
          >
            <Icon className="w-6 h-6 text-primary opacity-85 shrink-0 mt-0.5" aria-hidden="true" />
            <div>
              <div className="font-technical-data text-lg font-bold text-primary">{value}</div>
              <div className="font-technical-label text-[11px] text-on-surface font-semibold">
                {label}
              </div>
              <div className="font-technical-data text-[10px] text-on-surface-variant mt-0.5">
                {sub}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Lesion-Disjoint Split Verification Table */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
            <Shuffle className="w-4 h-4 text-primary" aria-hidden="true" /> Lesion-Disjoint 70 / 15 /
            15 Partition Verification
          </div>
          <span className="font-technical-data text-[11px] text-status-benign flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5" /> Train∩Val = 0 · Train∩Test = 0 · Val∩Test = 0
          </span>
        </div>
        <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
          HAM10000 contains 10,015 images derived from 7,470 unique{" "}
          <code className="text-on-surface">lesion_id</code> records. Because multiple serial or
          multi-magnification photographs exist for the same lesion, naive image-level splitting
          leaks identical lesions into both training and test partitions. Our stratified grouping
          places all photographs of a given <code className="text-on-surface">lesion_id</code>{" "}
          strictly into a single partition.
        </p>
        <div className="grid sm:grid-cols-3 gap-3">
          {[
            {
              title: "Training Split (splits/train.csv)",
              pct: `${DATASET.splits.train.pct}%`,
              images: DATASET.splits.train.images.toLocaleString(),
              lesions: DATASET.splits.train.lesions.toLocaleString(),
              role: "Backbone fine-tuning + OOD v2 prototype & Mahalanobis fitting",
              cls: "text-primary border-primary/30 bg-primary/5",
            },
            {
              title: "Validation Split (splits/val.csv)",
              pct: `${DATASET.splits.val.pct}%`,
              images: DATASET.splits.val.images.toLocaleString(),
              lesions: DATASET.splits.val.lesions.toLocaleString(),
              role: "Early stopping + OOD v2 threshold selection (τ = 0.2500)",
              cls: "text-secondary border-secondary/30 bg-secondary/5",
            },
            {
              title: "Held-Out Test Split (splits/test.csv)",
              pct: `${DATASET.splits.test.pct}%`,
              images: DATASET.splits.test.images.toLocaleString(),
              lesions: DATASET.splits.test.lesions.toLocaleString(),
              role: "Untouched evaluation for 6 models & OOD v2 test benchmark",
              cls: "text-status-benign border-status-benign/30 bg-status-benign/5",
            },
          ].map((s) => (
            <div key={s.title} className={`p-4 rounded-xl border space-y-1.5 ${s.cls}`}>
              <div className="flex items-baseline justify-between">
                <span className="font-headline-md text-2xl font-bold">{s.pct}</span>
                <span className="font-technical-data text-xs font-semibold">
                  {s.images} imgs · {s.lesions} lesions
                </span>
              </div>
              <div className="font-technical-label text-[11px] text-on-surface font-bold">
                {s.title}
              </div>
              <div className="font-body-sm text-[11px] text-on-surface-variant">{s.role}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Class Distribution with Split Breakdown */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high flex flex-wrap items-center justify-between gap-2">
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            7-Class Distribution Across Partitions (Severe 58.3 : 1 Imbalance)
          </span>
          <span className="font-technical-data text-[11px] text-on-surface-variant">
            Canonical Order: akiec (0), bcc (1), bkl (2), df (3), mel (4), nv (5), vasc (6)
          </span>
        </div>
        <div className="p-4 space-y-3.5">
          {DATASET.classes.map((cls) => (
            <div key={cls.code} className="space-y-1">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span
                    className={`font-technical-label text-[11px] font-bold px-1.5 py-0.5 rounded border shrink-0 ${
                      CATEGORY_COLOR[cls.category]
                    }`}
                  >
                    {cls.code}
                  </span>
                  <span className="font-body-sm text-xs text-on-surface font-medium">
                    {cls.name}
                  </span>
                </div>
                <div className="font-technical-data text-[11px] text-on-surface-variant">
                  Total: <strong className="text-on-surface">{cls.count.toLocaleString()}</strong> (
                  {cls.pct}%) · Train: {cls.train} · Val: {cls.val} · Test: {cls.test}
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
                  className="h-full bg-primary/75 rounded-full"
                  style={{ width: `${(cls.count / maxCount) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Histopathology Ground Truth Verification Modalities */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5 space-y-3">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
          <Microscope className="w-4 h-4 text-primary" aria-hidden="true" /> Diagnostic Ground-Truth
          Provenance (dx_type)
        </div>
        <div className="grid sm:grid-cols-4 gap-3">
          {DATASET.dxVerification.map((d) => (
            <div
              key={d.method}
              className="p-3.5 rounded-lg bg-surface-container-low border border-outline-variant/15 space-y-1"
            >
              <div className="font-technical-data text-base font-bold text-primary">
                {d.pct.toFixed(1)}%
              </div>
              <div className="font-technical-label text-[11px] text-on-surface font-semibold">
                {d.method}
              </div>
              <div className="font-technical-data text-[10px] text-on-surface-variant">
                {d.count.toLocaleString()} images
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Calibrated OOD Domain Gate v2 Section */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-primary" aria-hidden="true" />
            <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
              Calibrated Multi-Signal OOD Domain Gate v2.0 — Held-Out Evaluation
            </span>
          </div>
          <span className="font-technical-data text-[11px] text-primary font-semibold">
            AUROC: {OOD_EVALUATION_V2.auroc} (95% CI: {OOD_EVALUATION_V2.aurocCi95[0]}–
            {OOD_EVALUATION_V2.aurocCi95[1]})
          </span>
        </div>

        <div className="p-5 space-y-5">
          {/* Before vs After OOD Remediation */}
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl border border-status-warning/30 bg-status-warning/5 space-y-1.5">
              <div className="font-technical-label text-[11px] text-status-warning font-bold uppercase">
                Legacy OOD Gate v1 (Remediated Audit Finding)
              </div>
              <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                Built from only 7 demo files in <code className="text-on-surface">data/samples/</code>{" "}
                (which were matplotlib grid crops with white borders) and evaluated on the exact
                same 7 images (`N = 22` total). Produced circular 1.00 AUROC on prototypes while
                suffering a <strong>55.89% false rejection rate</strong> on real HAM10000 test
                images.
              </p>
            </div>
            <div className="p-4 rounded-xl border border-status-benign/30 bg-status-benign/5 space-y-1.5">
              <div className="font-technical-label text-[11px] text-status-benign font-bold uppercase">
                Calibrated OOD Gate v2.0 (Strict Train/Val/Test Separation)
              </div>
              <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                Fitted on <code className="text-on-surface">splits/train.csv (N=2,123)</code> using
                49 class/sub-cluster prototypes, diagonal Mahalanobis precision, and logit free
                energy. Threshold (<code className="text-on-surface">τ = 0.2500</code>) selected on{" "}
                <code className="text-on-surface">splits/val.csv (N=1,503)</code>. Evaluated once on{" "}
                <code className="text-on-surface">splits/test.csv (N=1,510 ID + 200 OOD)</code>:{" "}
                <strong>98.08% ID Acceptance</strong> (1.92% FRR) &amp;{" "}
                <strong>100.0% OOD Rejection</strong>.
              </p>
            </div>
          </div>

          {/* 10-Category OOD Breakdown */}
          <div className="overflow-x-auto rounded-lg border border-outline-variant/15">
            <table className="w-full text-left font-technical-data text-xs">
              <thead className="bg-surface-container-lowest text-on-surface-variant uppercase text-[10px] border-b border-outline-variant/15">
                <tr>
                  <th className="py-2 px-3">OOD Test Category (Held-Out N = 200)</th>
                  <th className="py-2 px-3">Samples</th>
                  <th className="py-2 px-3">Rejected</th>
                  <th className="py-2 px-3">Rejection Rate</th>
                  <th className="py-2 px-3">Mean Gate Score (τ = 0.2500)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/10">
                {OOD_EVALUATION_V2.categories.map((cat) => (
                  <tr key={cat.name}>
                    <td className="py-2 px-3 font-semibold text-on-surface">{cat.name}</td>
                    <td className="py-2 px-3">{cat.total}</td>
                    <td className="py-2 px-3 text-status-benign font-bold">{cat.rejected}</td>
                    <td className="py-2 px-3 text-status-benign font-bold">
                      {cat.rate.toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-on-surface-variant">
                      {cat.meanScore.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Verified Held-Out Demo Sample Provenance */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high">
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Verified Held-Out Demo Cohort (Genuine 600×450 HAM10000 Test Images)
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left font-technical-data text-xs">
            <thead className="bg-surface-container-lowest text-on-surface-variant uppercase text-[10px] border-b border-outline-variant/15">
              <tr>
                <th className="py-2 px-3">Class</th>
                <th className="py-2 px-3">ISIC Image ID</th>
                <th className="py-2 px-3">Lesion ID</th>
                <th className="py-2 px-3">Verification</th>
                <th className="py-2 px-3">Anatomical Site</th>
                <th className="py-2 px-3">Patient</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/10">
              {SAMPLE_CASES.map((s) => (
                <tr key={s.code}>
                  <td className="py-2 px-3 font-bold text-primary">{s.code}</td>
                  <td className="py-2 px-3 font-semibold text-on-surface">{s.isicId}</td>
                  <td className="py-2 px-3 text-on-surface-variant">{s.lesionId}</td>
                  <td className="py-2 px-3">{s.dxType}</td>
                  <td className="py-2 px-3">{s.localization}</td>
                  <td className="py-2 px-3 text-on-surface-variant">
                    {s.sex}, {s.age}y
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Preprocessing Pipeline */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high">
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            Deterministic Evaluation &amp; Live Inference Preprocessing Pipeline
          </span>
        </div>
        <div className="p-4 space-y-0">
          {PREPROCESSING_STEPS.map((step, i) => (
            <div
              key={step.label}
              className="flex items-center gap-4 py-3 border-b border-outline-variant/10 last:border-0"
            >
              <span className="w-6 h-6 rounded-full bg-primary/15 text-primary text-[10px] font-bold font-technical-label flex items-center justify-center shrink-0">
                {i + 1}
              </span>
              <div className="min-w-0">
                <div className="font-technical-label text-xs text-on-surface">{step.label}</div>
                <div className="font-technical-data text-[11px] text-on-surface-variant break-all">
                  {step.value}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
