import { Cpu, GitMerge, FlaskConical, Award, HelpCircle, Layers, Code2 } from "lucide-react";
import { BENCHMARK_DATA } from "@/lib/constants";

const champion = BENCHMARK_DATA.find((m) => m.id === "efficientnet_b4")!;
const dbV1Seed123 = BENCHMARK_DATA.find((m) => m.id === "dual_branch_seed123")!;
const dbV2Seed42 = BENCHMARK_DATA.find((m) => m.id === "dual_branch_seed42")!;

const SUBMODULE_BREAKDOWN = [
  {
    module: "ShallowWideBranch (Texture)",
    topology: "Conv(3→256) → Conv(256→512, Pool) → Conv(512→1024) → GAP(1024)",
    v1Params: "6,035,456",
    v2Params: "6,035,456",
    pretraining: "0% transferred (Kaiming normal init; 64ch vs 256ch mismatch)",
  },
  {
    module: "DeepNarrowBranch (Structure)",
    topology: "7×7 Stem(64) → 4 NarrowResStages (64→128→128→128) → 1×1 Conv(256) → GAP(256)",
    v1Params: "1,961,216",
    v2Params: "1,961,216",
    pretraining: "0.49% transferred (9,536 stem params from DenseNet-121 conv0/norm0)",
  },
  {
    module: "AttentionFusionHead (Gate + MLP)",
    topology: "v1: Linear(1280→1280) gate | v2: Linear(1280→1) gate + MLP(1280→512→256)",
    v1Params: "2,671,104",
    v2Params: "1,032,706",
    pretraining: "0% transferred (Xavier uniform init)",
  },
  {
    module: "Classification Head",
    topology: "Dropout(0.3) → Linear(256 → 7)",
    v1Params: "1,799",
    v2Params: "1,799",
    pretraining: "0% transferred (Xavier uniform init)",
  },
];

export default function ArchitecturePage() {
  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="font-headline-md text-xl font-bold text-on-surface">
          Architecture Comparator &amp; Fusion Topology Audit
        </h1>
        <p className="font-body-sm text-sm text-on-surface-variant mt-0.5">
          Structural analysis of compound-scaled baselines versus custom Decoupled Dual-Branch CNN{" "}
          <code className="text-on-surface">v1_matrix</code> and{" "}
          <code className="text-on-surface">v2_scalar</code> variants.
        </p>
      </div>

      {/* Research Question */}
      <div className="p-5 rounded-xl bg-research-violet/5 border border-research-violet/20 space-y-2">
        <div className="font-technical-label text-xs text-research-violet uppercase tracking-wider flex items-center gap-1.5">
          <HelpCircle className="w-4 h-4" aria-hidden="true" /> Primary Architectural Hypothesis
        </div>
        <p className="font-body-lg text-base text-on-surface leading-relaxed">
          Can physically decoupling high-frequency local <em>texture representations</em>{" "}
          (Shallow-Wide branch, 1,024-dim) from macroscopic <em>morphological structure</em>{" "}
          (Deep-Narrow branch, 256-dim) via learned sigmoid attention gating outperform single-stream
          compound-scaled backbones on HAM10000?
        </p>
      </div>

      {/* Pretrained Baselines */}
      <section className="space-y-3">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
          <Cpu className="w-4 h-4 text-primary" aria-hidden="true" /> ImageNet-Pretrained Single-Stream
          Baselines (100% Backbone Transfer)
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          {[
            {
              name: "EfficientNet-B4 (Champion)",
              params: "17,560,655 (17.56M)",
              balAcc: "79.23%",
              auc: "95.91%",
              f1: "69.37%",
              mps: "27.64 ms",
              desc: "Compound scaling of depth, width, and resolution via MBConv inverted residual blocks with Squeeze-and-Excitation attention.",
              champion: true,
            },
            {
              name: "DenseNet-121 (Rank #2)",
              params: "6,961,031 (6.96M)",
              balAcc: "75.60%",
              auc: "95.32%",
              f1: "66.62%",
              mps: "27.51 ms",
              desc: "Dense block concatenation where each layer receives feature maps from all preceding layers, promoting multi-scale feature reuse.",
              champion: false,
            },
            {
              name: "ResNet-50 (Rank #3)",
              params: "23,522,375 (23.52M)",
              balAcc: "71.15%",
              auc: "93.52%",
              f1: "62.80%",
              mps: "16.01 ms",
              desc: "50-layer bottleneck residual network with identity skip connections and 2,048-dimensional penultimate feature pooling.",
              champion: false,
            },
          ].map((b) => (
            <div
              key={b.name}
              className={`p-4 rounded-xl border space-y-2.5 ${
                b.champion
                  ? "border-primary/40 bg-primary/5"
                  : "border-outline-variant/20 bg-surface-container"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-technical-label text-xs font-bold text-on-surface">
                  {b.name}
                </span>
                {b.champion && (
                  <Award className="w-4 h-4 text-primary shrink-0" aria-label="Champion" />
                )}
              </div>
              <div className="font-technical-data text-[11px] text-primary font-semibold">
                Bal. Acc: {b.balAcc} · ROC-AUC: {b.auc} · F1: {b.f1}
              </div>
              <div className="font-technical-data text-[10px] text-on-surface-variant">
                {b.params} · M4 MPS: {b.mps}
              </div>
              <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                {b.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Dual-Branch Architecture Diagram & Version Reconciliation */}
      <section className="space-y-4">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
          <GitMerge className="w-4 h-4 text-research-violet" aria-hidden="true" /> Decoupled
          Dual-Branch CNN: Explicit Fusion Versioning (v1_matrix vs. v2_scalar)
        </div>

        <div className="grid lg:grid-cols-12 gap-4">
          <div className="lg:col-span-6 bg-surface-container rounded-xl border border-outline-variant/20 p-5">
            <div className="font-technical-label text-[10px] text-on-surface-variant uppercase mb-2">
              Dual-Stream Feature Extraction &amp; Gated Fusion Topology
            </div>
            <pre className="font-technical-data text-[11px] text-on-surface-variant leading-relaxed overflow-x-auto">{`
Dermoscopic Input (B, 3, 224, 224)
         │
   ┌─────┴─────────────────────────┐
   ▼                               ▼
ShallowWideBranch              DeepNarrowBranch
(Texture / High-Freq)          (Structure / Macroscopic)
3 → 256 → 512(Pool) → 1024     7×7 Stem(64) → 4 NarrowStages → 256
GAP → f_tex (B, 1024)          GAP → f_struct (B, 256)
   │                               │
   └─────────────┬─────────────────┘
                 ▼
       z = [f_tex ‖ f_struct] ∈ ℝ^(B × 1280)
                 │
   ┌─────────────┴─────────────────┐
   ▼ (fusion_version="v1_matrix")  ▼ (fusion_version="v2_scalar")
Linear(1280 → 1280) + Sigmoid   Linear(1280 → 1) + Sigmoid
g = z ⊙ σ(W_m z + b_m)          α = σ(w_s^T z + b_s)
(1,639,680 gate params)         g = [α·f_tex ‖ (1-α)·f_struct]
   │                            (1,281 gate params)
   └─────────────┬─────────────────┘
                 ▼
     MLP(1280 → 512 → 256) + Classifier(256 → 7)
`.trim()}</pre>
          </div>

          <div className="lg:col-span-6 space-y-4">
            <div className="p-4 rounded-xl bg-surface-container border border-research-violet/30 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-technical-label text-xs font-bold text-research-violet">
                  Fusion Version 1: v1_matrix (Seeds 123 &amp; 999)
                </span>
                <span className="font-technical-data text-[11px] text-on-surface font-semibold">
                  10,669,639 Params
                </span>
              </div>
              <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                Uses a full <code className="text-on-surface">Linear(1280 → 1280)</code>{" "}
                feature-wise sigmoid gate (<code className="text-on-surface">1,639,680</code> gate
                parameters) allowing each of the 1,280 concatenated dimensions to be modulated
                independently. Achieved{" "}
                <strong className="text-on-surface">
                  {dbV1Seed123.auc}% Macro ROC-AUC / {dbV1Seed123.balAcc}% Balanced Accuracy
                </strong>{" "}
                on Seed 123.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-container border border-primary/30 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-technical-label text-xs font-bold text-primary">
                  Fusion Version 2: v2_scalar (Seed 42)
                </span>
                <span className="font-technical-data text-[11px] text-on-surface font-semibold">
                  9,031,241 Params
                </span>
              </div>
              <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                Compresses the gate to a single scalar{" "}
                <code className="text-on-surface">Linear(1280 → 1)</code>{" "}
                (<code className="text-on-surface">1,281</code> gate parameters, saving 1.64M
                parameters) that trades off the entire texture branch (<code className="text-on-surface">α</code>)
                against the structure branch (<code className="text-on-surface">1 - α</code>).
                Achieved{" "}
                <strong className="text-on-surface">
                  {dbV2Seed42.auc}% Macro ROC-AUC / {dbV2Seed42.balAcc}% Balanced Accuracy
                </strong>{" "}
                on Seed 42.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-container border border-outline-variant/20 space-y-1.5">
              <div className="font-technical-label text-xs text-on-surface flex items-center gap-1.5">
                <Code2 className="w-3.5 h-3.5 text-primary" /> Strict Checkpoint Compatibility
              </div>
              <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
                Both architectures are now explicitly supported via{" "}
                <code className="text-on-surface">
                  DualBranchNet(fusion_version=&quot;v1_matrix&quot; | &quot;v2_scalar&quot;)
                </code>{" "}
                and <code className="text-on-surface">DualBranchNet.from_checkpoint()</code> with{" "}
                <code className="text-on-surface">strict=True</code> state-dict verification.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Parameter & Pretraining Audit Table */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/20 overflow-hidden">
        <div className="p-4 border-b border-outline-variant/15 bg-surface-container-high flex items-center gap-2">
          <Layers className="w-4 h-4 text-primary" aria-hidden="true" />
          <span className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
            DualBranchNet Submodule Parameter &amp; Pretraining Audit
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left font-technical-data text-xs">
            <thead className="bg-surface-container-lowest text-on-surface-variant uppercase text-[10px] border-b border-outline-variant/15">
              <tr>
                <th className="py-2.5 px-3">Submodule</th>
                <th className="py-2.5 px-3">Layer Topology</th>
                <th className="py-2.5 px-3">v1_matrix Params</th>
                <th className="py-2.5 px-3">v2_scalar Params</th>
                <th className="py-2.5 px-3">ImageNet Pretrained Weight Transfer</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/10">
              {SUBMODULE_BREAKDOWN.map((row) => (
                <tr key={row.module}>
                  <td className="py-2.5 px-3 font-bold text-on-surface">{row.module}</td>
                  <td className="py-2.5 px-3 text-on-surface-variant">{row.topology}</td>
                  <td className="py-2.5 px-3">{row.v1Params}</td>
                  <td className="py-2.5 px-3">{row.v2Params}</td>
                  <td className="py-2.5 px-3 text-status-warning">{row.pretraining}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Root-Cause Scientific Analysis */}
      <section className="p-5 rounded-xl bg-surface-container border border-outline-variant/20 space-y-4">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-primary" aria-hidden="true" /> Root-Cause Analysis:
          Why Compound Scaling Outperformed Custom Dual-Branch CNN
        </div>
        <div className="grid sm:grid-cols-2 gap-6 font-body-sm text-xs text-on-surface-variant leading-relaxed">
          <div className="space-y-2">
            <div className="font-technical-label text-xs font-bold text-on-surface">
              1. 99.88% Scratch Initialization Deficit
            </div>
            <p>
              In <code className="text-on-surface">ShallowWideBranch</code>, the first convolution
              expands directly from 3 to 256 channels (`3×3`), preventing weight transfer from{" "}
              <code className="text-on-surface">WideResNet-50-2</code> (`64×3×7×7`). Consequently,
              all 6.04M texture branch parameters were initialized from scratch via Kaiming normal.
              In <code className="text-on-surface">DeepNarrowBranch</code>, only the 7×7 stem
              (`9,536` params) matched DenseNet-121, leaving{" "}
              <strong className="text-on-surface">99.88% of DualBranchNet parameters</strong> to be
              learned solely from 7,002 HAM10000 training images.
            </p>
          </div>
          <div className="space-y-2">
            <div className="font-technical-label text-xs font-bold text-on-surface">
              2. Early Global Average Pooling Bottleneck
            </div>
            <p>
              <code className="text-on-surface">ShallowWideBranch</code> applies{" "}
              <code className="text-on-surface">AdaptiveAvgPool2d(1, 1)</code> directly to a{" "}
              <code className="text-on-surface">112 × 112</code> feature map after only 3
              convolutional layers (effective receptive field `9×9` pixels), spatially averaging out
              localized pigment network asymmetries before fusion. By contrast,{" "}
              <strong className="text-primary">
                EfficientNet-B4 ({champion.balAcc}% Balanced Accuracy)
              </strong>{" "}
              leverages progressive multi-stage spatial reduction with pretrained Squeeze-and-Excitation
              channels.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
