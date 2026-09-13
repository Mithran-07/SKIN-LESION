import { Cpu, GitMerge, FlaskConical, Award, HelpCircle } from "lucide-react";
import { BENCHMARK_DATA } from "@/lib/constants";

const champion  = BENCHMARK_DATA.find((m) => m.isChampion)!;
const dualBranch = BENCHMARK_DATA.filter((m) => m.name.includes("Dual-Branch"));
const bestDB     = dualBranch.reduce((a, b) => (a.auc > b.auc ? a : b), dualBranch[0]);

export default function ArchitecturePage() {
  return (
    <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-5xl mx-auto space-y-8">

      {/* Header */}
      <div>
        <h1 className="font-headline-md text-xl font-bold text-on-surface">Architecture Deep Dive</h1>
        <p className="font-body-sm text-sm text-on-surface-variant mt-0.5">
          Baseline models, the proposed Dual-Branch architecture, and empirical findings.
        </p>
      </div>

      {/* Research Question */}
      <div className="p-5 rounded-xl bg-research-violet/5 border border-research-violet/20 space-y-2">
        <div className="font-technical-label text-xs text-research-violet uppercase tracking-wider flex items-center gap-1.5">
          <HelpCircle className="w-4 h-4" aria-hidden="true" /> Research Question
        </div>
        <p className="font-body-lg text-base text-on-surface leading-relaxed">
          Can separating local <em>texture representation</em> from macroscopic{" "}
          <em>structural representation</em> improve multi-class dermoscopic classification over
          compound-scaled single-branch networks?
        </p>
      </div>

      {/* Baseline Architectures */}
      <section className="space-y-4">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
          <Cpu className="w-4 h-4 text-primary" aria-hidden="true" /> Baseline Architectures
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          {[
            {
              name: "ResNet-50",
              params: "23.52M",
              auc: 93.52,
              desc: "Residual skip connections enabling deep network training without vanishing gradients. 50 layers.",
              champion: false,
            },
            {
              name: "DenseNet-121",
              params: "6.96M",
              auc: 95.31,
              desc: "Dense feature reuse — each layer receives feature maps from all preceding layers.",
              champion: false,
            },
            {
              name: "EfficientNet-B4",
              params: "17.56M",
              auc: 95.92,
              desc: "Compound scaling of depth, width, and resolution via neural architecture search (NAS).",
              champion: true,
            },
          ].map((b) => (
            <div
              key={b.name}
              className={`p-4 rounded-xl border space-y-2 ${
                b.champion
                  ? "border-primary/40 bg-primary/5"
                  : "border-outline-variant/20 bg-surface-container"
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="font-technical-label text-sm font-bold text-on-surface">{b.name}</span>
                {b.champion && (
                  <Award className="w-3.5 h-3.5 text-primary" aria-label="Best evaluated" />
                )}
              </div>
              <div className="font-technical-data text-[10px] text-on-surface-variant">
                {b.params} params · {b.auc}% ROC-AUC
              </div>
              <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">{b.desc}</p>
              {b.champion && (
                <div className="font-technical-data text-[11px] text-primary mt-1">
                  ✓ Selected as deployed inference model
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Dual-Branch Architecture */}
      <section className="space-y-4">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
          <GitMerge className="w-4 h-4 text-research-violet" aria-hidden="true" /> Proposed: Decoupled Dual-Branch CNN
        </div>

        {/* ASCII diagram */}
        <div className="bg-surface-container rounded-xl border border-outline-variant/20 p-5">
          <pre className="font-technical-data text-[11px] text-on-surface-variant leading-relaxed overflow-x-auto">{`
Dermoscopic Image (224×224 RGB)
          │
  Preprocessing
(Resize 256 → CenterCrop 224)
          │
  ┌───────┴───────┐
  │               │
Texture        Structure
 Branch          Branch
(Shallow/Wide)  (Deep/Narrow)
  1024-dim        256-dim
  │               │
  └───────┬───────┘
          │
  Attention-Gated
    Feature Fusion
          │
   Classifier Head
          │
  7-Class Softmax
(AKIEC/BCC/BKL/DF/MEL/NV/VASC)
`.trim()}</pre>
        </div>

        <div className="grid sm:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-surface-container border border-outline-variant/20 space-y-2">
            <div className="font-technical-label text-xs text-on-surface">Texture Branch (Shallow / Wide)</div>
            <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
              Captures high-frequency local patterns — dermoscopic texture, pigmentation
              microstructure, and hair artifacts. Shallower layers preserve fine spatial detail.
              Output: 1,024-dimensional embedding.
            </p>
          </div>
          <div className="p-4 rounded-xl bg-surface-container border border-outline-variant/20 space-y-2">
            <div className="font-technical-label text-xs text-on-surface">Structure Branch (Deep / Narrow)</div>
            <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
              Captures macroscopic morphological representations — lesion boundary regularity,
              symmetry, and global shape characteristics. Deeper layers with reduced channel width.
              Output: 256-dimensional embedding.
            </p>
          </div>
        </div>
      </section>

      {/* Empirical Finding */}
      <section className="p-5 rounded-xl bg-surface-container border border-outline-variant/15 space-y-4">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-primary" aria-hidden="true" /> Empirical Finding
        </div>

        <div className="grid sm:grid-cols-2 gap-6">
          <div className="space-y-2">
            <div className="font-technical-label text-sm font-bold text-research-violet">
              Dual-Branch CNN (Best Seed)
            </div>
            <div className="font-technical-data text-xs text-on-surface-variant space-y-0.5">
              <div>ROC-AUC: {bestDB?.auc}%</div>
              <div>Balanced Accuracy: {bestDB?.balAcc}%</div>
              <div>Test Accuracy: {bestDB?.acc}%</div>
              <div>Macro F1: {bestDB?.f1}%</div>
              <div>Latency (LOQ CUDA): {bestDB?.latency} ms</div>
              <div>VRAM: {bestDB?.vram}</div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="font-technical-label text-sm font-bold text-primary">
              EfficientNet-B4 (Best Evaluated)
            </div>
            <div className="font-technical-data text-xs text-on-surface-variant space-y-0.5">
              <div>ROC-AUC: {champion.auc}%</div>
              <div>Balanced Accuracy: {champion.balAcc}%</div>
              <div>Test Accuracy: {champion.acc}%</div>
              <div>Macro F1: {champion.f1}%</div>
              <div>Latency (LOQ CUDA): {champion.latency} ms</div>
              <div>VRAM: {champion.vram}</div>
            </div>
          </div>
        </div>

        <div className="border-t border-outline-variant/15 pt-4 font-body-sm text-xs text-on-surface-variant leading-relaxed space-y-2">
          <p>
            The Dual-Branch architecture achieved competitive AUC (~{bestDB?.auc}%) but did not
            surpass EfficientNet-B4 (95.92%). Fusion gate diagnostics revealed the attention
            mechanism predominantly weighted the structural branch, with texture branch gradients
            showing higher variance — suggesting the two-branch decoupling did not produce
            independently complementary representations under this training regime.
          </p>
          <p>
            The pre-trained ImageNet compound scaling in EfficientNet-B4 provided richer
            dermoscopic feature representations out-of-the-box. This is a legitimate,
            documented research outcome — <strong className="text-on-surface">not a software failure</strong>.
          </p>
          <p className="text-research-violet">
            The Dual-Branch CNN remains the core research hypothesis investigated in this project.
            Its results provide directional insights into feature decoupling strategies for
            dermoscopic classification.
          </p>
        </div>
      </section>

      {/* Findings summary */}
      <section className="space-y-3">
        <div className="font-technical-label text-xs text-on-surface uppercase tracking-wider">
          Research Findings Summary
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          {[
            {
              color: "border-status-benign/30 bg-status-benign/5",
              label: "Confirmed",
              items: [
                "EfficientNet-B4 compound scaling effectively captures multi-scale dermoscopic patterns.",
                "Patient-level stratified splitting is essential for honest evaluation on HAM10000.",
                "Class imbalance correction (WeightedRandomSampler + focal loss) improves rare-class recall.",
                "Grad-CAM attribution is clinically interpretable for the dominant lesion region.",
              ],
            },
            {
              color: "border-research-violet/30 bg-research-violet/5",
              label: "Open Questions",
              items: [
                "Whether explicit texture–structure decoupling can outperform compound scaling with more training data or different fusion designs.",
                "Whether OOD detection via feature-space similarity is robust to non-dermoscopic skin photographs (close-up macro shots).",
                "Rare class (DF, VASC) performance — larger datasets or specific augmentation strategies may help.",
              ],
            },
          ].map(({ color, label, items }) => (
            <div key={label} className={`p-4 rounded-xl border ${color} space-y-2`}>
              <div className="font-technical-label text-xs text-on-surface">{label}</div>
              <ul className="space-y-1.5">
                {items.map((item, i) => (
                  <li key={i} className="font-body-sm text-xs text-on-surface-variant flex gap-2">
                    <span className="mt-0.5 shrink-0">·</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
