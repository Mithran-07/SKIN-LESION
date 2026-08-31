export default function ResearchPage() {
  const sections = [
    {
      id: "problem",
      title: "1. Problem Statement",
      content: `Skin cancer is among the most prevalent cancers globally. Early dermoscopic classification can
significantly improve outcomes, yet visual diagnosis requires high expertise. We investigate whether
deep learning can reliably classify 7 dermoscopic lesion categories from HAM10000, and whether a
dual-branch architecture explicitly separating texture and structure extraction outperforms standard
single-branch baselines.`,
    },
    {
      id: "dataset",
      title: "2. Dataset",
      content: `HAM10000 (Human Against Machine with 10000 training images) is a publicly available collection of
10,015 dermoscopic images spanning 7 categories: Actinic Keratosis/Bowen's Disease (AKIEC), Basal Cell
Carcinoma (BCC), Benign Keratosis (BKL), Dermatofibroma (DF), Melanoma (MEL), Melanocytic Nevi (NV),
and Vascular Lesions (VASC). The dataset exhibits extreme class imbalance: NV accounts for ~67% of all
samples. Patient-level splits were enforced to prevent data leakage (70/15/15 train/val/test). All images
were resized to 224×224 and normalized using ImageNet statistics.`,
    },
    {
      id: "methodology",
      title: "3. Methodology",
      content: `All models were trained using AdamW optimizer (lr=1e-4, weight_decay=1e-2), cosine annealing
scheduler, and mixed-precision training (AMP). Class imbalance was addressed via square-root inverse-frequency
class weights and WeightedRandomSampler. Early stopping was applied with patience=5 on validation
balanced accuracy. Label smoothing (ε=0.1) was applied in V1.1 and V2. All random seeds were fixed (42,
123, 999) for reproducibility.`,
    },
    {
      id: "baselines",
      title: "4. Baseline Models",
      content: `Three standard CNN architectures were evaluated as baselines:
      
• ResNet50 (23.5M params): Residual connections, strong general feature extractor. Test Accuracy: 56.62%.
• DenseNet121 (7.0M params): Dense connections encouraging feature reuse. Test Accuracy: 66.36%.
• EfficientNet-B4 (17.6M params): Compound scaling of depth, width, resolution. Test Accuracy: 73.64%.

EfficientNet-B4 achieved the strongest performance across all metrics and was selected as the final
demonstration model.`,
    },
    {
      id: "dual_branch",
      title: "5. Dual-Branch Architecture",
      content: `The core research contribution is a dual-branch CNN that explicitly separates feature extraction
into two specialized pathways:

• Shallow-Wide Branch (WideResNet-50-2): 1024-dimensional features emphasizing localized texture patterns.
• Deep-Narrow Branch (DenseNet-121): 256-dimensional features emphasizing global lesion structure.

These representations are merged via an attention fusion gate. The hypothesis is that explicitly decoupling
texture and structural understanding would yield richer, more clinically-motivated representations.`,
    },
    {
      id: "experiments",
      title: "6. Experiments",
      content: `V1 — Baseline Dual-Branch: Original attention gate (1280×1280 projection). Tested on seeds 42,
123, 999. Mean accuracy: 54.79%. Fusion diagnostics revealed that the attention gate consistently collapsed
toward the Structure branch across all seeds.

V1.1 — Training Improvements: Applied square-root class weights, label smoothing (ε=0.1), reduced early
stopping patience (5). No architectural changes. Accuracy improved to 65.76%, but fusion collapse persisted.

V2 — Fusion Redesign: Replaced the 1280×1280 joint projection with two independent scalar gates (gate_t,
gate_s) to eliminate dimensional bias. Gates initialized in balance (0.51 / 0.48). After training, the
structure gate still dominated (0.91 vs 0.35), confirming the collapse is optimization-driven, not
architecture-driven.`,
    },
    {
      id: "results",
      title: "7. Results",
      content: `EfficientNet-B4 achieved the strongest performance under all evaluated conditions:
Accuracy 73.64%, Balanced Accuracy 79.16%, Macro F1 69.19%, ROC-AUC 95.92%.

The Dual-Branch CNN framework did not outperform EfficientNet-B4 in any configuration. The maximum
Dual-Branch accuracy (65.76% in V1.1) remained ~8 absolute percentage points below the baseline.
Cohen's d effect size vs EfficientNet-B4 for V1 accuracy: d = -23.28, indicating the gap is
statistically decisive and not attributable to random seed variance.`,
    },
    {
      id: "limitations",
      title: "8. Limitations",
      content: `• HAM10000 is a limited, single-center dataset with known demographic biases.
• Extreme class imbalance (NV ~67%) affects model calibration for rare classes.
• External clinical validation has not been performed — generalization to other imaging protocols is unknown.
• The Dual-Branch CNN did not outperform EfficientNet-B4.
• Predictions should not be treated as medical diagnosis under any circumstances.
• Grad-CAM highlights are model-attribution maps, not clinical evidence of pathology.`,
    },
    {
      id: "future",
      title: "9. Future Work",
      content: `• Dataset expansion: Multi-center dermoscopy datasets (ISIC 2020, BCN20000) to improve generalization.
• Grad-CAM clinical validation: Expert dermatologist review of explainability maps.
• Cross-attention fusion: Replace simple scalar gates with cross-attention between branch embeddings.
• Self-supervised pretraining: Use dermatology-domain pretraining instead of ImageNet weights.
• Conformal prediction: Add coverage-guaranteed uncertainty quantification to predictions.`,
    },
  ];

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-white mb-2">Research Story</h1>
        <p className="text-slate-400">
          The complete narrative behind the Dual-Branch CNN for Non-Melanoma Dermoscopic Classification project.
        </p>
      </div>

      <div className="space-y-8">
        {sections.map((section) => (
          <div key={section.id} className="glass rounded-2xl p-6">
            <h2 className="text-lg font-semibold text-white mb-3">{section.title}</h2>
            <div className="text-slate-400 text-sm leading-relaxed whitespace-pre-line">
              {section.content}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-10 bg-sky-500/10 border border-sky-500/20 rounded-2xl p-6">
        <h2 className="text-lg font-semibold text-white mb-2">Final Conclusion</h2>
        <p className="text-slate-300 leading-relaxed">
          <strong className="text-sky-400">EfficientNet-B4 achieved the strongest performance under the evaluated
          experimental conditions.</strong> The Dual-Branch CNN framework introduced significant architectural
          complexity and 4-7× longer training times, while failing to surpass the EfficientNet-B4 baseline in
          any metric. The persistent attention gate collapse across all three versions (V1, V1.1, V2) suggests
          that forced branch specialization is counterproductive for this dataset — a modern single-branch
          network naturally extracts both textural and structural features more effectively.
        </p>
      </div>
    </div>
  );
}
