// ─── Verified LOQ Benchmark Data ─────────────────────────────────────────────
// Do NOT use any of these values to override live inference API responses.
// These are static research constants, measured on Lenovo LOQ CUDA environment.

export const BENCHMARK_DATA = [
  { name: "EfficientNet-B4",            rank: 1, params: "17.56M", auc: 95.92, balAcc: 79.16, acc: 73.64, f1: 69.19, latency: 8.83,  vram: "677.7 MB",  role: "Best Evaluated Baseline",  isChampion: true  },
  { name: "DenseNet-121",               rank: 2, params: "6.96M",  auc: 95.31, balAcc: 79.14, acc: 66.36, f1: 62.42, latency: 20.37, vram: "393.7 MB",  role: "Baseline",                 isChampion: false },
  { name: "Dual-Branch CNN (Seed 123)", rank: 3, params: "10.67M", auc: 90.98, balAcc: 70.31, acc: 55.50, f1: 48.55, latency: 27.23, vram: "1561.1 MB", role: "Experimental Architecture", isChampion: false },
  { name: "Dual-Branch CNN (Seed 999)", rank: 4, params: "10.67M", auc: 89.73, balAcc: 66.39, acc: 54.97, f1: 45.77, latency: 26.32, vram: "1561.1 MB", role: "Experimental Architecture", isChampion: false },
  { name: "Dual-Branch CNN (Seed 42)",  rank: 5, params: "10.67M", auc: 90.54, balAcc: 68.62, acc: 53.91, f1: 44.90, latency: 28.47, vram: "1561.1 MB", role: "Experimental Architecture", isChampion: false },
  { name: "ResNet-50",                  rank: 6, params: "23.52M", auc: 93.52, balAcc: 75.13, acc: 56.62, f1: 53.52, latency: 20.96, vram: "900.3 MB",  role: "Baseline",                 isChampion: false },
] as const;

export type BenchmarkRow = typeof BENCHMARK_DATA[number];

// ─── HAM10000 Dataset ─────────────────────────────────────────────────────────
export const DATASET = {
  total: 10015,
  classes: [
    { code: "NV",    name: "Melanocytic Nevi",                              count: 6705, category: "Benign"       as const, pct: 66.9 },
    { code: "MEL",   name: "Melanoma",                                      count: 1113, category: "Malignant"    as const, pct: 11.1 },
    { code: "BKL",   name: "Benign Keratosis-like Lesions",                 count: 1099, category: "Benign"       as const, pct: 11.0 },
    { code: "BCC",   name: "Basal Cell Carcinoma",                          count:  514, category: "Malignant"    as const, pct:  5.1 },
    { code: "AKIEC", name: "Actinic Keratoses / Intraepithelial Carcinoma", count:  327, category: "Pre-malignant" as const, pct:  3.3 },
    { code: "VASC",  name: "Vascular Lesions",                              count:  142, category: "Benign"       as const, pct:  1.4 },
    { code: "DF",    name: "Dermatofibroma",                                count:  115, category: "Benign"       as const, pct:  1.1 },
  ],
} as const;

// ─── Preprocessing pipeline (evaluation) ─────────────────────────────────────
export const PREPROCESSING_STEPS = [
  { label: "Input",        value: "Dermoscopic image (any resolution)" },
  { label: "Resize",       value: "256 × 256 px (bilinear interpolation)" },
  { label: "Center Crop",  value: "224 × 224 px" },
  { label: "ToTensor",     value: "[0, 255] → [0.0, 1.0] float32" },
  { label: "Normalize",    value: "ImageNet μ = [0.485, 0.456, 0.406]  σ = [0.229, 0.224, 0.225]" },
] as const;

// ─── Class metadata (canonical alphabetical order matches model output) ───────
export const CLASS_METADATA: Record<string, {
  code: string; name: string; category: "Malignant" | "Pre-malignant" | "Benign"; urgency: string;
}> = {
  akiec: { code: "AKIEC", name: "Actinic Keratoses / Intraepithelial Carcinoma", category: "Pre-malignant", urgency: "High" },
  bcc:   { code: "BCC",   name: "Basal Cell Carcinoma",                          category: "Malignant",    urgency: "High" },
  bkl:   { code: "BKL",   name: "Benign Keratosis-like Lesions",                 category: "Benign",       urgency: "Low"  },
  df:    { code: "DF",    name: "Dermatofibroma",                                category: "Benign",       urgency: "Low"  },
  mel:   { code: "MEL",   name: "Melanoma",                                      category: "Malignant",    urgency: "Critical" },
  nv:    { code: "NV",    name: "Melanocytic Nevi",                              category: "Benign",       urgency: "Low"  },
  vasc:  { code: "VASC",  name: "Vascular Lesions",                             category: "Benign",       urgency: "Low"  },
};

// Canonical output order from the trained model (alphabetical, matches HAM_LABEL_MAP)
export const CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"] as const;

// ─── OOD caveat ───────────────────────────────────────────────────────────────
export const OOD_NOTE =
  "OOD validation AUROC: 1.00 on 22-image synthetic evaluation set " +
  "(MobileNetV2 prototype-based, threshold = 0.55). " +
  "Primarily synthetic OOD — see validation report for scope limitations.";
