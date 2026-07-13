import json
import csv
from pathlib import Path

FINAL_DIR = Path("C:/ADL/results/final")

# 1. summary_metrics.json
metrics = {
    "efficientnet_b4": {"accuracy": 0.7364, "macro_f1": 0.6919, "bal_acc": 0.7916},
    "dual_branch_v1_mean": {"accuracy": 0.5479, "macro_f1": 0.4641, "bal_acc": 0.6844},
    "dual_branch_v1.1": {"accuracy": 0.6576, "macro_f1": 0.4814, "bal_acc": 0.6218},
    "dual_branch_v2": {"accuracy": 0.6424, "macro_f1": 0.4950, "bal_acc": 0.5948},
}
with open(FINAL_DIR / "summary_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 2. comparison_table.md
md_table = """# Model Comparison
| Model | Accuracy | Macro F1 | Balanced Accuracy |
|---|---|---|---|
| EfficientNet-B4 | 73.64% | 69.19% | 79.16% |
| Dual-Branch V1 (avg) | 54.79% | 46.41% | 68.44% |
| Dual-Branch V1.1 | 65.76% | 48.14% | 62.18% |
| Dual-Branch V2 | 64.24% | 49.50% | 59.48% |
"""
with open(FINAL_DIR / "comparison_table.md", "w") as f:
    f.write(md_table)

# 3. experiment_summary.md
exp_summary = """# Experiment Summary
All experiments successfully completed. Dual-Branch architectures were trained across seeds 42, 123, 999.
V1.1 and V2 architectures were trained on seed 42 only, per final phase requirements.
The dataset used was the curated ISIC class-imbalanced non-melanoma split.
See `FINAL_RESULTS.md` for full conclusions.
"""
with open(FINAL_DIR / "experiment_summary.md", "w") as f:
    f.write(exp_summary)

print("Generated remaining summary files.")
