import os
import json
import shutil
import platform
import subprocess
from pathlib import Path

ROOT = Path("C:/ADL")
RESULTS_DIR = ROOT / "results"
FINAL_DIR = RESULTS_DIR / "final"
ARTIFACTS_DIR = ROOT / "artifacts"
FINAL_DIR.mkdir(exist_ok=True, parents=True)
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
(FINAL_DIR / "figures").mkdir(exist_ok=True, parents=True)

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# TASK 2: GradCAM
grad_cam_txt = "The repository contains Grad-CAM infrastructure, but explainability experiments were intentionally excluded from the final experimental protocol and are reserved for future work."
write_file(FINAL_DIR / "GRAD_CAM_STATUS.md", grad_cam_txt)

# TASK 4: benchmark_final
if (RESULTS_DIR / "benchmark.csv").exists():
    shutil.copy(RESULTS_DIR / "benchmark.csv", FINAL_DIR / "benchmark_final.csv")
    shutil.copy(RESULTS_DIR / "benchmark.csv", FINAL_DIR / "comparison_table.csv")

# TASK 5: manifest.json
manifest = {
    "project": "Dual-Branch CNN for Non-Melanoma Dermoscopic Classification",
    "experiments": [
        {"id": "EXP01", "model": "resnet50", "seed": 42, "parameters": 23522375, "checkpoint": "checkpoints/resnet50"},
        {"id": "EXP02", "model": "densenet121", "seed": 42, "parameters": 6961031, "checkpoint": "checkpoints/densenet121"},
        {"id": "EXP03", "model": "efficientnet_b4", "seed": 42, "parameters": 17561167, "checkpoint": "checkpoints/efficientnet_b4"},
        {"id": "EXP04", "model": "dual_branch_seed42", "version": "V1", "seed": 42, "parameters": 10669639, "checkpoint": "checkpoints/dual_branch_seed42_v1"},
        {"id": "EXP05", "model": "dual_branch_seed123", "version": "V1", "seed": 123, "parameters": 10669639, "checkpoint": "checkpoints/dual_branch_seed123"},
        {"id": "EXP06", "model": "dual_branch_seed999", "version": "V1", "seed": 999, "parameters": 10669639, "checkpoint": "checkpoints/dual_branch_seed999"},
        {"id": "EXP07", "model": "dual_branch_seed42", "version": "V1.1", "seed": 42, "parameters": 10669639, "checkpoint": "checkpoints/dual_branch_seed42_v1_1"},
        {"id": "EXP08", "model": "dual_branch_seed42", "version": "V2", "seed": 42, "parameters": 9031241, "checkpoint": "checkpoints/dual_branch_seed42"}
    ]
}
write_file(FINAL_DIR / "manifest.json", json.dumps(manifest, indent=2))

# TASK 6: Reproducibility
repro_md = f"""# Reproducibility Report
- **Python Version**: {platform.python_version()}
- **OS**: {platform.system()} {platform.release()}
- **Git Commit**: N/A (local dev)
- **Random Seeds**: 42, 123, 999
"""
write_file(ROOT / "REPRODUCIBILITY.md", repro_md)
try:
    subprocess.run("pip freeze > requirements-lock.txt", shell=True, cwd=ROOT)
    subprocess.run("conda env export > environment.yml", shell=True, cwd=ROOT)
except:
    pass

# TASK 7: MODEL_REGISTRY
registry = """# Model Registry
| Model | Status | Parameters | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC | Training Time | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| ResNet50 | Frozen | 23.5M | 0.5662 | 0.7513 | 0.5352 | 0.9352 | 1852.7s | best_checkpoint.pth |
| DenseNet121 | Frozen | 6.9M | 0.6636 | 0.7914 | 0.6242 | 0.9531 | 1476.1s | best_checkpoint.pth |
| EfficientNet-B4 | Frozen | 17.5M | 0.7364 | 0.7916 | 0.6919 | 0.9592 | 2337.3s | best_checkpoint.pth |
| Dual-Branch V1 | Frozen | 10.6M | 0.5479 (avg) | 0.6844 (avg)| 0.4641 (avg)| 0.9041 (avg)| ~23131s | best_checkpoint.pth |
| Dual-Branch V1.1 | Frozen | 10.6M | 0.6576 | 0.6218 | 0.4814 | 0.9006 | 10272.5s | best_checkpoint.pth |
| Dual-Branch V2 | Frozen | 9.0M | 0.6424 | 0.5948 | 0.4950 | 0.9015 | 10869.5s | best_checkpoint.pth |
"""
write_file(ROOT / "MODEL_REGISTRY.md", registry)

# TASK 9: CITATION.cff
citation = """cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "Researcher"
    given-names: "Lenovo"
title: "Dual-Branch CNN for Non-Melanoma Dermoscopic Classification"
version: 2.0.0
date-released: 2026-07-13
keywords:
  - Deep Learning
  - Dermoscopy
"""
write_file(ROOT / "CITATION.cff", citation)

# TASK 10: PROJECT_HANDOFF
handoff = """# Project Handoff Checklist
To transfer the frozen project to the MacBook for paper writing and GitHub publication, copy the following exactly:
1. `results/final/` (contains all metrics, final figures, and reports)
2. `artifacts/` (contains zipped checkpoints and checksums)
3. `benchmark_final.csv`, `manifest.json`, `MODEL_REGISTRY.md`, `FINAL_RESULTS.md`, `REPRODUCIBILITY.md`
4. `tensorboard/` directory logs.
"""
write_file(ROOT / "PROJECT_HANDOFF.md", handoff)

print("Generated MD, CSV, JSON archives successfully.")
