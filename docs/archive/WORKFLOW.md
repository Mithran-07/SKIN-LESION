# Quick Start: Analysis Workflow

**When benchmark.csv arrives from Lenovo LOQ**, follow this workflow:

## Step 1: Setup (One-time)

```bash
# Navigate to workspace
cd /Users/mithran/Documents/ADL

# Ensure Python environment is active
source .venv/bin/activate

# Install analysis dependencies (if not already installed)
pip install pandas numpy scipy scikit-learn matplotlib
```

## Step 2: Receive Results from LOQ

Expected file: `results/benchmark.csv`

Expected format:
```csv
model_name,model_type,auc,f1,accuracy,macro_recall,balanced_accuracy
Original Dual-Branch,dual_branch,0.954,0.931,0.915,0.898,0.912
Optimized Dual-Branch,dual_branch_optimized,0.962,0.945,0.928,0.918,0.926
ResNet50,resnet50,0.941,0.912,0.901,0.875,0.895
DenseNet121,densenet121,0.948,0.925,0.910,0.891,0.908
EfficientNet-B4,efficientnet_b4,0.955,0.933,0.918,0.905,0.920
```

## Step 3: Auto-Generate Comparison Tables

```bash
# Generate publication-ready comparison tables
python scripts/compare_models.py \
  --results-csv results/benchmark.csv \
  --output-dir comparison/

# This produces:
# - comparison/comparison_report.md (markdown analysis)
# - comparison/comparison_table.csv (for spreadsheets)
# - comparison/comparison_table.tex (for LaTeX papers)
```

### Output Example:
```
✓ Updated: comparison/comparison_report.md
✓ Models processed: 5
✓ Best AUC: 0.9620
```

## Step 4: Auto-Update Paper

```bash
# Automatically update paper/sections/results.md with:
# - Results table
# - Best models by metric  
# - Figure references
# - Caption templates
python scripts/update_paper_results.py \
  --benchmark results/benchmark.csv \
  --paper paper/sections/results.md
```

### Output Example:
```
✓ Updated: paper/sections/results.md
✓ Tables: 5 models
✓ Best metrics identified
✓ Figure references inserted
```

## Step 5: Fill Analysis Templates

### For Each Template, Follow This Process:

#### a) Fusion Collapse Analysis
```bash
# Open template
open analysis_templates/fusion_collapse.md

# Fill in sections:
1. Run model with Grad-CAM analysis
2. Record attention gate weights from explainability/gradcam.py
3. Check if branches specialize or collapse
4. Document findings and recommendations
```

#### b) Overfitting Analysis
```bash
# Check training vs. validation metrics
1. Extract from training logs (scripts/train.py output)
2. Calculate gaps for each metric
3. Classify severity: HEALTHY / MILD / MODERATE / SEVERE
4. Apply mitigations if needed
```

#### c) Class Imbalance Analysis
```bash
# Analyze per-class performance
1. Extract confusion matrix from scripts/evaluate.py
2. Calculate per-class recall
3. Focus on minority classes (DF, VASC)
4. Clinical significance check for melanoma recall
```

#### d) Confusion Matrix Interpretation
```bash
# Detailed per-class error analysis
1. Open confusion matrix from benchmark
2. Analyze misclassification patterns
3. Identify critical errors (MEL→Benign)
4. Compare with baseline models
```

#### e) Grad-CAM Observations
```bash
# Visual interpretation of attention
1. Generate Grad-CAM for ~10 samples per class
2. Assess branch specialization (texture vs. structure)
3. Check alignment with clinical expertise
4. Evaluate trustworthiness for clinical use
```

## Step 6: Update Presentation

```bash
# Open presentation template
open presentations/results_analysis.md

# For each slide with data:
1. Copy table from comparison/comparison_table.csv or 
   paper/sections/results.md
2. Insert figures from comparison/ directory
3. Fill statistical significance results from analysis_templates/
4. Review slides for accuracy and flow
```

## Step 7: Generate Statistical Summaries

```python
# Use statistical utilities for detailed analysis
python3 << 'PYEND'
from utils.statistics import compute_statistics, compare_baselines, t_test

# Example: Dual-Branch vs. ResNet50
baseline_auc = [0.941]  # ResNet50
dual_branch_auc = [0.954]  # Dual-Branch

baseline_stats, new_stats, improvement = compare_baselines(
    baseline_auc, dual_branch_auc
)

print("Baseline (ResNet50):", baseline_stats)
print("Dual-Branch:", new_stats)
print("Improvement:", improvement)

# Statistical significance
t_stat, p_val = t_test(baseline_auc, dual_branch_auc)
print(f"t-test: t={t_stat:.4f}, p={p_val:.4f}")
PYEND
```

## Step 8: Medical Review

```bash
# Checklist before clinical deployment:
□ Melanoma recall ≥ 0.90? (Safety requirement)
□ Per-class recall ≥ 0.60 for rare classes?
□ Confusion matrix shows clinically acceptable errors?
□ Grad-CAM attention patterns are interpretable?
□ Overfitting is within acceptable range?
□ Class imbalance is addressed by focal loss?

If ALL check, proceed to deployment consideration.
If ANY fail, document findings in analysis templates.
```

## Step 9: Final Report

```bash
# Generate comprehensive report
cat > EXPERIMENT_RESULTS.md << 'REPORT'
# Experiment Results Summary

## Results
[Copy from comparison/comparison_report.md]

## Analysis
[Copy from analysis_templates/]

## Conclusions
[Add your conclusions here]

## Recommendations
[Based on analysis, recommend next steps]
REPORT

# Open for review
open EXPERIMENT_RESULTS.md
```

## Directory Structure During Analysis

```
ADL/
├── results/
│   └── benchmark.csv              ← Input from LOQ
├── comparison/                     ← AUTO-GENERATED
│   ├── comparison_report.md
│   ├── comparison_table.csv
│   └── comparison_table.tex
├── paper/sections/
│   └── results.md                 ← AUTO-UPDATED
├── analysis_templates/            ← MANUALLY FILLED
│   ├── fusion_collapse.md
│   ├── overfitting_analysis.md
│   ├── class_imbalance.md
│   ├── confusion_matrix.md
│   └── gradcam_observations.md
├── presentations/
│   └── results_analysis.md        ← UPDATED WITH RESULTS
└── EXPERIMENT_RESULTS.md          ← FINAL REPORT
```

## Common Issues & Solutions

### Issue: "No module named 'pandas'"
```bash
pip install pandas numpy scipy scikit-learn
```

### Issue: "No such file or directory: results/benchmark.csv"
```bash
# Check path is correct
ls -la results/

# Copy from LOQ if needed
scp user@lenovo_lq:~/ADL/results/benchmark.csv ./results/
```

### Issue: "Paper file not found"
```bash
# Create paper sections if missing
mkdir -p paper/sections
touch paper/sections/results.md
```

### Issue: Tables not formatted correctly
```bash
# Verify CSV column names match expected format:
# model_name, model_type, auc, f1, accuracy, macro_recall, balanced_accuracy
head results/benchmark.csv
```

## Workflow Checklist

- [ ] Receive benchmark.csv from LOQ
- [ ] Run compare_models.py to generate tables
- [ ] Run update_paper_results.py to update paper
- [ ] Fill fusion_collapse.md template
- [ ] Fill overfitting_analysis.md template
- [ ] Fill class_imbalance.md template
- [ ] Fill confusion_matrix.md template
- [ ] Fill gradcam_observations.md template
- [ ] Update presentations/results_analysis.md with figures
- [ ] Review statistical significance
- [ ] Conduct medical review
- [ ] Generate final EXPERIMENT_RESULTS.md

## Quick Commands Reference

```bash
# Compare models
python scripts/compare_models.py --results-csv results/benchmark.csv --output-dir comparison/

# Update paper
python scripts/update_paper_results.py --benchmark results/benchmark.csv --paper paper/sections/results.md

# Statistical analysis in Python
python3 -c "from utils.statistics import compute_statistics; print(compute_statistics([0.95, 0.94, 0.96]))"

# View template
cat analysis_templates/fusion_collapse.md

# View results
cat comparison/comparison_report.md
```

---

**Status**: Ready to execute  
**Last Updated**: July 12, 2026  
**Awaiting**: benchmark.csv from Lenovo LOQ
