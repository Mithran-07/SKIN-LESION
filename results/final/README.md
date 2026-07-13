# Dual-Branch CNN: Final Results Archive

This directory contains the finalized output for the Dual-Branch CNN for Non-Melanoma Dermoscopic Classification project.

## Directory Structure
- `figures/`: Contains publication-ready figures in PNG, SVG, and PDF formats (ROC, PR, Confusion Matrices, Learning Curves, Class Distribution, Fusion Diagnostics, Gate Weights).
- `benchmark_final.csv`: The final benchmark metrics across all tested architectures.
- `comparison_table.csv / .md`: A concise comparison of performance metrics.
- `summary_metrics.json`: JSON output of all calculated metric averages and stds.
- `experiment_summary.md`: A high-level view of the experiment manifest.
- `FINAL_RESULTS.md`: The complete, formalized research summary, statistical conclusions, and limitations.

## Interpretation
To interpret the outputs, begin with `FINAL_RESULTS.md` to understand the core findings (e.g. why the Dual-Branch architecture was rejected). You may cross-reference the raw performance data in `benchmark_final.csv` and view the visual distributions of the gate collapse in `figures/`.
