"""
Model Comparison Framework

Compares multiple model architectures (Dual-Branch variants, ResNet50, DenseNet121, EfficientNet-B4)
across their benchmark results. Generates publication-ready comparison tables in multiple formats.

Usage:
    python compare_models.py --results-dir results/ --output comparison/
    python compare_models.py --config config/comparison.yaml
"""

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelBenchmark:
    """Container for a single model's benchmark results."""
    
    name: str
    model_type: str  # 'dual_branch', 'dual_branch_optimized', 'resnet50', 'densenet121', 'efficientnet_b4'
    metrics: Dict[str, float]  # {'auc': 0.95, 'f1': 0.92, 'accuracy': 0.91, ...}
    seed: Optional[int] = None
    checkpoint_path: Optional[str] = None
    training_time_hours: Optional[float] = None
    
    def __repr__(self):
        return f"ModelBenchmark({self.name}, {self.model_type}, seed={self.seed})"


class BenchmarkLoader:
    """Load benchmark results from CSV or JSON files."""
    
    EXPECTED_METRICS = ['auc', 'f1', 'accuracy', 'macro_recall', 'balanced_accuracy']
    
    @staticmethod
    def from_csv(csv_path: Path) -> List[ModelBenchmark]:
        """Load benchmarks from CSV file.
        
        Expected CSV columns: model_name, model_type, auc, f1, accuracy, macro_recall, balanced_accuracy, [seed], [checkpoint], [training_time]
        """
        benchmarks = []
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                metrics = {}
                for metric in BenchmarkLoader.EXPECTED_METRICS:
                    if metric in df.columns:
                        metrics[metric] = float(row[metric])
                
                benchmark = ModelBenchmark(
                    name=row['model_name'],
                    model_type=row['model_type'],
                    metrics=metrics,
                    seed=int(row['seed']) if 'seed' in df.columns else None,
                    checkpoint_path=row.get('checkpoint', None),
                    training_time_hours=float(row['training_time']) if 'training_time' in df.columns else None
                )
                benchmarks.append(benchmark)
            logger.info(f"Loaded {len(benchmarks)} benchmarks from {csv_path}")
        except Exception as e:
            logger.error(f"Failed to load benchmarks from CSV: {e}")
            raise
        
        return benchmarks
    
    @staticmethod
    def from_json(json_path: Path) -> List[ModelBenchmark]:
        """Load benchmarks from JSON file."""
        benchmarks = []
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    benchmark = ModelBenchmark(
                        name=item['name'],
                        model_type=item['model_type'],
                        metrics=item['metrics'],
                        seed=item.get('seed', None),
                        checkpoint_path=item.get('checkpoint_path', None),
                        training_time_hours=item.get('training_time_hours', None)
                    )
                    benchmarks.append(benchmark)
            else:
                # Single benchmark
                benchmark = ModelBenchmark(
                    name=data['name'],
                    model_type=data['model_type'],
                    metrics=data['metrics'],
                    seed=data.get('seed', None),
                    checkpoint_path=data.get('checkpoint_path', None),
                    training_time_hours=data.get('training_time_hours', None)
                )
                benchmarks.append(benchmark)
            
            logger.info(f"Loaded {len(benchmarks)} benchmarks from {json_path}")
        except Exception as e:
            logger.error(f"Failed to load benchmarks from JSON: {e}")
            raise
        
        return benchmarks


class ComparisonTable:
    """Generate publication-ready comparison tables."""
    
    MODEL_ORDER = ['dual_branch', 'dual_branch_optimized', 'resnet50', 'densenet121', 'efficientnet_b4']
    METRIC_ORDER = ['auc', 'f1', 'accuracy', 'macro_recall', 'balanced_accuracy']
    
    @staticmethod
    def aggregate_by_model(benchmarks: List[ModelBenchmark]) -> Dict[str, Dict[str, List[float]]]:
        """Aggregate benchmark results by model type.
        
        Returns: {model_type: {metric: [values]}}
        """
        aggregated = {}
        for benchmark in benchmarks:
            if benchmark.model_type not in aggregated:
                aggregated[benchmark.model_type] = {metric: [] for metric in ComparisonTable.METRIC_ORDER}
            
            for metric in ComparisonTable.METRIC_ORDER:
                if metric in benchmark.metrics:
                    aggregated[benchmark.model_type][metric].append(benchmark.metrics[metric])
        
        return aggregated
    
    @staticmethod
    def to_dataframe(aggregated: Dict[str, Dict[str, List[float]]], metric: str = 'auc') -> pd.DataFrame:
        """Convert aggregated results to DataFrame with mean ± std format.
        
        Args:
            aggregated: Dict from aggregate_by_model()
            metric: Which metric to extract ('auc', 'f1', etc.)
        
        Returns:
            DataFrame with rows=models, columns=[metric value, std, count]
        """
        rows = []
        for model_type in ComparisonTable.MODEL_ORDER:
            if model_type not in aggregated:
                continue
            
            values = aggregated[model_type].get(metric, [])
            if not values:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            count = len(values)
            
            rows.append({
                'Model': model_type,
                'Mean': mean_val,
                'Std': std_val,
                'Count': count,
                'Formatted': f"{mean_val:.4f} ± {std_val:.4f}" if count > 1 else f"{mean_val:.4f}"
            })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def to_latex_table(aggregated: Dict[str, Dict[str, List[float]]], caption: str = "Model Comparison") -> str:
        """Generate LaTeX table for publication."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{l|" + "|".join(["c"] * len(ComparisonTable.METRIC_ORDER)) + "}",
            r"\hline",
            "Model & " + " & ".join(ComparisonTable.METRIC_ORDER).upper() + r" \\",
            r"\hline"
        ]
        
        for model_type in ComparisonTable.MODEL_ORDER:
            if model_type not in aggregated:
                continue
            
            row = [model_type]
            for metric in ComparisonTable.METRIC_ORDER:
                values = aggregated[model_type].get(metric, [])
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    if len(values) > 1:
                        row.append(f"${mean_val:.4f} \\pm {std_val:.4f}$")
                    else:
                        row.append(f"${mean_val:.4f}$")
                else:
                    row.append("—")
            
            lines.append(" & ".join(row) + r" \\")
        
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            r"\end{table}"
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def to_markdown_table(aggregated: Dict[str, Dict[str, List[float]]]) -> str:
        """Generate Markdown table for documentation."""
        lines = ["| Model | " + " | ".join(ComparisonTable.METRIC_ORDER) + " |"]
        lines.append("|" + "|".join(["-" * 15] * (len(ComparisonTable.METRIC_ORDER) + 1)) + "|")
        
        for model_type in ComparisonTable.MODEL_ORDER:
            if model_type not in aggregated:
                continue
            
            row = [f"**{model_type}**"]
            for metric in ComparisonTable.METRIC_ORDER:
                values = aggregated[model_type].get(metric, [])
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    if len(values) > 1:
                        row.append(f"{mean_val:.4f} ± {std_val:.4f}")
                    else:
                        row.append(f"{mean_val:.4f}")
                else:
                    row.append("—")
            
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    @staticmethod
    def to_csv_table(aggregated: Dict[str, Dict[str, List[float]]], output_path: Path) -> None:
        """Save comparison table to CSV."""
        rows = []
        for model_type in ComparisonTable.MODEL_ORDER:
            if model_type not in aggregated:
                continue
            
            row = {'model': model_type}
            for metric in ComparisonTable.METRIC_ORDER:
                values = aggregated[model_type].get(metric, [])
                if values:
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std'] = np.std(values)
                    row[f'{metric}_count'] = len(values)
                else:
                    row[f'{metric}_mean'] = None
                    row[f'{metric}_std'] = None
                    row[f'{metric}_count'] = 0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved CSV comparison table to {output_path}")


class ComparisonAnalyzer:
    """Analyze and report comparisons between models."""
    
    @staticmethod
    def generate_report(benchmarks: List[ModelBenchmark], output_dir: Path) -> str:
        """Generate comprehensive comparison report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        aggregated = ComparisonTable.aggregate_by_model(benchmarks)
        
        report_lines = [
            "# Model Comparison Report",
            "",
            f"Generated for {len(benchmarks)} benchmark(s) across {len(set(b.model_type for b in benchmarks))} model types",
            "",
            "## Summary Statistics by Metric",
            ""
        ]
        
        # Generate tables for each metric
        for metric in ComparisonTable.METRIC_ORDER:
            report_lines.append(f"### {metric.upper()}")
            report_lines.append("")
            
            df = ComparisonTable.to_dataframe(aggregated, metric=metric)
            report_lines.append(df.to_markdown(index=False))
            report_lines.append("")
        
        # Identify best model for each metric
        report_lines.append("## Best Performing Models (by Metric)")
        report_lines.append("")
        
        for metric in ComparisonTable.METRIC_ORDER:
            best_model = None
            best_val = -1.0
            
            for model_type in ComparisonTable.MODEL_ORDER:
                if model_type not in aggregated:
                    continue
                
                values = aggregated[model_type].get(metric, [])
                if values and np.mean(values) > best_val:
                    best_val = np.mean(values)
                    best_model = model_type
            
            if best_model:
                report_lines.append(f"- **{metric}**: {best_model} ({best_val:.4f})")
        
        report_lines.append("")
        report_lines.append("## Formatted Tables for Publication")
        report_lines.append("")
        
        # Save LaTeX table
        latex_table = ComparisonTable.to_latex_table(aggregated, caption="Model Performance Comparison")
        latex_path = output_dir / "comparison_table.tex"
        latex_path.write_text(latex_table)
        report_lines.append(f"LaTeX table saved to: `{latex_path.name}`")
        report_lines.append("")
        
        # Save Markdown table
        md_table = ComparisonTable.to_markdown_table(aggregated)
        report_lines.append("### Markdown Table")
        report_lines.append("")
        report_lines.append(md_table)
        report_lines.append("")
        
        # Save CSV
        csv_path = output_dir / "comparison_table.csv"
        ComparisonTable.to_csv_table(aggregated, csv_path)
        
        report_text = "\n".join(report_lines)
        report_path = output_dir / "comparison_report.md"
        report_path.write_text(report_text)
        logger.info(f"Saved comparison report to {report_path}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Compare model benchmark results and generate publication-ready tables"
    )
    parser.add_argument('--results-csv', type=Path, help='Path to benchmark CSV file')
    parser.add_argument('--results-json', type=Path, help='Path to benchmark JSON file')
    parser.add_argument('--output-dir', type=Path, default=Path('comparison'), help='Output directory for tables')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load benchmarks
    benchmarks = []
    if args.results_csv:
        benchmarks.extend(BenchmarkLoader.from_csv(args.results_csv))
    if args.results_json:
        benchmarks.extend(BenchmarkLoader.from_json(args.results_json))
    
    if not benchmarks:
        logger.error("No benchmarks loaded. Provide --results-csv or --results-json")
        return
    
    logger.info(f"Loaded {len(benchmarks)} total benchmarks")
    
    # Generate comparison report
    report = ComparisonAnalyzer.generate_report(benchmarks, args.output_dir)
    print(report)


if __name__ == '__main__':
    main()
