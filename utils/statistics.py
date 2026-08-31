"""
Statistical Analysis Utilities

Provides functions for comprehensive statistical analysis of model benchmarks:
- Mean, standard deviation
- Relative improvement over baseline
- Percentage improvement
- Confidence intervals (95% CI for multi-seed runs)

Usage:
    from utils.statistics import compute_ci, relative_improvement
    
    ci = compute_ci(values=[0.92, 0.93, 0.91])  # 95% CI
    improvement = relative_improvement(baseline_val=0.85, new_val=0.92)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class Statistics:
    """Container for statistical results."""
    
    mean: float
    std: float
    min: float
    max: float
    count: int
    ci_lower: Optional[float] = None  # 95% confidence interval lower bound
    ci_upper: Optional[float] = None  # 95% confidence interval upper bound
    
    def __str__(self):
        if self.ci_lower is not None and self.ci_upper is not None:
            return f"{self.mean:.4f} ± {self.std:.4f} (95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        else:
            return f"{self.mean:.4f} ± {self.std:.4f}"
    
    def to_dict(self):
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'count': self.count,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper
        }


@dataclass
class Improvement:
    """Container for improvement metrics."""
    
    absolute_diff: float
    percentage_improvement: float  # (new - baseline) / baseline * 100
    relative_improvement: float    # (new - baseline) / baseline
    
    def __str__(self):
        sign = "+" if self.absolute_diff >= 0 else ""
        return f"{sign}{self.percentage_improvement:.2f}% ({sign}{self.absolute_diff:.4f})"


def compute_statistics(values: List[float], confidence: float = 0.95) -> Statistics:
    """
    Compute comprehensive statistics for a list of values.
    
    Args:
        values: List of numeric values
        confidence: Confidence level for CI (default 0.95 for 95% CI)
    
    Returns:
        Statistics object with mean, std, min, max, CI bounds
    """
    if not values:
        raise ValueError("Cannot compute statistics for empty list")
    
    values_array = np.array(values)
    mean = np.mean(values_array)
    std = np.std(values_array, ddof=1) if len(values) > 1 else 0.0
    min_val = np.min(values_array)
    max_val = np.max(values_array)
    
    # Compute confidence interval
    ci_lower = None
    ci_upper = None
    if len(values) > 1:
        # t-distribution CI for small samples
        se = std / np.sqrt(len(values))
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha/2, df=len(values) - 1)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
    
    return Statistics(
        mean=float(mean),
        std=float(std),
        min=float(min_val),
        max=float(max_val),
        count=len(values),
        ci_lower=float(ci_lower) if ci_lower is not None else None,
        ci_upper=float(ci_upper) if ci_upper is not None else None
    )


def relative_improvement(
    baseline_val: float,
    new_val: float,
    as_percentage: bool = False
) -> float:
    """
    Compute relative improvement: (new - baseline) / baseline
    
    Args:
        baseline_val: Baseline (original) value
        new_val: New value
        as_percentage: If True, return as percentage (multiply by 100)
    
    Returns:
        Relative improvement as decimal or percentage
    """
    if baseline_val == 0:
        raise ValueError("Baseline value cannot be zero for relative improvement")
    
    improvement = (new_val - baseline_val) / baseline_val
    return improvement * 100 if as_percentage else improvement


def compute_improvement(baseline_val: float, new_val: float) -> Improvement:
    """
    Compute absolute and relative improvements.
    
    Args:
        baseline_val: Baseline value
        new_val: New value
    
    Returns:
        Improvement object with absolute, percentage, and relative metrics
    """
    absolute_diff = new_val - baseline_val
    percentage_imp = relative_improvement(baseline_val, new_val, as_percentage=True)
    relative_imp = relative_improvement(baseline_val, new_val, as_percentage=False)
    
    return Improvement(
        absolute_diff=absolute_diff,
        percentage_improvement=percentage_imp,
        relative_improvement=relative_imp
    )


def compare_baselines(
    baseline_values: List[float],
    new_values: List[float],
    confidence: float = 0.95
) -> Tuple[Statistics, Statistics, Optional[Improvement]]:
    """
    Compare baseline and new values with statistical tests.
    
    Args:
        baseline_values: List of baseline measurements
        new_values: List of new measurements
        confidence: Confidence level for CI
    
    Returns:
        Tuple of (baseline_stats, new_stats, improvement_over_mean)
    """
    baseline_stats = compute_statistics(baseline_values, confidence=confidence)
    new_stats = compute_statistics(new_values, confidence=confidence)
    
    improvement = compute_improvement(baseline_stats.mean, new_stats.mean)
    
    return baseline_stats, new_stats, improvement


def t_test(
    baseline_values: List[float],
    new_values: List[float]
) -> Tuple[float, float]:
    """
    Perform independent t-test to check if means are significantly different.
    
    Args:
        baseline_values: Baseline measurements
        new_values: New measurements
    
    Returns:
        Tuple of (t_statistic, p_value)
    """
    t_stat, p_val = stats.ttest_ind(baseline_values, new_values)
    return float(t_stat), float(p_val)


def effect_size_cohens_d(
    baseline_values: List[float],
    new_values: List[float]
) -> float:
    """
    Compute Cohen's d effect size.
    
    Effect size interpretation:
    - 0.0-0.2: negligible
    - 0.2-0.5: small
    - 0.5-0.8: medium
    - 0.8+: large
    
    Args:
        baseline_values: Baseline measurements
        new_values: New measurements
    
    Returns:
        Cohen's d value
    """
    mean_baseline = np.mean(baseline_values)
    mean_new = np.mean(new_values)
    
    # Pooled standard deviation
    n1, n2 = len(baseline_values), len(new_values)
    std_baseline = np.std(baseline_values, ddof=1) if n1 > 1 else 0
    std_new = np.std(new_values, ddof=1) if n2 > 1 else 0
    
    pooled_std = np.sqrt(((n1 - 1) * std_baseline**2 + (n2 - 1) * std_new**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (mean_new - mean_baseline) / pooled_std
    return float(cohens_d)


def effect_size_interpretation(cohens_d: float) -> str:
    """Interpret Cohen's d value."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def format_confidence_interval(
    mean: float,
    ci_lower: float,
    ci_upper: float,
    decimals: int = 4
) -> str:
    """Format confidence interval for reporting."""
    return f"{mean:.{decimals}f} [95% CI: {ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.
    
    Args:
        values: List of values
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    if seed is not None:
        np.random.seed(seed)
    
    values_array = np.array(values)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(values_array, size=len(values_array), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return float(ci_lower), float(ci_upper)


def summarize_multi_seed_run(
    seed_results: dict,
    baseline_type: str = 'dual_branch',
    confidence: float = 0.95
) -> dict:
    """
    Summarize multi-seed experiment results.
    
    Args:
        seed_results: Dict mapping seed -> dict of metrics
        baseline_type: Model type to use as baseline for relative improvement
        confidence: Confidence level
    
    Returns:
        Summary dict with statistics and improvements
    """
    summary = {}
    
    # Compute statistics for each model type
    model_stats = {}
    for seed, metrics_dict in seed_results.items():
        for model_type, metrics in metrics_dict.items():
            if model_type not in model_stats:
                model_stats[model_type] = []
            # Assuming metrics is a dict with 'auc', 'f1', etc.
            model_stats[model_type].append(metrics)
    
    # Aggregate by model type
    aggregated = {}
    for model_type, metrics_list in model_stats.items():
        aggregated[model_type] = {
            'auc': [m.get('auc', 0) for m in metrics_list],
            'f1': [m.get('f1', 0) for m in metrics_list],
            'accuracy': [m.get('accuracy', 0) for m in metrics_list]
        }
    
    # Compute summary statistics
    for model_type, metric_values in aggregated.items():
        summary[model_type] = {}
        for metric_name, values in metric_values.items():
            if values:
                stats_obj = compute_statistics(values, confidence=confidence)
                summary[model_type][metric_name] = stats_obj.to_dict()
    
    return summary


if __name__ == '__main__':
    # Example usage
    baseline = [0.85, 0.84, 0.86]
    new_model = [0.92, 0.93, 0.91]
    
    baseline_stats, new_stats, improvement = compare_baselines(baseline, new_model)
    
    print("Baseline Statistics:", baseline_stats)
    print("New Model Statistics:", new_stats)
    print("Improvement:", improvement)
    
    t_stat, p_val = t_test(baseline, new_model)
    print(f"t-test: t={t_stat:.4f}, p={p_val:.6f}")
    
    cohens_d = effect_size_cohens_d(baseline, new_model)
    print(f"Cohen's d: {cohens_d:.4f} ({effect_size_interpretation(cohens_d)})")
