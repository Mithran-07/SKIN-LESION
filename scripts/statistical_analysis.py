"""
Statistical Analysis Utilities for ADL Experimental Results.

Provides utility functions to calculate:
- Mean
- Standard Deviation
- Relative improvement over baseline
- Percentage improvement
- Confidence intervals (for multi-seed experiments)
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np

def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
    """Calculate mean and standard deviation of a list of values."""
    if not values:
        return 0.0, 0.0
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return float(mean_val), float(std_val)

def calculate_relative_improvement(baseline: float, current: float) -> float:
    """Calculate absolute improvement over baseline."""
    return current - baseline

def calculate_percentage_improvement(baseline: float, current: float) -> float:
    """Calculate percentage improvement over baseline."""
    if baseline == 0:
        return 0.0
    return ((current - baseline) / abs(baseline)) * 100.0

def calculate_confidence_interval(values: List[float], confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a list of values.
    Returns: (mean, lower_bound, upper_bound)
    """
    import scipy.stats as st
    
    if not values:
        return 0.0, 0.0, 0.0
    
    mean_val = np.mean(values)
    if len(values) < 2:
        return float(mean_val), float(mean_val), float(mean_val)
        
    sem = st.sem(values) # Standard error of mean
    ci = st.t.interval(confidence_level, len(values)-1, loc=mean_val, scale=sem)
    
    return float(mean_val), float(ci[0]), float(ci[1])

def analyze_experiment_metrics(
    baseline_metrics: List[float], 
    proposed_metrics: List[float], 
    metric_name: str = "AUC"
) -> Dict[str, Any]:
    """Perform comprehensive statistical analysis between a baseline and proposed model."""
    b_mean, b_std = calculate_mean_std(baseline_metrics)
    p_mean, p_std = calculate_mean_std(proposed_metrics)
    
    relative_imp = calculate_relative_improvement(b_mean, p_mean)
    percent_imp = calculate_percentage_improvement(b_mean, p_mean)
    
    _, b_ci_lower, b_ci_upper = calculate_confidence_interval(baseline_metrics)
    _, p_ci_lower, p_ci_upper = calculate_confidence_interval(proposed_metrics)
    
    return {
        "metric": metric_name,
        "baseline": {
            "mean": b_mean,
            "std": b_std,
            "ci_95": (b_ci_lower, b_ci_upper)
        },
        "proposed": {
            "mean": p_mean,
            "std": p_std,
            "ci_95": (p_ci_lower, p_ci_upper)
        },
        "improvement": {
            "relative": relative_imp,
            "percentage": percent_imp
        }
    }
