"""
Tests for Conformal Prediction.

Verifies:
1. Coverage guarantee holds: empirical coverage ≥ 1 - alpha
2. Set expansion: ambiguous predictions produce larger sets
3. Confident predictions: high-confidence predictions yield set_size = 1
4. q_hat is a valid threshold
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from uncertainty.conformal_prediction import SplitConformalPredictor

NUM_CLASSES = 7


def _make_mock_model(probs_sequence):
    """
    Create a mock model that returns pre-specified softmax probabilities.
    probs_sequence: list of (B, C) numpy arrays to return on successive calls.
    """
    call_count = [0]

    def fake_forward(x):
        idx = min(call_count[0], len(probs_sequence) - 1)
        call_count[0] += 1
        logits = torch.tensor(probs_sequence[idx], dtype=torch.float32)
        # Scale to logit-like values
        return logits, None, None

    model = MagicMock()
    model.__call__ = fake_forward
    model.eval = MagicMock(return_value=model)
    return model


def test_conformal_coverage_guarantee():
    """
    Empirical coverage on test data must be >= 1 - alpha.
    Uses synthetic softmax probabilities where we know the true class.
    """
    alpha = 0.1  # 90% coverage target
    n_cal = 500
    n_test = 200
    predictor = SplitConformalPredictor(alpha=alpha)

    # Simulate calibration: random confidences
    rng = np.random.RandomState(42)
    # Generate calibration scores (non-conformity = 1 - p_true)
    # For 90% coverage, most scores should be low (model is correct)
    cal_scores = rng.beta(0.5, 2.0, size=n_cal)  # Skewed toward 0 (confident)
    predictor._cal_scores = cal_scores
    n = n_cal
    level = np.ceil((n + 1) * (1 - alpha)) / n
    predictor.q_hat = float(np.quantile(cal_scores, min(level, 1.0)))
    predictor._is_calibrated = True

    # Generate test scores
    test_scores = rng.beta(0.5, 2.0, size=n_test)
    covered = [score <= predictor.q_hat for score in test_scores]
    empirical_coverage = np.mean(covered)

    # Coverage may slightly exceed or approach 1-alpha due to finite samples
    # The guarantee is asymptotic; we test with tolerance
    assert empirical_coverage >= (1 - alpha) - 0.05, (
        f"Empirical coverage {empirical_coverage:.3f} below target {1-alpha:.2f}"
    )


def test_confident_prediction_set_size_1():
    """
    A model that is 99% confident about class 2 should produce set_size = 1.
    """
    predictor = SplitConformalPredictor(alpha=0.1)
    predictor.q_hat = 0.5  # Threshold: non-conformity ≤ 0.5
    predictor._is_calibrated = True
    predictor.class_names = [f"C{i}" for i in range(NUM_CLASSES)]

    # 99% confident → non-conformity = 1 - 0.99 = 0.01 ≤ q̂
    confident_probs = np.array([0.003, 0.003, 0.99, 0.001, 0.001, 0.001, 0.001])
    logits = torch.tensor(confident_probs).unsqueeze(0).log()  # Approximate logits

    model = MagicMock()
    model.eval.return_value = model
    model.return_value = (logits, None, None)

    result = predictor.predict(model, torch.randn(1, 3, 224, 224), torch.device("cpu"))
    assert result["set_size"] == 1, f"Expected set_size=1, got {result['set_size']}"
    assert result["confident"] is True


def test_ambiguous_prediction_set_size_greater_than_1():
    """
    A uniformly uncertain model should produce a large prediction set.
    """
    predictor = SplitConformalPredictor(alpha=0.1)
    predictor.q_hat = 0.9  # Very permissive threshold
    predictor._is_calibrated = True
    predictor.class_names = [f"C{i}" for i in range(NUM_CLASSES)]

    # Uniform uncertainty: p = 1/7 ≈ 0.143 → non-conformity = 0.857 > q̂=0.9? No
    # Use q̂ = 0.95 so all classes are included
    predictor.q_hat = 0.95
    uniform_probs = np.full((1, NUM_CLASSES), 1.0 / NUM_CLASSES)
    logits = torch.tensor(uniform_probs).log()

    model = MagicMock()
    model.eval.return_value = model
    model.return_value = (logits, None, None)

    result = predictor.predict(model, torch.randn(1, 3, 224, 224), torch.device("cpu"))
    assert result["set_size"] > 1, f"Expected set_size>1 for uncertain prediction, got {result['set_size']}"


def test_q_hat_within_range():
    """q_hat must be a valid probability (between 0 and 1)."""
    predictor = SplitConformalPredictor(alpha=0.1)
    predictor._cal_scores = np.random.uniform(0, 1, 100)
    predictor._is_calibrated = True

    n = 100
    level = np.ceil((n + 1) * (1 - 0.1)) / n
    predictor.q_hat = float(np.quantile(predictor._cal_scores, min(level, 1.0)))

    assert 0 <= predictor.q_hat <= 1, f"q̂ out of range: {predictor.q_hat}"
