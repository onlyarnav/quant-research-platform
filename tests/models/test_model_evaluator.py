"""
Tests for the ModelEvaluator utility.

Exercises regression metrics (MSE, RMSE, MAE, R2) and financial metrics (Directional Accuracy, IC),
ensuring correctness against hand-computable values and robust input validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.model_evaluator import ModelEvaluator

@pytest.fixture
def evaluator():
    return ModelEvaluator()

def test_mse_zero_for_perfect_predictions(evaluator):
    """Assert MSE is 0.0 when predictions match exactly."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert evaluator.mse(y_true, y_pred) == 0.0

def test_mse_correct_value(evaluator):
    """Assert MSE is 1.0 for specific non-trivial predictions."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    # (1-2)^2 + (2-3)^2 + (3-4)^2 / 3 = (1+1+1)/3 = 1.0
    assert evaluator.mse(y_true, y_pred) == 1.0

def test_rmse_is_sqrt_of_mse(evaluator):
    """Assert RMSE is the square root of MSE."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([2.1, 3.2, 4.3])
    mse = evaluator.mse(y_true, y_pred)
    rmse = evaluator.rmse(y_true, y_pred)
    assert rmse == pytest.approx(mse ** 0.5)

def test_mae_correct_value(evaluator):
    """Assert MAE is 1.0 for specific non-trivial predictions."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    # (|1-2| + |2-3| + |3-4|) / 3 = 3/3 = 1.0
    assert evaluator.mae(y_true, y_pred) == 1.0

def test_r2_perfect_predictions_returns_one(evaluator):
    """Assert R2 returns 1.0 for perfect predictions."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert evaluator.r2(y_true, y_pred) == pytest.approx(1.0)

def test_r2_negative_for_poor_predictions(evaluator):
    """Assert R2 is negative when predictions are worse than the mean."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert evaluator.r2(y_true, y_pred) < 0

def test_directional_accuracy_all_correct(evaluator):
    """Assert 1.0 accuracy when all signs match."""
    y_true = pd.Series([1.0, -1.0, 2.0, -2.0])
    y_pred = np.array([0.5, -0.5, 1.0, -1.0])
    assert evaluator.directional_accuracy(y_true, y_pred) == 1.0

def test_directional_accuracy_all_wrong(evaluator):
    """Assert 0.0 accuracy when all signs are flipped."""
    y_true = pd.Series([1.0, -1.0])
    y_pred = np.array([-1.0, 1.0])
    assert evaluator.directional_accuracy(y_true, y_pred) == 0.0

def test_directional_accuracy_partial(evaluator):
    """Assert correct fraction for mixed sign matches."""
    y_true = pd.Series([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([0.5, 0.5, 0.5, -0.5])
    # matches: 1st (pos/pos), 4th (neg/neg). 2nd (neg/pos), 3rd (pos/pos - wait, y_true=1, y_pred=0.5 is match)
    # indices:
    # 0: 1, 0.5 -> match (1)
    # 1: -1, 0.5 -> mismatch (0)
    # 2: 1, 0.5 -> match (1)
    # 3: -1, -0.5 -> match (1)
    # Correct: 3/4 = 0.75
    assert evaluator.directional_accuracy(y_true, y_pred) == 0.75

def test_directional_accuracy_zero_true_value_edge_case(evaluator):
    """
    Document behavior for zero values. np.sign(0) is 0.
    A match occurs only if both are zero or both have same sign.
    """
    # Case 1: Both zero -> match
    y_true_0 = pd.Series([0.0])
    y_pred_0 = np.array([0.0])
    assert evaluator.directional_accuracy(y_true_0, y_pred_0) == 1.0

    # Case 2: True is zero, Pred is positive -> mismatch
    y_true_1 = pd.Series([0.0])
    y_pred_1 = np.array([0.5])
    assert evaluator.directional_accuracy(y_true_1, y_pred_1) == 0.0

def test_ic_perfect_positive_correlation(evaluator):
    """Assert IC is 1.0 for perfect rank correlation."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert evaluator.information_coefficient(y_true, y_pred) == pytest.approx(1.0)

def test_ic_perfect_negative_correlation(evaluator):
    """Assert IC is -1.0 for perfect inverse rank correlation."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert evaluator.information_coefficient(y_true, y_pred) == pytest.approx(-1.0)

def test_ic_within_valid_bounds(evaluator):
    """Assert IC stays within [-1, 1] for random data."""
    rng = np.random.default_rng(42)
    y_true = pd.Series(rng.standard_normal(100))
    y_pred = np.array(rng.standard_normal(100))
    ic = evaluator.information_coefficient(y_true, y_pred)
    assert -1.0 <= ic <= 1.0

def test_evaluate_returns_all_six_keys(evaluator):
    """Assert the evaluate() method returns all expected metrics."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    result = evaluator.evaluate(y_true, y_pred)
    expected_keys = {"mse", "rmse", "mae", "r2", "directional_accuracy", "ic"}
    assert set(result.keys()) == expected_keys

@pytest.mark.parametrize("metric_fn", [
    "mse", "rmse", "mae", "r2", "directional_accuracy", "information_coefficient"
])
def test_all_methods_raise_on_empty_y_true(evaluator, metric_fn):
    """Assert every public metric method raises ValueError on empty inputs."""
    fn = getattr(evaluator, metric_fn)
    with pytest.raises(ValueError, match="y_true cannot be empty"):
        fn(pd.Series([]), np.array([]))

@pytest.mark.parametrize("metric_fn", [
    "mse", "rmse", "mae", "r2", "directional_accuracy", "information_coefficient"
])
def test_all_methods_raise_on_length_mismatch(evaluator, metric_fn):
    """Assert every public metric method raises ValueError on length mismatch."""
    fn = getattr(evaluator, metric_fn)
    with pytest.raises(ValueError, match="Length mismatch"):
        fn(pd.Series([1, 2, 3]), np.array([1, 2]))
