"""
Extended tests for ModelEvaluator: Pearson IC, daily rank IC, extended metrics, and HIGHER_IS_BETTER.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ConstantInputWarning

from src.models.model_evaluator import HIGHER_IS_BETTER, ModelEvaluator


def test_constant_predictions_gives_nan_ic_and_warning_no_scipy_warning(caplog) -> None:
    """Assert constant predictions give NaN IC, log a constant warning, and raise no ConstantInputWarning."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_constant = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

    with (
        caplog.at_level(logging.WARNING),
        warnings.catch_warnings(record=True) as recorded_warnings,
    ):
        warnings.simplefilter("always")
        ic = ModelEvaluator.information_coefficient(y_true, y_pred_constant)

        # Assert no ConstantInputWarning was raised
        assert not any(
            issubclass(w.category, ConstantInputWarning) for w in recorded_warnings
        )

    assert np.isnan(ic)
    assert "constant" in caplog.text.lower()


def test_pearson_ic_equals_np_corrcoef() -> None:
    """Assert Pearson IC matches np.corrcoef on random numbers."""
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(100)
    y_pred = rng.standard_normal(100)

    expected = np.corrcoef(y_true, y_pred)[0, 1]
    actual = ModelEvaluator.pearson_information_coefficient(y_true, y_pred)
    assert actual == pytest.approx(expected)


def test_rank_ic_alias_equals_information_coefficient() -> None:
    """Assert rank_information_coefficient is an exact alias of information_coefficient."""
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(50)
    y_pred = rng.standard_normal(50)

    ic = ModelEvaluator.information_coefficient(y_true, y_pred)
    rank_ic = ModelEvaluator.rank_information_coefficient(y_true, y_pred)
    assert ic == pytest.approx(rank_ic)


def test_daily_rank_ic_golden_two_dates() -> None:
    """
    Assert daily_rank_ic golden: two dates with 5 symbols each,
    one perfectly aligned (+1.0) and one inverted (-1.0), with small groups dropped.
    """
    # Date 1: 5 symbols perfectly ordered: y_true=[1,2,3,4,5], y_pred=[10,20,30,40,50] -> IC = +1.0
    # Date 2: 5 symbols inverted: y_true=[1,2,3,4,5], y_pred=[50,40,30,20,10] -> IC = -1.0
    # Date 3: only 3 symbols (below min_group_size=5) -> dropped
    d1 = ["2023-01-01"] * 5
    d2 = ["2023-01-02"] * 5
    d3 = ["2023-01-03"] * 3

    y_true = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0]
    y_pred = [10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 40.0, 30.0, 20.0, 10.0, 1.0, 2.0, 3.0]
    dates = d1 + d2 + d3

    daily_ic = ModelEvaluator.daily_rank_ic(y_true, y_pred, dates, min_group_size=5)

    assert len(daily_ic) == 2
    assert "2023-01-01" in daily_ic
    assert "2023-01-02" in daily_ic
    assert "2023-01-03" not in daily_ic
    assert daily_ic["2023-01-01"] == pytest.approx(1.0)
    assert daily_ic["2023-01-02"] == pytest.approx(-1.0)


def test_extended_metrics_without_dates() -> None:
    """Assert extended_metrics without dates returns exactly pearson_ic and rank_ic."""
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(40)
    y_pred = rng.standard_normal(40)

    evaluator = ModelEvaluator()
    metrics = evaluator.extended_metrics(y_true, y_pred, dates=None)
    assert set(metrics.keys()) == {"pearson_ic", "rank_ic"}


def test_extended_metrics_with_dates_and_ic_ir() -> None:
    """Assert extended_metrics with dates adds 4 daily metrics and verifies ic_ir."""
    # 2 dates: date 1 IC = 1.0, date 2 IC = 0.5
    # mean = (1.0 + 0.5) / 2 = 0.75
    # std (ddof=0) = sqrt(((1-0.75)^2 + (0.5-0.75)^2) / 2) = sqrt((0.0625 + 0.0625)/2) = 0.25
    # ic_ir = 0.75 / 0.25 = 3.0
    # pos_rate = 2 / 2 = 1.0
    dates = ["2023-01-01"] * 5 + ["2023-01-02"] * 5
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # date 1 perfect (1.0), date 2 partially aligned (y_pred=[1, 2, 4, 3, 5])
    # Let's verify with daily_rank_ic directly:
    y_pred = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 4.0, 3.0, 5.0]

    evaluator = ModelEvaluator()
    metrics = evaluator.extended_metrics(y_true, y_pred, dates=pd.Series(dates))

    assert "mean_daily_rank_ic" in metrics
    assert "daily_rank_ic_std" in metrics
    assert "ic_ir" in metrics
    assert "daily_rank_ic_positive_rate" in metrics

    daily_ic = evaluator.daily_rank_ic(y_true, y_pred, dates)
    expected_mean = float(daily_ic.mean())
    expected_std = float(daily_ic.std(ddof=0))
    expected_ir = expected_mean / expected_std
    assert metrics["mean_daily_rank_ic"] == pytest.approx(expected_mean)
    assert metrics["daily_rank_ic_std"] == pytest.approx(expected_std)
    assert metrics["ic_ir"] == pytest.approx(expected_ir)


def test_evaluate_returns_exactly_six_keys() -> None:
    """Assert evaluate() returns exactly the six base metrics."""
    evaluator = ModelEvaluator()
    res = evaluator.evaluate(pd.Series([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert set(res.keys()) == {
        "mse",
        "rmse",
        "mae",
        "r2",
        "directional_accuracy",
        "ic",
    }


def test_higher_is_better_covers_all_metrics() -> None:
    """Assert HIGHER_IS_BETTER covers every key produced by evaluate and extended_metrics."""
    evaluator = ModelEvaluator()
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = pd.Series(["2023-01-01"] * 5)

    base_keys = set(evaluator.evaluate(y, p).keys())
    ext_keys = set(evaluator.extended_metrics(y, p, dates=d).keys())
    all_keys = base_keys | ext_keys

    for key in all_keys:
        assert key in HIGHER_IS_BETTER
        assert isinstance(HIGHER_IS_BETTER[key], bool)
