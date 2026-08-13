"""
Unit tests for BenchmarkComparison class.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.benchmark_comparison import BenchmarkComparison
from src.analytics.performance_metrics import PerformanceMetrics


def _sample_history_df(multiplier: float = 1.0) -> pd.DataFrame:
    """Helper to generate sample portfolio history DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    base_returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
    returns = base_returns * multiplier
    values = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": values,
            "daily_return": returns,
            "drawdown": [0.0, 0.0, -0.01, 0.0, 0.0],
        }
    )


def test_initialization_with_injected_metrics() -> None:
    """Test initialization with default and custom PerformanceMetrics."""
    bc_default = BenchmarkComparison()
    assert bc_default.performance_metrics is not None

    custom_pm = PerformanceMetrics(risk_free_rate=0.05, trading_days_per_year=250)
    bc_custom = BenchmarkComparison(performance_metrics=custom_pm)
    assert bc_custom.performance_metrics.risk_free_rate == pytest.approx(0.05)
    assert bc_custom.performance_metrics.trading_days_per_year == 250


def test_input_validation() -> None:
    """Test empty DataFrames, missing columns, and non-overlapping dates raise ValueError."""
    bc = BenchmarkComparison()
    valid_df = _sample_history_df()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="strategy_history_df cannot be None or empty"):
        bc.compare(empty_df, valid_df)

    with pytest.raises(ValueError, match="benchmark_history_df cannot be None or empty"):
        bc.compare(valid_df, empty_df)

    missing_cols_df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError, match="missing required columns"):
        bc.compare(missing_cols_df, valid_df)

    non_overlapping_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "portfolio_value": [100.0] * 5,
            "daily_return": [0.0] * 5,
            "drawdown": [0.0] * 5,
        }
    )
    with pytest.raises(ValueError, match="No overlapping dates"):
        bc.compare(valid_df, non_overlapping_df)


def test_identical_strategy_and_benchmark() -> None:
    """
    Test comparison where strategy mirrors benchmark exactly.

    Must yield alpha = 0.0, beta = 1.0, outperformance = 0.0.
    """
    bc = BenchmarkComparison()
    bench_df = _sample_history_df(multiplier=1.0)
    strat_df = bench_df.copy()

    res = bc.compare(strat_df, bench_df)

    assert res["alpha"] == pytest.approx(0.0, abs=1e-9)
    assert res["beta"] == pytest.approx(1.0, abs=1e-6)
    assert res["outperformance"] == pytest.approx(0.0, abs=1e-9)
    assert set(res.keys()) == {
        "strategy_metrics",
        "benchmark_metrics",
        "alpha",
        "beta",
        "information_ratio",
        "outperformance",
    }


def test_distinct_strategy_and_benchmark_golden() -> None:
    """
    Test comparison between distinct strategy and benchmark.

    Hand verification:
    strat returns 2x benchmark returns.
    beta should be 2.0.
    outperformance > 0.
    """
    bc = BenchmarkComparison()
    bench_df = _sample_history_df(multiplier=1.0)
    strat_df = _sample_history_df(multiplier=2.0)

    res = bc.compare(strat_df, bench_df)

    assert res["beta"] == pytest.approx(2.0, abs=1e-5)
    assert res["outperformance"] > 0.0
    assert res["alpha"] > 0.0
    assert not math.isnan(res["information_ratio"])


def test_flat_benchmark_zero_variance_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Test flat benchmark (zero return variance) produces NaN beta with warning logged."""
    bc = BenchmarkComparison()
    strat_df = _sample_history_df()
    flat_bench_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "portfolio_value": [100.0] * 5,
            "daily_return": [0.0] * 5,
            "drawdown": [0.0] * 5,
        }
    )

    res = bc.compare(strat_df, flat_bench_df)
    assert math.isnan(res["beta"])
    assert "Cannot compute beta: benchmark variance is zero or undefined" in caplog.text
