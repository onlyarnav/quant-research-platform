"""
Unit tests for extended performance metrics.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.performance_metrics import PerformanceMetrics


def _sample_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": [100.0, 105.0, 95.0, 90.0, 102.0, 110.0],
            "daily_return": [0.0, 0.05, -0.095238, -0.052632, 0.133333, 0.078431],
            "drawdown": [0.0, 0.0, -0.095238, -0.142857, -0.028571, 0.0],
        }
    )


def test_extended_scalar_methods() -> None:
    pm = PerformanceMetrics()
    df = _sample_df()

    assert pm.best_day(df) == pytest.approx(0.133333, rel=1e-4)
    assert pm.worst_day(df) == pytest.approx(-0.095238, rel=1e-4)
    assert pm.positive_days_ratio(df) == pytest.approx(3 / 6)  # 3 positive out of 6


def test_max_drawdown_duration_days() -> None:
    pm = PerformanceMetrics()
    df = _sample_df()

    # Peak is 2024-01-02 (row 1, dd=0.0). Underwater: rows 2, 3, 4. Recovery: 2024-01-06 (row 5, dd=0.0).
    # Duration from peak (2024-01-02) to recovery (2024-01-06) is 4 days.
    dur = pm.max_drawdown_duration_days(df)
    assert dur == 4


def test_compute_all_extended_keys() -> None:
    pm = PerformanceMetrics()
    df = _sample_df()

    res = pm.compute_all(df, extended=True)

    expected_keys = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "value_at_risk_95",
        "conditional_value_at_risk_95",
        "best_day",
        "worst_day",
        "positive_days_ratio",
        "max_drawdown_duration_days",
    }
    assert set(res.keys()) == expected_keys
    assert res["max_drawdown_duration_days"] == 4.0
