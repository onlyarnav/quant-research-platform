"""
Unit tests for BenchmarkBuilder and extended BenchmarkComparison.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.benchmark_builder import BenchmarkBuilder
from src.analytics.benchmark_comparison import BenchmarkComparison


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    records = []
    for d in dates:
        records.append({"date": d, "symbol": "AAPL", "close": 150.0 + (d.day * 2)})
        records.append({"date": d, "symbol": "MSFT", "close": 300.0 + (d.day * 1.5)})
    return pd.DataFrame(records)


def test_benchmark_builder_equal_weight_universe() -> None:
    builder = BenchmarkBuilder()
    prices = _sample_prices()

    bench = builder.equal_weight_universe(prices, initial_capital=100_000.0)
    assert len(bench) == 10
    assert list(bench.columns) == [
        "date",
        "portfolio_value",
        "cash",
        "invested_capital",
        "daily_return",
        "drawdown",
        "total_return",
    ]
    assert bench["portfolio_value"].iloc[0] == pytest.approx(100_000.0)
    assert bench["cash"].iloc[0] == 0.0


def test_benchmark_builder_build_fallback_universe() -> None:
    builder = BenchmarkBuilder()
    prices = _sample_prices()
    dates = pd.date_range("2024-01-03", periods=5, freq="D")

    bench, label = builder.build(
        prices_df=prices, initial_capital=50_000.0, dates=dates, prefer_index=False
    )
    assert label == "Equal-Weight Universe"
    assert len(bench) == 5
    assert (bench["date"] == dates).all()


def test_benchmark_comparison_extended() -> None:
    bc = BenchmarkComparison()
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    strat_ret = [0.01, -0.005, 0.02, 0.015, -0.01, 0.005, 0.01, -0.002, 0.03, 0.005]
    bench_ret = [0.005, -0.002, 0.01, 0.008, -0.005, 0.002, 0.005, -0.001, 0.015, 0.002]

    strat_pv = 100.0 * np.cumprod(1.0 + np.array(strat_ret))
    bench_pv = 100.0 * np.cumprod(1.0 + np.array(bench_ret))

    s_df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": strat_pv,
            "daily_return": strat_ret,
            "drawdown": 0.0,
        }
    )
    b_df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": bench_pv,
            "daily_return": bench_ret,
            "drawdown": 0.0,
        }
    )

    res = bc.compare(s_df, b_df, extended=True)

    expected_extended_keys = {
        "strategy_metrics",
        "benchmark_metrics",
        "alpha",
        "beta",
        "information_ratio",
        "outperformance",
        "excess_return",
        "alpha_jensen",
        "tracking_error",
        "correlation",
        "up_capture",
        "down_capture",
    }
    assert expected_extended_keys.issubset(set(res.keys()))
    assert res["beta"] == pytest.approx(2.0, rel=1e-2)
    assert res["tracking_error"] > 0.0
    assert res["correlation"] > 0.99


def test_identical_strategy_and_benchmark_extended() -> None:
    bc = BenchmarkComparison()
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    bench_ret = [0.005, -0.002, 0.01, 0.008, -0.005, 0.002, 0.005, -0.001, 0.015, 0.002]
    bench_pv = 100.0 * np.cumprod(1.0 + np.array(bench_ret))

    df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": bench_pv,
            "daily_return": bench_ret,
            "drawdown": 0.0,
        }
    )

    res = bc.compare(df, df.copy(), extended=True)

    assert res["beta"] == pytest.approx(1.0, abs=1e-5)
    assert res["alpha_jensen"] == pytest.approx(0.0, abs=1e-5)
    assert res["excess_return"] == pytest.approx(0.0, abs=1e-5)
    assert res["tracking_error"] == pytest.approx(0.0, abs=1e-5)
    assert math.isnan(res["information_ratio"])
