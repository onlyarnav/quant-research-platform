"""
Unit tests for ReportBuilder and StrategyReport.
"""

from __future__ import annotations

import json

import pandas as pd

from src.analytics.report_builder import ReportBuilder, StrategyReport


def _sample_hist() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "portfolio_value": [100.0 + i for i in range(10)],
            "daily_return": [0.0] + [0.01] * 9,
            "drawdown": [0.0] * 10,
        }
    )


def _sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": [1],
            "symbol": ["AAPL"],
            "entry_date": [pd.Timestamp("2024-01-01")],
            "exit_date": [pd.Timestamp("2024-01-05")],
            "entry_price": [100.0],
            "exit_price": [110.0],
            "position_size": [10.0],
            "pnl": [100.0],
            "return_pct": [0.10],
        }
    )


def test_report_builder_with_trades_and_benchmark() -> None:
    builder = ReportBuilder()
    hist = _sample_hist()
    trades = _sample_trades()
    bench = hist.copy()

    report = builder.build(
        portfolio_history_df=hist,
        trades_df=trades,
        benchmark_history_df=bench,
        benchmark_label="SPY",
    )

    assert isinstance(report, StrategyReport)
    assert report.benchmark_label == "SPY"
    assert report.benchmark is not None
    assert report.trade_stats["total_trades"] == 1
    assert "sharpe_ratio" in report.metrics

    d = report.to_dict()
    # Must be serializable to JSON without error
    json_str = json.dumps(d)
    assert len(json_str) > 0


def test_report_builder_without_trades_and_benchmark() -> None:
    builder = ReportBuilder()
    hist = _sample_hist()

    report = builder.build(
        portfolio_history_df=hist,
        trades_df=pd.DataFrame(),
        benchmark_history_df=None,
    )

    assert report.benchmark is None
    assert report.trade_stats == {}
    assert report.trade_distribution.empty
    assert report.pnl_by_symbol.empty

    d = report.to_dict()
    json_str = json.dumps(d)
    assert len(json_str) > 0
