"""
Unit tests for Streamlit dashboard views.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.dashboard.views import (
    allocation,
    benchmark,
    experiments,
    overview,
    performance,
    pipeline,
    run_backtest,
    signals,
    trades,
)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.list_runs.return_value = [
        {
            "run_id": "run_1",
            "strategy_name": "test_strat",
            "created_at": "2024-01-01T00:00:00",
            "final_equity": 105.0,
        }
    ]
    client.get_run.return_value = {
        "run_id": "run_1",
        "strategy_name": "test_strat",
        "created_at": "2024-01-01T00:00:00",
        "final_equity": 105.0,
        "config": {"initial_capital": 100000.0},
    }
    client.get_metrics.return_value = {
        "metrics": {
            "total_return": 0.05,
            "annualized_return": 0.12,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
        },
        "trade_stats": {"total_trades": 10, "win_rate": 0.6, "profit_factor": 1.8},
        "benchmark": {"alpha": 0.03, "beta": 1.1},
    }
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    client.get_equity.return_value = pd.DataFrame(
        {"date": dates, "portfolio_value": [100.0, 102.0, 101.0, 103.0, 105.0]}
    )
    client.get_drawdown.return_value = pd.DataFrame(
        {"date": dates, "drawdown": [0.0, 0.0, -0.01, 0.0, 0.0]}
    )
    client.get_rolling.return_value = pd.DataFrame(
        {"date": dates, "rolling_sharpe": [1.5] * 5, "rolling_volatility": [0.15] * 5}
    )
    client.get_monthly_returns.return_value = pd.DataFrame(
        {"Year": [2024], "Jan": [0.05], "Total": [0.05]}
    )
    client.get_trades.return_value = (
        [
            {
                "trade_id": 1,
                "symbol": "AAPL",
                "entry_date": "2024-01-01",
                "exit_date": "2024-01-02",
                "pnl": 100.0,
                "return_pct": 0.05,
            }
        ],
        1,
    )
    client.get_trade_distribution.return_value = pd.DataFrame(
        {"bin_left": [0.0], "bin_right": [0.1], "count": [1]}
    )
    client.get_allocation.return_value = pd.DataFrame(
        {"symbol": ["AAPL"], "weight": [1.0], "market_value": [10000.0]}
    )
    client.get_benchmark.return_value = {"alpha": 0.02, "beta": 1.05}
    client.latest_signals.return_value = pd.DataFrame(
        {"date": dates, "symbol": ["AAPL"] * 5, "signal": [1] * 5}
    )
    client.signal_counts.return_value = (
        "2024-01-05",
        {"buy": 1, "neutral": 0, "sell": 0},
    )
    client.list_models.return_value = [
        {"name": "model_xgb", "version": "1.0", "created_at": "2024-01-01"}
    ]
    client.pipeline_runs.return_value = [
        {
            "run_id": "pipe_1",
            "started_at": "2024-01-01",
            "finished_at": "2024-01-01",
            "all_succeeded": True,
            "total_duration_seconds": 10.0,
        }
    ]
    client.health.return_value = {
        "overall": "ok",
        "checks": [{"name": "db", "status": "ok", "message": "connected"}],
    }
    client.job_status.return_value = {"job_id": "job_1", "status": "running"}
    return client


def test_views_render_without_exception(mock_client) -> None:
    # Exercise each view function with mock client
    overview.render(mock_client, run_id="run_1")
    performance.render(mock_client, run_id="run_1")
    trades.render(mock_client, run_id="run_1")
    allocation.render(mock_client, run_id="run_1")
    benchmark.render(mock_client, run_id="run_1")
    signals.render(mock_client, run_id="run_1")
    experiments.render(mock_client, run_id="run_1")
    pipeline.render(mock_client, run_id="run_1")
    run_backtest.render(mock_client, run_id="run_1")
