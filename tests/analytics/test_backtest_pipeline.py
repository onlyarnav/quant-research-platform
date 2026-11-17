"""
Unit tests for BacktestPipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.backtest_pipeline import BacktestPipeline
from src.backtesting.backtest_config import (
    AllocationMethod,
    BacktestConfig,
    RebalanceFrequency,
)
from src.backtesting.backtest_store import BacktestStore


def _sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    px_records = []
    sig_records = []
    for d in dates:
        px_records.append(
            {
                "date": d,
                "symbol": "AAPL",
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0 + d.day,
                "volume": 1000,
            }
        )
        sig_records.append(
            {
                "date": d,
                "symbol": "AAPL",
                "signal": 1 if d.day in [1, 2] else (0 if d.day in [3, 4] else -1),
            }
        )
    return pd.DataFrame(px_records), pd.DataFrame(sig_records)


def test_backtest_pipeline_legacy_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BacktestStore(root=Path(tmp_dir))
        pipeline = BacktestPipeline(store=store)
        prices_df, signals_df = _sample_data()

        config = BacktestConfig(
            strategy_name="test_legacy",
            allocation_method=AllocationMethod.FIXED_NOTIONAL,
            initial_capital=50_000.0,
            rebalance_frequency=RebalanceFrequency.ON_SIGNAL,
            rebalance_tolerance=0.0,
            stop_loss_pct=None,
            take_profit_pct=None,
            execution_lag=0,
            allow_short=False,
            cash_reserve=0.0,
        )

        result = pipeline.run(
            config=config,
            signals_df=signals_df,
            prices_df=prices_df,
            persist=True,
            persist_db=False,
        )

        assert result.run_id.startswith("run_")
        assert result.run_dir is not None
        assert result.output.portfolio_history_df is not None
        assert not result.output.portfolio_history_df.empty
        assert "sharpe_ratio" in result.report.metrics

        # Rebuild report
        rebuilt = pipeline.rebuild_report(result.run_id)
        assert rebuilt.metrics["total_return"] == pytest.approx(
            result.report.metrics["total_return"]
        )


def test_backtest_pipeline_weighted_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BacktestStore(root=Path(tmp_dir))
        pipeline = BacktestPipeline(store=store)
        prices_df, signals_df = _sample_data()

        config = BacktestConfig(
            strategy_name="test_weighted",
            allocation_method=AllocationMethod.EQUAL_WEIGHT,
            initial_capital=100_000.0,
            execution_lag=1,
            rebalance_frequency="daily",
        )

        result = pipeline.run(
            config=config,
            signals_df=signals_df,
            prices_df=prices_df,
            persist=True,
            persist_db=False,
        )

        assert result.output.daily_details_df is not None
        assert len(result.output.daily_details_df) == len(
            result.output.portfolio_history_df
        )
        assert result.report.trade_stats is not None
