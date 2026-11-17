"""
End-to-end integration tests for quantitative platform.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.backtest_pipeline import BacktestPipeline
from src.backtesting.backtest_config import (
    AllocationMethod,
    BacktestConfig,
    RebalanceFrequency,
)
from src.backtesting.backtest_store import BacktestStore


@pytest.mark.integration
def test_full_platform_end_to_end_run() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = BacktestStore(root=Path(tmp_dir))
        pipeline = BacktestPipeline(store=store)

        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        rng = np.random.default_rng(123)
        symbols = ["TCS", "INFY", "RELIANCE"]

        px_records = []
        sig_records = []
        for s in symbols:
            base = 100.0
            rets = rng.normal(0.001, 0.02, len(dates))
            px = base * np.exp(np.cumsum(rets))
            sigs = rng.choice([1, 0, -1], size=len(dates), p=[0.25, 0.50, 0.25])
            for d, p, sig in zip(dates, px, sigs):
                px_records.append({"date": d, "symbol": s, "close": float(p)})
                sig_records.append(
                    {
                        "date": d,
                        "symbol": s,
                        "signal": int(sig),
                        "confidence": float(rng.uniform(0.6, 0.9)),
                    }
                )

        prices_df = pd.DataFrame(px_records)
        signals_df = pd.DataFrame(sig_records)

        config = BacktestConfig(
            strategy_name="e2e_integration_test",
            allocation_method=AllocationMethod.EQUAL_WEIGHT,
            rebalance_frequency=RebalanceFrequency.DAILY,
            initial_capital=100_000.0,
            execution_lag=1,
            max_position_size=0.30,
            cash_reserve=0.05,
            stop_loss_pct=0.08,
            take_profit_pct=0.15,
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
        assert result.run_dir.exists()

        rep = result.report
        assert "sharpe_ratio" in rep.metrics
        assert "total_return" in rep.metrics
        assert not rep.equity.empty
        assert not rep.drawdown.empty

        # Verify disk artifacts
        assert (result.run_dir / "report.json").exists()
        assert (result.run_dir / "metrics.json").exists()
        assert (result.run_dir / "portfolio_history.parquet").exists()

        # Rebuild report from disk
        rebuilt = pipeline.rebuild_report(result.run_id)
        assert rebuilt.metrics["total_return"] == pytest.approx(
            rep.metrics["total_return"]
        )
