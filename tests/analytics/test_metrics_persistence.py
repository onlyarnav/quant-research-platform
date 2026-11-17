"""
Database persistence tests for metrics and trades.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.metrics_persistence import persist_metrics, persist_trades
from src.database.repository.portfolio_metric_repository import (
    PortfolioMetricRepository,
)
from src.database.sessions import get_session


@pytest.mark.db
def test_persist_metrics_and_trades_to_db() -> None:
    strategy_name = "test_persistence_strategy"
    metrics = {
        "sharpe_ratio": 1.75,
        "total_return": 0.25,
        "invalid_metric": float("nan"),
    }

    trades_df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "entry_date": pd.Timestamp("2024-01-01"),
                "exit_date": pd.Timestamp("2024-01-05"),
                "entry_price": 100.0,
                "exit_price": 105.0,
                "quantity": 10.0,
                "fees": 1.0,
                "slippage_cost": 0.5,
                "pnl": 50.0,
                "return_pct": 0.05,
            }
        ]
    )

    with get_session() as session:
        # Clean previous if existing
        metric_repo = PortfolioMetricRepository(session)
        metric_repo.delete_by_strategy(strategy_name)

        n_metrics = persist_metrics(session, strategy_name, metrics)
        assert n_metrics == 2  # skipped the NaN one

        n_trades = persist_trades(session, trades_df)
        assert n_trades == 1

        # Query back
        stored_metrics = metric_repo.get_by_strategy(strategy_name)
        assert len(stored_metrics) == 2
        metric_map = {m.metric: m.value for m in stored_metrics}
        assert metric_map["sharpe_ratio"] == pytest.approx(1.75)
        assert metric_map["total_return"] == pytest.approx(0.25)

        # Cleanup
        metric_repo.delete_by_strategy(strategy_name)
