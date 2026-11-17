"""
Unit tests for extended trade analytics.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.trade_analytics import TradeAnalytics


def _sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": [1, 2, 3, 4, 5],
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT", "GOOG"],
            "entry_date": pd.date_range("2024-01-01", periods=5, freq="2D"),
            "exit_date": pd.date_range("2024-01-02", periods=5, freq="2D"),
            "entry_price": [100.0, 105.0, 200.0, 210.0, 150.0],
            "exit_price": [110.0, 100.0, 220.0, 200.0, 160.0],
            "position_size": [10.0, 10.0, 5.0, 5.0, 8.0],
            "pnl": [100.0, -50.0, 100.0, -50.0, 80.0],
            "return_pct": [0.10, -0.0476, 0.10, -0.0476, 0.0667],
        }
    )


def test_payoff_and_best_worst() -> None:
    ta = TradeAnalytics()
    df = _sample_trades()

    # Winning: 100, 100, 80 -> avg = 93.333
    # Losing: -50, -50 -> avg = -50.0
    # Payoff = 93.333 / 50 = 1.8666
    assert ta.payoff_ratio(df) == pytest.approx(93.33333 / 50.0, rel=1e-3)
    assert ta.best_trade(df) == pytest.approx(100.0)
    assert ta.worst_trade(df) == pytest.approx(-50.0)


def test_consecutive_streaks() -> None:
    ta = TradeAnalytics()
    df = _sample_trades()

    # Sequence of PnL: +100, -50, +100, -50, +80
    assert ta.max_consecutive_wins(df) == 1
    assert ta.max_consecutive_losses(df) == 1

    streak_df = df.copy()
    streak_df["pnl"] = [10.0, 20.0, 30.0, -5.0, -10.0]
    assert ta.max_consecutive_wins(streak_df) == 3
    assert ta.max_consecutive_losses(streak_df) == 2


def test_pnl_by_symbol_and_distribution() -> None:
    ta = TradeAnalytics()
    df = _sample_trades()

    sym_df = ta.pnl_by_symbol(df)
    assert set(sym_df["symbol"]) == {"AAPL", "MSFT", "GOOG"}
    assert "total_pnl" in sym_df.columns
    assert "win_rate" in sym_df.columns

    dist_df = ta.return_distribution(df, bins=5)
    assert len(dist_df) == 5
    assert "bin_left" in dist_df.columns
    assert "count" in dist_df.columns
    assert dist_df["count"].sum() == 5


def test_compute_all_extended() -> None:
    ta = TradeAnalytics()
    df = _sample_trades()

    res = ta.compute_all(df, extended=True)
    expected_keys = {
        "total_trades",
        "win_rate",
        "loss_rate",
        "average_win",
        "average_loss",
        "profit_factor",
        "expectancy",
        "average_trade_duration_days",
        "total_pnl",
        "payoff_ratio",
        "best_trade",
        "worst_trade",
        "max_consecutive_wins",
        "max_consecutive_losses",
    }
    assert set(res.keys()) == expected_keys
    assert isinstance(res["average_trade_duration_days"], float)
