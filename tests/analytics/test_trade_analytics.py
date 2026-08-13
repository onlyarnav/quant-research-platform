"""
Unit tests for TradeAnalytics class.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.analytics.trade_analytics import TradeAnalytics


def _sample_trades_df() -> pd.DataFrame:
    """Helper to create a standard synthetic trades DataFrame with hand-calculable values."""
    return pd.DataFrame(
        {
            "pnl": [100.0, 200.0, -50.0, 0.0],
            "return_pct": [0.10, 0.20, -0.05, 0.00],
            "entry_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "exit_date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]),
        }
    )


def test_trade_analytics_instantiation() -> None:
    """Test TradeAnalytics instantiation."""
    ta = TradeAnalytics()
    assert ta is not None


def test_input_validation_empty_and_missing_columns() -> None:
    """Test empty DataFrame and missing columns raise ValueError."""
    ta = TradeAnalytics()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="cannot be None or empty"):
        ta.win_rate(empty_df)

    missing_cols_df = pd.DataFrame({"pnl": [100.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        ta.win_rate(missing_cols_df)


def test_golden_metrics() -> None:
    """
    Test golden metric calculations with hand-calculated expected values.

    Hand calculations for trades PnL [100.0, 200.0, -50.0, 0.0]:
    total_trades = 4
    win_rate = 2 / 4 = 0.50
    loss_rate = 1 / 4 = 0.25 (win_rate + loss_rate = 0.75 due to 1 breakeven trade)
    average_win = (100 + 200) / 2 = 150.0
    average_loss = -50.0 / 1 = -50.0
    profit_factor = 300.0 / abs(-50.0) = 6.0
    expectancy = (0.50 * 150.0) + (0.25 * -50.0) = 75.0 - 12.5 = 62.5
    total_pnl = 100 + 200 - 50 + 0 = 250.0
    average_trade_duration = 2 days
    """
    ta = TradeAnalytics()
    df = _sample_trades_df()

    assert ta.total_trades(df) == 4
    assert ta.win_rate(df) == pytest.approx(0.50)
    assert ta.loss_rate(df) == pytest.approx(0.25)
    assert ta.average_win(df) == pytest.approx(150.0)
    assert ta.average_loss(df) == pytest.approx(-50.0)
    assert ta.profit_factor(df) == pytest.approx(6.0)
    assert ta.expectancy(df) == pytest.approx(62.5)
    assert ta.total_pnl(df) == pytest.approx(250.0)
    assert ta.average_trade_duration(df) == pd.Timedelta(2, unit="D")


def test_zero_winning_trades_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Test zero winning trades logs warning and returns 0.0 for average_win."""
    ta = TradeAnalytics()
    losing_df = pd.DataFrame(
        {
            "pnl": [-50.0, -100.0],
            "return_pct": [-0.05, -0.10],
            "entry_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "exit_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )
    res = ta.average_win(losing_df)
    assert res == pytest.approx(0.0)
    assert "No winning trades found; average win is 0.0" in caplog.text


def test_zero_losing_trades_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Test zero losing trades logs warnings and returns 0.0 for average_loss and inf for profit_factor."""
    ta = TradeAnalytics()
    winning_df = pd.DataFrame(
        {
            "pnl": [100.0, 200.0],
            "return_pct": [0.10, 0.20],
            "entry_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "exit_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )
    res_avg_loss = ta.average_loss(winning_df)
    assert res_avg_loss == pytest.approx(0.0)
    assert "No losing trades found; average loss is 0.0" in caplog.text

    res_pf = ta.profit_factor(winning_df)
    assert math.isinf(res_pf) and res_pf > 0
    assert "No losing trades; profit factor undefined (infinite)" in caplog.text


def test_compute_all_keys_and_spot_check() -> None:
    """Test compute_all returns dict with expected keys and matching values."""
    ta = TradeAnalytics()
    df = _sample_trades_df()
    result = ta.compute_all(df)

    expected_keys = {
        "total_trades",
        "win_rate",
        "loss_rate",
        "average_win",
        "average_loss",
        "profit_factor",
        "expectancy",
        "average_trade_duration",
        "total_pnl",
    }
    assert set(result.keys()) == expected_keys
    assert result["total_trades"] == 4
    assert result["expectancy"] == pytest.approx(ta.expectancy(df))
    assert result["average_trade_duration"] == ta.average_trade_duration(df)
