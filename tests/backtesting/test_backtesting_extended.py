"""
Extended tests for the backtesting and portfolio simulation layer.

Verifies:
- Hand-computed single-trade cost, fee, slippage, and PnL to the paisa
- Empty signals producing a flat equity curve at initial capital
- Strict schema and column completeness for trade log and portfolio history
- Drawdown calculations on a crafted price curve
- Cash non-negativity invariant across randomized market trajectories
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtesting_engine import BacktestingEngine
from src.backtesting.portfolio_simulator import PortfolioSimulator


def test_hand_computed_single_trade_to_the_paisa() -> None:
    """
    Verify fees, slippage, pnl, and return_pct down to 2 decimal places (paisa).

    Setup:
      Initial capital: 1,000,000 INR
      Max position size: 5% -> 50,000 INR
      Slippage: 0.0005 (5 bps)
      Transaction cost: 0.001 (10 bps)
      Raw entry price: 100.00 INR -> Executed entry: 100.05 INR
      Position size: 50,000 / 100.05 = 499.750125 shares
      Raw exit price: 120.00 INR -> Executed exit: 119.94 INR
      Gross PnL: (119.94 - 100.05) * pos = 9,940.03 INR
      Turnover: (100.05 + 119.94) * pos = 109,940.03 INR
      Fees: 0.001 * 109,940.03 = 109.94 INR
      Net PnL: 9,940.03 - 109.94 = 9,830.09 INR
      Slippage cost: 0.11 * pos = 54.97 INR
    """
    sim = PortfolioSimulator(
        initial_capital=1_000_000.0,
        max_position_size=0.05,
        transaction_cost=0.001,
        slippage=0.0005,
    )
    signals = pd.DataFrame(
        {
            "symbol": ["INFY", "INFY"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "signal": [1, -1],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["INFY", "INFY"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "close": [100.0, 120.0],
        }
    )

    result = sim.run(signals, prices)
    assert len(result.trades) == 1
    t = result.trades[0]

    assert round(t.entry_price, 4) == 100.0500
    assert round(t.exit_price, 4) == 119.9400
    assert round(t.position_size, 4) == 499.7501
    assert round(t.fees, 2) == 109.94
    assert round(t.slippage_cost, 2) == 54.97
    assert round(t.pnl, 2) == 9830.09
    assert round(t.return_pct, 4) == round(9830.09 / 50000.0, 4)


def test_empty_signals_returns_flat_equity_curve() -> None:
    """Holding no positions across time returns an exactly flat equity curve."""
    sim = PortfolioSimulator(initial_capital=500_000.0)
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    signals = pd.DataFrame(
        {
            "symbol": ["TCS"] * 5,
            "date": dates,
            "signal": [0, 0, 0, 0, 0],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["TCS"] * 5,
            "date": dates,
            "close": [3000.0, 3050.0, 2980.0, 3100.0, 3120.0],
        }
    )

    result = sim.run(signals, prices)
    assert len(result.trades) == 0
    df = result.portfolio_history_df
    assert len(df) == 5
    assert (df["cash"] == 500_000.0).all()
    assert (df["invested_capital"] == 0.0).all()
    assert (df["portfolio_value"] == 500_000.0).all()
    assert (df["daily_return"] == 0.0).all()
    assert (df["cumulative_return"] == 0.0).all()
    assert (df["drawdown"] == 0.0).all()


def test_schema_and_column_completeness() -> None:
    """Verify trade log and portfolio history strictly match required schemas."""
    engine = BacktestingEngine()
    signals = pd.DataFrame(
        {
            "symbol": ["SBIN", "SBIN"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "signal": [1, -1],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["SBIN", "SBIN"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "close": [500.0, 520.0],
        }
    )

    result = engine.run(signals, prices)
    expected_trade_cols = {
        "trade_id",
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "position_size",
        "fees",
        "slippage_cost",
        "pnl",
        "return_pct",
    }
    assert set(result.trades_df.columns) == expected_trade_cols

    sim = PortfolioSimulator()
    sim_res = sim.run(signals, prices)
    expected_history_cols = {
        "date",
        "cash",
        "invested_capital",
        "portfolio_value",
        "daily_return",
        "cumulative_return",
        "drawdown",
    }
    assert set(sim_res.portfolio_history_df.columns) == expected_history_cols


def test_drawdown_calculation_on_crafted_curve() -> None:
    """Verify drawdown peak tracking and negative drawdown percentages."""
    sim = PortfolioSimulator(
        initial_capital=100_000.0,
        max_position_size=1.0,
        transaction_cost=0.0,
        slippage=0.0,
    )
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    signals = pd.DataFrame(
        {
            "symbol": ["RELIANCE"] * 4,
            "date": dates,
            "signal": [1, 0, 0, -1],
        }
    )
    # Peak on day 2 at 120 -> 25% drop to 90 on day 3
    prices = pd.DataFrame(
        {
            "symbol": ["RELIANCE"] * 4,
            "date": dates,
            "close": [100.0, 120.0, 90.0, 110.0],
        }
    )

    result = sim.run(signals, prices)
    history = result.portfolio_history
    # Day 0: value = 100k, dd = 0
    assert history[0].drawdown == 0.0
    # Day 1: value = 120k (peak), dd = 0
    assert history[1].drawdown == 0.0
    # Day 2: value = 90k, peak = 120k -> dd = (90 - 120) / 120 = -0.25 (-25%)
    assert history[2].drawdown == pytest.approx(-0.25)
    # Day 3: value = 110k, peak = 120k -> dd = (110 - 120) / 120 = -0.0833
    assert history[3].drawdown == pytest.approx(-10.0 / 120.0)


def test_cash_never_negative_under_random_signals() -> None:
    """Verify cash pool remains strictly non-negative during random trading."""
    rng = np.random.default_rng(42)
    sim = PortfolioSimulator(
        initial_capital=50_000.0,
        max_position_size=0.10,
        transaction_cost=0.002,
        slippage=0.001,
    )
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    symbols = ["SYM_A", "SYM_B", "SYM_C"]

    records_sig = []
    records_px = []
    for sym in symbols:
        price = 100.0
        for d in dates:
            sig = rng.choice([-1, 0, 1], p=[0.25, 0.50, 0.25])
            price = max(10.0, price + rng.normal(0, 2))
            records_sig.append({"symbol": sym, "date": d, "signal": sig})
            records_px.append({"symbol": sym, "date": d, "close": price})

    signals_df = pd.DataFrame(records_sig)
    prices_df = pd.DataFrame(records_px)

    res = sim.run(signals_df, prices_df)
    for snap in res.portfolio_history:
        assert snap.cash >= 0.0, f"Cash negative on {snap.date}: {snap.cash}"
