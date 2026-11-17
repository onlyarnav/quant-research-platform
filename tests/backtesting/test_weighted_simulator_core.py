"""
Core unit and golden path tests for WeightedPortfolioSimulator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.backtest_config import (
    AllocationMethod,
    BacktestConfig,
    RebalanceFrequency,
)
from src.backtesting.weighted_simulator import WeightedPortfolioSimulator


def test_fixed_notional_golden_single_symbol() -> None:
    """
    Test fixed-notional execution for a single symbol.

    Capital: 1,000,000, max_pos: 0.10 -> notional = 100,000
    Entry @ 100.0, Exit @ 110.0, cost = 0, slippage = 0
    Shares = 100,000 / 100 = 1,000
    Gross PnL = (110 - 100) * 1000 = 10,000
    Return pct = 10,000 / 100,000 = 10.0%
    """
    cfg = BacktestConfig(
        initial_capital=1_000_000.0,
        max_position_size=0.10,
        transaction_cost=0.0,
        slippage=0.0,
        allocation_method=AllocationMethod.FIXED_NOTIONAL,
        rebalance_frequency=RebalanceFrequency.ON_SIGNAL,
    )
    sim = WeightedPortfolioSimulator(cfg)

    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    signals_df = pd.DataFrame(
        {
            "symbol": ["XYZ", "XYZ", "XYZ"],
            "date": dates,
            "signal": [1, 0, -1],
        }
    )
    prices_df = pd.DataFrame(
        {
            "symbol": ["XYZ", "XYZ", "XYZ"],
            "date": dates,
            "close": [100.0, 105.0, 110.0],
        }
    )

    out = sim.run(signals_df, prices_df)
    assert len(out.trades) == 1
    t = out.trades[0]
    assert t.symbol == "XYZ"
    assert t.position_size == pytest.approx(1000.0)
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(110.0)
    assert t.pnl == pytest.approx(10000.0)
    assert t.return_pct == pytest.approx(0.10)
    assert out.summary["final_equity"] == pytest.approx(1_010_000.0)


def test_frozen_validation_messages() -> None:
    """Test required platform error messages match frozen specs."""
    cfg = BacktestConfig(allocation_method=AllocationMethod.FIXED_NOTIONAL)
    sim = WeightedPortfolioSimulator(cfg)

    valid_px = pd.DataFrame(
        {"symbol": ["A"], "date": [pd.Timestamp("2024-01-01")], "close": [100.0]}
    )
    valid_sig = pd.DataFrame(
        {"symbol": ["A"], "date": [pd.Timestamp("2024-01-01")], "signal": [1]}
    )

    with pytest.raises(ValueError, match="signals_df cannot be empty"):
        sim.run(pd.DataFrame(), valid_px)

    with pytest.raises(ValueError, match="prices_df cannot be empty"):
        sim.run(valid_sig, pd.DataFrame())

    with pytest.raises(ValueError, match="signals_df missing required columns"):
        sim.run(pd.DataFrame({"symbol": ["A"]}), valid_px)

    with pytest.raises(ValueError, match="prices_df missing required columns"):
        sim.run(valid_sig, pd.DataFrame({"symbol": ["A"]}))

    # Non-overlapping
    px_diff = pd.DataFrame(
        {"symbol": ["B"], "date": [pd.Timestamp("2024-01-02")], "close": [100.0]}
    )
    with pytest.raises(ValueError, match="No overlapping \\(symbol, date\\) rows"):
        sim.run(valid_sig, px_diff)


def test_zero_trades_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test simulation with no BUY signals logs zero trades warning."""
    cfg = BacktestConfig(allocation_method=AllocationMethod.FIXED_NOTIONAL)
    sim = WeightedPortfolioSimulator(cfg)

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    sig = pd.DataFrame({"symbol": ["A", "A"], "date": dates, "signal": [0, 0]})
    px = pd.DataFrame({"symbol": ["A", "A"], "date": dates, "close": [100.0, 102.0]})

    out = sim.run(sig, px)
    assert len(out.trades) == 0
    assert "zero trades" in caplog.text


def test_force_close_at_end_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test open positions at end are force-closed with Force-closing warning."""
    cfg = BacktestConfig(
        allocation_method=AllocationMethod.FIXED_NOTIONAL,
        force_close_at_end=True,
    )
    sim = WeightedPortfolioSimulator(cfg)

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    sig = pd.DataFrame({"symbol": ["A", "A"], "date": dates, "signal": [1, 0]})
    px = pd.DataFrame({"symbol": ["A", "A"], "date": dates, "close": [100.0, 105.0]})

    out = sim.run(sig, px)
    assert len(out.trades) == 1
    assert "Force-closing" in caplog.text
    assert out.trade_details_df["exit_reason"].iloc[0] == "forced_close_end"
