"""
Tests for stop-loss, take-profit, execution lag, and shorting in WeightedPortfolioSimulator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.backtest_config import (
    AllocationMethod,
    BacktestConfig,
)
from src.backtesting.weighted_simulator import WeightedPortfolioSimulator


def test_stop_loss_trigger() -> None:
    """Test stop-loss closes position when close drops by more than stop_loss_pct."""
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        max_position_size=1.0,
        transaction_cost=0.0,
        slippage=0.0,
        allocation_method=AllocationMethod.FIXED_NOTIONAL,
        stop_loss_pct=0.05,  # 5% stop
        force_close_at_end=False,
    )
    sim = WeightedPortfolioSimulator(cfg)

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    # Buy on day 1, day 2 drop 3%, day 3 drop 6% (triggers stop), day 4 rebound
    sig = pd.DataFrame({"symbol": ["A"] * 4, "date": dates, "signal": [1, 0, 0, 0]})
    px = pd.DataFrame(
        {"symbol": ["A"] * 4, "date": dates, "close": [100.0, 97.0, 94.0, 99.0]}
    )

    out = sim.run(sig, px)
    assert len(out.trades) == 1
    t = out.trades[0]
    assert t.exit_date == dates[2]
    assert t.exit_price == pytest.approx(94.0)
    assert out.trade_details_df["exit_reason"].iloc[0] == "stop_loss"


def test_take_profit_trigger() -> None:
    """Test take-profit closes position when close rises by more than take_profit_pct."""
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        max_position_size=1.0,
        transaction_cost=0.0,
        slippage=0.0,
        allocation_method=AllocationMethod.FIXED_NOTIONAL,
        take_profit_pct=0.10,  # 10% TP
        force_close_at_end=False,
    )
    sim = WeightedPortfolioSimulator(cfg)

    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    sig = pd.DataFrame({"symbol": ["A"] * 3, "date": dates, "signal": [1, 0, 0]})
    px = pd.DataFrame(
        {"symbol": ["A"] * 3, "date": dates, "close": [100.0, 105.0, 112.0]}
    )

    out = sim.run(sig, px)
    assert len(out.trades) == 1
    t = out.trades[0]
    assert t.exit_date == dates[2]
    assert t.exit_price == pytest.approx(112.0)
    assert out.trade_details_df["exit_reason"].iloc[0] == "take_profit"


def test_execution_lag() -> None:
    """Test signal execution is lagged by execution_lag bars."""
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        max_position_size=1.0,
        transaction_cost=0.0,
        slippage=0.0,
        allocation_method=AllocationMethod.FIXED_NOTIONAL,
        execution_lag=1,
        force_close_at_end=False,
    )
    sim = WeightedPortfolioSimulator(cfg)

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    # Signal on day 1 (date 0), should execute on day 2 (date 1)
    sig = pd.DataFrame({"symbol": ["A"] * 4, "date": dates, "signal": [1, 0, -1, 0]})
    px = pd.DataFrame(
        {"symbol": ["A"] * 4, "date": dates, "close": [100.0, 102.0, 108.0, 105.0]}
    )

    out = sim.run(sig, px)
    assert len(out.trades) == 1
    t = out.trades[0]
    assert t.entry_date == dates[1]
    assert t.entry_price == pytest.approx(102.0)
    # Sell signal on day 3 (dates[2]) executes on day 4 (dates[3])
    assert t.exit_date == dates[3]
    assert t.exit_price == pytest.approx(105.0)


def test_shorting_pnl_golden() -> None:
    """
    Test short position execution and PnL calculation.

    Initial capital: 100,000, max_pos: 1.0 -> notional = 100,000
    Open short @ 100.0, Cover short @ 90.0, zero cost/slippage.
    Shares = 1000
    Gross PnL = (100 - 90) * 1000 = +10,000
    """
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        max_position_size=1.0,
        transaction_cost=0.0,
        slippage=0.0,
        allow_short=True,
        allocation_method=AllocationMethod.FIXED_NOTIONAL,
        force_close_at_end=False,
    )
    sim = WeightedPortfolioSimulator(cfg)

    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    sig = pd.DataFrame({"symbol": ["A"] * 3, "date": dates, "signal": [-1, 0, 1]})
    px = pd.DataFrame(
        {"symbol": ["A"] * 3, "date": dates, "close": [100.0, 95.0, 90.0]}
    )

    out = sim.run(sig, px)
    assert len(out.trades) == 1
    t = out.trades[0]
    assert t.pnl == pytest.approx(10000.0)
    assert t.return_pct == pytest.approx(0.10)
    assert out.trade_details_df["side"].iloc[0] == "short"
