"""
Tests for PortfolioSimulator.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.backtesting.portfolio_simulator import PortfolioSimulator


def make_signals(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    symbols, dates, signals = zip(*rows)
    return pd.DataFrame({
        "symbol": symbols,
        "date": pd.to_datetime(dates),
        "signal": signals,
    })


def make_prices(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    symbols, dates, closes = zip(*rows)
    return pd.DataFrame({
        "symbol": symbols,
        "date": pd.to_datetime(dates),
        "close": closes,
    })


# =============================================================================
# Input validation
# =============================================================================

def test_run_raises_on_empty_signals_df():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    prices = make_prices([("AAPL", "2024-01-01", 100)])
    with pytest.raises(ValueError, match="signals_df cannot be empty"):
        sim.run(pd.DataFrame(), prices)


def test_run_raises_on_empty_prices_df():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([("AAPL", "2024-01-01", 1)])
    with pytest.raises(ValueError, match="prices_df cannot be empty"):
        sim.run(signals, pd.DataFrame())


def test_run_raises_on_no_overlapping_rows():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([("AAPL", "2024-01-01", 1)])
    prices = make_prices([("AAPL", "2025-01-01", 100)])
    with pytest.raises(ValueError, match="No overlapping"):
        sim.run(signals, prices)


# =============================================================================
# Core single-symbol behavior
# =============================================================================

def test_simple_buy_sell_produces_one_trade_and_correct_history_length():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", 0),
        ("AAPL", "2024-01-03", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 105),
        ("AAPL", "2024-01-03", 110),
    ])

    result = sim.run(signals, prices)

    assert len(result.trades) == 1
    assert len(result.portfolio_history) == 3
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(100)
    assert trade.exit_price == pytest.approx(110)


def test_portfolio_value_increases_on_profitable_trade():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", 0),
        ("AAPL", "2024-01-03", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 105),
        ("AAPL", "2024-01-03", 110),
    ])

    result = sim.run(signals, prices)
    final_value = result.portfolio_history[-1].portfolio_value

    assert final_value > 100_000


def test_mark_to_market_reflected_mid_trade():
    """While holding, invested_capital should track the latest known price, not entry price."""
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", 0),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 120),
    ])

    result = sim.run(signals, prices)
    day2 = result.portfolio_history[1]

    # position_size = 50,000 / 100 = 500 shares; mark-to-market at 120 = 60,000
    assert day2.invested_capital == pytest.approx(60_000)
    assert day2.portfolio_value == pytest.approx(50_000 + 60_000)


# =============================================================================
# Capital constraints — the core value-add of this class
# =============================================================================

def test_insufficient_cash_skips_second_buy():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.6, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("MSFT", "2024-01-01", 1),  # 60% already allocated, 60% more needed -> insufficient
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("MSFT", "2024-01-01", 200),
    ])

    result = sim.run(signals, prices)

    assert len(result.trades) == 1
    assert result.trades[0].symbol == "AAPL"


def test_duplicate_buy_signal_is_noop():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.3, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", 1),
        ("AAPL", "2024-01-03", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 105),
        ("AAPL", "2024-01-03", 110),
    ])

    result = sim.run(signals, prices)

    assert len(result.trades) == 1
    assert result.trades[0].entry_price == pytest.approx(100)


def test_sell_without_open_position_is_noop():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.3, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", -1),
        ("AAPL", "2024-01-02", 1),
        ("AAPL", "2024-01-03", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 105),
        ("AAPL", "2024-01-03", 110),
    ])

    result = sim.run(signals, prices)

    assert len(result.trades) == 1
    assert result.trades[0].entry_price == pytest.approx(105)


# =============================================================================
# Multi-symbol shared cash
# =============================================================================

def test_multiple_symbols_share_cash_pool():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.3, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("MSFT", "2024-01-01", 1),
        ("AAPL", "2024-01-02", -1),
        ("MSFT", "2024-01-02", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("MSFT", "2024-01-01", 200),
        ("AAPL", "2024-01-02", 110),
        ("MSFT", "2024-01-02", 220),
    ])

    result = sim.run(signals, prices)

    assert len(result.trades) == 2
    assert set(result.trades_df["symbol"]) == {"AAPL", "MSFT"}
    # Both positions profitable -> cash should exceed initial capital
    assert sim.position_manager.cash > 100_000


# =============================================================================
# Force-close at end of data
# =============================================================================

def test_force_closes_open_position_at_end_of_data(caplog):
    caplog.set_level(logging.WARNING)
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", 0),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 110),
    ])

    result = sim.run(signals, prices)

    assert len(result.trades) == 1
    assert result.trades[0].exit_date == pd.Timestamp("2024-01-02")
    assert any("Force-closing" in rec.message for rec in caplog.records)


# =============================================================================
# Portfolio history fields
# =============================================================================

def test_cumulative_return_matches_value_ratio():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 110),
    ])

    result = sim.run(signals, prices)
    snapshot = result.portfolio_history[-1]

    expected = snapshot.portfolio_value / 100_000 - 1
    assert snapshot.cumulative_return == pytest.approx(expected)


def test_drawdown_is_zero_when_at_running_max():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 110),
    ])

    result = sim.run(signals, prices)
    # Profitable trade -> final value is a new running max -> drawdown == 0
    assert result.portfolio_history[-1].drawdown == pytest.approx(0.0)


def test_drawdown_negative_after_loss():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 90),  # losing trade
    ])

    result = sim.run(signals, prices)
    assert result.portfolio_history[-1].drawdown < 0.0


def test_portfolio_history_df_matches_history_list():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 110),
    ])
    result = sim.run(signals, prices)
    assert len(result.portfolio_history_df) == len(result.portfolio_history)
    for i, snapshot in enumerate(result.portfolio_history):
        row = result.portfolio_history_df.iloc[i]
        assert row["date"] == snapshot.date
        assert row["cash"] == pytest.approx(snapshot.cash)
        assert row["invested_capital"] == pytest.approx(snapshot.invested_capital)
        assert row["portfolio_value"] == pytest.approx(snapshot.portfolio_value)
        assert row["daily_return"] == pytest.approx(snapshot.daily_return)
        assert row["cumulative_return"] == pytest.approx(snapshot.cumulative_return)
        assert row["drawdown"] == pytest.approx(snapshot.drawdown)


# =============================================================================
# Additional coverage & validation tests (to bring total to 20 tests)
# =============================================================================

def test_run_raises_on_missing_signal_columns():
    sim = PortfolioSimulator(initial_capital=100_000)
    prices = make_prices([("AAPL", "2024-01-01", 100)])
    # signals_df is missing 'signal' column
    signals = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
    })
    with pytest.raises(ValueError, match="signals_df missing required columns"):
        sim.run(signals, prices)


def test_run_raises_on_missing_price_columns():
    sim = PortfolioSimulator(initial_capital=100_000)
    signals = make_signals([("AAPL", "2024-01-01", 1)])
    # prices_df is missing 'close' column
    prices = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": pd.to_datetime(["2024-01-01"]),
    })
    with pytest.raises(ValueError, match="prices_df missing required columns"):
        sim.run(signals, prices)


def test_transaction_cost_applied_to_trades_and_cash():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0.01, slippage=0)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 120),
    ])
    result = sim.run(signals, prices)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.fees == pytest.approx(1100.0)
    assert sim.position_manager.cash == pytest.approx(108900.0)


def test_slippage_applied_to_entry_exit_prices():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0, slippage=0.02)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 150),
    ])
    result = sim.run(signals, prices)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(102.0)
    assert trade.exit_price == pytest.approx(147.0)
    assert trade.slippage_cost == pytest.approx(5 * trade.position_size)


def test_trades_df_matches_trades_list():
    sim = PortfolioSimulator(initial_capital=100_000, max_position_size=0.5, transaction_cost=0.005, slippage=0.01)
    signals = make_signals([
        ("AAPL", "2024-01-01", 1),
        ("AAPL", "2024-01-02", -1),
    ])
    prices = make_prices([
        ("AAPL", "2024-01-01", 100),
        ("AAPL", "2024-01-02", 110),
    ])
    result = sim.run(signals, prices)
    assert len(result.trades_df) == len(result.trades)
    for i, trade in enumerate(result.trades):
        row = result.trades_df.iloc[i]
        assert row["trade_id"] == trade.trade_id
        assert row["symbol"] == trade.symbol
        assert row["entry_date"] == trade.entry_date
        assert row["exit_date"] == trade.exit_date
        assert row["entry_price"] == pytest.approx(trade.entry_price)
        assert row["exit_price"] == pytest.approx(trade.exit_price)
        assert row["position_size"] == pytest.approx(trade.position_size)
        assert row["fees"] == pytest.approx(trade.fees)
        assert row["slippage_cost"] == pytest.approx(trade.slippage_cost)
        assert row["pnl"] == pytest.approx(trade.pnl)
        assert row["return_pct"] == pytest.approx(trade.return_pct)