"""
Tests for BacktestingEngine.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.backtesting.backtesting_engine import BacktestingEngine


# =============================================================================
# Helpers
# =============================================================================

def make_signals(symbol: str, dates: list[str], signals: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": [symbol] * len(dates),
        "date": pd.to_datetime(dates),
        "signal": signals,
    })


def make_prices(
    symbol: str,
    dates: list[str],
    closes: list[float],
    adj_closes: list[float] | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "symbol": [symbol] * len(dates),
        "date": pd.to_datetime(dates),
        "close": closes,
    })
    if adj_closes is not None:
        df["adj_close"] = adj_closes
    return df


DATES_3 = ["2024-01-01", "2024-01-02", "2024-01-03"]
DATES_4 = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]


# =============================================================================
# Input validation
# =============================================================================

def test_run_raises_on_empty_signals_df():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    with pytest.raises(ValueError, match="signals_df cannot be empty"):
        engine.run(pd.DataFrame(), prices)


def test_run_raises_on_empty_prices_df():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])

    with pytest.raises(ValueError, match="prices_df cannot be empty"):
        engine.run(signals, pd.DataFrame())


def test_run_raises_on_missing_signal_columns():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1]).drop(columns=["signal"])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    with pytest.raises(ValueError, match="signals_df missing required columns"):
        engine.run(signals, prices)


def test_run_raises_on_missing_price_columns():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110]).drop(columns=["close"])

    with pytest.raises(ValueError, match="prices_df missing required columns"):
        engine.run(signals, prices)


def test_run_raises_on_no_overlapping_dates():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices(
        "AAPL",
        ["2025-01-01", "2025-01-02", "2025-01-03"],
        [100, 105, 110],
    )

    with pytest.raises(ValueError, match="No overlapping"):
        engine.run(signals, prices)


# =============================================================================
# Core trade lifecycle — single symbol, hand-verifiable
# =============================================================================

def test_simple_buy_sell_produces_one_trade():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    result = engine.run(signals, prices)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(100)
    assert trade.exit_price == pytest.approx(110)


def test_pnl_calculation_with_zero_costs():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    result = engine.run(signals, prices)
    trade = result.trades[0]

    position_size = 10000 / 100
    expected_pnl = (110 - 100) * position_size

    assert trade.pnl == pytest.approx(expected_pnl)


def test_hold_signal_does_not_trigger_trade(caplog):
    caplog.set_level(logging.WARNING)
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_4, [1, 0, 0, 0])
    prices = make_prices("AAPL", DATES_4, [100, 101, 102, 103])

    result = engine.run(signals, prices)

    # Position never explicitly closed via SELL -> force-closed at end
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_date == pd.Timestamp("2024-01-04")
    assert any("Force-closing" in rec.message for rec in caplog.records)


# =============================================================================
# State machine edge cases
# =============================================================================

def test_duplicate_buy_signal_is_noop():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 1, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    result = engine.run(signals, prices)

    assert len(result.trades) == 1
    # Entry price must come from the FIRST buy (100), not the second (105)
    assert result.trades[0].entry_price == pytest.approx(100)


def test_duplicate_sell_signal_is_noop():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, -1, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    result = engine.run(signals, prices)

    assert len(result.trades) == 1
    assert result.trades[0].exit_price == pytest.approx(105)


def test_sell_without_prior_buy_is_noop():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_4, [-1, 0, 1, -1])
    prices = make_prices("AAPL", DATES_4, [100, 101, 102, 103])

    result = engine.run(signals, prices)

    # First SELL (while flat) is ignored; only the BUY/SELL pair on
    # days 3 and 4 produces a trade.
    assert len(result.trades) == 1
    assert result.trades[0].entry_price == pytest.approx(102)
    assert result.trades[0].exit_price == pytest.approx(103)


def test_force_closes_open_position_at_end_of_data(caplog):
    caplog.set_level(logging.WARNING)
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, 0])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    result = engine.run(signals, prices)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_date == pd.Timestamp("2024-01-03")
    assert trade.exit_price == pytest.approx(110)
    assert any("Force-closing" in rec.message for rec in caplog.records)


def test_multiple_buy_sell_cycles_same_symbol():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_4, [1, -1, 1, -1])
    prices = make_prices("AAPL", DATES_4, [100, 105, 110, 115])

    result = engine.run(signals, prices)

    assert len(result.trades) == 2

    first, second = result.trades
    assert first.entry_price == pytest.approx(100)
    assert first.exit_price == pytest.approx(105)
    assert second.entry_price == pytest.approx(110)
    assert second.exit_price == pytest.approx(115)


# =============================================================================
# Slippage and fees correctness
# =============================================================================

def test_slippage_increases_entry_decreases_exit():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0.01)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    result = engine.run(signals, prices)
    trade = result.trades[0]

    assert trade.entry_price == pytest.approx(101.0)
    assert trade.exit_price == pytest.approx(108.9)


def test_transaction_cost_reduces_pnl():
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    engine_no_cost = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    engine_with_cost = BacktestingEngine(capital_per_trade=10000, transaction_cost=0.01, slippage=0)

    pnl_no_cost = engine_no_cost.run(signals, prices).trades[0].pnl
    pnl_with_cost = engine_with_cost.run(signals, prices).trades[0].pnl

    assert pnl_with_cost < pnl_no_cost


def test_zero_or_negative_price_raises_value_error():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 0.0])

    with pytest.raises(ValueError, match="must be positive"):
        engine.run(signals, prices)


# =============================================================================
# adj_close fallback rule
# =============================================================================

def test_uses_adj_close_when_available():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices(
        "AAPL", DATES_3,
        closes=[100, 105, 110],
        adj_closes=[90, 95, 99],
    )

    result = engine.run(signals, prices)
    trade = result.trades[0]

    assert trade.entry_price == pytest.approx(90)
    assert trade.exit_price == pytest.approx(99)


def test_falls_back_to_close_when_adj_close_is_nan():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices(
        "AAPL", DATES_3,
        closes=[100, 105, 110],
        adj_closes=[float("nan"), 95, 99],
    )

    result = engine.run(signals, prices)
    trade = result.trades[0]

    # Entry row's adj_close is NaN -> falls back to close (100)
    assert trade.entry_price == pytest.approx(100)
    # Exit row has a valid adj_close -> uses it (99)
    assert trade.exit_price == pytest.approx(99)


# =============================================================================
# Multi-symbol independence
# =============================================================================

def test_multiple_symbols_produce_independent_trades():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)

    signals = pd.concat([
        make_signals("AAPL", DATES_3, [1, 0, -1]),
        make_signals("MSFT", DATES_3, [1, 0, -1]),
    ], ignore_index=True)

    prices = pd.concat([
        make_prices("AAPL", DATES_3, [100, 105, 110]),
        make_prices("MSFT", DATES_3, [200, 210, 220]),
    ], ignore_index=True)

    result = engine.run(signals, prices)

    assert set(result.trades_df["symbol"].unique()) == {"AAPL", "MSFT"}
    assert len(result.trades) == 2

    aapl_trade = next(t for t in result.trades if t.symbol == "AAPL")
    msft_trade = next(t for t in result.trades if t.symbol == "MSFT")

    assert aapl_trade.entry_price == pytest.approx(100)
    assert aapl_trade.exit_price == pytest.approx(110)
    assert msft_trade.entry_price == pytest.approx(200)
    assert msft_trade.exit_price == pytest.approx(220)


# =============================================================================
# Output integrity
# =============================================================================

def test_trade_id_is_unique_and_sequential():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)

    signals = pd.concat([
        make_signals("AAPL", DATES_4, [1, -1, 1, -1]),
        make_signals("MSFT", DATES_3, [1, 0, -1]),
    ], ignore_index=True)

    prices = pd.concat([
        make_prices("AAPL", DATES_4, [100, 105, 110, 115]),
        make_prices("MSFT", DATES_3, [200, 210, 220]),
    ], ignore_index=True)

    result = engine.run(signals, prices)

    trade_ids = [t.trade_id for t in result.trades]
    assert trade_ids == sorted(trade_ids)
    assert trade_ids == list(range(1, len(trade_ids) + 1))


def test_trades_df_matches_trades_list():
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)
    signals = make_signals("AAPL", DATES_3, [1, 0, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    result = engine.run(signals, prices)

    assert len(result.trades) == len(result.trades_df)

    trade = result.trades[0]
    row = result.trades_df.iloc[0]

    assert row["trade_id"] == trade.trade_id
    assert row["symbol"] == trade.symbol
    assert row["entry_price"] == pytest.approx(trade.entry_price)
    assert row["exit_price"] == pytest.approx(trade.exit_price)
    assert row["pnl"] == pytest.approx(trade.pnl)


def test_run_logs_zero_trades_warning_when_no_trades_generated(caplog):
    caplog.set_level(logging.WARNING)
    engine = BacktestingEngine(capital_per_trade=10000, transaction_cost=0, slippage=0)

    # SELL with no prior BUY is a no-op; no position ever opened,
    # so nothing to force-close either.
    signals = make_signals("AAPL", DATES_3, [0, 0, -1])
    prices = make_prices("AAPL", DATES_3, [100, 105, 110])

    result = engine.run(signals, prices)

    assert len(result.trades) == 0
    assert any("zero trades" in rec.message for rec in caplog.records)