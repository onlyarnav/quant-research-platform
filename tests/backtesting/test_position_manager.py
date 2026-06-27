"""
Tests for PositionManager.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.position_manager import OpenPosition, PositionManager


# =============================================================================
# __init__()
# =============================================================================

def test_init_sets_cash_to_initial_capital():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    assert manager.cash == 100_000


def test_init_starts_with_no_open_positions():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    assert manager.get_open_positions() == {}


# =============================================================================
# can_open_position()
# =============================================================================

def test_can_open_position_true_when_cash_sufficient():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    assert manager.can_open_position("AAPL") is True


def test_can_open_position_false_when_symbol_already_open():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    assert manager.can_open_position("AAPL") is False

def test_can_open_position_false_when_cash_below_required():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.6)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    # 60% allocated (60,000), 40,000 cash remains.
    # A second position requires another 60,000 -> insufficient.
    assert manager.can_open_position("MSFT") is False


def test_can_open_position_does_not_mutate_state():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    cash_before = manager.cash
    positions_before = manager.get_open_positions()

    manager.can_open_position("AAPL")

    assert manager.cash == cash_before
    assert manager.get_open_positions() == positions_before


# =============================================================================
# open_position()
# =============================================================================

def test_open_position_returns_open_position_instance():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    position = manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    assert isinstance(position, OpenPosition)


def test_open_position_correct_capital_allocated():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    position = manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    assert position.capital_allocated == pytest.approx(10_000)


def test_open_position_correct_position_size():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    position = manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    # capital_allocated=10,000 / entry_price=100 -> 100 shares
    assert position.position_size == pytest.approx(100.0)


def test_open_position_deducts_cash():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    assert manager.cash == pytest.approx(90_000)


def test_open_position_adds_to_open_positions():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    open_positions = manager.get_open_positions()
    assert "AAPL" in open_positions


def test_open_position_raises_if_symbol_already_open():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)

    with pytest.raises(ValueError, match="already open"):
        manager.open_position("AAPL", pd.Timestamp("2024-01-02"), 105.0)


def test_open_position_raises_on_zero_entry_price():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    with pytest.raises(ValueError, match="must be positive"):
        manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 0.0)


def test_open_position_raises_on_negative_entry_price():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    with pytest.raises(ValueError, match="must be positive"):
        manager.open_position("AAPL", pd.Timestamp("2024-01-01"), -50.0)


def test_open_position_raises_runtime_error_on_insufficient_cash():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.6)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    # 60,000 allocated, 40,000 remains; another 60,000 position fails.
    with pytest.raises(RuntimeError, match="Insufficient cash"):
        manager.open_position("MSFT", pd.Timestamp("2024-01-02"), 200.0)


# =============================================================================
# close_position()
# =============================================================================

def test_close_position_returns_closed_position():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    opened = manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)

    closed = manager.close_position("AAPL", 110.0)

    assert closed is opened


def test_close_position_removes_from_open_positions():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    manager.close_position("AAPL", 110.0)

    assert "AAPL" not in manager.get_open_positions()


def test_close_position_adds_proceeds_to_cash():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    # cash = 90,000 after open; position_size = 100 shares
    manager.close_position("AAPL", 110.0)
    # proceeds = 100 * 110 = 11,000 -> cash = 90,000 + 11,000 = 101,000
    assert manager.cash == pytest.approx(101_000)


def test_close_position_at_loss_reduces_cash_relative_to_entry():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    # cash = 90,000 after open; position_size = 100 shares
    manager.close_position("AAPL", 90.0)
    # proceeds = 100 * 90 = 9,000 -> cash = 90,000 + 9,000 = 99,000
    assert manager.cash == pytest.approx(99_000)


def test_close_position_raises_if_no_position_open():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    with pytest.raises(ValueError, match="No open position"):
        manager.close_position("AAPL", 110.0)


def test_close_position_raises_on_zero_exit_price():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    with pytest.raises(ValueError, match="must be positive"):
        manager.close_position("AAPL", 0.0)


def test_close_position_raises_on_negative_exit_price():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    with pytest.raises(ValueError, match="must be positive"):
        manager.close_position("AAPL", -10.0)


# =============================================================================
# get_open_positions()
# =============================================================================

def test_get_open_positions_returns_copy_not_reference():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)

    snapshot = manager.get_open_positions()
    snapshot["MSFT"] = "tampered"

    assert "MSFT" not in manager.get_open_positions()


def test_get_open_positions_reflects_multiple_symbols():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    manager.open_position("MSFT", pd.Timestamp("2024-01-01"), 200.0)

    open_positions = manager.get_open_positions()
    assert set(open_positions.keys()) == {"AAPL", "MSFT"}


# =============================================================================
# total_invested_capital()
# =============================================================================

def test_total_invested_capital_zero_when_no_positions():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    assert manager.total_invested_capital() == 0.0


def test_total_invested_capital_sums_across_positions():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    manager.open_position("MSFT", pd.Timestamp("2024-01-01"), 200.0)

    # Each position allocates 10,000 -> total = 20,000
    assert manager.total_invested_capital() == pytest.approx(20_000)


def test_total_invested_capital_decreases_after_close():
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)
    manager.open_position("MSFT", pd.Timestamp("2024-01-01"), 200.0)
    manager.close_position("AAPL", 110.0)

    assert manager.total_invested_capital() == pytest.approx(10_000)


# =============================================================================
# Cash + position invariant sanity check
# =============================================================================

def test_cash_plus_invested_equals_initial_capital_before_any_pnl():
    """
    Immediately after opening (no price movement yet), cash plus
    invested capital should equal the original initial_capital,
    since no gains/losses have been realized.
    """
    manager = PositionManager(initial_capital=100_000, max_position_size=0.1)
    manager.open_position("AAPL", pd.Timestamp("2024-01-01"), 100.0)

    assert manager.cash + manager.total_invested_capital() == pytest.approx(100_000)