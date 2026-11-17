"""
Unit tests for PositionBook average-cost accounting and exposure tracking.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.position_book import PositionBook


def test_add_and_average_up_golden() -> None:
    """
    Test adding to an existing position averages entry price and accumulates costs.

    Hand calculation:
    Fill 1: 100 @ 10, fee=1.0, slip=0.5
    Fill 2: 100 @ 12, fee=1.2, slip=0.6
    Total qty = 200
    Avg price = (100 * 10 + 100 * 12) / 200 = 2200 / 200 = 11.0
    Cumulative fees = 2.2, slippage = 1.1
    """
    book = PositionBook()
    d1 = pd.Timestamp("2024-01-01")
    d2 = pd.Timestamp("2024-01-02")

    book.add(
        "XYZ",
        side=1,
        date=d1,
        quantity=100.0,
        executed_price=10.0,
        fee=1.0,
        slippage_cost=0.5,
    )
    pos = book.add(
        "XYZ",
        side=1,
        date=d2,
        quantity=100.0,
        executed_price=12.0,
        fee=1.2,
        slippage_cost=0.6,
    )

    assert pos.quantity == pytest.approx(200.0)
    assert pos.avg_entry_price == pytest.approx(11.0)
    assert pos.entry_fees_paid == pytest.approx(2.2)
    assert pos.entry_slippage_paid == pytest.approx(1.1)
    assert pos.entry_date == d1


def test_reduce_partial_and_full_with_cost_share() -> None:
    """Test partial and full position reductions allocate pro-rata entry costs."""
    book = PositionBook()
    d = pd.Timestamp("2024-01-01")
    book.add(
        "XYZ",
        side=1,
        date=d,
        quantity=200.0,
        executed_price=10.0,
        fee=2.0,
        slippage_cost=1.0,
    )

    # Reduce half (100 shares)
    lot1 = book.reduce("XYZ", 100.0)
    assert lot1.quantity == pytest.approx(100.0)
    assert lot1.entry_fee_share == pytest.approx(1.0)
    assert lot1.entry_slippage_share == pytest.approx(0.5)

    pos_remaining = book.get("XYZ")
    assert pos_remaining is not None
    assert pos_remaining.quantity == pytest.approx(100.0)
    assert pos_remaining.entry_fees_paid == pytest.approx(1.0)

    # Reduce remaining with tiny residual
    _ = book.reduce("XYZ", 99.9999999999)
    assert book.has("XYZ") is False


def test_side_mismatch_raises() -> None:
    """Test attempting to add to a position of opposite side raises ValueError."""
    book = PositionBook()
    book.add(
        "XYZ",
        side=1,
        date=pd.Timestamp("2024-01-01"),
        quantity=10.0,
        executed_price=100.0,
        fee=0.1,
        slippage_cost=0.05,
    )
    with pytest.raises(
        ValueError, match="Cannot add to a position of the opposite side"
    ):
        book.add(
            "XYZ",
            side=-1,
            date=pd.Timestamp("2024-01-02"),
            quantity=10.0,
            executed_price=100.0,
            fee=0.1,
            slippage_cost=0.05,
        )


def test_market_value_unrealized_pnl_and_exposures() -> None:
    """
    Test market value, unrealized PnL, and exposure calculation for long and short.

    Hand calculation:
    Long A: 100 @ 10, mark = 12 -> mkt_val = 1200, unreal = +200
    Short B: 50 @ 20, mark = 18 -> mkt_val = 900, unreal = (20 - 18) * 50 = +100
    Total unrealized PnL = 300
    Gross market value = 1200 + 900 = 2100
    Net market value = 1200 - 900 = 300
    Equity = 2000
    Gross exposure = 2100 / 2000 = 1.05
    Net exposure = 300 / 2000 = 0.15
    """
    book = PositionBook()
    d = pd.Timestamp("2024-01-01")
    book.add(
        "A",
        side=1,
        date=d,
        quantity=100.0,
        executed_price=10.0,
        fee=0.0,
        slippage_cost=0.0,
    )
    book.add(
        "B",
        side=-1,
        date=d,
        quantity=50.0,
        executed_price=20.0,
        fee=0.0,
        slippage_cost=0.0,
    )

    prices = {"A": 12.0, "B": 18.0}
    assert book.unrealized_pnl(prices) == pytest.approx(300.0)
    assert book.market_value(prices, side=1) == pytest.approx(1200.0)
    assert book.market_value(prices, side=-1) == pytest.approx(900.0)
    assert book.market_value(prices) == pytest.approx(2100.0)

    exp = book.exposures(prices, equity=2000.0)
    assert exp["gross"] == pytest.approx(1.05)
    assert exp["net"] == pytest.approx(0.15)
    assert exp["long"] == pytest.approx(0.60)
    assert exp["short"] == pytest.approx(0.45)
