"""
Unit tests for LegacyAllocatorAdapter.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.allocation_interface import AllocationContext
from src.portfolio.adapters import LegacyAllocatorAdapter
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.equal_weight import EqualWeightAllocator
from src.portfolio.volatility_weighted import VolatilityWeightedAllocator


def test_equal_weight_adapter() -> None:
    """Test EqualWeightAllocator adapted to AllocationContext with constraints."""
    adapter = LegacyAllocatorAdapter(
        "equal_weight",
        EqualWeightAllocator(),
        needs_prices=False,
        constraints=PortfolioConstraints(max_position_size=0.40),
    )
    ctx = AllocationContext(
        as_of_date=pd.Timestamp("2024-01-01"),
        signals=pd.DataFrame({"symbol": ["A", "B", "C"], "signal": [1, 1, 1]}),
        prices_history=pd.DataFrame(),
    )
    res = adapter.allocate(ctx)
    assert len(res.weights) == 3
    # Equal 1/3 = ~0.333 <= 0.40
    for w in res.weights.values():
        assert w == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_volatility_weighted_adapter_and_price_truncation() -> None:
    """Test VolatilityWeightedAllocator receives prices history and computes weights."""
    adapter = LegacyAllocatorAdapter(
        "volatility_weighted",
        VolatilityWeightedAllocator(lookback_window=5),
        needs_prices=True,
        constraints=PortfolioConstraints(max_position_size=0.70),
    )
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    px_a = [100.0, 101.0, 99.0, 102.0, 100.0, 101.0, 103.0, 102.0, 104.0, 105.0]
    px_b = [100.0, 105.0, 95.0, 110.0, 90.0, 115.0, 85.0, 120.0, 80.0, 125.0]

    px_df = pd.DataFrame(
        {
            "symbol": ["A"] * 10 + ["B"] * 10,
            "date": list(dates) * 2,
            "close": px_a + px_b,
        }
    )
    ctx = AllocationContext(
        as_of_date=dates[-1],
        signals=pd.DataFrame({"symbol": ["A", "B"], "signal": [1, 1]}),
        prices_history=px_df,
    )
    res = adapter.allocate(ctx)
    assert len(res.weights) == 2
    # B has much higher volatility, so A should have higher weight
    assert res.weights["A"] > res.weights["B"]


def test_short_side_returns_magnitudes() -> None:
    """Test side='short' filters signal == -1 and returns positive weight magnitudes."""
    adapter = LegacyAllocatorAdapter(
        "equal_weight",
        EqualWeightAllocator(),
        needs_prices=False,
        constraints=PortfolioConstraints(max_position_size=1.0),
    )
    ctx = AllocationContext(
        as_of_date=pd.Timestamp("2024-01-01"),
        signals=pd.DataFrame({"symbol": ["A", "B", "C"], "signal": [-1, -1, 1]}),
        prices_history=pd.DataFrame(),
        side="short",
    )
    res = adapter.allocate(ctx)
    assert set(res.weights.keys()) == {"A", "B"}
    assert res.weights["A"] == pytest.approx(0.50)
    assert res.weights["B"] == pytest.approx(0.50)


def test_empty_candidates_returns_empty_result() -> None:
    """Test no active candidates returns empty AllocationResult without raising."""
    adapter = LegacyAllocatorAdapter(
        "equal_weight",
        EqualWeightAllocator(),
        needs_prices=False,
        constraints=PortfolioConstraints(max_position_size=1.0),
    )
    ctx = AllocationContext(
        as_of_date=pd.Timestamp("2024-01-01"),
        signals=pd.DataFrame({"symbol": ["A"], "signal": [0]}),
        prices_history=pd.DataFrame(),
    )
    res = adapter.allocate(ctx)
    assert res.weights == {}
    assert res.cash_weight == 1.0
    assert "A" in res.excluded
