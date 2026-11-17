"""
Unit tests for allocator factory and registry functions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.allocation_interface import AllocationContext, AllocationResult
from src.backtesting.backtest_config import AllocationMethod
from src.portfolio.allocator_factory import (
    ALLOCATOR_NAMES,
    available_allocators,
    create_allocator,
    register_allocator,
)


def test_available_allocators_and_names() -> None:
    """Test all expected allocator names are registered and available."""
    avail = available_allocators()
    for name in ALLOCATOR_NAMES:
        assert name in avail


def test_fixed_notional_rejected() -> None:
    """Test fixed_notional raises clear ValueError."""
    with pytest.raises(ValueError, match="fixed_notional does not use an allocator"):
        create_allocator("fixed_notional")

    with pytest.raises(ValueError, match="fixed_notional does not use an allocator"):
        create_allocator(AllocationMethod.FIXED_NOTIONAL)


def test_unknown_allocator_raises_with_available_list() -> None:
    """Test unknown allocator name raises ValueError containing available choices."""
    with pytest.raises(ValueError, match="Unknown allocation method 'magic_allocator'"):
        create_allocator("magic_allocator")


def test_register_duplicate_raises() -> None:
    """Test registering an existing name raises ValueError."""
    with pytest.raises(ValueError, match="is already registered"):
        register_allocator("equal_weight", lambda **kw: None)


def test_create_and_satisfy_protocol() -> None:
    """Test every built-in allocator instantiates and executes allocate() properly."""
    for method in ALLOCATOR_NAMES:
        alloc = create_allocator(method, lookback_window=5)
        assert hasattr(alloc, "allocate")

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        px_df = pd.DataFrame(
            {
                "symbol": ["A"] * 10 + ["B"] * 10,
                "date": list(dates) * 2,
                "close": [100.0 + i for i in range(10)]
                + [200.0 + i * 2 for i in range(10)],
            }
        )
        ctx = AllocationContext(
            as_of_date=dates[-1],
            signals=pd.DataFrame(
                {
                    "symbol": ["A", "B"],
                    "signal": [1, 1],
                    "confidence": [0.8, 0.6],
                    "predicted_return": [0.02, 0.01],
                }
            ),
            prices_history=px_df,
        )
        res = alloc.allocate(ctx)
        assert isinstance(res, AllocationResult)
        assert res.method is not None
        assert 0.0 <= res.cash_weight <= 1.0
