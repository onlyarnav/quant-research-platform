"""
Unit tests for SignalWeightedAllocator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.allocation_interface import AllocationContext
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.signal_weighted import SignalWeightedAllocator


def test_confidence_basis_golden() -> None:
    """
    Test weights proportional to confidence.

    Symbol A: conf=0.8, Symbol B: conf=0.4
    Total = 1.2
    Raw: A = 0.8/1.2 = 2/3 (~0.667), B = 0.4/1.2 = 1/3 (~0.333)
    """
    alloc = SignalWeightedAllocator(
        constraints=PortfolioConstraints(max_position_size=1.0),
        basis="confidence",
    )
    sig = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "signal": [1, 1],
            "confidence": [0.8, 0.4],
        }
    )
    ctx = AllocationContext(
        as_of_date=pd.Timestamp("2024-01-01"),
        signals=sig,
        prices_history=pd.DataFrame(),
    )
    res = alloc.allocate(ctx)
    assert res.weights["A"] == pytest.approx(2.0 / 3.0, abs=1e-5)
    assert res.weights["B"] == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_nan_or_negative_basis_excluded() -> None:
    """Test symbols with NaN or non-positive basis are excluded."""
    alloc = SignalWeightedAllocator(basis="confidence")
    sig = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "signal": [1, 1, 1],
            "confidence": [0.8, -0.2, None],
        }
    )
    ctx = AllocationContext(
        as_of_date=pd.Timestamp("2024-01-01"),
        signals=sig,
        prices_history=pd.DataFrame(),
    )
    res = alloc.allocate(ctx)
    assert set(res.weights.keys()) == {"A"}
    assert "B" in res.excluded
    assert "C" in res.excluded


def test_all_zero_falls_back_to_equal_weight(caplog: pytest.LogCaptureFixture) -> None:
    """Test all zero basis falls back to equal weight with warning."""
    alloc = SignalWeightedAllocator(
        constraints=PortfolioConstraints(max_position_size=1.0),
        basis="confidence",
    )
    sig = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "signal": [1, 1],
            "confidence": [0.0, 0.0],
        }
    )
    ctx = AllocationContext(
        as_of_date=pd.Timestamp("2024-01-01"),
        signals=sig,
        prices_history=pd.DataFrame(),
    )
    res = alloc.allocate(ctx)
    assert "All signal weights are zero; falling back to equal weighting" in caplog.text
    assert res.weights["A"] == pytest.approx(0.50)
    assert res.weights["B"] == pytest.approx(0.50)


def test_predicted_return_basis() -> None:
    """Test weighting based on predicted_return."""
    alloc = SignalWeightedAllocator(
        constraints=PortfolioConstraints(max_position_size=1.0),
        basis="predicted_return",
    )
    sig = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "signal": [1, 1],
            "predicted_return": [0.03, 0.01],
        }
    )
    ctx = AllocationContext(
        as_of_date=pd.Timestamp("2024-01-01"),
        signals=sig,
        prices_history=pd.DataFrame(),
    )
    res = alloc.allocate(ctx)
    # A has 3x predicted return of B: 75% vs 25%
    assert res.weights["A"] == pytest.approx(0.75, abs=1e-5)
    assert res.weights["B"] == pytest.approx(0.25, abs=1e-5)
