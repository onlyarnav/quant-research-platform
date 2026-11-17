"""
Unit tests for RiskParityAllocator equal risk contribution solver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.allocation_interface import AllocationContext
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.risk_parity import RiskParityAllocator


def test_two_assets_different_volatilities() -> None:
    """
    Test two uncorrelated assets with volatilities 1:2 receive inverse-vol risk parity weights.

    Asset A has vol ~0.01, Asset B has vol ~0.02.
    For uncorrelated assets, ERC weights are w_A * vol_A = w_B * vol_B.
    So w_A / w_B = vol_B / vol_A = 2 / 1 -> w_A = 2/3 (~0.667), w_B = 1/3 (~0.333).
    """
    alloc = RiskParityAllocator(
        lookback_window=20,
        constraints=PortfolioConstraints(max_position_size=1.0),
    )
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=25, freq="B")

    rets_a = rng.normal(0.0, 0.01, 25)
    rets_b = rng.normal(0.0, 0.02, 25)
    px_a = 100.0 * np.exp(np.cumsum(rets_a))
    px_b = 100.0 * np.exp(np.cumsum(rets_b))

    df = pd.DataFrame(
        {
            "symbol": ["A"] * 25 + ["B"] * 25,
            "date": list(dates) * 2,
            "close": list(px_a) + list(px_b),
        }
    )
    ctx = AllocationContext(
        as_of_date=dates[-1],
        signals=pd.DataFrame({"symbol": ["A", "B"], "signal": [1, 1]}),
        prices_history=df,
    )

    res = alloc.allocate(ctx)
    assert res.weights["A"] > res.weights["B"]
    # Ratio of weights should be approximately 2:1
    ratio = res.weights["A"] / res.weights["B"]
    assert ratio == pytest.approx(2.0, rel=0.25)


def test_identical_assets_yield_equal_weights() -> None:
    """Test identical assets receive equal weights."""
    alloc = RiskParityAllocator(
        lookback_window=10,
        constraints=PortfolioConstraints(max_position_size=1.0),
    )
    dates = pd.date_range("2024-01-01", periods=15, freq="B")
    px = [100.0 + i for i in range(15)]
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 15 + ["B"] * 15,
            "date": list(dates) * 2,
            "close": px + px,
        }
    )
    ctx = AllocationContext(
        as_of_date=dates[-1],
        signals=pd.DataFrame({"symbol": ["A", "B"], "signal": [1, 1]}),
        prices_history=df,
    )
    res = alloc.allocate(ctx)
    assert res.weights["A"] == pytest.approx(0.50, abs=1e-4)
    assert res.weights["B"] == pytest.approx(0.50, abs=1e-4)


def test_insufficient_history_excluded() -> None:
    """Test asset with fewer rows than lookback_window+1 is excluded."""
    alloc = RiskParityAllocator(lookback_window=10)
    dates = pd.date_range("2024-01-01", periods=5, freq="B")  # only 5 rows, need 11
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 5,
            "date": dates,
            "close": [100.0] * 5,
        }
    )
    ctx = AllocationContext(
        as_of_date=dates[-1],
        signals=pd.DataFrame({"symbol": ["A"], "signal": [1]}),
        prices_history=df,
    )
    res = alloc.allocate(ctx)
    assert "A" in res.excluded
    assert res.weights == {}


def test_solver_failure_fallback_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test optimizer failure falls back to inverse volatility with warning."""
    import src.portfolio.risk_parity as rp

    def mock_minimize(*args, **kwargs):
        raise RuntimeError("Optimization exploded")

    monkeypatch.setattr(rp, "minimize", mock_minimize)

    alloc = RiskParityAllocator(
        lookback_window=5,
        constraints=PortfolioConstraints(max_position_size=1.0),
    )
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    px_a = [100.0, 101.0, 99.0, 102.0, 100.0, 101.0, 103.0, 102.0, 104.0, 105.0]
    px_b = [100.0, 105.0, 95.0, 110.0, 90.0, 115.0, 85.0, 120.0, 80.0, 125.0]
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 10 + ["B"] * 10,
            "date": list(dates) * 2,
            "close": px_a + px_b,
        }
    )
    ctx = AllocationContext(
        as_of_date=dates[-1],
        signals=pd.DataFrame({"symbol": ["A", "B"], "signal": [1, 1]}),
        prices_history=df,
    )
    res = alloc.allocate(ctx)
    assert "Risk parity optimisation failed" in caplog.text
    assert len(res.weights) == 2
    assert res.weights["A"] > res.weights["B"]
