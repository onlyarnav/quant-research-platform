"""
Unit and property tests for PortfolioConstraints and iterative projection algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.portfolio.constraints import PortfolioConstraints


def test_scaling_to_investable_and_cash_reserve() -> None:
    """Test raw weights are scaled to (1 - cash_reserve)."""
    constraints = PortfolioConstraints(
        max_position_size=1.0,
        cash_reserve=0.20,
    )
    raw = {"A": 0.5, "B": 0.5}
    res = constraints.apply(raw)

    assert res.cash_weight == pytest.approx(0.20)
    assert res.weights["A"] == pytest.approx(0.40)
    assert res.weights["B"] == pytest.approx(0.40)
    assert sum(res.weights.values()) == pytest.approx(0.80)


def test_position_cap_with_redistribution() -> None:
    """
    Test max_position_size caps large weights and redistributes excess.

    Raw: A=0.8, B=0.1, C=0.1. Cap=0.5.
    A gets capped at 0.5. Excess 0.3 redistributed equally to B and C (0.15 each).
    Final: A=0.5, B=0.25, C=0.25.
    """
    constraints = PortfolioConstraints(
        max_position_size=0.5,
        cash_reserve=0.0,
    )
    raw = {"A": 0.8, "B": 0.1, "C": 0.1}
    res = constraints.apply(raw)

    assert res.weights["A"] == pytest.approx(0.50)
    assert res.weights["B"] == pytest.approx(0.25)
    assert res.weights["C"] == pytest.approx(0.25)
    assert "max_position_size:A" in res.binding


def test_sector_cap_binding_golden() -> None:
    """
    Test sector caps constrain aggregate sector weight.

    Sector tech: A, B. Sector finance: C.
    Raw: A=0.4, B=0.4, C=0.2. Max sector = 0.60.
    Tech sum 0.8 exceeds 0.6. Scale tech down to 0.6 (A=0.3, B=0.3).
    Excess 0.2 moves to C (up to max_position_size 0.5).
    Final: A=0.3, B=0.3, C=0.4.
    """
    constraints = PortfolioConstraints(
        max_position_size=0.5,
        max_sector_weight=0.60,
        sector_map={"A": "Tech", "B": "Tech", "C": "Finance"},
        cash_reserve=0.0,
    )
    raw = {"A": 0.4, "B": 0.4, "C": 0.2}
    res = constraints.apply(raw)

    assert res.weights["A"] == pytest.approx(0.30)
    assert res.weights["B"] == pytest.approx(0.30)
    assert res.weights["C"] == pytest.approx(0.40)
    assert "max_sector_weight:Tech" in res.binding


def test_unknown_sector_not_capped() -> None:
    """Test symbols not in sector_map are grouped into UNKNOWN and not capped."""
    constraints = PortfolioConstraints(
        max_position_size=1.0,
        max_sector_weight=0.40,
        sector_map={"A": "Tech"},
    )
    raw = {"A": 0.3, "UNKNOWN_SYM": 0.7}
    res = constraints.apply(raw)

    assert res.weights["A"] == pytest.approx(0.3)
    assert res.weights["UNKNOWN_SYM"] == pytest.approx(0.7)


def test_infeasible_budget_leaves_cash() -> None:
    """Test fewer symbols than needed to absorb budget leaves residual cash."""
    # 2 symbols with max cap 0.30 each -> max total 0.60, residual 0.40 cash
    constraints = PortfolioConstraints(
        max_position_size=0.30,
        cash_reserve=0.0,
    )
    raw = {"A": 0.5, "B": 0.5}
    res = constraints.apply(raw)

    assert res.weights["A"] == pytest.approx(0.30)
    assert res.weights["B"] == pytest.approx(0.30)
    assert res.cash_weight == pytest.approx(0.40)


def test_min_position_weight_dropping() -> None:
    """Test weights below min_position_weight are dropped to 0 and redistributed."""
    constraints = PortfolioConstraints(
        max_position_size=0.6,
        min_position_weight=0.05,
        cash_reserve=0.0,
    )
    raw = {"A": 0.50, "B": 0.48, "C": 0.02}  # C is 2% < 5%
    res = constraints.apply(raw)

    assert "C" not in res.weights
    assert "A" in res.weights
    assert "B" in res.weights
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert res.cash_weight == pytest.approx(0.0)


def test_validation_errors() -> None:
    """Test invalid constraints parameters raise ValueError."""
    with pytest.raises(ValueError, match="max_position_size must be in \\(0, 1\\]"):
        PortfolioConstraints(max_position_size=0.0).validate()

    with pytest.raises(ValueError, match="cash_reserve must be in \\[0, 1\\)"):
        PortfolioConstraints(cash_reserve=1.0).validate()

    with pytest.raises(
        ValueError, match="min_position_weight .* cannot exceed max_position_size"
    ):
        PortfolioConstraints(min_position_weight=0.6, max_position_size=0.5).validate()


def test_property_random_weights_convergence() -> None:
    """Property test verifying constraints hold over 20 randomized weight profiles."""
    rng = np.random.default_rng(42)
    constraints = PortfolioConstraints(
        max_position_size=0.25,
        cash_reserve=0.10,
        max_sector_weight=0.50,
        sector_map={f"S_{i}": "SectorA" if i < 3 else "SectorB" for i in range(8)},
    )

    for _ in range(20):
        raw = {f"S_{i}": float(rng.uniform(0.1, 1.0)) for i in range(8)}
        res = constraints.apply(raw)

        # Check all weights non-negative and <= cap
        for sym, w in res.weights.items():
            assert w >= 0.0
            assert w <= 0.25 + 1e-7

        # Check total + cash == 1.0
        tot = sum(res.weights.values()) + res.cash_weight
        assert tot == pytest.approx(1.0, abs=1e-6)
        assert res.cash_weight >= 0.10 - 1e-7
