"""
Unit tests for BacktestConfig dataclass, resolution, validation, and hashing.
"""

from __future__ import annotations

from typing import Any

import pytest

from config.settings import settings
from src.backtesting.backtest_config import (
    AllocationMethod,
    BacktestConfig,
    RebalanceFrequency,
)


def test_defaults_resolve_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test BacktestConfig resolves None values from settings."""
    monkeypatch.setattr(settings, "INITIAL_CAPITAL", 500_000.0)
    monkeypatch.setattr(settings, "TRANSACTION_COST", 0.002)
    monkeypatch.setattr(settings, "SLIPPAGE", 0.001)
    monkeypatch.setattr(settings, "MAX_POSITION_SIZE", 0.10)
    monkeypatch.setattr(settings, "BACKTEST_CASH_RESERVE", 0.05)
    monkeypatch.setattr(settings, "BACKTEST_ALLOCATION_METHOD", "equal_weight")
    monkeypatch.setattr(settings, "BACKTEST_REBALANCE_FREQUENCY", "weekly")

    cfg = BacktestConfig(strategy_name="test_strat")
    resolved = cfg.resolved()

    assert resolved.initial_capital == pytest.approx(500_000.0)
    assert resolved.transaction_cost == pytest.approx(0.002)
    assert resolved.slippage == pytest.approx(0.001)
    assert resolved.max_position_size == pytest.approx(0.10)
    assert resolved.cash_reserve == pytest.approx(0.05)
    assert resolved.allocation_method == AllocationMethod.EQUAL_WEIGHT
    assert resolved.rebalance_frequency == RebalanceFrequency.WEEKLY


def test_validation_rules_raise_with_value_in_message() -> None:
    """Test validation raises ValueError with offending value in message."""
    # initial_capital <= 0
    with pytest.raises(ValueError, match="initial_capital must be positive, got 0.0"):
        BacktestConfig(initial_capital=0.0).validate()

    # transaction_cost not in [0, 0.5]
    with pytest.raises(
        ValueError, match="transaction_cost must be in \\[0, 0.5\\], got 0.6"
    ):
        BacktestConfig(transaction_cost=0.6).validate()

    # slippage not in [0, 0.5]
    with pytest.raises(ValueError, match="slippage must be in \\[0, 0.5\\], got -0.01"):
        BacktestConfig(slippage=-0.01).validate()

    # max_position_size not in (0, 1]
    with pytest.raises(
        ValueError, match="max_position_size must be in \\(0, 1\\], got 1.5"
    ):
        BacktestConfig(max_position_size=1.5).validate()

    # cash_reserve not in [0, 1)
    with pytest.raises(ValueError, match="cash_reserve must be in \\[0, 1\\), got 1.0"):
        BacktestConfig(cash_reserve=1.0).validate()

    # rebalance_tolerance < 0
    with pytest.raises(ValueError, match="rebalance_tolerance must be >= 0, got -0.01"):
        BacktestConfig(rebalance_tolerance=-0.01).validate()

    # stop_loss_pct not in (0, 1]
    with pytest.raises(
        ValueError, match="stop_loss_pct must be in \\(0, 1\\] or None, got 1.5"
    ):
        BacktestConfig(stop_loss_pct=1.5).validate()

    # take_profit_pct not in (0, 1]
    with pytest.raises(
        ValueError, match="take_profit_pct must be in \\(0, 1\\] or None, got 0.0"
    ):
        BacktestConfig(take_profit_pct=0.0).validate()

    # execution_lag < 0
    with pytest.raises(ValueError, match="execution_lag must be >= 0, got -1"):
        BacktestConfig(execution_lag=-1).validate()

    # max_open_positions < 1
    with pytest.raises(
        ValueError, match="max_open_positions must be >= 1 or None, got 0"
    ):
        BacktestConfig(max_open_positions=0).validate()

    # lookback_window < 2
    with pytest.raises(ValueError, match="lookback_window must be >= 2, got 1"):
        BacktestConfig(lookback_window=1).validate()


def test_allow_short_with_unsupported_allocation_raises() -> None:
    """Test allow_short raises when allocation is not fixed_notional or equal_weight."""
    with pytest.raises(
        ValueError,
        match="allow_short is only supported with fixed_notional or equal_weight allocation",
    ):
        BacktestConfig(
            allow_short=True,
            allocation_method=AllocationMethod.RISK_PARITY,
        ).validate()


def test_to_dict_and_from_dict_round_trip() -> None:
    """Test to_dict and from_dict serialize enums as strings and reconstruct accurately."""
    cfg = BacktestConfig(
        strategy_name="momentum_alpha",
        initial_capital=2_000_000.0,
        allocation_method=AllocationMethod.SIGNAL_WEIGHTED,
        rebalance_frequency=RebalanceFrequency.MONTHLY,
        allow_short=False,
    )
    d = cfg.to_dict()
    assert d["allocation_method"] == "signal_weighted"
    assert d["rebalance_frequency"] == "monthly"

    reconstructed = BacktestConfig.from_dict(d)
    assert reconstructed == cfg


def test_config_hash_stability_and_sensitivity() -> None:
    """Test config_hash is 8 chars hex, deterministic, and sensitive to differences."""
    cfg1 = BacktestConfig(strategy_name="alpha", initial_capital=1_000_000.0)
    cfg2 = BacktestConfig(strategy_name="alpha", initial_capital=1_000_000.0)
    cfg3 = BacktestConfig(strategy_name="alpha", initial_capital=1_000_001.0)

    h1 = cfg1.config_hash()
    h2 = cfg2.config_hash()
    h3 = cfg3.config_hash()

    assert len(h1) == 8
    assert h1 == h2
    assert h1 != h3


def test_is_legacy_compatible_truth_table() -> None:
    """Test legacy compatibility condition detection."""
    leg = BacktestConfig(
        allocation_method=AllocationMethod.FIXED_NOTIONAL,
        rebalance_frequency=RebalanceFrequency.ON_SIGNAL,
        allow_short=False,
        stop_loss_pct=None,
        take_profit_pct=None,
        execution_lag=0,
        cash_reserve=0.0,
        max_open_positions=None,
        force_close_at_end=True,
    )
    assert leg.is_legacy_compatible() is True

    # Variations make it not legacy-compatible
    assert leg._replace_field(allow_short=True).is_legacy_compatible() is False
    assert leg._replace_field(execution_lag=1).is_legacy_compatible() is False
    assert leg._replace_field(stop_loss_pct=0.05).is_legacy_compatible() is False
    assert leg._replace_field(cash_reserve=0.1).is_legacy_compatible() is False
    assert (
        leg._replace_field(
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        ).is_legacy_compatible()
        is False
    )


# Helper for testing immutability with replacements
def _replace(cfg: BacktestConfig, **kwargs: Any) -> BacktestConfig:
    d = cfg.to_dict()
    d.update(kwargs)
    return BacktestConfig.from_dict(d)


BacktestConfig._replace_field = _replace  # type: ignore[attr-defined]
