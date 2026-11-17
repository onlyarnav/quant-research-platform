"""
Configuration data structures and validation for backtest execution.

Defines the BacktestConfig dataclass, allocation and rebalancing enums,
parameter resolution against system defaults, validation rules, and hashing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from config.settings import settings


class AllocationMethod(str, Enum):
    """Supported capital allocation and weighting methods."""

    FIXED_NOTIONAL = "fixed_notional"
    EQUAL_WEIGHT = "equal_weight"
    SIGNAL_WEIGHTED = "signal_weighted"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"


class RebalanceFrequency(str, Enum):
    """Portfolio rebalancing frequency schedules."""

    ON_SIGNAL = "on_signal"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Immutable configuration container for backtesting simulation runs."""

    strategy_name: str = "strategy"
    initial_capital: float | None = None
    transaction_cost: float | None = None
    slippage: float | None = None
    max_position_size: float | None = None
    cash_reserve: float | None = None
    allocation_method: AllocationMethod | None = None
    rebalance_frequency: RebalanceFrequency | None = None
    rebalance_tolerance: float | None = None
    allow_short: bool = False
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    execution_lag: int | None = None
    max_open_positions: int | None = None
    min_trade_value: float = 0.0
    lookback_window: int | None = None
    force_close_at_end: bool = True
    risk_free_rate: float | None = None

    def resolved(self) -> BacktestConfig:
        """
        Return a new BacktestConfig where all None values are replaced with defaults.

        Returns:
            BacktestConfig with all optional fields resolved.
        """
        if isinstance(self.allocation_method, str):
            alloc_val = AllocationMethod(self.allocation_method)
        elif self.allocation_method is not None:
            alloc_val = self.allocation_method
        else:
            alloc_val = AllocationMethod(settings.BACKTEST_ALLOCATION_METHOD)

        if isinstance(self.rebalance_frequency, str):
            reb_val = RebalanceFrequency(self.rebalance_frequency)
        elif self.rebalance_frequency is not None:
            reb_val = self.rebalance_frequency
        else:
            reb_val = RebalanceFrequency(settings.BACKTEST_REBALANCE_FREQUENCY)

        return BacktestConfig(
            strategy_name=self.strategy_name,
            initial_capital=(
                self.initial_capital
                if self.initial_capital is not None
                else float(settings.INITIAL_CAPITAL)
            ),
            transaction_cost=(
                self.transaction_cost
                if self.transaction_cost is not None
                else float(settings.TRANSACTION_COST)
            ),
            slippage=(
                self.slippage if self.slippage is not None else float(settings.SLIPPAGE)
            ),
            max_position_size=(
                self.max_position_size
                if self.max_position_size is not None
                else float(settings.MAX_POSITION_SIZE)
            ),
            cash_reserve=(
                self.cash_reserve
                if self.cash_reserve is not None
                else float(settings.BACKTEST_CASH_RESERVE)
            ),
            allocation_method=alloc_val,
            rebalance_frequency=reb_val,
            rebalance_tolerance=(
                self.rebalance_tolerance
                if self.rebalance_tolerance is not None
                else float(settings.BACKTEST_REBALANCE_TOLERANCE)
            ),
            allow_short=self.allow_short,
            stop_loss_pct=(
                self.stop_loss_pct
                if self.stop_loss_pct is not None
                else (
                    float(settings.BACKTEST_STOP_LOSS)
                    if settings.BACKTEST_STOP_LOSS is not None
                    else None
                )
            ),
            take_profit_pct=(
                self.take_profit_pct
                if self.take_profit_pct is not None
                else (
                    float(settings.BACKTEST_TAKE_PROFIT)
                    if settings.BACKTEST_TAKE_PROFIT is not None
                    else None
                )
            ),
            execution_lag=(
                self.execution_lag
                if self.execution_lag is not None
                else int(settings.BACKTEST_EXECUTION_LAG)
            ),
            max_open_positions=self.max_open_positions,
            min_trade_value=self.min_trade_value,
            lookback_window=(
                self.lookback_window
                if self.lookback_window is not None
                else int(settings.BACKTEST_LOOKBACK_WINDOW)
            ),
            force_close_at_end=self.force_close_at_end,
            risk_free_rate=(
                self.risk_free_rate
                if self.risk_free_rate is not None
                else float(settings.PORTFOLIO_RISK_FREE_RATE)
            ),
        )

    def validate(self) -> None:
        """
        Validate all backtest configuration parameters.

        Raises:
            ValueError: If any parameter is out of its allowable financial range.
        """
        res = self.resolved()

        if res.initial_capital <= 0:
            raise ValueError(
                f"initial_capital must be positive, got {res.initial_capital}"
            )
        if not (0.0 <= res.transaction_cost <= 0.5):
            raise ValueError(
                f"transaction_cost must be in [0, 0.5], got {res.transaction_cost}"
            )
        if not (0.0 <= res.slippage <= 0.5):
            raise ValueError(f"slippage must be in [0, 0.5], got {res.slippage}")
        if not (0.0 < res.max_position_size <= 1.0):
            raise ValueError(
                f"max_position_size must be in (0, 1], got {res.max_position_size}"
            )
        if not (0.0 <= res.cash_reserve < 1.0):
            raise ValueError(f"cash_reserve must be in [0, 1), got {res.cash_reserve}")
        if res.rebalance_tolerance < 0:
            raise ValueError(
                f"rebalance_tolerance must be >= 0, got {res.rebalance_tolerance}"
            )
        if res.min_trade_value < 0:
            raise ValueError(f"min_trade_value must be >= 0, got {res.min_trade_value}")
        if res.stop_loss_pct is not None and not (0.0 < res.stop_loss_pct <= 1.0):
            raise ValueError(
                f"stop_loss_pct must be in (0, 1] or None, got {res.stop_loss_pct}"
            )
        if res.take_profit_pct is not None and not (0.0 < res.take_profit_pct <= 1.0):
            raise ValueError(
                f"take_profit_pct must be in (0, 1] or None, got {res.take_profit_pct}"
            )
        if res.execution_lag < 0:
            raise ValueError(f"execution_lag must be >= 0, got {res.execution_lag}")
        if res.max_open_positions is not None and res.max_open_positions < 1:
            raise ValueError(
                f"max_open_positions must be >= 1 or None, got {res.max_open_positions}"
            )
        if res.lookback_window < 2:
            raise ValueError(f"lookback_window must be >= 2, got {res.lookback_window}")

        if res.allow_short and res.allocation_method not in (
            AllocationMethod.FIXED_NOTIONAL,
            AllocationMethod.EQUAL_WEIGHT,
        ):
            raise ValueError(
                "allow_short is only supported with fixed_notional or equal_weight allocation"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to a JSON-serializable dictionary.

        Returns:
            Dictionary with enum values converted to strings.
        """
        raw = asdict(self)
        if isinstance(self.allocation_method, AllocationMethod):
            raw["allocation_method"] = self.allocation_method.value
        elif self.allocation_method is not None:
            raw["allocation_method"] = str(self.allocation_method)

        if isinstance(self.rebalance_frequency, RebalanceFrequency):
            raw["rebalance_frequency"] = self.rebalance_frequency.value
        elif self.rebalance_frequency is not None:
            raw["rebalance_frequency"] = str(self.rebalance_frequency)
        return raw

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BacktestConfig:
        """Construct a BacktestConfig instance from a dictionary."""
        copy_data = dict(data)
        if (
            "allocation_method" in copy_data
            and copy_data["allocation_method"] is not None
        ):
            copy_data["allocation_method"] = AllocationMethod(
                copy_data["allocation_method"]
            )
        if (
            "rebalance_frequency" in copy_data
            and copy_data["rebalance_frequency"] is not None
        ):
            copy_data["rebalance_frequency"] = RebalanceFrequency(
                copy_data["rebalance_frequency"]
            )
        return cls(**copy_data)

    def config_hash(self) -> str:
        """
        Compute an 8-character hex hash of the canonical resolved configuration.

        Returns:
            First 8 hex characters of the SHA-256 digest.
        """
        resolved_dict = self.resolved().to_dict()
        canonical_json = json.dumps(resolved_dict, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:8]

    def is_legacy_compatible(self) -> bool:
        """
        Determine whether this config matches the legacy engine execution model.

        Legacy engine criteria:
            - allocation_method: fixed_notional
            - rebalance_frequency: on_signal
            - allow_short: False
            - stop_loss_pct: None
            - take_profit_pct: None
            - execution_lag: 0
            - cash_reserve: 0.0
            - max_open_positions: None
            - force_close_at_end: True

        Returns:
            True if config is compatible with legacy PortfolioSimulator.
        """
        res = self.resolved()
        return (
            res.allocation_method == AllocationMethod.FIXED_NOTIONAL
            and res.rebalance_frequency == RebalanceFrequency.ON_SIGNAL
            and not res.allow_short
            and res.stop_loss_pct is None
            and res.take_profit_pct is None
            and res.execution_lag == 0
            and abs(res.cash_reserve) < 1e-9
            and res.max_open_positions is None
            and res.force_close_at_end
        )
