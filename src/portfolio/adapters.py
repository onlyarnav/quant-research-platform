"""
Adapter layer connecting legacy portfolio allocators to the unified Allocator protocol.

Wraps EqualWeightAllocator, VolatilityWeightedAllocator, and MeanVarianceAllocator
to accept AllocationContext and return standardized AllocationResult.
"""

from __future__ import annotations

from typing import Any

from src.backtesting.allocation_interface import (
    AllocationContext,
    AllocationResult,
)
from src.portfolio.constraints import PortfolioConstraints
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LegacyAllocatorAdapter:
    """Adapts existing platform allocators to the unified Allocator interface."""

    def __init__(
        self,
        name: str,
        legacy: Any,
        needs_prices: bool = False,
        constraints: PortfolioConstraints | None = None,
    ) -> None:
        """
        Initialize LegacyAllocatorAdapter.

        Args:
            name: Identifier name of the allocation strategy.
            legacy: Underlying legacy allocator instance.
            needs_prices: True if legacy allocator expects prices_df and as_of_date.
            constraints: PortfolioConstraints projecter.
        """
        self.name = name
        self.legacy = legacy
        self.needs_prices = needs_prices
        self.constraints = constraints or PortfolioConstraints()

    def allocate(self, context: AllocationContext) -> AllocationResult:
        """
        Transform AllocationContext into legacy arguments and apply constraints.

        Args:
            context: Market data and signal snapshot as of decision date.

        Returns:
            Standardized AllocationResult.
        """
        if context.signals is None or context.signals.empty:
            return AllocationResult(
                weights={}, cash_weight=1.0, excluded={}, method=self.name
            )

        # Build candidate signals DataFrame: symbols with non-zero signal (or signal 1)
        sig_df = context.signals.copy()
        if "signal" not in sig_df.columns:
            raise ValueError("signals missing required columns: {'signal'}")
        if "symbol" not in sig_df.columns:
            raise ValueError("signals missing required columns: {'symbol'}")

        # When side is short, treat signal == -1 as active candidates by mapping to 1
        if context.side == "short":
            candidates = sig_df[sig_df["signal"] == -1].copy()
            candidates["signal"] = 1
        else:
            candidates = sig_df[sig_df["signal"] == 1].copy()

        if candidates.empty:
            return AllocationResult(
                weights={},
                cash_weight=1.0,
                excluded={s: "no active signal" for s in sig_df["symbol"]},
                method=self.name,
            )

        try:
            if self.needs_prices:
                raw_weights = self.legacy.allocate(
                    candidates,
                    prices_df=context.prices_history,
                    as_of_date=context.as_of_date,
                )
            else:
                raw_weights = self.legacy.allocate(candidates)
        except ValueError as e:
            msg = str(e)
            if "cannot be empty" in msg or "missing required columns" in msg:
                logger.warning(
                    "Legacy allocator %s failed with ValueError: %s", self.name, msg
                )
                return AllocationResult(
                    weights={},
                    cash_weight=1.0,
                    excluded={s: msg for s in candidates["symbol"]},
                    method=self.name,
                )
            raise

        if not raw_weights:
            return AllocationResult(
                weights={},
                cash_weight=1.0,
                excluded={
                    s: "zero allocation from legacy" for s in candidates["symbol"]
                },
                method=self.name,
            )

        proj = self.constraints.apply(raw_weights)
        return AllocationResult(
            weights=proj.weights,
            cash_weight=proj.cash_weight,
            excluded={},
            method=self.name,
        )
