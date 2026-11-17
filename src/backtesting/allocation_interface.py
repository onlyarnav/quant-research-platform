"""
Protocol and context data structures for portfolio allocation.

Defines the protocol connecting backtesting simulation to portfolio construction
allocators without introducing reverse imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass(slots=True)
class AllocationContext:
    """
    Context provided to an allocator to determine portfolio weights.

    Attributes:
        as_of_date: Decision date as of which data is available (no lookahead).
        signals: DataFrame of candidate symbols for this date.
        prices_history: Historical price DataFrame for dates <= as_of_date.
        current_weights: Current portfolio symbol weights.
        portfolio_value: Current total portfolio equity.
        side: Position side to allocate ("long" or "short").
    """

    as_of_date: pd.Timestamp
    signals: pd.DataFrame
    prices_history: pd.DataFrame
    current_weights: dict[str, float] = field(default_factory=dict)
    portfolio_value: float = 0.0
    side: str = "long"


@dataclass(slots=True)
class AllocationResult:
    """
    Target weights and diagnostic metadata returned by an allocator.

    Attributes:
        weights: Symbol -> target weight fraction of total equity (non-negative).
        cash_weight: Unallocated cash fraction (e.g. cash reserve or unspent budget).
        excluded: Mapping of symbol -> exclusion reason string.
        method: Identifier name of the allocation algorithm applied.
    """

    weights: dict[str, float]
    cash_weight: float
    excluded: dict[str, str]
    method: str


class Allocator(Protocol):
    """Protocol implemented by all portfolio allocation models."""

    def allocate(self, context: AllocationContext) -> AllocationResult:
        """
        Compute target portfolio weights for candidate symbols.

        Args:
            context: Market, signal, and portfolio state as of the decision date.

        Returns:
            AllocationResult containing target weights and exclusions.
        """
        ...
