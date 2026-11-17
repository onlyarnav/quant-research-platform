"""
Base allocator abstraction implementing the template method pattern.

Provides AllocatorBase for input validation, raw weight computation delegation,
constraint application, and unified AllocationResult assembly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.backtesting.allocation_interface import (
    AllocationContext,
    AllocationResult,
)
from src.portfolio.constraints import PortfolioConstraints
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AllocatorBase(ABC):
    """Abstract base class for quantitative portfolio allocators."""

    def __init__(
        self,
        constraints: PortfolioConstraints | None = None,
        name: str = "base_allocator",
    ) -> None:
        """
        Initialize AllocatorBase.

        Args:
            constraints: Portfolio concentration and exposure constraints.
            name: Identifier string for this allocator method.
        """
        self.constraints = constraints or PortfolioConstraints()
        self.name = name

    def allocate(self, context: AllocationContext) -> AllocationResult:
        """
        Execute template allocation workflow: validate, calculate raw weights, project constraints.

        Args:
            context: Market data and portfolio state as of decision date.

        Returns:
            AllocationResult containing final weights and diagnostic metadata.

        Raises:
            ValueError: If context signals DataFrame is missing required columns.
        """
        if context.signals is None or context.signals.empty:
            logger.warning("AllocationContext signals DataFrame is empty.")
            return AllocationResult(
                weights={}, cash_weight=1.0, excluded={}, method=self.name
            )

        required_cols = {"symbol", "signal"}
        missing = required_cols - set(context.signals.columns)
        if missing:
            raise ValueError(f"signals missing required columns: {missing}")

        raw_weights, excluded = self._raw_weights(context)
        if not raw_weights:
            return AllocationResult(
                weights={}, cash_weight=1.0, excluded=excluded, method=self.name
            )

        constraint_result = self.constraints.apply(raw_weights)
        return AllocationResult(
            weights=constraint_result.weights,
            cash_weight=constraint_result.cash_weight,
            excluded=excluded,
            method=self.name,
        )

    @abstractmethod
    def _raw_weights(
        self, context: AllocationContext
    ) -> tuple[dict[str, float], dict[str, str]]:
        """
        Compute unconstrained weights summing to 1.0 over eligible symbols.

        Args:
            context: Allocation context as of decision date.

        Returns:
            Tuple of (raw_weights_dict, excluded_reasons_dict).
        """
        ...
