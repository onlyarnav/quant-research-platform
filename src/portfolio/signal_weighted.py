"""
Signal-strength weighted portfolio allocation strategy.

Allocates capital proportional to ML model confidence or predicted return,
falling back to equal weighting when signals lack distinctive positive magnitudes.
"""

from __future__ import annotations

import pandas as pd

from src.backtesting.allocation_interface import AllocationContext
from src.portfolio.base import AllocatorBase
from src.portfolio.constraints import PortfolioConstraints
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SignalWeightedAllocator(AllocatorBase):
    """Allocates portfolio weights proportional to signal conviction or return predictions."""

    def __init__(
        self,
        constraints: PortfolioConstraints | None = None,
        basis: str = "confidence",
    ) -> None:
        """
        Initialize SignalWeightedAllocator.

        Args:
            constraints: Portfolio concentration and exposure constraints.
            basis: Weighting basis: 'confidence' or 'predicted_return'.

        Raises:
            ValueError: If basis is not 'confidence' or 'predicted_return'.
        """
        super().__init__(constraints=constraints, name=f"signal_weighted_{basis}")
        if basis not in ("confidence", "predicted_return"):
            raise ValueError(
                f"basis must be 'confidence' or 'predicted_return', got '{basis}'"
            )
        self.basis = basis

    def _raw_weights(
        self, context: AllocationContext
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Compute unconstrained weights proportional to chosen signal basis."""
        sig_df = context.signals
        target_sig = -1 if context.side == "short" else 1
        active = sig_df[sig_df["signal"] == target_sig].copy()

        if active.empty:
            return {}, {s: "no active signal" for s in sig_df["symbol"]}

        excluded: dict[str, str] = {}
        magnitudes: dict[str, float] = {}

        for row in active.itertuples(index=False):
            sym = row.symbol
            val = getattr(row, self.basis, None)
            if val is None or pd.isna(val) or float(val) <= 0.0:
                excluded[sym] = f"non-positive or NaN {self.basis}: {val}"
                continue
            magnitudes[sym] = float(val)

        total_mag = sum(magnitudes.values())
        if total_mag <= 1e-12:
            logger.warning(
                "All signal weights are zero; falling back to equal weighting"
            )
            # Fall back to equal weight across all active symbols
            symbols = list(active["symbol"].unique())
            eq_w = 1.0 / len(symbols)
            return {s: eq_w for s in symbols}, {}

        raw_weights = {s: mag / total_mag for s, mag in magnitudes.items()}
        return raw_weights, excluded
