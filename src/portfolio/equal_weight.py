"""
Equal-weight portfolio allocation.

Allocates capital equally across all symbols with an active BUY
signal, ignoring any signal strength, volatility, or correlation
information. The simplest possible allocation baseline.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EqualWeightAllocator:
    """
    Allocates capital equally across all symbols with a BUY signal.

    Every symbol receives weight = 1 / n, where n is the count of
    symbols with signal == 1 on the given date.
    """

    def allocate(self, signals_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute equal weights for all BUY-signaled symbols.

        Args:
            signals_df: DataFrame with at least 'symbol' and 'signal'
                columns, already filtered to a single date by the
                caller. Only rows with signal == 1 are allocated.

        Returns:
            Dict mapping symbol -> weight, where weights sum to 1.0.
            Returns an empty dict if no symbols have signal == 1.

        Raises:
            ValueError: If signals_df is empty or missing required columns.
        """
        if signals_df.empty:
            raise ValueError("signals_df cannot be empty.")

        required_cols = {"symbol", "signal"}
        missing = required_cols - set(signals_df.columns)
        if missing:
            raise ValueError(f"signals_df missing required columns: {missing}")

        buy_symbols = signals_df.loc[signals_df["signal"] == 1, "symbol"].unique()

        if len(buy_symbols) == 0:
            logger.warning("No BUY signals found; returning empty allocation.")
            return {}

        weight = 1.0 / len(buy_symbols)
        allocation = {symbol: weight for symbol in buy_symbols}

        logger.info(
            "Equal-weight allocation across %d symbols (weight=%.4f each).",
            len(buy_symbols),
            weight,
        )

        return allocation
