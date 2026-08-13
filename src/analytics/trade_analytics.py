"""
Trade analytics module for evaluating trade-level performance statistics.

Provides TradeAnalytics class to calculate win rates, profit factors,
expectancy, trade duration, and aggregate PnL statistics from trade logs.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradeAnalytics:
    """Computes trade-level performance statistics from completed trades data."""

    def __init__(self) -> None:
        """Initialize TradeAnalytics (stateless)."""
        pass

    def _validate_trades_df(self, trades_df: pd.DataFrame) -> None:
        """
        Validate trades DataFrame for non-emptiness and required columns.

        Args:
            trades_df: DataFrame containing trade records.

        Raises:
            ValueError: If trades_df is None, empty, or missing required columns.
        """
        if trades_df is None or trades_df.empty:
            raise ValueError("trades_df cannot be None or empty.")

        required_cols = {"pnl", "return_pct", "entry_date", "exit_date"}
        missing = required_cols - set(trades_df.columns)
        if missing:
            raise ValueError(f"trades_df missing required columns: {missing}")

    def win_rate(self, trades_df: pd.DataFrame) -> float:
        """
        Compute fraction of trades with pnl > 0.

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Win rate fraction.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        return float((trades_df["pnl"] > 0).sum() / len(trades_df))

    def loss_rate(self, trades_df: pd.DataFrame) -> float:
        """
        Compute fraction of trades with pnl < 0.

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Loss rate fraction.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        return float((trades_df["pnl"] < 0).sum() / len(trades_df))
