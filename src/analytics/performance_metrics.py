"""
Performance metrics computation module for portfolio history.

Provides PerformanceMetrics class to compute risk-adjusted portfolio metrics.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """Computes portfolio-level, risk-adjusted performance metrics."""

    def __init__(
        self,
        risk_free_rate: float | None = None,
        trading_days_per_year: int = 252,
    ) -> None:
        """
        Initialize PerformanceMetrics.

        Args:
            risk_free_rate: Annualized risk-free rate. Defaults to settings.PORTFOLIO_RISK_FREE_RATE.
            trading_days_per_year: Number of trading days per year. Defaults to 252.

        Raises:
            ValueError: If trading_days_per_year <= 0.
        """
        if trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive.")

        self.risk_free_rate: float = (
            settings.PORTFOLIO_RISK_FREE_RATE if risk_free_rate is None else risk_free_rate
        )
        self.trading_days_per_year: int = trading_days_per_year

    def _validate_portfolio_history(
        self,
        portfolio_history_df: pd.DataFrame,
        required_extra_cols: set[str] | None = None,
    ) -> None:
        """Validate portfolio history DataFrame for non-emptiness and required columns."""
        if portfolio_history_df is None or portfolio_history_df.empty:
            raise ValueError("portfolio_history_df cannot be None or empty.")

        required_cols = {"date", "portfolio_value", "daily_return"}
        if required_extra_cols:
            required_cols = required_cols.union(required_extra_cols)

        missing = required_cols - set(portfolio_history_df.columns)
        if missing:
            raise ValueError(f"portfolio_history_df missing required columns: {missing}")

    def total_return(self, portfolio_history_df: pd.DataFrame) -> float:
        """Compute cumulative total return over portfolio history."""
        self._validate_portfolio_history(portfolio_history_df)
        initial_val = float(portfolio_history_df["portfolio_value"].iloc[0])
        final_val = float(portfolio_history_df["portfolio_value"].iloc[-1])
        if initial_val <= 0:
            raise ValueError("Initial portfolio value must be greater than 0.")
        return (final_val / initial_val) - 1.0
