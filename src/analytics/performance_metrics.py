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

    def annualized_return(self, portfolio_history_df: pd.DataFrame) -> float:
        """Compute annualized compound return over portfolio history."""
        self._validate_portfolio_history(portfolio_history_df)
        tot_ret = self.total_return(portfolio_history_df)
        num_days = len(portfolio_history_df)
        if num_days == 1:
            logger.warning(
                "Annualization is not meaningful with a single data point; returning total return directly."
            )
            return tot_ret
        return float((1.0 + tot_ret) ** (self.trading_days_per_year / num_days) - 1.0)

    def annualized_volatility(self, portfolio_history_df: pd.DataFrame) -> float:
        """Compute annualized volatility of daily returns."""
        self._validate_portfolio_history(portfolio_history_df)
        if len(portfolio_history_df) < 2:
            logger.warning(
                "Fewer than 2 rows in portfolio history; annualized volatility is undefined."
            )
            return float("nan")

        daily_std = float(portfolio_history_df["daily_return"].std(ddof=1))
        if math.isnan(daily_std):
            logger.warning(
                "Daily return standard deviation is NaN; annualized volatility is undefined."
            )
            return float("nan")
        return daily_std * math.sqrt(self.trading_days_per_year)

    def sharpe_ratio(self, portfolio_history_df: pd.DataFrame) -> float:
        """Compute annualized Sharpe ratio."""
        self._validate_portfolio_history(portfolio_history_df)
        ann_ret = self.annualized_return(portfolio_history_df)
        ann_vol = self.annualized_volatility(portfolio_history_df)
        if math.isnan(ann_vol) or ann_vol == 0.0:
            logger.warning(
                "Cannot compute Sharpe ratio: annualized volatility is zero or undefined."
            )
            return float("nan")
        return (ann_ret - self.risk_free_rate) / ann_vol

    def sortino_ratio(self, portfolio_history_df: pd.DataFrame) -> float:
        """Compute annualized Sortino ratio using negative returns (< 0) for downside risk."""
        self._validate_portfolio_history(portfolio_history_df)
        ann_ret = self.annualized_return(portfolio_history_df)
        neg_returns = portfolio_history_df[portfolio_history_df["daily_return"] < 0]["daily_return"]
        if len(neg_returns) == 0:
            logger.warning(
                "No downside deviation observed (no negative daily returns); Sortino ratio undefined."
            )
            return float("nan")

        downside_std = float(neg_returns.std(ddof=1))
        if math.isnan(downside_std) or downside_std == 0.0:
            logger.warning(
                "Cannot compute Sortino ratio: downside volatility is zero or undefined."
            )
            return float("nan")

        downside_vol = downside_std * math.sqrt(self.trading_days_per_year)
        return (ann_ret - self.risk_free_rate) / downside_vol
