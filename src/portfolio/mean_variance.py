"""
Mean-variance portfolio allocation using PyPortfolioOpt.

Allocates capital to maximize the Sharpe ratio subject to a
long-only, fully-invested constraint, using historical price data
to estimate expected returns and the covariance matrix.

Falls back to equal weighting if optimization fails to converge
or if fewer than 2 eligible symbols have sufficient price history,
since mean-variance optimization is not meaningful with 0 or 1 assets.
"""

from __future__ import annotations

import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MeanVarianceAllocator:
    """
    Allocates capital via mean-variance (Markowitz) optimization,
    maximizing Sharpe ratio subject to long-only, fully-invested
    constraints.
    """

    def __init__(
        self,
        lookback_window: int = 252,
        risk_free_rate: float | None = None,
    ) -> None:
        """
        Args:
            lookback_window: Number of trailing trading days of price
                history used to estimate returns and covariance.
            risk_free_rate: Annual risk-free rate used in Sharpe ratio
                maximization. Defaults to settings.PORTFOLIO_RISK_FREE_RATE.

        Raises:
            ValueError: If lookback_window < 2.
        """
        if lookback_window < 2:
            raise ValueError(f"lookback_window must be at least 2, got {lookback_window}")

        self.lookback_window = lookback_window

        if risk_free_rate is None:
            from config.settings import settings
            risk_free_rate = settings.PORTFOLIO_RISK_FREE_RATE

        self.risk_free_rate = risk_free_rate

    def allocate(
        self,
        signals_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> dict[str, float]:
        """
        Compute mean-variance optimal weights for all BUY-signaled symbols.

        Args:
            signals_df: DataFrame with 'symbol', 'signal' columns,
                already filtered to a single deat by the caller.
            prices_df: Historical price DataFrame with 'symbol', 'date',
                'close' columns.
            as_of_date: The date to compute allocation as of (inclusive).

        Returns:
            Dict mapping symbol -> weight, summing to 1.0. Returns an
            empty dict if no symbols have signal == 1. Falls back to
            equal weighting across eligible symbols if fewer than 2
            symbols have sufficient history, or if optimization fails.

        Raises:
            ValueError: If signals_df or prices_df is empty, or missing
                required columns.
        """
        if signals_df.empty:
            raise ValueError("signals_df cannot be empty.")
        if prices_df.empty:
            raise ValueError("prices_df cannot be empty.")

        required_signal_cols = {"symbol", "signal"}
        missing = required_signal_cols - set(signals_df.columns)
        if missing:
            raise ValueError(f"signals_df missing required columns: {missing}")

        required_price_cols = {"symbol", "date", "close"}
        missing = required_price_cols - set(prices_df.columns)
        if missing:
            raise ValueError(f"prices_df missing required columns: {missing}")

        buy_symbols = list(signals_df.loc[signals_df["signal"] == 1, "symbol"].unique())

        if len(buy_symbols) == 0:
            logger.warning("No BUY signals found; returning empty allocation.")
            return {}

        history = prices_df[
            (prices_df["symbol"].isin(buy_symbols)) & (prices_df["date"] <= as_of_date)
        ]

        price_matrix = history.pivot(index="date", columns="symbol", values="close")
        price_matrix = price_matrix.sort_index().tail(self.lookback_window)

        eligible_symbols = [
            symbol for symbol in buy_symbols
            if symbol in price_matrix.columns and price_matrix[symbol].notna().sum() >= self.lookback_window
        ]

        excluded = set(buy_symbols) - set(eligible_symbols)
        if excluded:
            logger.warning(
                "Excluding symbols with insufficient price history: %s", sorted(excluded)
            )

        price_matrix = price_matrix[eligible_symbols]

        if len(eligible_symbols) < 2:
            logger.warning(
                "Fewer than 2 eligible symbols (%d); falling back to equal weighting.",
                len(eligible_symbols),
            )
            return self._equal_weight_fallback(eligible_symbols)

        try:
            mu = expected_returns.mean_historical_return(price_matrix, frequency=252)
            S = risk_models.sample_cov(price_matrix, frequency=252)

            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            cleaned_weights = ef.clean_weights()

            allocation = {
                symbol: weight for symbol, weight in cleaned_weights.items() if weight > 0
            }

            if not allocation:
                logger.warning(
                    "Optimization returned all-zero weights; falling back to equal weighting."
                )
                return self._equal_weight_fallback(eligible_symbols)

            total = sum(allocation.values())
            allocation = {symbol: weight / total for symbol, weight in allocation.items()}

            logger.info(
                "Mean-variance allocation across %d symbols (excluded %d).",
                len(allocation),
                len(buy_symbols) - len(allocation),
            )

            return allocation

        except Exception as exc:
            logger.warning(
                "Mean-variance optimization failed (%s); falling back to equal weighting.",
                exc,
            )
            return self._equal_weight_fallback(eligible_symbols)

    def _equal_weight_fallback(self, symbols: list[str]) -> dict[str, float]:
        """Equal-weight allocation used when optimization is not viable."""
        if not symbols:
            return {}
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
