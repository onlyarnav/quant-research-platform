"""
Volatility-weighted portfolio allocation.

Allocates capital inversely proportional to each symbol's historical
return volatility — lower-volatility symbols receive higher weight.
Requires historical price data to compute volatility; signals_df alone
is insufficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VolatilityWeightedAllocator:
    """
    Allocates capital inversely proportional to historical volatility.

    Volatility is computed as the standard deviation of daily returns
    over a trailing lookback window, per symbol, using price history
    up to (and including) the allocation date.
    """

    def __init__(self, lookback_window: int = 20) -> None:
        """
        Args:
            lookback_window: Number of trailing trading days used to
                compute volatility.

        Raises:
            ValueError: If lookback_window < 2 (need at least 2 prices
                to compute at least 1 return).
        """
        if lookback_window < 2:
            raise ValueError(f"lookback_window must be at least 2, got {lookback_window}")
        self.lookback_window = lookback_window

    def allocate(
        self,
        signals_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> dict[str, float]:
        """
        Compute inverse-volatility weights for all BUY-signaled symbols.

        Args:
            signals_df: DataFrame with 'symbol', 'signal' columns,
                already filtered to a single date by the caller.
            prices_df: Historical price DataFrame with 'symbol', 'date',
                'close' columns, covering at least lookback_window+1
                trading days up to as_of_date for each eligible symbol.
            as_of_date: The date to compute volatility as of (inclusive).

        Returns:
            Dict mapping symbol -> weight, where weights sum to 1.0.
            Returns an empty dict if no symbols have signal == 1.
            Symbols with insufficient price history (fewer than
            lookback_window+1 rows on or before as_of_date) or with
            exactly zero volatility are excluded and logged, with
            remaining weights renormalized to sum to 1.0.

        Raises:
            ValueError: If signals_df or prices_df is empty, or missing
                required columns.
        """
        if signals_df.empty:
            raise ValueError("signals_df cannot be empty.")
        if prices_df.empty:
            raise ValueError("prices_df cannot be empty.")

        required_signal_cols = {"symbol", "signal"}
        missing_signals = required_signal_cols - set(signals_df.columns)
        if missing_signals:
            raise ValueError(f"signals_df missing required columns: {missing_signals}")

        required_price_cols = {"symbol", "date", "close"}
        missing_prices = required_price_cols - set(prices_df.columns)
        if missing_prices:
            raise ValueError(f"prices_df missing required columns: {missing_prices}")

        buy_symbols = signals_df.loc[signals_df["signal"] == 1, "symbol"].unique()

        if len(buy_symbols) == 0:
            logger.warning("No BUY signals found; returning empty allocation.")
            return {}

        volatilities: dict[str, float] = {}

        for symbol in buy_symbols:
            symbol_prices = (
                prices_df[(prices_df["symbol"] == symbol) & (prices_df["date"] <= as_of_date)]
                .sort_values("date")
                .tail(self.lookback_window + 1)
            )

            if len(symbol_prices) < self.lookback_window + 1:
                logger.warning(
                    "Symbol=%s has insufficient price history (%d rows, need %d). Excluding.",
                    symbol,
                    len(symbol_prices),
                    self.lookback_window + 1,
                )
                continue

            returns = symbol_prices["close"].pct_change().dropna()
            vol = returns.std()

            if vol == 0 or np.isnan(vol):
                logger.warning("Symbol=%s has zero or undefined volatility. Excluding.", symbol)
                continue

            volatilities[symbol] = vol

        if not volatilities:
            logger.warning("No symbols with valid volatility; returning empty allocation.")
            return {}

        inverse_vols = {symbol: 1.0 / vol for symbol, vol in volatilities.items()}
        total_inverse_vol = sum(inverse_vols.values())

        allocation = {
            symbol: inv_vol / total_inverse_vol
            for symbol, inv_vol in inverse_vols.items()
        }

        logger.info(
            "Volatility-weighted allocation across %d symbols (excluded %d).",
            len(allocation),
            len(buy_symbols) - len(allocation),
        )

        return allocation
