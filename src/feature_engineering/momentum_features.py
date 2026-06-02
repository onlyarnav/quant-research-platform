"""
Momentum feature generation.

Generates momentum-based technical indicators used by downstream
machine learning models and quantitative research pipelines.

Features:
    - RSI (14)
    - MACD
    - MACD Signal
    - ROC (5, 10)
    - Momentum (5, 10)

Rules:
    - Uses adj_close if available, otherwise falls back to close
    - Computed independently per symbol
    - No lookahead bias
    - Returns only symbol, date, and generated feature columns
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MomentumFeatureGenerator:
    """
    Generate momentum-based technical indicators.

    Input DataFrame requirements:
        symbol
        date
        close

    Optional:
        adj_close

    Output:
        symbol
        date
        rsi_14
        macd
        macd_signal
        roc_5
        roc_10
        momentum_5
        momentum_10
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "symbol",
        "date",
        "close",
    )

    FEATURE_COLUMNS: tuple[str, ...] = (
        "rsi_14",
        "macd",
        "macd_signal",
        "roc_5",
        "roc_10",
        "momentum_5",
        "momentum_10",
    )

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all momentum features.

        Parameters
        ----------
        df : pd.DataFrame
            Processed market data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing symbol, date, and momentum features.
        """

        self._validate_input(df)

        logger.info(
            "Generating momentum features for %s rows across %s symbols.",
            len(df),
            df["symbol"].nunique(),
        )

        result = df.copy()

        result = result.sort_values(
            ["symbol", "date"]
        ).reset_index(drop=True)

        price_column = self._get_price_column(result)

        logger.info(
            "Using '%s' as source price column for momentum features.",
            price_column,
        )

        result = self._generate_rsi(
            result,
            price_column,
        )

        result = self._generate_macd(
            result,
            price_column,
        )

        result = self._generate_roc(
            result,
            price_column,
        )

        result = self._generate_momentum(
            result,
            price_column,
        )

        logger.info(
            "Successfully generated %s momentum features.",
            len(self.FEATURE_COLUMNS),
        )

        return result[
            [
                "symbol",
                "date",
                *self.FEATURE_COLUMNS,
            ]
        ]

    def _generate_rsi(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate RSI(14) using standard Wilder smoothing.
        """

        def calculate_rsi(prices: pd.Series) -> pd.Series:
            delta = prices.diff()

            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)

            avg_gain = gains.ewm(
                alpha=1 / 14,
                adjust=False,
                min_periods=14,
            ).mean()

            avg_loss = losses.ewm(
                alpha=1 / 14,
                adjust=False,
                min_periods=14,
            ).mean()

            rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))

            return rsi

        df["rsi_14"] = (
            df.groupby("symbol")[price_column]
            .transform(calculate_rsi)
        )

        return df

    def _generate_macd(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate MACD and MACD signal line.

        MACD = EMA(12) - EMA(26)
        Signal = EMA(9) of MACD
        """

        def calculate_macd(
            prices: pd.Series,
        ) -> pd.DataFrame:
            ema_12 = prices.ewm(
                span=12,
                adjust=False,
            ).mean()

            ema_26 = prices.ewm(
                span=26,
                adjust=False,
            ).mean()

            macd = ema_12 - ema_26

            macd_signal = macd.ewm(
                span=9,
                adjust=False,
            ).mean()

            return pd.DataFrame(
                {
                    "macd": macd,
                    "macd_signal": macd_signal,
                }
            )

        macd_features = (
            df.groupby("symbol")[price_column]
            .apply(calculate_macd)
            .reset_index(level=0, drop=True)
        )

        df["macd"] = macd_features["macd"]
        df["macd_signal"] = macd_features["macd_signal"]

        return df

    def _generate_roc(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate Rate of Change features.

        ROC = (price(t) / price(t-n) - 1) * 100
        """

        for period in (5, 10):
            df[f"roc_{period}"] = (
                df.groupby("symbol")[price_column]
                .transform(
                    lambda x, p=period: (
                        (x / x.shift(p)) - 1
                    )
                    * 100
                )
            )

        return df

    def _generate_momentum(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate momentum features.

        Momentum = price(t) - price(t-n)
        """

        for period in (5, 10):
            df[f"momentum_{period}"] = (
                df.groupby("symbol")[price_column]
                .transform(
                    lambda x, p=period: x - x.shift(p)
                )
            )

        return df

    def _get_price_column(
        self,
        df: pd.DataFrame,
    ) -> str:
        """
        Select source price column.

        Preference:
            1. adj_close
            2. close
        """

        if (
            "adj_close" in df.columns
            and not df["adj_close"].isna().all()
        ):
            return "adj_close"

        return "close"

    def _validate_input(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate input DataFrame.
        """

        if df.empty:
            raise ValueError(
                "Input DataFrame is empty."
            )

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        if df["symbol"].isna().any():
            raise ValueError(
                "Column 'symbol' contains null values."
            )

        if df["date"].isna().any():
            raise ValueError(
                "Column 'date' contains null values."
            )

        if df["close"].isna().any():
            raise ValueError(
                "Column 'close' contains null values."
            )