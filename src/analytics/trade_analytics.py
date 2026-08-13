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

    def average_win(self, trades_df: pd.DataFrame) -> float:
        """
        Compute mean PnL of winning trades (pnl > 0).

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Mean winning trade PnL float, or 0.0 if zero winning trades.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        winning = trades_df[trades_df["pnl"] > 0]["pnl"]
        if winning.empty:
            logger.warning("No winning trades found; average win is 0.0.")
            return 0.0
        return float(winning.mean())

    def average_loss(self, trades_df: pd.DataFrame) -> float:
        """
        Compute mean PnL of losing trades (pnl < 0).

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Mean losing trade PnL float (negative number), or 0.0 if zero losing trades.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        losing = trades_df[trades_df["pnl"] < 0]["pnl"]
        if losing.empty:
            logger.warning("No losing trades found; average loss is 0.0.")
            return 0.0
        return float(losing.mean())

    def profit_factor(self, trades_df: pd.DataFrame) -> float:
        """
        Compute ratio of gross profits to gross losses.

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Profit factor float, or float("inf") if zero losing trades.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        total_win = float(trades_df[trades_df["pnl"] > 0]["pnl"].sum())
        total_loss = float(trades_df[trades_df["pnl"] < 0]["pnl"].sum())

        if total_loss == 0.0 or abs(total_loss) == 0.0:
            logger.warning("No losing trades; profit factor undefined (infinite).")
            return float("inf")

        return total_win / abs(total_loss)

    def expectancy(self, trades_df: pd.DataFrame) -> float:
        """
        Compute expected PnL per trade: (win_rate * average_win) + (loss_rate * average_loss).

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Expected PnL float per trade.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        w_rate = self.win_rate(trades_df)
        l_rate = self.loss_rate(trades_df)
        avg_w = self.average_win(trades_df)
        avg_l = self.average_loss(trades_df)
        return (w_rate * avg_w) + (l_rate * avg_l)

    def average_trade_duration(self, trades_df: pd.DataFrame) -> pd.Timedelta:
        """
        Compute mean duration between entry_date and exit_date.

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Average trade duration as pd.Timedelta.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        durations = pd.to_datetime(trades_df["exit_date"]) - pd.to_datetime(
            trades_df["entry_date"]
        )
        mean_dur = durations.mean()
        return mean_dur if isinstance(mean_dur, pd.Timedelta) else pd.Timedelta(mean_dur)

    def total_trades(self, trades_df: pd.DataFrame) -> int:
        """
        Get total number of trades.

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Number of trades integer.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        return int(len(trades_df))

    def total_pnl(self, trades_df: pd.DataFrame) -> float:
        """
        Compute cumulative sum of trade PnL.

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Total PnL float.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        return float(trades_df["pnl"].sum())

    def compute_all(
        self, trades_df: pd.DataFrame
    ) -> dict[str, float | int | pd.Timedelta]:
        """
        Compute all trade analytics metrics into a dictionary.

        Args:
            trades_df: DataFrame containing trade records.

        Returns:
            Dictionary containing trade statistics.

        Raises:
            ValueError: If trades_df is empty or missing required columns.
        """
        self._validate_trades_df(trades_df)
        return {
            "total_trades": self.total_trades(trades_df),
            "win_rate": self.win_rate(trades_df),
            "loss_rate": self.loss_rate(trades_df),
            "average_win": self.average_win(trades_df),
            "average_loss": self.average_loss(trades_df),
            "profit_factor": self.profit_factor(trades_df),
            "expectancy": self.expectancy(trades_df),
            "average_trade_duration": self.average_trade_duration(trades_df),
            "total_pnl": self.total_pnl(trades_df),
        }
