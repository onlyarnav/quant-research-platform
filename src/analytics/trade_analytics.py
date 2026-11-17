"""
Trade analytics module for evaluating trade-level performance statistics.

Provides TradeAnalytics class to calculate win rates, profit factors,
expectancy, trade duration, and aggregate PnL statistics from trade logs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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
        """Compute fraction of trades with pnl > 0."""
        self._validate_trades_df(trades_df)
        return float((trades_df["pnl"] > 0).sum() / len(trades_df))

    def loss_rate(self, trades_df: pd.DataFrame) -> float:
        """Compute fraction of trades with pnl < 0."""
        self._validate_trades_df(trades_df)
        return float((trades_df["pnl"] < 0).sum() / len(trades_df))

    def average_win(self, trades_df: pd.DataFrame) -> float:
        """Compute mean PnL of winning trades (pnl > 0)."""
        self._validate_trades_df(trades_df)
        winning = trades_df[trades_df["pnl"] > 0]["pnl"]
        if winning.empty:
            logger.warning("No winning trades found; average win is 0.0.")
            return 0.0
        return float(winning.mean())

    def average_loss(self, trades_df: pd.DataFrame) -> float:
        """Compute mean PnL of losing trades (pnl < 0)."""
        self._validate_trades_df(trades_df)
        losing = trades_df[trades_df["pnl"] < 0]["pnl"]
        if losing.empty:
            logger.warning("No losing trades found; average loss is 0.0.")
            return 0.0
        return float(losing.mean())

    def profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Compute ratio of gross profits to gross losses."""
        self._validate_trades_df(trades_df)
        total_win = float(trades_df[trades_df["pnl"] > 0]["pnl"].sum())
        total_loss = float(trades_df[trades_df["pnl"] < 0]["pnl"].sum())

        if total_loss == 0.0 or abs(total_loss) == 0.0:
            logger.warning("No losing trades; profit factor undefined (infinite).")
            return float("inf")

        return total_win / abs(total_loss)

    def expectancy(self, trades_df: pd.DataFrame) -> float:
        """Compute expected PnL per trade: (win_rate * avg_win) + (loss_rate * avg_loss)."""
        self._validate_trades_df(trades_df)
        w_rate = self.win_rate(trades_df)
        l_rate = self.loss_rate(trades_df)
        avg_w = self.average_win(trades_df)
        avg_l = self.average_loss(trades_df)
        return (w_rate * avg_w) + (l_rate * avg_l)

    def average_trade_duration(self, trades_df: pd.DataFrame) -> pd.Timedelta:
        """Compute mean duration between entry_date and exit_date."""
        self._validate_trades_df(trades_df)
        durations = pd.to_datetime(trades_df["exit_date"]) - pd.to_datetime(
            trades_df["entry_date"]
        )
        mean_dur = durations.mean()
        return (
            mean_dur if isinstance(mean_dur, pd.Timedelta) else pd.Timedelta(mean_dur)
        )

    def total_trades(self, trades_df: pd.DataFrame) -> int:
        """Get total number of trades."""
        self._validate_trades_df(trades_df)
        return int(len(trades_df))

    def total_pnl(self, trades_df: pd.DataFrame) -> float:
        """Compute cumulative sum of trade PnL."""
        self._validate_trades_df(trades_df)
        return float(trades_df["pnl"].sum())

    def payoff_ratio(self, trades_df: pd.DataFrame) -> float:
        """Compute payoff ratio: average_win / abs(average_loss)."""
        self._validate_trades_df(trades_df)
        avg_w = self.average_win(trades_df)
        avg_l = self.average_loss(trades_df)
        if avg_l == 0.0 or abs(avg_l) < 1e-12:
            return float("nan")
        return avg_w / abs(avg_l)

    def best_trade(self, trades_df: pd.DataFrame) -> float:
        """Compute highest single trade PnL."""
        self._validate_trades_df(trades_df)
        return float(trades_df["pnl"].max())

    def worst_trade(self, trades_df: pd.DataFrame) -> float:
        """Compute lowest single trade PnL."""
        self._validate_trades_df(trades_df)
        return float(trades_df["pnl"].min())

    def _sort_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        sort_cols = ["exit_date"]
        if "trade_id" in trades_df.columns:
            sort_cols.append("trade_id")
        return trades_df.sort_values(sort_cols).reset_index(drop=True)

    def max_consecutive_wins(self, trades_df: pd.DataFrame) -> int:
        """Compute maximum consecutive winning trades (pnl > 0)."""
        self._validate_trades_df(trades_df)
        df = self._sort_trades(trades_df)
        max_c = cur = 0
        for pnl in df["pnl"]:
            if pnl > 0:
                cur += 1
                if cur > max_c:
                    max_c = cur
            else:
                cur = 0
        return max_c

    def max_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """Compute maximum consecutive losing trades (pnl < 0)."""
        self._validate_trades_df(trades_df)
        df = self._sort_trades(trades_df)
        max_c = cur = 0
        for pnl in df["pnl"]:
            if pnl < 0:
                cur += 1
                if cur > max_c:
                    max_c = cur
            else:
                cur = 0
        return max_c

    def pnl_by_symbol(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trade performance by symbol."""
        self._validate_trades_df(trades_df)
        cols = ["symbol", "n_trades", "total_pnl", "win_rate", "avg_return_pct"]
        if "symbol" not in trades_df.columns or trades_df.empty:
            return pd.DataFrame(columns=cols)
        records = []
        for sym, grp in trades_df.groupby("symbol"):
            n = len(grp)
            wins = (grp["pnl"] > 0).sum()
            records.append(
                {
                    "symbol": sym,
                    "n_trades": int(n),
                    "total_pnl": float(grp["pnl"].sum()),
                    "win_rate": float(wins / n) if n > 0 else 0.0,
                    "avg_return_pct": float(grp["return_pct"].mean()),
                }
            )
        return pd.DataFrame(records, columns=cols)

    def return_distribution(
        self, trades_df: pd.DataFrame, bins: int = 20
    ) -> pd.DataFrame:
        """Compute return percentage histogram bins."""
        self._validate_trades_df(trades_df)
        cols = ["bin_left", "bin_right", "count"]
        if trades_df.empty:
            return pd.DataFrame(columns=cols)
        counts, edges = np.histogram(trades_df["return_pct"], bins=bins)
        records = [
            {
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]
        return pd.DataFrame(records, columns=cols)

    def compute_all(
        self, trades_df: pd.DataFrame, extended: bool = False
    ) -> dict[str, Any]:
        """Compute trade analytics metrics into a dictionary."""
        self._validate_trades_df(trades_df)
        dur = self.average_trade_duration(trades_df)
        if not extended:
            return {
                "total_trades": self.total_trades(trades_df),
                "win_rate": self.win_rate(trades_df),
                "loss_rate": self.loss_rate(trades_df),
                "average_win": self.average_win(trades_df),
                "average_loss": self.average_loss(trades_df),
                "profit_factor": self.profit_factor(trades_df),
                "expectancy": self.expectancy(trades_df),
                "average_trade_duration": dur,
                "total_pnl": self.total_pnl(trades_df),
            }
        dur_days = float(dur.total_seconds() / 86400.0)
        return {
            "total_trades": self.total_trades(trades_df),
            "win_rate": self.win_rate(trades_df),
            "loss_rate": self.loss_rate(trades_df),
            "average_win": self.average_win(trades_df),
            "average_loss": self.average_loss(trades_df),
            "profit_factor": self.profit_factor(trades_df),
            "expectancy": self.expectancy(trades_df),
            "average_trade_duration_days": dur_days,
            "total_pnl": self.total_pnl(trades_df),
            "payoff_ratio": self.payoff_ratio(trades_df),
            "best_trade": self.best_trade(trades_df),
            "worst_trade": self.worst_trade(trades_df),
            "max_consecutive_wins": self.max_consecutive_wins(trades_df),
            "max_consecutive_losses": self.max_consecutive_losses(trades_df),
        }
