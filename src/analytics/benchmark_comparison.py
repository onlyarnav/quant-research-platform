"""
Benchmark comparison module for strategy vs benchmark analytics.

Provides BenchmarkComparison class to evaluate excess performance, alpha,
beta, and information ratio against a passive benchmark history.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.performance_metrics import PerformanceMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkComparison:
    """Compares strategy performance history against a passive benchmark history."""

    def __init__(
        self, performance_metrics: PerformanceMetrics | None = None
    ) -> None:
        """
        Initialize BenchmarkComparison.

        Args:
            performance_metrics: Injected PerformanceMetrics instance. If None,
                instantiates default PerformanceMetrics().
        """
        self.performance_metrics: PerformanceMetrics = (
            PerformanceMetrics()
            if performance_metrics is None
            else performance_metrics
        )

    def _validate_history_df(self, df: pd.DataFrame, df_name: str) -> None:
        """
        Validate history DataFrame for non-emptiness and required columns.

        Args:
            df: DataFrame to validate.
            df_name: Descriptive name for error messages.

        Raises:
            ValueError: If df is None, empty, or missing required columns.
        """
        if df is None or df.empty:
            raise ValueError(f"{df_name} cannot be None or empty.")

        required_cols = {"date", "portfolio_value", "daily_return"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

    def compare(
        self,
        strategy_history_df: pd.DataFrame,
        benchmark_history_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Compare strategy performance against benchmark performance.

        Computes strategy and benchmark performance metrics, as well as relative
        metrics: alpha (excess annualized return over benchmark), beta (sensitivity
        to benchmark returns), information ratio (annualized risk-adjusted excess
        return), and outperformance (total return excess). Note: alpha is simple
        excess annualized return rather than CAPM regression alpha.

        Args:
            strategy_history_df: Portfolio history DataFrame for the strategy.
            benchmark_history_df: Portfolio history DataFrame for the benchmark.

        Returns:
            Dictionary containing strategy_metrics, benchmark_metrics, alpha,
            beta, information_ratio, and outperformance.

        Raises:
            ValueError: If either DataFrame is invalid or there are no overlapping dates.
        """
        self._validate_history_df(strategy_history_df, "strategy_history_df")
        self._validate_history_df(benchmark_history_df, "benchmark_history_df")

        aligned_df = pd.merge(
            strategy_history_df,
            benchmark_history_df,
            on="date",
            suffixes=("_strat", "_bench"),
        )

        if aligned_df.empty:
            raise ValueError(
                "No overlapping dates between strategy_history_df and benchmark_history_df."
            )

        strat_metrics = self.performance_metrics.compute_all(strategy_history_df)
        bench_metrics = self.performance_metrics.compute_all(benchmark_history_df)

        alpha = strat_metrics["annualized_return"] - bench_metrics["annualized_return"]
        outperformance = strat_metrics["total_return"] - bench_metrics["total_return"]

        strat_ret = aligned_df["daily_return_strat"]
        bench_ret = aligned_df["daily_return_bench"]

        bench_var = float(bench_ret.var(ddof=1))
        if len(aligned_df) < 2 or math.isnan(bench_var) or bench_var == 0.0:
            logger.warning(
                "Cannot compute beta: benchmark variance is zero or undefined."
            )
            beta = float("nan")
        else:
            cov = float(strat_ret.cov(bench_ret))
            beta = cov / bench_var

        diff_ret = strat_ret - bench_ret
        diff_std = float(diff_ret.std(ddof=1))

        if len(aligned_df) < 2 or math.isnan(diff_std) or diff_std == 0.0:
            logger.warning(
                "Cannot compute information ratio: tracking error standard deviation is zero or undefined."
            )
            information_ratio = float("nan")
        else:
            diff_mean = float(diff_ret.mean())
            trading_days = self.performance_metrics.trading_days_per_year
            information_ratio = (diff_mean / diff_std) * math.sqrt(trading_days)

        return {
            "strategy_metrics": strat_metrics,
            "benchmark_metrics": bench_metrics,
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "outperformance": outperformance,
        }
