"""
Strategy report builder module.

Consolidates portfolio performance, trade analytics, benchmark comparisons,
and time-series curves into a structured StrategyReport.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.benchmark_comparison import BenchmarkComparison
from src.analytics.performance_metrics import PerformanceMetrics
from src.analytics.rolling_metrics import (
    drawdown_series,
    equity_series,
    exposure_series,
    monthly_returns_table,
    rolling_sharpe,
    rolling_volatility,
)
from src.analytics.trade_analytics import TradeAnalytics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sanitize_for_json(val: Any) -> Any:
    """Recursively convert values to JSON-serializable Python native types."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (datetime, pd.Timestamp, date)):
        return val.isoformat()
    if isinstance(val, pd.Timedelta):
        return val.total_seconds()
    if isinstance(val, pd.DataFrame):
        records = val.to_dict(orient="records")
        return [_sanitize_for_json(r) for r in records]
    if isinstance(val, dict):
        return {str(k): _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in val]
    return val


@dataclass(slots=True)
class StrategyReport:
    """Structured container of all backtest performance reports and visual series."""

    metrics: dict[str, float]
    trade_stats: dict[str, float]
    benchmark: dict[str, Any] | None
    benchmark_label: str | None
    equity: pd.DataFrame
    drawdown: pd.DataFrame
    rolling_sharpe: pd.DataFrame
    rolling_volatility: pd.DataFrame
    monthly_returns: pd.DataFrame
    trade_distribution: pd.DataFrame
    pnl_by_symbol: pd.DataFrame
    exposure: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-ready native dictionary."""
        return {
            "metrics": _sanitize_for_json(self.metrics),
            "trade_stats": _sanitize_for_json(self.trade_stats),
            "benchmark": _sanitize_for_json(self.benchmark),
            "benchmark_label": self.benchmark_label,
            "equity": _sanitize_for_json(self.equity),
            "drawdown": _sanitize_for_json(self.drawdown),
            "rolling_sharpe": _sanitize_for_json(self.rolling_sharpe),
            "rolling_volatility": _sanitize_for_json(self.rolling_volatility),
            "monthly_returns": _sanitize_for_json(self.monthly_returns),
            "trade_distribution": _sanitize_for_json(self.trade_distribution),
            "pnl_by_symbol": _sanitize_for_json(self.pnl_by_symbol),
            "exposure": _sanitize_for_json(self.exposure),
        }


class ReportBuilder:
    """Builder for constructing StrategyReport from simulator outputs."""

    def __init__(
        self,
        metrics: PerformanceMetrics | None = None,
        trade_analytics: TradeAnalytics | None = None,
        benchmark_comparison: BenchmarkComparison | None = None,
    ) -> None:
        """Initialize ReportBuilder with optional analytics engines."""
        self.metrics = metrics or PerformanceMetrics()
        self.trade_analytics = trade_analytics or TradeAnalytics()
        self.benchmark_comparison = benchmark_comparison or BenchmarkComparison(
            performance_metrics=self.metrics
        )

    def build(
        self,
        portfolio_history_df: pd.DataFrame,
        trades_df: pd.DataFrame | None = None,
        benchmark_history_df: pd.DataFrame | None = None,
        benchmark_label: str | None = None,
        daily_details_df: pd.DataFrame | None = None,
    ) -> StrategyReport:
        """
        Build complete strategy report from simulation history.

        Args:
            portfolio_history_df: Canonical portfolio history DataFrame.
            trades_df: Optional completed trades DataFrame.
            benchmark_history_df: Optional aligned benchmark history DataFrame.
            benchmark_label: Optional label for the benchmark (e.g. "SPY").
            daily_details_df: Optional daily breakdown DataFrame.

        Returns:
            StrategyReport instance.
        """
        # 1. Performance metrics
        pm_dict = self.metrics.compute_all(portfolio_history_df, extended=True)

        # 2. Trade analytics
        trade_stats: dict[str, float] = {}
        trade_dist = pd.DataFrame(columns=["bin_left", "bin_right", "count"])
        sym_pnl = pd.DataFrame(
            columns=["symbol", "n_trades", "total_pnl", "win_rate", "avg_return_pct"]
        )

        if trades_df is not None and not trades_df.empty:
            try:
                trade_stats = self.trade_analytics.compute_all(trades_df, extended=True)
                trade_dist = self.trade_analytics.return_distribution(trades_df)
                sym_pnl = self.trade_analytics.pnl_by_symbol(trades_df)
            except Exception as exc:
                logger.warning("Error computing trade analytics: %s", exc)

        # 3. Benchmark comparison
        bench_res = None
        if benchmark_history_df is not None and not benchmark_history_df.empty:
            try:
                bench_res = self.benchmark_comparison.compare(
                    portfolio_history_df, benchmark_history_df, extended=True
                )
            except Exception as exc:
                logger.warning("Error comparing with benchmark: %s", exc)

        # 4. Time series curves
        eq = equity_series(portfolio_history_df)
        dd = drawdown_series(portfolio_history_df)
        r_sharpe = rolling_sharpe(portfolio_history_df)
        r_vol = rolling_volatility(portfolio_history_df)
        m_table = monthly_returns_table(portfolio_history_df)
        exp_s = exposure_series(daily_details_df)

        return StrategyReport(
            metrics=pm_dict,
            trade_stats=trade_stats,
            benchmark=bench_res,
            benchmark_label=benchmark_label,
            equity=eq,
            drawdown=dd,
            rolling_sharpe=r_sharpe,
            rolling_volatility=r_vol,
            monthly_returns=m_table,
            trade_distribution=trade_dist,
            pnl_by_symbol=sym_pnl,
            exposure=exp_s,
        )
