"""
Performance and trade analytics package.
"""

from src.analytics.backtest_pipeline import BacktestPipeline, BacktestRunResult
from src.analytics.benchmark_builder import BenchmarkBuilder
from src.analytics.benchmark_comparison import BenchmarkComparison
from src.analytics.metrics_persistence import persist_metrics, persist_trades
from src.analytics.performance_metrics import PerformanceMetrics
from src.analytics.report_builder import ReportBuilder, StrategyReport
from src.analytics.rolling_metrics import (
    drawdown_series,
    equity_series,
    exposure_series,
    monthly_returns_table,
    rolling_sharpe,
    rolling_volatility,
)
from src.analytics.trade_analytics import TradeAnalytics

__all__ = [
    "PerformanceMetrics",
    "TradeAnalytics",
    "BenchmarkComparison",
    "BenchmarkBuilder",
    "StrategyReport",
    "ReportBuilder",
    "BacktestPipeline",
    "BacktestRunResult",
    "persist_metrics",
    "persist_trades",
    "rolling_sharpe",
    "rolling_volatility",
    "drawdown_series",
    "equity_series",
    "monthly_returns_table",
    "exposure_series",
]
