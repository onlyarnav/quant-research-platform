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
