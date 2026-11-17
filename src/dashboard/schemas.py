"""
Pydantic v2 schemas and sanitization helpers for Dashboard API endpoints.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

T = TypeVar("T")


def sanitize(obj: Any) -> Any:
    """Recursively convert float NaN/inf to None, Timestamps to ISO, numpy to Python."""
    if obj is None:
        return None
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if isinstance(obj, pd.DataFrame):
        records = obj.to_dict(orient="records")
        return [sanitize(r) for r in records]
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    return obj


class HealthResponse(BaseModel):
    """System health inspection response schema."""

    status: str
    checks: list[dict[str, Any]]
    generated_at: str


class RunSummary(BaseModel):
    """Metadata summary of a stored backtest run."""

    run_id: str
    strategy_name: str
    created_at: str
    n_trades: int
    final_equity: float
    start_date: str | None = None
    end_date: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Performance, trade, and benchmark metrics response."""

    metrics: dict[str, Any]
    trade_stats: dict[str, Any] = Field(default_factory=dict)
    benchmark: dict[str, Any] | None = None


class SeriesResponse(BaseModel):
    """Time-series curve response schema."""

    columns: list[str]
    rows: list[dict[str, Any]]


class Page(BaseModel, Generic[T]):
    """Generic paginated response schema."""

    items: list[T]
    total: int
    limit: int
    offset: int


class SignalRow(BaseModel):
    """Individual trade signal model."""

    date: str
    symbol: str
    signal: int
    confidence: float | None = None
    rank: float | None = None


class SignalCounts(BaseModel):
    """Counts of buy, sell, and neutral signals as of a given date."""

    date: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Trained machine learning model summary."""

    name: str
    version: str | None = None
    created_at: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class PipelineRunInfo(BaseModel):
    """Pipeline orchestration run summary."""

    run_id: str
    started_at: str
    finished_at: str
    all_succeeded: bool
    total_duration_seconds: float
    steps: list[dict[str, Any]] = Field(default_factory=dict)


class BacktestRequest(BaseModel):
    """Job submission request for launching a backtest."""

    strategy_name: str = "strategy"
    symbols: list[str] | None = None
    asset_class: str = "equity"
    timeframe: str = "daily"
    start: str | None = None
    end: str | None = None
    allocation_method: str | None = None
    rebalance_frequency: str | None = None
    initial_capital: float | None = None
    transaction_cost: float | None = None
    slippage: float | None = None
    max_position_size: float | None = None
    cash_reserve: float | None = None
    allow_short: bool = False
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    execution_lag: int = 1


class BacktestJobStatus(BaseModel):
    """Background backtest execution job state."""

    job_id: str
    status: str  # queued | running | succeeded | failed
    run_id: str | None = None
    error: str | None = None
