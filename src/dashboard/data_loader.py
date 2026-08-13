"""Centralized data loader for dashboard read-only database queries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.database.repository import (
    AssetRepository,
    PortfolioMetricRepository,
    SignalRepository,
    TradeRepository,
)
from src.database.sessions import get_session
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _orm_list_to_dataframe_impl(records: list[Any], columns: list[str]) -> pd.DataFrame:
    """
    Extract attributes from a list of ORM objects into a pandas DataFrame.

    Args:
        records: List of SQLAlchemy ORM instances.
        columns: List of attribute names to extract for each column.

    Returns:
        DataFrame containing extracted attributes or empty DataFrame with columns.
    """
    if not records:
        return pd.DataFrame(columns=columns)

    data: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        for col in columns:
            if col == "trade_id" and not hasattr(record, "trade_id"):
                val = getattr(record, "id", None)
            else:
                val = getattr(record, col, None)
            row[col] = val
        data.append(row)

    return pd.DataFrame(data, columns=columns)


def orm_list_to_dataframe(records: list[Any], columns: list[str]) -> pd.DataFrame:
    """Public wrapper for ORM list to DataFrame conversion."""
    return _orm_list_to_dataframe_impl(records, columns)


def _load_trades_impl(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Uncached implementation to load trade records for a symbol within a date range.

    Args:
        symbol: Equity ticker symbol.
        start_date: Start date threshold.
        end_date: End date threshold.

    Returns:
        DataFrame containing trades matching Trade schema.
    """
    columns = [
        "trade_id",
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "position_size",
        "fees",
        "slippage_cost",
        "pnl",
        "return_pct",
    ]
    start_dt = start_date.to_pydatetime() if isinstance(start_date, pd.Timestamp) else start_date
    end_dt = end_date.to_pydatetime() if isinstance(end_date, pd.Timestamp) else end_date

    with get_session() as session:
        repo = TradeRepository(session)
        records = repo.get_by_date_range(symbol, start_dt, end_dt)
        logger.info("Loaded %s trades for symbol=%s", len(records), symbol)
        return _orm_list_to_dataframe_impl(records, columns)


@st.cache_data(ttl=300)
def load_trades(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Load trade records for a symbol within a date range (cached 5 min)."""
    return _load_trades_impl(symbol, start_date, end_date)


def _load_portfolio_history_impl(strategy_name: str) -> pd.DataFrame:
    """
    Uncached implementation to load and pivot portfolio metrics for a strategy.

    Args:
        strategy_name: Unique identifier for strategy.

    Returns:
        Wide-format DataFrame with date and portfolio metrics columns.
    """
    expected_cols = [
        "date",
        "portfolio_value",
        "daily_return",
        "cumulative_return",
        "drawdown",
    ]

    with get_session() as session:
        repo = PortfolioMetricRepository(session)
        records = repo.get_by_strategy(strategy_name)
        logger.info("Loaded %s portfolio metric records for strategy=%s", len(records), strategy_name)

        if not records:
            return pd.DataFrame(columns=expected_cols)

        raw_df = _orm_list_to_dataframe_impl(
            records, ["strategy_name", "metric", "value", "calculated_at"]
        )

    pivoted = raw_df.pivot(index="calculated_at", columns="metric", values="value").reset_index()
    pivoted.rename(columns={"calculated_at": "date"}, inplace=True)

    for col in expected_cols:
        if col not in pivoted.columns:
            pivoted[col] = None

    return pivoted


@st.cache_data(ttl=300)
def load_portfolio_history(strategy_name: str) -> pd.DataFrame:
    """Load and pivot portfolio metrics for a strategy (cached 5 min)."""
    return _load_portfolio_history_impl(strategy_name)


def _load_signals_impl(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Uncached implementation to load signals for a symbol within a date range.

    Args:
        symbol: Equity ticker symbol.
        start_date: Start date threshold.
        end_date: End date threshold.

    Returns:
        DataFrame containing signal records.
    """
    columns = ["symbol", "date", "signal", "predicted_return", "model_version"]
    start_dt = start_date.to_pydatetime() if isinstance(start_date, pd.Timestamp) else start_date
    end_dt = end_date.to_pydatetime() if isinstance(end_date, pd.Timestamp) else end_date

    with get_session() as session:
        repo = SignalRepository(session)
        records = repo.get_by_date_range(symbol, start_dt, end_dt)
        logger.info("Loaded %s signals for symbol=%s", len(records), symbol)
        return _orm_list_to_dataframe_impl(records, columns)


@st.cache_data(ttl=300)
def load_signals(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Load signals for a symbol within a date range (cached 5 min)."""
    return _load_signals_impl(symbol, start_date, end_date)


def _load_available_symbols_impl() -> list[str]:
    """
    Uncached implementation to load active equity symbols.

    Returns:
        Sorted list of active equity symbol strings.
    """
    with get_session() as session:
        repo = AssetRepository(session)
        records = repo.get_by_asset_class("equity", active_only=True)
        symbols = sorted([rec.symbol for rec in records if rec.symbol])
        logger.info("Loaded %s active equity symbols", len(symbols))
        return symbols


@st.cache_data(ttl=600)
def load_available_symbols() -> list[str]:
    """Load active equity symbols (cached 10 min)."""
    return _load_available_symbols_impl()
