"""
Metrics and trades database persistence module.

Persists computed strategy metrics and executed trade records into PostgreSQL
using PortfolioMetricRepository and TradeRepository.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Mapping

import pandas as pd
from sqlalchemy.orm import Session

from src.database.repository.portfolio_metric_repository import (
    PortfolioMetricRepository,
)
from src.database.repository.trade_repository import TradeRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


def persist_metrics(
    session: Session,
    strategy_name: str,
    metrics: Mapping[str, float],
    calculated_at: datetime | None = None,
) -> int:
    """
    Persist scalar metrics into portfolio_metrics table.

    Skips NaN and infinite values, logging the count of skipped metrics.

    Args:
        session: Active SQLAlchemy session.
        strategy_name: Name of the strategy.
        metrics: Key-value mapping of metric names and numeric values.
        calculated_at: Timestamp of metric calculation. Defaults to current UTC time.

    Returns:
        Number of records successfully inserted.
    """
    if not metrics:
        return 0

    ts = calculated_at or datetime.now(UTC)
    records = []
    skipped = 0

    for k, v in metrics.items():
        if v is None:
            skipped += 1
            continue
        try:
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                skipped += 1
                continue
            records.append(
                {
                    "strategy_name": strategy_name,
                    "metric": str(k),
                    "value": val,
                    "calculated_at": ts,
                }
            )
        except (TypeError, ValueError):
            skipped += 1

    if skipped > 0:
        logger.info(
            "Skipped %s NaN/infinite/invalid metrics for strategy %s",
            skipped,
            strategy_name,
        )

    if not records:
        return 0

    repo = PortfolioMetricRepository(session)
    return repo.bulk_insert(records)


def persist_trades(
    session: Session,
    trades_df: pd.DataFrame,
) -> int:
    """
    Persist trade records into trades table.

    Args:
        session: Active SQLAlchemy session.
        trades_df: DataFrame containing trade records.

    Returns:
        Number of trade records inserted.
    """
    if trades_df is None or trades_df.empty:
        return 0

    records = []
    for _, row in trades_df.iterrows():
        entry_dt = pd.to_datetime(row["entry_date"])
        exit_dt = pd.to_datetime(row["exit_date"])

        if hasattr(entry_dt, "to_pydatetime"):
            entry_dt = entry_dt.to_pydatetime()
        if hasattr(exit_dt, "to_pydatetime"):
            exit_dt = exit_dt.to_pydatetime()

        pos_size = row.get("position_size", row.get("quantity", 0.0))

        records.append(
            {
                "symbol": str(row["symbol"]),
                "entry_date": entry_dt,
                "exit_date": exit_dt,
                "entry_price": float(row["entry_price"]),
                "exit_price": float(row["exit_price"]),
                "position_size": float(pos_size),
                "fees": float(row.get("fees", 0.0)),
                "slippage_cost": float(row.get("slippage_cost", 0.0)),
                "pnl": float(row["pnl"]),
                "return_pct": float(row["return_pct"]),
            }
        )

    repo = TradeRepository(session)
    return repo.bulk_insert(records)
