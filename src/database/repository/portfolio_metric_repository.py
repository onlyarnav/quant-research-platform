"""
Repository for PortfolioMetric database operations.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from src.database.models.portfolio_metric import PortfolioMetric
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioMetricRepository:
    """Data access layer for PortfolioMetric entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def insert(
        self,
        data: dict,
    ) -> PortfolioMetric:
        """
        Insert a new portfolio metric record.
        """

        metric = PortfolioMetric(**data)

        self._session.add(metric)
        self._session.flush()

        logger.info(
            "Inserted portfolio metric id=%s strategy=%s metric=%s",
            metric.id,
            metric.strategy_name,
            metric.metric,
        )

        return metric

    def get_by_strategy(
        self,
        strategy_name: str,
    ) -> list[PortfolioMetric]:
        """
        Fetch all metrics for a strategy.
        """

        stmt = (
            select(PortfolioMetric)
            .where(
                PortfolioMetric.strategy_name == strategy_name,
            )
            .order_by(
                PortfolioMetric.calculated_at.asc(),
            )
        )

        return list(
            self._session.scalars(stmt)
        )

    def get_by_date_range(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[PortfolioMetric]:
        """
        Fetch metrics within a calculation date range.
        """

        stmt = (
            select(PortfolioMetric)
            .where(
                PortfolioMetric.strategy_name == strategy_name,
                PortfolioMetric.calculated_at >= start_date,
                PortfolioMetric.calculated_at <= end_date,
            )
            .order_by(
                PortfolioMetric.calculated_at.asc(),
            )
        )

        return list(
            self._session.scalars(stmt)
        )

    def get_by_metric(
        self,
        metric: str,
    ) -> list[PortfolioMetric]:
        """
        Fetch a metric across all strategies.

        Useful for comparing strategy performance
        on a specific metric.
        """

        stmt = (
            select(PortfolioMetric)
            .where(
                PortfolioMetric.metric == metric,
            )
            .order_by(
                PortfolioMetric.strategy_name.asc(),
                PortfolioMetric.calculated_at.asc(),
            )
        )

        return list(
            self._session.scalars(stmt)
        )

    def bulk_insert(
        self,
        records: list[dict],
    ) -> int:
        """
        Bulk insert portfolio metrics.
        """

        if not records:
            return 0

        self._session.execute(
            insert(PortfolioMetric),
            records,
        )

        logger.info(
            "Bulk inserted %s portfolio metric records.",
            len(records),
        )

        return len(records)

    def delete_by_strategy(
        self,
        strategy_name: str,
    ) -> int:
        """
        Delete all metrics for a strategy.
        """

        result = self._session.execute(
            delete(PortfolioMetric).where(
                PortfolioMetric.strategy_name == strategy_name,
            )
        )

        deleted = result.rowcount or 0

        logger.info(
            "Deleted %s portfolio metric rows for strategy=%s",
            deleted,
            strategy_name,
        )

        return deleted