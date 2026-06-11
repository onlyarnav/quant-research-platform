"""
Repository for Feature metadata database operations.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.database.models.feature import Feature
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureRepository:
    """Data access layer for Feature entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, data: dict) -> Feature:
        """
        Insert or update a feature metadata record.

        Conflict key:
            (symbol, date)
        """

        stmt = (
            insert(Feature)
            .values(**data)
            .on_conflict_do_update(
                index_elements=["symbol", "date"],
                set_={
                    "feature_set_version": data[
                        "feature_set_version"
                    ],
                },
            )
        )

        self._session.execute(stmt)

        feature = self._session.scalar(
            select(Feature).where(
                Feature.symbol == data["symbol"],
                Feature.date == data["date"],
            )
        )

        if feature is None:
            raise RuntimeError(
                f"Failed to fetch feature metadata after upsert: "
                f"{data['symbol']} {data['date']}"
            )

        logger.info(
            "Upserted feature metadata symbol=%s date=%s version=%s",
            data["symbol"],
            data["date"],
            data["feature_set_version"],
        )

        return feature

    def get_by_symbol(
        self,
        symbol: str,
    ) -> list[Feature]:
        """
        Fetch all feature metadata records for a symbol.

        Ordered by date ascending.
        """

        stmt = (
            select(Feature)
            .where(
                Feature.symbol == symbol,
            )
            .order_by(
                Feature.date.asc(),
            )
        )

        return list(
            self._session.scalars(stmt)
        )

    def get_by_date_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[Feature]:
        """
        Fetch feature metadata records within a date range.

        Inclusive on both ends.
        Ordered by date ascending.
        """

        stmt = (
            select(Feature)
            .where(
                Feature.symbol == symbol,
                Feature.date >= start_date,
                Feature.date <= end_date,
            )
            .order_by(
                Feature.date.asc(),
            )
        )

        return list(
            self._session.scalars(stmt)
        )

    def get_by_feature_set_version(
        self,
        version: str,
    ) -> list[Feature]:
        """
        Fetch all records for a feature set version.

        Ordered by symbol then date.
        """

        stmt = (
            select(Feature)
            .where(
                Feature.feature_set_version == version,
            )
            .order_by(
                Feature.symbol.asc(),
                Feature.date.asc(),
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
        Bulk insert feature metadata.

        Duplicate (symbol, date) rows are ignored.
        """

        if not records:
            return 0

        stmt = (
            insert(Feature)
            .on_conflict_do_nothing(
                index_elements=[
                    "symbol",
                    "date",
                ]
            )
        )

        self._session.execute(
            stmt,
            records,
        )

        logger.info(
            "Bulk inserted %s feature metadata records.",
            len(records),
        )

        return len(records)

    def delete_by_symbol(
        self,
        symbol: str,
    ) -> int:
        """
        Delete all feature metadata for a symbol.
        """

        result = self._session.execute(
            delete(Feature).where(
                Feature.symbol == symbol,
            )
        )

        deleted = result.rowcount or 0

        logger.info(
            "Deleted %s feature metadata rows for symbol=%s",
            deleted,
            symbol,
        )

        return deleted