"""
Repository for Trade database operations.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from src.database.models.trade import Trade
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradeRepository:
    """Data access layer for Trade entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def insert(
        self,
        data: dict,
    ) -> Trade:
        """
        Insert a new trade record.

        Trades are append-only records and do not support upserts.
        """

        trade = Trade(**data)

        self._session.add(trade)
        self._session.flush()

        logger.info(
            "Inserted trade id=%s symbol=%s",
            trade.id,
            trade.symbol,
        )

        return trade

    def get_by_symbol(
        self,
        symbol: str,
    ) -> list[Trade]:
        """
        Fetch all trades for a symbol.

        Ordered by entry_date ascending.
        """

        stmt = (
            select(Trade)
            .where(
                Trade.symbol == symbol,
            )
            .order_by(
                Trade.entry_date.asc(),
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
    ) -> list[Trade]:
        """
        Fetch trades within an entry date range.

        Filters on entry_date because it is indexed.
        """

        stmt = (
            select(Trade)
            .where(
                Trade.symbol == symbol,
                Trade.entry_date >= start_date,
                Trade.entry_date <= end_date,
            )
            .order_by(
                Trade.entry_date.asc(),
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
        Bulk insert trade records.
        """

        if not records:
            return 0

        self._session.execute(
            insert(Trade),
            records,
        )

        logger.info(
            "Bulk inserted %s trade records.",
            len(records),
        )

        return len(records)

    def delete_by_symbol(
        self,
        symbol: str,
    ) -> int:
        """
        Delete all trades for a symbol.
        """

        result = self._session.execute(
            delete(Trade).where(
                Trade.symbol == symbol,
            )
        )

        deleted = result.rowcount or 0

        logger.info(
            "Deleted %s trade rows for symbol=%s",
            deleted,
            symbol,
        )

        return deleted