"""
Repository for Price database operations.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.database.models.price import Price
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PriceRepository:
    """Data access layer for Price entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, data: dict) -> Price:
        """
        Insert or update a price record.

        Conflict key:
            (symbol, date)
        """

        stmt = (
            insert(Price)
            .values(**data)
            .on_conflict_do_update(
                index_elements=["symbol", "date"],
                set_={
                    "open": data["open"],
                    "high": data["high"],
                    "low": data["low"],
                    "close": data["close"],
                    "adj_close": data.get("adj_close"),
                    "volume": data.get("volume"),
                    "source": data["source"],
                },
            )
        )

        self._session.execute(stmt)

        price = self._session.scalar(
            select(Price).where(
                Price.symbol == data["symbol"],
                Price.date == data["date"],
            )
        )

        if price is None:
            raise RuntimeError(
                f"Upsert succeeded but fetch failed: "
                f"symbol={data['symbol']} date={data['date']}"
            )

        logger.info(
            "Upserted price symbol=%s date=%s",
            data["symbol"],
            data["date"],
        )

        return price

    def get_by_symbol(
        self,
        symbol: str,
        order_asc: bool = True,
    ) -> list[Price]:
        """
        Fetch all prices for a symbol.

        Parameters
        ----------
        symbol : str
            Asset symbol.

        order_asc : bool
            Sort oldest->newest when True,
            newest->oldest when False.
        """

        stmt = select(Price).where(
            Price.symbol == symbol,
        )

        stmt = stmt.order_by(
            Price.date.asc()
            if order_asc
            else Price.date.desc()
        )

        return list(
            self._session.scalars(stmt)
        )

    def get_by_date_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[Price]:
        """
        Fetch prices within a date range.

        Results are always ordered ascending by date.
        """

        stmt = (
            select(Price)
            .where(
                Price.symbol == symbol,
                Price.date >= start_date,
                Price.date <= end_date,
            )
            .order_by(
                Price.date.asc()
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
        Bulk insert price records.

        Duplicate (symbol, date) rows are ignored.
        """

        if not records:
            return 0

        stmt = (
            insert(Price)
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
            "Bulk inserted %s price records.",
            len(records),
        )

        return len(records)

    def delete_by_symbol(
        self,
        symbol: str,
    ) -> int:
        """
        Delete all prices for a symbol.
        """

        result = self._session.execute(
            delete(Price).where(
                Price.symbol == symbol,
            )
        )

        deleted = result.rowcount or 0

        logger.info(
            "Deleted %s price rows for symbol=%s",
            deleted,
            symbol,
        )

        return deleted