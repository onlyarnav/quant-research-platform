"""
Repository for Asset database operations.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.utils.logger import get_logger
from src.database.models.asset import Asset

logger = get_logger(__name__)

class AssetRepository:
    """Data access layer for Asset entities."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, data: dict) -> Asset:
        """
        Insert or update an asset using (symbol, exchange) conflict key.
        """

        stmt = (
            insert(Asset)
            .values(**data)
            .on_conflict_do_update(
                index_elements=["symbol", "exchange"],
                set_={
                    "name": data["name"],
                    "asset_class": data["asset_class"],
                    "currency": data["currency"],
                    "sector": data.get("sector"),
                    "is_active": data.get("is_active", True),
                    "updated_at": datetime.now(UTC),
                },
            )
        )

        self._session.execute(stmt)

        asset = self._session.scalar(
            select(Asset).where(
                Asset.symbol == data["symbol"],
                Asset.exchange == data["exchange"],
            )
        )

        logger.info(
            "Upserted asset symbol=%s exchange=%s",
            data["symbol"],
            data["exchange"],
        )

        return asset

    def get_by_symbol(self, symbol: str) -> list[Asset]:
        """
        Fetch all assets for a symbol.
        """

        stmt = select(Asset).where(
            Asset.symbol == symbol,
        )

        return list(self._session.scalars(stmt))

    def get_by_asset_class(
        self,
        asset_class: str,
        active_only: bool = True,
    ) -> list[Asset]:
        """
        Fetch assets by asset class.
        """

        stmt = select(Asset).where(
            Asset.asset_class == asset_class,
        )

        if active_only:
            stmt = stmt.where(
                Asset.is_active.is_(True),
            )

        return list(
            self._session.scalars(
                stmt.order_by(Asset.symbol)
            )
        )

    def bulk_insert(
        self,
        records: list[dict],
    ) -> int:
        """
        Bulk insert assets.
        """

        if not records:
            return 0

        self._session.execute(
            insert(Asset),
            records,
        )

        logger.info(
            "Inserted %s asset records.",
            len(records),
        )

        return len(records)

    def delete_by_symbol(
        self,
        symbol: str,
    ) -> int:
        """
        Delete all rows matching symbol.
        """

        result = self._session.execute(
            delete(Asset).where(
                Asset.symbol == symbol,
            )
        )

        deleted = result.rowcount or 0

        logger.info(
            "Deleted %s asset rows for symbol=%s",
            deleted,
            symbol,
        )

        return deleted