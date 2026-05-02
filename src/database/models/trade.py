"""
Trade database model.
"""
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(50))
    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    position_size: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float)
    slippage_cost: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float)
    return_pct: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        Index("ix_trades_symbol", "symbol"),
        Index("ix_trades_entry_date", "entry_date"),
        Index("ix_trades_symbol_entry_date", "symbol", "entry_date"),
    )