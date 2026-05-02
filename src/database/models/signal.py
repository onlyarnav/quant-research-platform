"""
Signal database model.
"""
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(50))
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    signal: Mapped[int] = mapped_column(Integer)
    predicted_return: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(100))

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_signals_symbol_date"),
        Index("ix_signals_symbol", "symbol"),
        Index("ix_signals_date", "date"),
        Index("ix_signals_model_version", "model_version"),
    )