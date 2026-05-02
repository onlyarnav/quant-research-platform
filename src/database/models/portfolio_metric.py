"""
Portfolio metric database model.
"""
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base


class PortfolioMetric(Base):
    __tablename__ = "portfolio_metrics"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    strategy_name: Mapped[str] = mapped_column(String(100))
    metric: Mapped[str] = mapped_column(String(100))
    value: Mapped[float] = mapped_column(Float)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_portfolio_metrics_strategy_name", "strategy_name"),
        Index("ix_portfolio_metrics_metric", "metric"),
        Index("ix_portfolio_metrics_calculated_at", "calculated_at"),
    )