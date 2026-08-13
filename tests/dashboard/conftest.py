"""Pytest fixtures for dashboard tests using in-memory database isolation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.database.base import Base
from src.database.models import (  # noqa: F401
    Asset,
    Feature,
    PortfolioMetric,
    Price,
    Signal,
    Trade,
)
from src.database.repository import (
    AssetRepository,
    PortfolioMetricRepository,
    SignalRepository,
    TradeRepository,
)


@pytest.fixture
def db_session(mocker) -> Generator[Session, None, None]:
    """Provide an in-memory SQLite session and patch data_loader.get_session."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            try:
                table.create(conn, checkfirst=True)
            except Exception:
                pass

    session = Session(engine)

    @contextmanager
    def _mock_get_session() -> Generator[Session, None, None]:
        yield session

    mocker.patch("src.dashboard.data_loader.get_session", _mock_get_session)
    yield session
    session.close()


@pytest.fixture
def trade_repo(db_session: Session) -> TradeRepository:
    """Trade repository fixture."""
    return TradeRepository(db_session)


@pytest.fixture
def portfolio_repo(db_session: Session) -> PortfolioMetricRepository:
    """Portfolio metric repository fixture."""
    return PortfolioMetricRepository(db_session)


@pytest.fixture
def signal_repo(db_session: Session) -> SignalRepository:
    """Signal repository fixture."""
    return SignalRepository(db_session)


@pytest.fixture
def asset_repo(db_session: Session) -> AssetRepository:
    """Asset repository fixture."""
    return AssetRepository(db_session)
