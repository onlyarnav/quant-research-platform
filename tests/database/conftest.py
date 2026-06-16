"""
Pytest fixtures for database testing.

Creates tables once per test session and provides
transaction-isolated sessions for every test.
"""

from __future__ import annotations

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from config.settings import settings

from src.database.repository import (
    AssetRepository,
    FeatureRepository,
    PortfolioMetricRepository,
    PriceRepository,
    SignalRepository,
    TradeRepository,
)


@pytest.fixture()
def engine():
    _engine = create_engine(settings.DATABASE_URL)
    yield _engine
    _engine.dispose()


@pytest.fixture()
def session(engine):
    """
    Provide a transaction-isolated session.

    Each test runs inside a transaction that
    is rolled back after completion.
    """

    connection = engine.connect()
    transaction = connection.begin()

    session = Session(connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def asset_repo(session: Session) -> AssetRepository:
    """Asset repository fixture."""
    return AssetRepository(session)


@pytest.fixture
def price_repo(session: Session) -> PriceRepository:
    """Price repository fixture."""
    return PriceRepository(session)


@pytest.fixture
def feature_repo(session: Session) -> FeatureRepository:
    """Feature repository fixture."""
    return FeatureRepository(session)


@pytest.fixture
def signal_repo(session: Session) -> SignalRepository:
    """Signal repository fixture."""
    return SignalRepository(session)


@pytest.fixture
def trade_repo(session: Session) -> TradeRepository:
    """Trade repository fixture."""
    return TradeRepository(session)


@pytest.fixture
def portfolio_repo(
    session: Session,
) -> PortfolioMetricRepository:
    """Portfolio metric repository fixture."""
    return PortfolioMetricRepository(session)