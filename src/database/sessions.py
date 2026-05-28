"""
Database session management.

This module owns:
- SQLAlchemy engine creation
- SessionLocal factory
- transactional session lifecycle
- optional database initialization helper

Application code should use get_session() instead of creating sessions directly.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings


logger = logging.getLogger(__name__)


# Create the SQLAlchemy engine.
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10,
)


# Create a configured SessionLocal class.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.

    Behavior:
    - opens a new SQLAlchemy session
    - yields it to the caller
    - commits if no exception occurs
    - rolls back if an exception occurs
    - always closes the session

    Usage:
        from src.database.sessions import get_session

        with get_session() as session:
            session.add(some_object)
    """

    session = SessionLocal()
    logger.debug("Database session opened.")

    try:
        yield session
        session.commit()
        logger.debug("Database session committed successfully.")

    except Exception:
        logger.exception("Database session rollback triggered due to an exception.")
        session.rollback()
        raise

    finally:
        session.close()
        logger.debug("Database session closed.")


def init_db() -> None:
    """
    Create all database tables registered in SQLAlchemy metadata.

    This helper is useful for local development and quick experiments.

    Production rule:
    Use Alembic migrations instead of Base.metadata.create_all().
    """

    logger.info("Initializing database tables from SQLAlchemy metadata.")

    from src.database.base import Base

    # Import models so they are registered with Base.metadata.
    from src.database.models import (  # noqa: F401
        Asset,
        Feature,
        PortfolioMetric,
        Price,
        Signal,
        Trade,
    )

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")

    except Exception:
        logger.exception("Database initialization failed.")
        raise