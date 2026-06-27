"""
Position manager for tracking open positions and enforcing
capital allocation constraints during portfolio simulation.

This module owns position state only — it does not execute trades
or calculate P&L. It answers whether a new position can be opened
given current cash and exposure constraints, and tracks what is
currently open.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class OpenPosition:
    """A currently open position held by the portfolio."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    position_size: float
    capital_allocated: float


class PositionManager:
    """
    Tracks open positions and enforces capital allocation constraints.

    Cash and open positions are the only mutable state owned by this
    class. All other attributes are set once at construction and
    treated as read-only for the lifetime of the instance.
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        max_position_size: float | None = None,
    ) -> None:
        """
        Args:
            initial_capital: Total capital available to the portfolio.
                Defaults to settings.INITIAL_CAPITAL.
            max_position_size: Maximum fraction of total capital that
                may be allocated to a single position. Defaults to
                settings.MAX_POSITION_SIZE.
        """
        self.initial_capital = (
            initial_capital if initial_capital is not None else settings.INITIAL_CAPITAL
        )
        self.max_position_size = (
            max_position_size if max_position_size is not None else settings.MAX_POSITION_SIZE
        )

        self.cash = self.initial_capital
        self._open_positions: dict[str, OpenPosition] = {}

    def can_open_position(self, symbol: str) -> bool:
        """
        Check whether a new position can be opened for symbol.

        Returns False if:
            - a position is already open for this symbol
            - available cash is below the capital required for a
              max_position_size-sized allocation
        """
        if symbol in self._open_positions:
            return False

        required_capital = self.initial_capital * self.max_position_size
        return self.cash >= required_capital

    def open_position(
        self,
        symbol: str,
        entry_date: pd.Timestamp,
        entry_price: float,
    ) -> OpenPosition:
        """
        Open a new position for symbol, deducting allocated capital
        from cash.

        Args:
            symbol: Trading symbol.
            entry_date: Date the position is opened.
            entry_price: Execution price used to size the position.

        Returns:
            The newly created OpenPosition.

        Raises:
            ValueError: If a position is already open for symbol, or
                if entry_price is not positive.
            RuntimeError: If insufficient cash is available. Callers
                should check can_open_position() first to avoid this.
        """
        if symbol in self._open_positions:
            raise ValueError(f"Position already open for symbol={symbol}.")

        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        capital_allocated = self.initial_capital * self.max_position_size

        if capital_allocated > self.cash:
            raise RuntimeError(
                f"Insufficient cash to open position for symbol={symbol}: "
                f"required={capital_allocated}, available={self.cash}. "
                f"Call can_open_position() before open_position()."
            )

        position_size = capital_allocated / entry_price

        position = OpenPosition(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price,
            position_size=position_size,
            capital_allocated=capital_allocated,
        )

        self._open_positions[symbol] = position
        self.cash -= capital_allocated

        logger.info(
            "Opened position symbol=%s capital=%s cash_remaining=%s",
            symbol,
            capital_allocated,
            self.cash,
        )

        return position

    def close_position(self, symbol: str, exit_price: float) -> OpenPosition:
        """
        Close an open position for symbol, returning capital plus/minus
        P&L to cash.

        Args:
            symbol: Trading symbol.
            exit_price: Execution price used to compute proceeds.

        Returns:
            The OpenPosition that was closed, for the caller to use
            when building a Trade record.

        Raises:
            ValueError: If no position is open for symbol, or
                exit_price is not positive.
        """
        if symbol not in self._open_positions:
            raise ValueError(f"No open position for symbol={symbol} to close.")

        if exit_price <= 0:
            raise ValueError(f"exit_price must be positive, got {exit_price}")

        position = self._open_positions.pop(symbol)

        proceeds = position.position_size * exit_price
        self.cash += proceeds

        logger.info(
            "Closed position symbol=%s proceeds=%s cash_after=%s",
            symbol,
            proceeds,
            self.cash,
        )

        return position

    def get_open_positions(self) -> dict[str, OpenPosition]:
        """Return a shallow copy of currently open positions."""
        return dict(self._open_positions)

    def total_invested_capital(self) -> float:
        """Sum of capital currently allocated across all open positions."""
        return sum(p.capital_allocated for p in self._open_positions.values())