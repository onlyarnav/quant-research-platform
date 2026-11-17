"""
Position book module for average-cost accounting and portfolio exposure tracking.

Provides BookPosition and PositionBook to support long and short positions,
multiple fills (averaging up/down), partial lot reductions, and exposure metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BookPosition:
    """
    Open position tracked using average-cost basis.

    Attributes:
        symbol: Ticker symbol.
        side: +1 for long, -1 for short.
        quantity: Current position magnitude (always positive).
        avg_entry_price: Volume-weighted executed price basis.
        entry_date: Timestamp of the initial fill.
        entry_fees_paid: Cumulative entry fees attributed to remaining quantity.
        entry_slippage_paid: Cumulative entry slippage attributed to remaining quantity.
    """

    symbol: str
    side: int
    quantity: float
    avg_entry_price: float
    entry_date: pd.Timestamp
    entry_fees_paid: float
    entry_slippage_paid: float


@dataclass(slots=True)
class ReducedLot:
    """
    Metadata of a partially or fully closed lot for trade accounting.

    Attributes:
        entry_date: Original entry date of the position.
        avg_entry_price: Weighted average entry price basis.
        quantity: Quantity closed.
        entry_fee_share: Proportion of entry fees allocated to this closed lot.
        entry_slippage_share: Proportion of entry slippage allocated to this closed lot.
    """

    entry_date: pd.Timestamp
    avg_entry_price: float
    quantity: float
    entry_fee_share: float
    entry_slippage_share: float


class PositionBook:
    """Multi-asset position book supporting partial fills and average-cost accounting."""

    def __init__(self) -> None:
        """Initialize an empty PositionBook."""
        self._positions: dict[str, BookPosition] = {}

    def has(self, symbol: str) -> bool:
        """Check whether a position currently exists for symbol."""
        return symbol in self._positions

    def get(self, symbol: str) -> BookPosition | None:
        """Retrieve the BookPosition for symbol, or None if not held."""
        return self._positions.get(symbol)

    def symbols(self, side: int | None = None) -> list[str]:
        """
        Return list of open symbols, optionally filtered by side.

        Args:
            side: Optional filter (+1 for long, -1 for short, None for all).

        Returns:
            List of symbol strings.
        """
        if side is None:
            return list(self._positions.keys())
        return [sym for sym, pos in self._positions.items() if pos.side == side]

    def add(
        self,
        symbol: str,
        side: int,
        date: pd.Timestamp,
        quantity: float,
        executed_price: float,
        fee: float,
        slippage_cost: float,
    ) -> BookPosition:
        """
        Open a new position or average-up/down into an existing position.

        Args:
            symbol: Asset ticker.
            side: Position side (+1 long, -1 short).
            date: Execution timestamp.
            quantity: Filled quantity magnitude (positive).
            executed_price: Execution price basis.
            fee: Fee paid on entry leg.
            slippage_cost: Slippage cost incurred on entry leg.

        Returns:
            Updated BookPosition.

        Raises:
            ValueError: If attempting to add to a position of the opposite side.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity to add must be positive, got {quantity}")

        if symbol not in self._positions:
            pos = BookPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                avg_entry_price=executed_price,
                entry_date=date,
                entry_fees_paid=fee,
                entry_slippage_paid=slippage_cost,
            )
            self._positions[symbol] = pos
            return pos

        existing = self._positions[symbol]
        if existing.side != side:
            raise ValueError("Cannot add to a position of the opposite side")

        new_qty = existing.quantity + quantity
        new_avg = (
            existing.quantity * existing.avg_entry_price + quantity * executed_price
        ) / new_qty

        existing.quantity = new_qty
        existing.avg_entry_price = new_avg
        existing.entry_fees_paid += fee
        existing.entry_slippage_paid += slippage_cost
        return existing

    def reduce(self, symbol: str, quantity: float) -> ReducedLot:
        """
        Partially or fully close an existing position.

        Args:
            symbol: Asset ticker.
            quantity: Quantity to reduce.

        Returns:
            ReducedLot detailing the closed lot's basis and allocated entry costs.

        Raises:
            ValueError: If symbol has no open position or reduction exceeds position size.
        """
        if symbol not in self._positions:
            raise ValueError(f"No open position for {symbol}")

        pos = self._positions[symbol]
        if quantity > pos.quantity + 1e-9:
            raise ValueError(
                f"Cannot reduce {quantity} from position of size {pos.quantity}"
            )

        qty_to_reduce = min(quantity, pos.quantity)
        frac = qty_to_reduce / pos.quantity

        fee_share = frac * pos.entry_fees_paid
        slippage_share = frac * pos.entry_slippage_paid

        lot = ReducedLot(
            entry_date=pos.entry_date,
            avg_entry_price=pos.avg_entry_price,
            quantity=qty_to_reduce,
            entry_fee_share=fee_share,
            entry_slippage_share=slippage_share,
        )

        remaining_qty = pos.quantity - qty_to_reduce
        if remaining_qty < 1e-9:
            del self._positions[symbol]
        else:
            pos.quantity = remaining_qty
            pos.entry_fees_paid -= fee_share
            pos.entry_slippage_paid -= slippage_share

        return lot

    def market_value(
        self, prices: Mapping[str, float], side: int | None = None
    ) -> float:
        """
        Calculate gross market value of positions.

        Args:
            prices: Symbol -> current price mapping.
            side: Filter (+1 long, -1 short, None for gross sum of both).

        Returns:
            Gross market value magnitude.
        """
        total = 0.0
        for sym, pos in self._positions.items():
            if side is not None and pos.side != side:
                continue
            px = prices.get(sym, pos.avg_entry_price)
            total += pos.quantity * px
        return total

    def unrealized_pnl(self, prices: Mapping[str, float]) -> float:
        """
        Calculate total unrealized PnL across all open positions.

        Args:
            prices: Symbol -> mark-to-market price mapping.

        Returns:
            Total net unrealized PnL.
        """
        total_pnl = 0.0
        for sym, pos in self._positions.items():
            px = prices.get(sym, pos.avg_entry_price)
            if pos.side == 1:
                total_pnl += (px - pos.avg_entry_price) * pos.quantity
            else:
                total_pnl += (pos.avg_entry_price - px) * pos.quantity
        return total_pnl

    def exposures(self, prices: Mapping[str, float], equity: float) -> dict[str, float]:
        """
        Compute normalized portfolio exposures.

        Args:
            prices: Symbol -> mark-to-market price mapping.
            equity: Total portfolio equity value.

        Returns:
            Dict containing long, short, gross, and net exposures as fractions of equity.
        """
        if equity <= 0:
            return {"long": 0.0, "short": 0.0, "gross": 0.0, "net": 0.0}

        long_val = self.market_value(prices, side=1)
        short_val = self.market_value(prices, side=-1)
        gross_val = long_val + short_val
        net_val = long_val - short_val

        return {
            "long": long_val / equity,
            "short": short_val / equity,
            "gross": gross_val / equity,
            "net": net_val / equity,
        }
