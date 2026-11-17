"""
Factory and extensible registry for quantitative portfolio allocators.

Provides available_allocators, create_allocator, and register_allocator for
instantiating and wrapping allocators under the unified Allocator protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.backtesting.allocation_interface import Allocator
from src.backtesting.backtest_config import AllocationMethod
from src.portfolio.adapters import LegacyAllocatorAdapter
from src.portfolio.constraints import PortfolioConstraints
from src.portfolio.equal_weight import EqualWeightAllocator
from src.portfolio.mean_variance import MeanVarianceAllocator
from src.portfolio.risk_parity import RiskParityAllocator
from src.portfolio.signal_weighted import SignalWeightedAllocator
from src.portfolio.volatility_weighted import VolatilityWeightedAllocator

ALLOCATOR_NAMES = (
    "equal_weight",
    "signal_weighted",
    "volatility_weighted",
    "risk_parity",
    "mean_variance",
)

_REGISTRY: dict[str, Callable[..., Allocator]] = {}


def _init_default_registry() -> None:
    """Initialize built-in platform allocator factory functions."""
    _REGISTRY["equal_weight"] = lambda constraints=None, **kw: (
        LegacyAllocatorAdapter(
            "equal_weight",
            EqualWeightAllocator(),
            needs_prices=False,
            constraints=constraints,
        )
    )
    _REGISTRY["signal_weighted"] = (
        lambda constraints=None, lookback_window=None, **kw: (
            SignalWeightedAllocator(constraints=constraints, **kw)
        )
    )
    _REGISTRY["volatility_weighted"] = (
        lambda constraints=None, lookback_window=None, **kw: (
            LegacyAllocatorAdapter(
                "volatility_weighted",
                VolatilityWeightedAllocator(lookback_window=lookback_window or 20),
                needs_prices=True,
                constraints=constraints,
            )
        )
    )
    _REGISTRY["risk_parity"] = lambda constraints=None, lookback_window=None, **kw: (
        RiskParityAllocator(
            lookback_window=lookback_window or 60,
            constraints=constraints,
            **kw,
        )
    )
    _REGISTRY["mean_variance"] = lambda constraints=None, lookback_window=None, **kw: (
        LegacyAllocatorAdapter(
            "mean_variance",
            MeanVarianceAllocator(lookback_window=lookback_window or 60),
            needs_prices=True,
            constraints=constraints,
        )
    )


_init_default_registry()


def available_allocators() -> list[str]:
    """Return list of all registered allocator method names."""
    return sorted(_REGISTRY.keys())


def register_allocator(name: str, factory: Callable[..., Allocator]) -> None:
    """
    Register a custom allocator factory.

    Args:
        name: Unique method identifier string.
        factory: Callable taking constraints, lookback_window, **kwargs and returning Allocator.

    Raises:
        ValueError: If allocator name is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Allocator '{name}' is already registered.")
    _REGISTRY[name] = factory


def create_allocator(
    method: str | AllocationMethod,
    constraints: PortfolioConstraints | None = None,
    lookback_window: int | None = None,
    **kwargs: Any,
) -> Allocator:
    """
    Instantiate an allocator matching the specified allocation method.

    Args:
        method: Allocation method name or AllocationMethod enum.
        constraints: Optional portfolio concentration constraints.
        lookback_window: Historical window parameter for risk/volatility models.
        kwargs: Additional strategy parameters passed to constructor.

    Returns:
        Instantiated Allocator satisfying the protocol.

    Raises:
        ValueError: If method is 'fixed_notional' or not found in registry.
    """
    name = method.value if isinstance(method, AllocationMethod) else str(method)
    if name == "fixed_notional":
        raise ValueError("fixed_notional does not use an allocator")

    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown allocation method '{name}'. Available: {available_allocators()}"
        )

    return _REGISTRY[name](
        constraints=constraints,
        lookback_window=lookback_window,
        **kwargs,
    )
