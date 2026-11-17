"""
Portfolio construction and capital allocation models layer.

Provides equal-weight, volatility-weighted, risk parity, signal-weighted,
and mean-variance allocators, constrained by position, sector, and asset class rules.
"""

from __future__ import annotations

from src.portfolio.adapters import LegacyAllocatorAdapter
from src.portfolio.allocator_factory import (
    ALLOCATOR_NAMES,
    available_allocators,
    create_allocator,
    register_allocator,
)
from src.portfolio.base import AllocatorBase
from src.portfolio.constraints import ConstraintResult, PortfolioConstraints
from src.portfolio.equal_weight import EqualWeightAllocator
from src.portfolio.mean_variance import MeanVarianceAllocator
from src.portfolio.risk_parity import RiskParityAllocator
from src.portfolio.signal_weighted import SignalWeightedAllocator
from src.portfolio.volatility_weighted import VolatilityWeightedAllocator

__all__ = [
    "ALLOCATOR_NAMES",
    "AllocatorBase",
    "ConstraintResult",
    "EqualWeightAllocator",
    "LegacyAllocatorAdapter",
    "MeanVarianceAllocator",
    "PortfolioConstraints",
    "RiskParityAllocator",
    "SignalWeightedAllocator",
    "VolatilityWeightedAllocator",
    "available_allocators",
    "create_allocator",
    "register_allocator",
]
