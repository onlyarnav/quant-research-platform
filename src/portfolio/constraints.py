"""
Portfolio constraints specification and projection algorithm.

Enforces maximum position sizes, minimum position weights, sector and asset class
concentration limits, and cash reserves using an iterative redistribution solver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ConstraintResult:
    """Projected weights, residual cash fraction, and binding constraints list."""

    weights: dict[str, float]
    cash_weight: float
    binding: list[str]


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    """
    Portfolio constraints definition container.

    Attributes:
        max_position_size: Maximum capital fraction allowed in any single asset.
        min_position_weight: Minimum threshold weight below which positions are dropped.
        cash_reserve: Mandatory cash buffer held uninvested.
        sector_map: Mapping of symbol -> sector name.
        max_sector_weight: Maximum total allocation per sector.
        asset_class_map: Mapping of symbol -> asset class name.
        max_asset_class_weight: Maximum total allocation per asset class.
    """

    max_position_size: float = field(
        default_factory=lambda: float(settings.MAX_POSITION_SIZE)
    )
    min_position_weight: float = 0.0
    cash_reserve: float = 0.0
    sector_map: Mapping[str, str] = field(default_factory=dict)
    max_sector_weight: float | None = None
    asset_class_map: Mapping[str, str] = field(default_factory=dict)
    max_asset_class_weight: float | None = None

    def validate(self) -> None:
        """
        Validate constraint parameters.

        Raises:
            ValueError: If any constraint threshold is outside allowable ranges.
        """
        if not (0.0 < self.max_position_size <= 1.0):
            raise ValueError(
                f"max_position_size must be in (0, 1], got {self.max_position_size}"
            )
        if not (0.0 <= self.min_position_weight < 1.0):
            raise ValueError(
                f"min_position_weight must be in [0, 1), got {self.min_position_weight}"
            )
        if not (0.0 <= self.cash_reserve < 1.0):
            raise ValueError(f"cash_reserve must be in [0, 1), got {self.cash_reserve}")
        if self.min_position_weight > self.max_position_size:
            raise ValueError(
                f"min_position_weight ({self.min_position_weight}) cannot exceed "
                f"max_position_size ({self.max_position_size})"
            )
        if self.max_sector_weight is not None and not (
            0.0 < self.max_sector_weight <= 1.0
        ):
            raise ValueError(
                f"max_sector_weight must be in (0, 1], got {self.max_sector_weight}"
            )
        if self.max_asset_class_weight is not None and not (
            0.0 < self.max_asset_class_weight <= 1.0
        ):
            raise ValueError(
                f"max_asset_class_weight must be in (0, 1], got {self.max_asset_class_weight}"
            )

    def apply(self, weights: Mapping[str, float]) -> ConstraintResult:
        """
        Project candidate weights onto the feasible constrained polytope.

        Args:
            weights: Mapping of symbol -> raw desired weight (non-negative).

        Returns:
            ConstraintResult with compliant weights, cash weight, and binding constraints.
        """
        self.validate()
        if not weights:
            return ConstraintResult(weights={}, cash_weight=1.0, binding=[])

        symbols = sorted(weights.keys())
        w = {s: max(0.0, float(weights[s])) for s in symbols}
        tot = sum(w.values())
        if tot <= 1e-12:
            return ConstraintResult(weights={}, cash_weight=1.0, binding=[])

        investable = 1.0 - self.cash_reserve
        scale = investable / tot
        w = {s: val * scale for s, val in w.items()}

        binding_set: set[str] = set()

        for _ in range(100):
            excess = 0.0
            # 1. Position cap
            for s in symbols:
                if w[s] > self.max_position_size + 1e-9:
                    excess += w[s] - self.max_position_size
                    w[s] = self.max_position_size
                    binding_set.add(f"max_position_size:{s}")

            # 2. Sector cap
            if self.max_sector_weight is not None:
                sectors: dict[str, list[str]] = {}
                for s in symbols:
                    sec = self.sector_map.get(s, "UNKNOWN")
                    sectors.setdefault(sec, []).append(s)

                for sec, sec_syms in sectors.items():
                    if sec == "UNKNOWN":
                        continue
                    sec_tot = sum(w[s] for s in sec_syms)
                    if sec_tot > self.max_sector_weight + 1e-9:
                        factor = self.max_sector_weight / sec_tot
                        sec_excess = sec_tot - self.max_sector_weight
                        excess += sec_excess
                        for s in sec_syms:
                            w[s] *= factor
                        binding_set.add(f"max_sector_weight:{sec}")

            # 3. Asset class cap
            if self.max_asset_class_weight is not None:
                ac_groups: dict[str, list[str]] = {}
                for s in symbols:
                    ac = self.asset_class_map.get(s, "UNKNOWN")
                    ac_groups.setdefault(ac, []).append(s)

                for ac, ac_syms in ac_groups.items():
                    if ac == "UNKNOWN":
                        continue
                    ac_tot = sum(w[s] for s in ac_syms)
                    if ac_tot > self.max_asset_class_weight + 1e-9:
                        factor = self.max_asset_class_weight / ac_tot
                        ac_excess = ac_tot - self.max_asset_class_weight
                        excess += ac_excess
                        for s in ac_syms:
                            w[s] *= factor
                        binding_set.add(f"max_asset_class_weight:{ac}")

            if excess < 1e-9:
                break

            # Find symbols with headroom
            headroom: dict[str, float] = {}
            for s in symbols:
                room = self.max_position_size - w[s]
                if room > 1e-9:
                    headroom[s] = room

            tot_room = sum(headroom.values())
            if tot_room < 1e-9:
                # Infeasible budget (e.g. fewer symbols than needed)
                break

            redist_amount = min(excess, tot_room)
            for s, r in headroom.items():
                w[s] += redist_amount * (r / tot_room)

        # 4. Filter minimum position weight
        if self.min_position_weight > 0.0:
            dropped = False
            for s in list(symbols):
                if 0.0 < w[s] < self.min_position_weight:
                    w[s] = 0.0
                    dropped = True

            if dropped:
                active_syms = [s for s in symbols if w[s] > 0.0]
                act_sum = sum(w[s] for s in active_syms)
                if act_sum > 0.0 and act_sum < investable:
                    rem = investable - act_sum
                    for s in active_syms:
                        w[s] += rem * (w[s] / act_sum)
                        if w[s] > self.max_position_size:
                            w[s] = self.max_position_size
                            binding_set.add(f"max_position_size:{s}")

        final_weights = {s: round(val, 9) for s, val in w.items() if val > 1e-9}
        total_allocated = sum(final_weights.values())
        cash_weight = max(0.0, 1.0 - total_allocated)

        return ConstraintResult(
            weights=final_weights,
            cash_weight=cash_weight,
            binding=sorted(binding_set),
        )
