"""
Equal risk contribution (risk parity) portfolio allocator.

Solves for asset weights such that each asset contributes an identical amount
to total portfolio return variance using SLSQP optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.backtesting.allocation_interface import AllocationContext
from src.portfolio.base import AllocatorBase
from src.portfolio.constraints import PortfolioConstraints
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskParityAllocator(AllocatorBase):
    """Equal Risk Contribution (ERC) portfolio allocation solver."""

    def __init__(
        self,
        lookback_window: int = 60,
        constraints: PortfolioConstraints | None = None,
        max_iter: int = 200,
    ) -> None:
        """
        Initialize RiskParityAllocator.

        Args:
            lookback_window: Number of trailing price return days.
            constraints: Portfolio concentration and exposure constraints.
            max_iter: Maximum iterations for the SLSQP optimizer.
        """
        super().__init__(constraints=constraints, name="risk_parity")
        if lookback_window < 2:
            raise ValueError(
                f"lookback_window must be at least 2, got {lookback_window}"
            )
        self.lookback_window = lookback_window
        self.max_iter = max_iter

    def _raw_weights(
        self, context: AllocationContext
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Calculate ERC weights via numerical optimization."""
        sig_df = context.signals
        target_sig = -1 if context.side == "short" else 1
        active = sig_df[sig_df["signal"] == target_sig].copy()

        if active.empty:
            return {}, {s: "no active signal" for s in sig_df["symbol"]}

        prices_df = context.prices_history
        as_of = context.as_of_date
        symbols = sorted(active["symbol"].unique())

        excluded: dict[str, str] = {}
        returns_dict: dict[str, pd.Series] = {}

        for sym in symbols:
            sub = (
                prices_df[(prices_df["symbol"] == sym) & (prices_df["date"] <= as_of)]
                .sort_values("date")
                .tail(self.lookback_window + 1)
            )
            if len(sub) < self.lookback_window + 1:
                excluded[sym] = "insufficient history"
                continue
            rets = sub["close"].pct_change().dropna()
            if (
                len(rets) < self.lookback_window
                or rets.std() == 0
                or np.isnan(rets.std())
            ):
                excluded[sym] = "zero or undefined volatility"
                continue
            returns_dict[sym] = rets.reset_index(drop=True)

        eligible = sorted(returns_dict.keys())
        n = len(eligible)

        if n == 0:
            return {}, excluded

        if n == 1:
            return {eligible[0]: 1.0}, excluded

        # Covariance matrix calculation
        ret_matrix = pd.DataFrame(returns_dict)[eligible]
        cov = np.array(ret_matrix.cov(), dtype=float, copy=True)
        # Add tiny regularization to diagonal for numerical stability
        cov += np.eye(n) * 1e-10

        # Initial point: inverse-volatility weights
        vols = np.sqrt(np.diag(cov))
        inv_vols = 1.0 / vols
        w0 = inv_vols / np.sum(inv_vols)

        # Objective: minimize sum_i (w_i * (Sigma @ w)_i - total_var / n)^2
        def objective(w: np.ndarray) -> float:
            sig_w = cov @ w
            port_var = float(w @ sig_w)
            target_risk = port_var / n
            risk_contribs = w * sig_w
            diff = risk_contribs - target_risk
            return float(np.sum(diff**2))

        bounds = [(0.0, 1.0) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        try:
            res = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.max_iter, "ftol": 1e-12},
            )
            if not res.success or np.any(np.isnan(res.x)):
                logger.warning(
                    "Risk parity optimisation failed; falling back to inverse volatility: %s",
                    getattr(res, "message", "unknown"),
                )
                weights = w0
            else:
                weights = res.x / np.sum(res.x)
        except Exception as e:
            logger.warning(
                "Risk parity optimisation failed with exception: %s. Falling back to inverse-vol.",
                e,
            )
            weights = w0

        raw_weights = {sym: float(w) for sym, w in zip(eligible, weights)}
        return raw_weights, excluded
