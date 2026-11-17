"""
Background backtest job execution manager.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Mapping

from src.backtesting.backtest_config import BacktestConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestJobManager:
    """Manages asynchronous background execution of backtests."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, req: Mapping[str, Any], pipeline: Any) -> str:
        """Submit a backtest task to execute in a worker thread."""
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "run_id": None,
                "error": None,
            }

        def _worker() -> None:
            with self._lock:
                self._jobs[job_id]["status"] = "running"
            try:
                cfg = BacktestConfig(
                    strategy_name=req.get("strategy_name", "strategy"),
                    initial_capital=req.get("initial_capital"),
                    transaction_cost=req.get("transaction_cost"),
                    slippage=req.get("slippage"),
                    max_position_size=req.get("max_position_size"),
                    cash_reserve=req.get("cash_reserve"),
                    allocation_method=req.get("allocation_method"),
                    rebalance_frequency=req.get("rebalance_frequency"),
                    allow_short=req.get("allow_short", False),
                    stop_loss_pct=req.get("stop_loss_pct"),
                    take_profit_pct=req.get("take_profit_pct"),
                    execution_lag=req.get("execution_lag", 1),
                )
                res = pipeline.run(
                    config=cfg,
                    symbols=req.get("symbols"),
                    asset_class=req.get("asset_class", "equity"),
                    timeframe=req.get("timeframe", "daily"),
                    start=req.get("start"),
                    end=req.get("end"),
                    persist=True,
                    persist_db=False,
                )
                with self._lock:
                    self._jobs[job_id]["status"] = "succeeded"
                    self._jobs[job_id]["run_id"] = res.run_id
            except Exception as exc:
                logger.error("Backtest job %s failed: %s", job_id, exc)
                with self._lock:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return job_id

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Fetch current execution status of a job."""
        with self._lock:
            if job_id not in self._jobs:
                raise FileNotFoundError(f"Job {job_id} not found.")
            return dict(self._jobs[job_id])
