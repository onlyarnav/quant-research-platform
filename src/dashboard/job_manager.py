"""
Background backtest job execution manager with disk persistence and concurrency limits.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from config.settings import settings
from src.backtesting.backtest_config import BacktestConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ACTIVE_THREADS: dict[str, threading.Thread] = {}


class BacktestJobManager:
    """Manages asynchronous background execution of backtests with disk persistence."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or (settings.BACKTEST_RESULTS_PATH / "_jobs"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._recover_stale_jobs()

    def _job_file(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def _atomic_write_job(self, job_data: dict[str, Any]) -> None:
        file_path = self._job_file(job_data["job_id"])
        with tempfile.NamedTemporaryFile(
            "w", dir=self.root, delete=False, encoding="utf-8"
        ) as tf:
            json.dump(job_data, tf, indent=2)
            temp_name = tf.name
        os.replace(temp_name, file_path)

    def _recover_stale_jobs(self) -> None:
        """Mark running jobs from prior process lifecycles as failed."""
        try:
            for f in self.root.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if data.get("status") in {"running", "queued"}:
                        jid = data.get("job_id")
                        t = _ACTIVE_THREADS.get(jid)
                        if t and t.is_alive():
                            continue
                        try:
                            created = datetime.fromisoformat(data.get("created_at", ""))
                            if (datetime.now(timezone.utc) - created).total_seconds() < 300:
                                continue
                        except Exception:
                            pass
                        data["status"] = "failed"
                        data["error"] = "Job interrupted by process termination."
                        data["updated_at"] = datetime.now(timezone.utc).isoformat()
                        self._atomic_write_job(data)
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Failed to recover stale jobs: %s", exc)

    def _count_active_jobs(self) -> int:
        active = 0
        for f in self.root.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("status") == "running":
                    active += 1
            except Exception:
                continue
        return active

    def submit(self, req: Mapping[str, Any], pipeline: Any) -> str:
        """Submit a backtest task to execute in a worker thread."""
        with self._lock:
            active = self._count_active_jobs()
            max_concurrent = getattr(settings, "BACKTEST_MAX_CONCURRENT_JOBS", 1)
            if active >= max_concurrent:
                raise RuntimeError(
                    f"Maximum concurrent backtest jobs ({max_concurrent}) reached."
                )

            job_id = f"job_{uuid.uuid4().hex[:10]}"
            now = datetime.now(timezone.utc).isoformat()
            job_data: dict[str, Any] = {
                "job_id": job_id,
                "status": "queued",
                "run_id": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
            self._atomic_write_job(job_data)

        def _worker() -> None:
            with self._lock:
                job_data["status"] = "running"
                job_data["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._atomic_write_job(job_data)
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
                    persist_db=req.get("persist_db", True),
                )
                with self._lock:
                    job_data["status"] = "succeeded"
                    job_data["run_id"] = res.run_id
                    job_data["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._atomic_write_job(job_data)
            except Exception as exc:
                logger.error("Backtest job %s failed: %s", job_id, exc)
                with self._lock:
                    job_data["status"] = "failed"
                    job_data["error"] = str(exc)
                    job_data["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._atomic_write_job(job_data)

            finally:
                _ACTIVE_THREADS.pop(job_id, None)

        thread = threading.Thread(target=_worker, daemon=True)
        _ACTIVE_THREADS[job_id] = thread
        thread.start()
        return job_id

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Fetch current execution status of a job from disk."""
        path = self._job_file(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Job {job_id} not found.")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read job {job_id}: {exc}") from exc
