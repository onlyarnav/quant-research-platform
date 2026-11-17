"""
Storage manager for persisting and loading backtest run artifacts and results.

Saves run configurations, trade logs, portfolio histories, details, and metadata
to disk using atomic writes and path-traversal validation.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from src.backtesting.backtest_config import BacktestConfig
from src.backtesting.backtesting_engine import Trade
from src.backtesting.simulator_models import SimulationOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert float NaN/inf to None and Timestamps to ISO strings."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _atomic_write_json(file_path: Path, payload: Any) -> None:
    """Write JSON data to a file atomically via a temporary file."""
    sanitized = _sanitize_for_json(payload)
    parent = file_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=parent, delete=False, encoding="utf-8"
    ) as tf:
        json.dump(sanitized, tf, indent=2, default=str)
        temp_name = tf.name
    os.replace(temp_name, file_path)


def _atomic_write_parquet(file_path: Path, df: pd.DataFrame) -> None:
    """Write DataFrame to parquet atomically via a temporary file."""
    parent = file_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=parent, delete=False) as tf:
        temp_name = tf.name
    df.to_parquet(temp_name, engine="pyarrow", index=False)
    os.replace(temp_name, file_path)


@dataclass(slots=True)
class RunRecord:
    """Metadata summary record for a stored backtest simulation."""

    run_id: str
    strategy_name: str
    created_at: str
    config: dict[str, Any]
    n_trades: int
    final_equity: float
    start_date: str | None
    end_date: str | None
    extras: dict[str, Any]


class BacktestStore:
    """Manages disk persistence, retrieval, and discovery of backtest runs."""

    def __init__(self, root: Path | None = None) -> None:
        """
        Initialize BacktestStore.

        Args:
            root: Root storage directory. Defaults to settings.BACKTEST_RESULTS_PATH.
        """
        self.root = (
            Path(root) if root is not None else Path(settings.BACKTEST_RESULTS_PATH)
        )
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_run_id(
        strategy_name: str, config_hash: str, now: datetime | None = None
    ) -> str:
        """
        Generate a unique, sorted run ID string.

        Format: YYYYMMDDTHHMMSSZ_{slug}_{hash}
        """
        dt = now or datetime.now(timezone.utc)
        ts_str = dt.strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", strategy_name)
        return f"{ts_str}_{slug}_{config_hash}"

    def _validate_safe_name(self, name: str) -> None:
        """Validate string matches safe alphanumeric format to prevent traversal."""
        if not SAFE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid path identifier '{name}'. Must match ^[A-Za-z0-9_.-]+$"
            )

    def save_run(
        self,
        run_id: str,
        config: BacktestConfig,
        output: SimulationOutput,
        extras: dict[str, Any] | None = None,
    ) -> Path:
        """
        Persist a full simulation run artifacts to disk.

        Returns:
            Path to the saved run directory.
        """
        self._validate_safe_name(run_id)
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Config
        cfg_dict = config.to_dict()
        _atomic_write_json(run_dir / "config.json", cfg_dict)

        # 2. Parquet files
        _atomic_write_parquet(run_dir / "trades.parquet", output.trades_df)
        _atomic_write_parquet(
            run_dir / "trade_details.parquet", output.trade_details_df
        )
        _atomic_write_parquet(
            run_dir / "portfolio_history.parquet", output.portfolio_history_df
        )
        _atomic_write_parquet(
            run_dir / "daily_details.parquet", output.daily_details_df
        )
        _atomic_write_parquet(
            run_dir / "positions_history.parquet", output.positions_history_df
        )

        # 3. Record
        start_date = None
        end_date = None
        if not output.portfolio_history_df.empty:
            start_date = str(output.portfolio_history_df["date"].iloc[0])
            end_date = str(output.portfolio_history_df["date"].iloc[-1])

        record = RunRecord(
            run_id=run_id,
            strategy_name=config.strategy_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            config=cfg_dict,
            n_trades=len(output.trades),
            final_equity=output.summary.get("final_equity", 0.0),
            start_date=start_date,
            end_date=end_date,
            extras=extras or {},
        )
        _atomic_write_json(run_dir / "record.json", asdict(record))
        logger.info(
            "Saved backtest run %s with %d trades to %s",
            run_id,
            len(output.trades),
            run_dir,
        )
        return run_dir

    def load_run(self, run_id: str) -> tuple[RunRecord, SimulationOutput]:
        """Load a backtest run and reconstruct RunRecord and SimulationOutput."""
        self._validate_safe_name(run_id)
        run_dir = self.root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Backtest run '{run_id}' not found")

        with open(run_dir / "record.json", "r", encoding="utf-8") as f:
            rec_dict = json.load(f)
        record = RunRecord(**rec_dict)

        trades_df = pd.read_parquet(run_dir / "trades.parquet")
        trade_details_df = pd.read_parquet(run_dir / "trade_details.parquet")
        portfolio_history_df = pd.read_parquet(run_dir / "portfolio_history.parquet")
        daily_details_df = pd.read_parquet(run_dir / "daily_details.parquet")
        positions_history_df = pd.read_parquet(run_dir / "positions_history.parquet")

        trades: list[Trade] = []
        trade_fields = {
            "trade_id",
            "symbol",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "position_size",
            "fees",
            "slippage_cost",
            "pnl",
            "return_pct",
        }
        for r in trades_df.to_dict(orient="records"):
            if "position_size" not in r and "quantity" in r:
                r["position_size"] = r["quantity"]
            filtered = {k: v for k, v in r.items() if k in trade_fields}
            trades.append(Trade(**filtered))

        fin_eq = (
            float(portfolio_history_df["portfolio_value"].iloc[-1])
            if not portfolio_history_df.empty
            else record.final_equity
        )
        avg_turn = (
            float(daily_details_df["turnover"].mean() * 252)
            if not daily_details_df.empty and "turnover" in daily_details_df
            else 0.0
        )

        summary = {
            "n_trades": float(len(trades)),
            "total_fees": (
                float(trades_df["fees"].sum()) if not trades_df.empty else 0.0
            ),
            "total_slippage": (
                float(trades_df["slippage_cost"].sum()) if not trades_df.empty else 0.0
            ),
            "avg_turnover_annualised": avg_turn,
            "final_equity": fin_eq,
        }

        output = SimulationOutput(
            trades=trades,
            trades_df=trades_df,
            trade_details_df=trade_details_df,
            portfolio_history_df=portfolio_history_df,
            daily_details_df=daily_details_df,
            positions_history_df=positions_history_df,
            summary=summary,
        )
        return record, output

    def list_runs(self) -> list[RunRecord]:
        """List all stored runs ordered newest first by created_at / run_id."""
        records: list[RunRecord] = []
        if not self.root.exists():
            return []

        for p in self.root.iterdir():
            if p.is_dir() and (p / "record.json").exists():
                try:
                    with open(p / "record.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    records.append(RunRecord(**data))
                except Exception as e:
                    logger.warning("Corrupt record in %s: %s", p, e)

        records.sort(key=lambda r: (r.created_at, r.run_id), reverse=True)
        return records

    def latest_run_id(self) -> str | None:
        """Return run_id of the most recent backtest, or None if none exist."""
        runs = self.list_runs()
        return runs[0].run_id if runs else None

    def delete_run(self, run_id: str) -> None:
        """Delete a backtest run directory and all its files."""
        self._validate_safe_name(run_id)
        run_dir = self.root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Backtest run '{run_id}' not found")
        for item in run_dir.iterdir():
            item.unlink()
        run_dir.rmdir()
        logger.info("Deleted backtest run %s", run_id)

    def save_json(self, run_id: str, name: str, payload: dict[str, Any]) -> Path:
        """Save an arbitrary JSON artifact inside a run directory."""
        self._validate_safe_name(run_id)
        self._validate_safe_name(name)
        run_dir = self.root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Backtest run '{run_id}' not found")
        file_path = run_dir / (name if name.endswith(".json") else f"{name}.json")
        _atomic_write_json(file_path, payload)
        return file_path

    def load_json(self, run_id: str, name: str) -> dict[str, Any]:
        """Load an arbitrary JSON artifact from a run directory."""
        self._validate_safe_name(run_id)
        self._validate_safe_name(name)
        run_dir = self.root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Backtest run '{run_id}' not found")
        file_path = run_dir / (name if name.endswith(".json") else f"{name}.json")
        if not file_path.exists():
            raise FileNotFoundError(
                f"JSON artifact '{name}' not found for run '{run_id}'"
            )
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
