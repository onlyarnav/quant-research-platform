"""
Pluggable pipeline step adapters integrating core platform subsystems.

Provides PipelineContext, PipelineStep Protocol, and concrete steps:
IngestionStep, FeatureStep, PredictionStep, SignalStep, BacktestStep, AnalyticsStep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from config.settings import settings
from src.backtesting.backtest_config import BacktestConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineContext:
    """Execution context and configuration shared across pipeline steps."""

    run_id: str
    asset_class: str = "equity"
    timeframe: str = "daily"
    symbols: list[str] | None = None
    dry_run: bool = False
    models: list[str] = field(default_factory=list)
    backtest_config: BacktestConfig | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class PipelineStep(Protocol):
    """Protocol defining the interface for an orchestrator pipeline step."""

    name: str

    def run(self, context: PipelineContext) -> dict[str, Any]:
        """Execute the step, returning details dictionary or raising an exception."""
        ...


class IngestionStep:
    """Incremental market data ingestion adapter."""

    name = "ingestion"

    def run(self, context: PipelineContext) -> dict[str, Any]:
        """Execute incremental data ingestion."""
        if context.dry_run:
            return {"dry_run": True, "mode": "incremental"}

        from src.data_pipeline.ingestion import IngestionPipeline
        from src.data_pipeline.universe import get_nse_equity_universe

        universe = get_nse_equity_universe(symbols=context.symbols)
        if not universe:
            raise ValueError(f"No assets discovered for symbols: {context.symbols}")

        pipeline = IngestionPipeline()
        report = pipeline.run(universe)
        return {
            "succeeded": len(report.succeeded),
            "failed": len(report.failed),
            "total": report.total,
        }


class FeatureStep:
    """Feature generation adapter executing FeaturePipeline."""

    name = "feature_engineering"

    def run(self, context: PipelineContext) -> dict[str, Any]:
        """Generate features across configured symbols."""
        if context.dry_run:
            return {"dry_run": True}

        from src.feature_engineering.feature_pipeline import FeaturePipeline

        pipe = FeaturePipeline()
        report = pipe.run()

        if report.failed > 0 and report.succeeded == 0:
            raise RuntimeError(
                f"Feature engineering failed completely ({report.failed} files failed)."
            )

        return {
            "succeeded": report.succeeded,
            "failed": report.failed,
            "total_files": report.total,
        }


class PredictionStep:
    """Machine learning inference adapter generating out-of-sample predictions."""

    name = "prediction"

    def run(self, context: PipelineContext) -> dict[str, Any]:
        """Run ML predictions with latest fitted models."""
        if context.dry_run:
            return {"dry_run": True}

        from src.models.model_store import ModelStore
        from src.models.prediction_service import PredictionService

        store = ModelStore()
        target_models = context.models or [
            getattr(settings, "MODEL_NAME", "linear_regression")
        ]

        for m_name in target_models:
            if not store.exists(name=m_name):
                raise RuntimeError(
                    f"No trained model found for '{m_name}' in model registry. Do not train implicitly."
                )

        service = PredictionService(model_store=store)
        results: dict[str, int] = {}
        for m_name in target_models:
            df = service.run(
                model_name=m_name,
                asset_class=context.asset_class,
                timeframe=context.timeframe,
                scope="latest",
            )
            results[m_name] = len(df)

        return {"predictions_generated": results}


class SignalStep:
    """Trading signal generation adapter executing SignalPipeline."""

    name = "signal_generation"

    def run(self, context: PipelineContext) -> dict[str, Any]:
        """Generate trading signals from model predictions."""
        if context.dry_run:
            return {"dry_run": True}

        from src.signals.rank_signal import RankSignalGenerator
        from src.signals.signal_pipeline import SignalPipeline
        from src.signals.threshold_signal import ThresholdSignalGenerator
        from src.signals.topn_signal import (
            TopNLongShortSignalGenerator as TopNSignalGenerator,
        )

        gen_name = context.extras.get("signal_generator", "threshold")
        generator: Any
        if gen_name == "rank":
            generator = RankSignalGenerator()
        elif gen_name == "topn":
            generator = TopNSignalGenerator(top_n=context.extras.get("top_n", 5))
        else:
            generator = ThresholdSignalGenerator()

        pipeline = SignalPipeline()
        syms = context.symbols or pipeline.discover_symbols(
            asset_class=context.asset_class, timeframe=context.timeframe
        )
        if not syms:
            raise ValueError(
                f"No prediction symbols found for signal generation in {context.asset_class}."
            )

        df = pipeline.run(
            generator=generator,
            symbols=syms,
            asset_class=context.asset_class,
            timeframe=context.timeframe,
        )
        return {"symbols_processed": len(syms), "signals_count": len(df)}


class BacktestStep:
    """Backtesting execution adapter running BacktestPipeline."""

    name = "backtesting"

    def run(self, context: PipelineContext) -> dict[str, Any]:
        """Execute simulation run and persist outputs."""
        if context.dry_run:
            return {"dry_run": True}

        from src.analytics.backtest_pipeline import BacktestPipeline

        pipe = BacktestPipeline()
        cfg = context.backtest_config or BacktestConfig(strategy_name="daily")
        res = pipe.run(
            config=cfg,
            symbols=context.symbols,
            asset_class=context.asset_class,
            timeframe=context.timeframe,
            persist=True,
            persist_db=False,
        )
        context.extras["run_id"] = res.run_id
        return {"run_id": res.run_id, "strategy_name": cfg.strategy_name}


class AnalyticsStep:
    """Analytics computation and database persistence adapter."""

    name = "analytics"

    def run(self, context: PipelineContext) -> dict[str, Any]:
        """Recompute performance analytics and persist to database if available."""
        if context.dry_run:
            return {"dry_run": True}

        run_id = context.extras.get("run_id")
        if not run_id:
            return {"status": "no_run_id_to_analyze"}

        from src.analytics.backtest_pipeline import BacktestPipeline
        from src.analytics.metrics_persistence import persist_metrics
        from src.database.sessions import get_session

        pipe = BacktestPipeline()
        report = pipe.rebuild_report(run_id)
        details: dict[str, Any] = {
            "run_id": run_id,
            "metrics_count": len(report.metrics),
        }

        try:
            with get_session() as session:
                strat_name = (
                    context.backtest_config.strategy_name
                    if context.backtest_config
                    else "daily"
                )
                persist_metrics(session, strat_name, report.metrics)
                details["db"] = "persisted"
        except Exception as exc:
            details["db"] = "skipped"
            logger.warning("Could not persist metrics to DB in AnalyticsStep: %s", exc)

        return details
