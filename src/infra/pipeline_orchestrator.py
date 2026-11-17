"""
Pipeline orchestrator coordinating daily batch execution workflows.

Implements strict sequential failure propagation: when any step fails, subsequent
steps are marked as SKIPPED, and the final PipelineRunReport is persisted atomically.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Sequence

from src.infra.pipeline_models import PipelineRunReport, StepResult, StepStatus
from src.infra.pipeline_steps import (
    AnalyticsStep,
    BacktestStep,
    FeatureStep,
    IngestionStep,
    PipelineContext,
    PipelineStep,
    PredictionStep,
    SignalStep,
)
from src.infra.run_history import RunHistory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Sequential pipeline runner executing ingestion through analytics."""

    def __init__(
        self,
        steps: Sequence[PipelineStep] | None = None,
        history: RunHistory | None = None,
    ) -> None:
        """Initialize orchestrator with custom steps and run history repository."""
        self.steps: Sequence[PipelineStep] = (
            steps
            if steps is not None
            else [
                IngestionStep(),
                FeatureStep(),
                PredictionStep(),
                SignalStep(),
                BacktestStep(),
                AnalyticsStep(),
            ]
        )
        self.history = history or RunHistory()

    def run(self, context: PipelineContext) -> PipelineRunReport:
        """
        Execute full pipeline sequentially.

        On failure of any step, downstream steps are skipped. Never raises:
        always constructs, logs, and persists PipelineRunReport.
        """
        logger.info("Starting pipeline execution for run_id=%s", context.run_id)
        start_time_iso = datetime.now(UTC).isoformat()

        step_results: list[StepResult] = []
        failed_step_name: str | None = None

        for step in self.steps:
            if failed_step_name is not None:
                step_results.append(
                    StepResult(
                        name=step.name,
                        status=StepStatus.SKIPPED,
                        started_at=datetime.now(UTC).isoformat(),
                        finished_at=datetime.now(UTC).isoformat(),
                        duration_seconds=0.0,
                        message=f"Skipped: upstream step '{failed_step_name}' failed.",
                    )
                )
                continue

            step_start = time.perf_counter()
            step_start_iso = datetime.now(UTC).isoformat()

            try:
                logger.info("Executing pipeline step: %s", step.name)
                details = step.run(context)
                duration = time.perf_counter() - step_start
                step_results.append(
                    StepResult(
                        name=step.name,
                        status=StepStatus.SUCCESS,
                        started_at=step_start_iso,
                        finished_at=datetime.now(UTC).isoformat(),
                        duration_seconds=duration,
                        message="Completed successfully.",
                        details=details or {},
                    )
                )
            except Exception as exc:
                duration = time.perf_counter() - step_start
                logger.error("Pipeline step %s failed: %s", step.name, exc)
                failed_step_name = step.name
                step_results.append(
                    StepResult(
                        name=step.name,
                        status=StepStatus.FAILED,
                        started_at=step_start_iso,
                        finished_at=datetime.now(UTC).isoformat(),
                        duration_seconds=duration,
                        message=str(exc),
                    )
                )

        end_time_iso = datetime.now(UTC).isoformat()
        report = PipelineRunReport(
            run_id=context.run_id,
            started_at=start_time_iso,
            finished_at=end_time_iso,
            steps=step_results,
            config={
                "asset_class": context.asset_class,
                "timeframe": context.timeframe,
                "symbols": context.symbols,
                "dry_run": context.dry_run,
                "models": context.models,
            },
        )

        try:
            self.history.save(report)
        except Exception as exc:
            logger.error("Failed to persist pipeline run report: %s", exc)

        logger.info(
            "Pipeline run %s finished in %.2fs. All succeeded: %s",
            context.run_id,
            report.total_duration_seconds,
            report.all_succeeded,
        )
        return report

    def run_retraining(
        self,
        model_type: str = "ridge",
        model_name: str | None = None,
        horizon: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Trigger model retraining check and execution.

        Re-raises exceptions on failure.
        """
        from src.models.dataset_builder import DatasetBuilder
        from src.models.drift_detector import DriftDetector
        from src.models.model_evaluator import ModelEvaluator
        from src.models.model_store import ModelStore
        from src.models.retraining_pipeline import RetrainingPipeline
        from src.models.training_pipeline import TrainingPipeline

        dataset_builder = DatasetBuilder()
        model_store = ModelStore()
        evaluator = ModelEvaluator()
        training_pipeline = TrainingPipeline(
            dataset_builder=dataset_builder,
            evaluator=evaluator,
            model_store=model_store,
        )
        drift_detector = DriftDetector()

        pipeline = RetrainingPipeline(
            training_pipeline=training_pipeline,
            model_store=model_store,
            drift_detector=drift_detector,
            dataset_builder=dataset_builder,
        )
        result = pipeline.run(
            model_type=model_type,
            model_name=model_name,
            horizon=horizon,
            force=force,
        )
        return {
            "retrained": result.trained,
            "promoted": result.promoted,
            "metrics": result.candidate_metrics,
            "decision": (
                result.decision.name
                if hasattr(result.decision, "name")
                else str(result.decision)
            ),
        }
