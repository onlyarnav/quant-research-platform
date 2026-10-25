"""Retraining pipeline orchestrating drift detection, training, and model promotion.

Determines whether retraining is required based on model age, feature drift,
or manual force flags. Trains candidate models, evaluates them against the
current champion model out-of-sample, and promotes winning models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from src.models.dataset_builder import DatasetBuilder
from src.models.drift_detector import DriftDetector, DriftReport
from src.models.model_evaluator import HIGHER_IS_BETTER, ModelEvaluator
from src.models.model_factory import create_model, default_model_name
from src.models.model_store import ModelStore
from src.models.training_pipeline import TrainingPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrainDecision:
    """Decision indicating whether retraining should proceed and why."""

    should_retrain: bool
    reasons: list[str]


@dataclass(slots=True)
class RetrainingResult:
    """Results of a retraining run."""

    decision: RetrainDecision
    trained: bool
    promoted: bool
    candidate_version: str | None
    champion_version: str | None
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None
    drift_report: DriftReport | None


class RetrainingPipeline:
    """Orchestrates periodic model health checks, retraining, and promotion."""

    def __init__(
        self,
        training_pipeline: TrainingPipeline,
        model_store: ModelStore,
        drift_detector: DriftDetector | None = None,
        dataset_builder: DatasetBuilder | None = None,
        retrain_interval_days: int | None = None,
        promotion_metric: str = "ic",
        promotion_tolerance: float | None = None,
    ) -> None:
        """Initialize RetrainingPipeline.

        Args:
            training_pipeline: TrainingPipeline to train candidate models.
            model_store: ModelStore for querying champions and promoting winners.
            drift_detector: Optional DriftDetector for feature drift checks.
            dataset_builder: Optional DatasetBuilder for loading splits.
            retrain_interval_days: Model age threshold in days triggering retrain.
            promotion_metric: Metric name used for champion-challenger comparison.
            promotion_tolerance: Tolerance margin allowed before rejecting challenger.
        """
        if promotion_metric not in HIGHER_IS_BETTER:
            raise ValueError(
                f"Unknown promotion_metric '{promotion_metric}'. "
                f"Must be one of {sorted(HIGHER_IS_BETTER.keys())}."
            )

        self.training_pipeline = training_pipeline
        self.model_store = model_store
        self.drift_detector = drift_detector or DriftDetector()
        self.dataset_builder = dataset_builder or DatasetBuilder()
        self.retrain_interval_days = (
            retrain_interval_days
            if retrain_interval_days is not None
            else settings.MODEL_RETRAIN_INTERVAL_DAYS
        )
        self.promotion_metric = promotion_metric
        self.promotion_tolerance = (
            promotion_tolerance
            if promotion_tolerance is not None
            else settings.RETRAIN_PROMOTION_TOLERANCE
        )

    def needs_retraining(
        self,
        model_name: str,
        now: datetime | None = None,
        drift_report: DriftReport | None = None,
    ) -> RetrainDecision:
        """Evaluate if a model needs retraining based on age, drift, or absence.

        Args:
            model_name: Registered model family name.
            now: Current datetime reference. Defaults to UTC now.
            drift_report: Optional DriftReport for feature drift evaluation.

        Returns:
            RetrainDecision with boolean flag and specific reason list.
        """
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        reasons: list[str] = []
        if not self.model_store.exists(model_name):
            return RetrainDecision(should_retrain=True, reasons=["no_model"])

        try:
            _, meta = self.model_store.load(model_name, version=None)
            raw_ts = meta.created_at
            created = (
                datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                if isinstance(raw_ts, str)
                else raw_ts
            )
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)

            age_days = (current_time - created).total_seconds() / 86400.0
            if age_days >= self.retrain_interval_days:
                reasons.append(
                    f"model_age_days={age_days:.1f}>={self.retrain_interval_days}"
                )
        except Exception as e:
            logger.warning("Could not inspect existing model %s: %s", model_name, e)
            reasons.append("no_model")

        if drift_report is not None and drift_report.drift_detected:
            reasons.append("feature_drift")

        return RetrainDecision(should_retrain=bool(reasons), reasons=reasons)

    def run(
        self,
        model_type: str,
        horizon: int | None = None,
        model_name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        force: bool = False,
        rolling_window_dates: int | None = None,
        embargo_dates: int = 0,
        now: datetime | None = None,
    ) -> RetrainingResult:
        """Execute retraining evaluation, training, and promotion workflow."""
        hz = horizon if horizon is not None else settings.MODEL_PREDICTION_HORIZON
        name = model_name or default_model_name(model_type, hz)

        if rolling_window_dates is None and embargo_dates == 0:
            splits = self.dataset_builder.build(horizon=hz)
        else:
            splits = self.dataset_builder.build(
                horizon=hz,
                embargo_dates=embargo_dates,
                lookback_dates=rolling_window_dates,
            )

        drift = self.drift_detector.detect(
            splits.X_train, splits.X_test, splits.feature_columns
        )
        decision = self.needs_retraining(name, now=now, drift_report=drift)
        if force:
            if not decision.should_retrain:
                decision = RetrainDecision(should_retrain=True, reasons=["forced"])
            elif "forced" not in decision.reasons:
                decision.reasons.append("forced")

        champion_ver = (
            self.model_store.latest_version(name)
            if self.model_store.exists(name)
            else None
        )
        if not decision.should_retrain:
            logger.info(
                "Retraining not required for %s: reasons=%s", name, decision.reasons
            )
            return RetrainingResult(
                decision=decision,
                trained=False,
                promoted=False,
                candidate_version=None,
                champion_version=champion_ver,
                candidate_metrics={},
                champion_metrics=None,
                drift_report=drift,
            )

        model = create_model(model_type, **(hyperparameters or {}))
        cand_metrics = self.training_pipeline.run(
            model, horizon=hz, splits=splits, model_name=name, promote=False
        )
        cand_ver = (
            self.training_pipeline.last_run_report.model_version
            if self.training_pipeline.last_run_report
            else None
        )

        champ_metrics: dict[str, float] | None = None
        if champion_ver is not None:
            champ_model, champ_meta = self.model_store.load(name, version=champion_ver)
            if set(champ_meta.feature_columns).issubset(set(splits.X_test.columns)):
                y_champ = champ_model.predict(
                    splits.X_test[list(champ_meta.feature_columns)]
                )
                ev = self.training_pipeline.evaluator or ModelEvaluator()
                champ_base = ev.evaluate(splits.y_test, y_champ)
                dates = (
                    splits.meta_test["date"]
                    if splits.meta_test is not None
                    and "date" in splits.meta_test.columns
                    else None
                )
                champ_metrics = {
                    **champ_base,
                    **ev.extended_metrics(splits.y_test, y_champ, dates=dates),
                }

        cand_val = cand_metrics.get(self.promotion_metric, float("nan"))
        if math.isnan(cand_val):
            promoted = False
        elif champ_metrics is None or champion_ver is None:
            promoted = True
        else:
            champ_val = champ_metrics.get(self.promotion_metric, float("nan"))
            degraded = DriftDetector.performance_degraded(
                baseline=champ_val,
                current=cand_val,
                tolerance=self.promotion_tolerance,
                higher_is_better=HIGHER_IS_BETTER[self.promotion_metric],
            )
            promoted = not degraded

        if promoted and cand_ver:
            self.model_store.promote(name, cand_ver)

        logger.info(
            "Retraining finished for %s: promoted=%s, cand=%s, champ=%s",
            name,
            promoted,
            cand_ver,
            champion_ver,
        )
        return RetrainingResult(
            decision=decision,
            trained=True,
            promoted=promoted,
            candidate_version=cand_ver,
            champion_version=champion_ver,
            candidate_metrics=cand_metrics,
            champion_metrics=champ_metrics,
            drift_report=drift,
        )
