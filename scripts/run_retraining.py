"""CLI script for model health checking, periodic retraining, and promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.models.dataset_builder import DatasetBuilder
from src.models.drift_detector import DriftDetector
from src.models.model_evaluator import ModelEvaluator
from src.models.model_factory import available_model_types
from src.models.model_store import ModelStore
from src.models.retraining_pipeline import RetrainingPipeline
from src.models.training_pipeline import TrainingPipeline
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for retraining."""
    parser = argparse.ArgumentParser(
        description="Check model drift and trigger retraining."
    )
    parser.add_argument(
        "--model-type",
        choices=available_model_types(),
        default="xgboost",
        help="Model architecture key.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 10],
        default=settings.MODEL_PREDICTION_HORIZON,
        help="Prediction return horizon in days.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional specific model family name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if drift and age criteria are not met.",
    )
    parser.add_argument(
        "--rolling-window-dates",
        type=int,
        default=None,
        help="Rolling lookback window in dates.",
    )
    parser.add_argument(
        "--embargo-dates",
        type=int,
        default=None,
        help="Embargo dates between splits. Defaults to horizon.",
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default=None,
        help="JSON string specifying candidate hyperparameters.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute retraining workflow."""
    args = parse_args(argv)
    try:
        hyperparams: dict[str, Any] = {}
        if args.hyperparameters:
            hyperparams = json.loads(args.hyperparameters)

        embargo = args.embargo_dates if args.embargo_dates is not None else args.horizon

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
            model_type=args.model_type,
            horizon=args.horizon,
            model_name=args.model_name,
            hyperparameters=hyperparams,
            force=args.force,
            rolling_window_dates=args.rolling_window_dates,
            embargo_dates=embargo,
        )

        print("=" * 60)
        print("RETRAINING SUMMARY")
        print(f"Should Retrain:    {result.decision.should_retrain}")
        print(f"Decision Reasons:  {result.decision.reasons}")
        print(f"Candidate Trained: {result.trained}")
        print(f"Candidate Promoted:{result.promoted}")
        print(f"Candidate Version: {result.candidate_version}")
        print(f"Champion Version:  {result.champion_version}")

        if result.drift_report:
            d_det = result.drift_report.drift_detected
            m_psi = result.drift_report.max_psi
            print(f"Drift Detected:    {d_det} (Max PSI: {m_psi:.4f})")
            if result.drift_report.drifted_features:
                print(f"Drifted Features:  {result.drift_report.drifted_features}")

        print("=" * 60)
        return 0
    except Exception:
        logger.exception("Retraining pipeline failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
