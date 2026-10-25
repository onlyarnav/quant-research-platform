"""CLI script for model training and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.models.dataset_builder import DatasetBuilder
from src.models.model_evaluator import ModelEvaluator
from src.models.model_factory import available_model_types, create_model
from src.models.model_store import ModelStore
from src.models.training_pipeline import TrainingPipeline
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


MODEL_ALIASES = {
    "ridge": "linear",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate predictive models."
    )
    parser.add_argument(
        "--model-type",
        choices=available_model_types() + list(MODEL_ALIASES.keys()),
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
        "--embargo-dates",
        type=int,
        default=None,
        help="Number of embargo dates between splits. Defaults to horizon.",
    )
    parser.add_argument(
        "--lookback-dates",
        type=int,
        default=None,
        help="Rolling lookback window in dates. If omitted, uses expanding window.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional custom identifier name for the model.",
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default=None,
        help="JSON string specifying model hyperparameters.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable persisting model artifacts to disk.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute model training workflow."""
    args = parse_args(argv)
    try:
        hyperparams: dict[str, Any] = {}
        if args.hyperparameters:
            hyperparams = json.loads(args.hyperparameters)

        embargo = args.embargo_dates if args.embargo_dates is not None else args.horizon

        model_store = None if args.no_save else ModelStore()
        pipeline = TrainingPipeline(
            dataset_builder=DatasetBuilder(),
            evaluator=ModelEvaluator(),
            model_store=model_store,
        )

        resolved_type = MODEL_ALIASES.get(args.model_type, args.model_type)
        model = create_model(resolved_type, **hyperparams)
        metrics = pipeline.run(
            model=model,
            horizon=args.horizon,
            embargo_dates=embargo,
            lookback_dates=args.lookback_dates,
            model_name=args.model_name,
        )

        print("=" * 60)
        print(f"TRAINING COMPLETE: {model.model_name} (Horizon: {args.horizon}d)")
        print(f"Test MSE:                   {metrics.get('mse', float('nan')):.6f}")
        print(f"Test RMSE:                  {metrics.get('rmse', float('nan')):.6f}")
        print(f"Test MAE:                   {metrics.get('mae', float('nan')):.6f}")
        print(f"Test R2:                    {metrics.get('r2', float('nan')):.6f}")
        da_val = metrics.get("directional_accuracy", float("nan"))
        print(f"Test Directional Accuracy:  {da_val:.4f}")
        print(f"Test IC:                    {metrics.get('ic', float('nan')):.4f}")
        print("=" * 60)
        return 0
    except Exception:
        logger.exception("Training pipeline failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
