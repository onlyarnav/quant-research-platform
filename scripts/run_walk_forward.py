"""CLI script for walk-forward model validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.models.dataset_builder import DatasetBuilder
from src.models.model_factory import available_model_types
from src.models.walk_forward import WalkForwardSplitter, WalkForwardValidator
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for walk-forward validation."""
    parser = argparse.ArgumentParser(description="Run walk-forward model validation.")
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
        "--n-splits",
        type=int,
        default=settings.WALK_FORWARD_SPLITS,
        help="Number of chronological folds.",
    )
    parser.add_argument(
        "--gap-dates",
        type=int,
        default=None,
        help="Gap dates between train and test. Defaults to horizon.",
    )
    parser.add_argument(
        "--min-train-dates",
        type=int,
        default=60,
        help="Minimum required training dates per fold (default: 60).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output fold metrics CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute walk-forward validation workflow."""
    args = parse_args(argv)
    try:
        gap = args.gap_dates if args.gap_dates is not None else args.horizon
        builder = DatasetBuilder()
        df = builder.load_all_features()

        splitter = WalkForwardSplitter(
            n_splits=args.n_splits,
            gap_dates=gap,
            min_train_dates=args.min_train_dates,
        )
        validator = WalkForwardValidator(
            splitter=splitter,
            dataset_builder=builder,
        )

        result = validator.run(
            df=df,
            model_type=args.model_type,
            horizon=args.horizon,
        )

        print("=" * 60)
        print(f"WALK-FORWARD VALIDATION SUMMARY: {args.model_type} ({args.horizon}d)")
        print(result.fold_metrics.to_string(index=False))
        print("-" * 60)
        print("SUMMARY METRICS:")
        for k, v in result.summary.items():
            print(f"  {k:20s}: {v:.6f}")
        print("=" * 60)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result.fold_metrics.to_csv(out_path, index=False)
            print(f"Saved fold metrics to {out_path}")

        return 0
    except Exception:
        logger.exception("Walk-forward validation failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
