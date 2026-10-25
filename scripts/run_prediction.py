"""CLI script for generating and storing model predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.models.prediction_service import PredictionService
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for running predictions."""
    parser = argparse.ArgumentParser(description="Generate asset return predictions.")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Registered model name in the ModelStore.",
    )
    parser.add_argument(
        "--asset-class",
        type=str,
        default="equity",
        help="Asset class (default: 'equity').",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="daily",
        help="Data timeframe (default: 'daily').",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "oos", "test", "latest"],
        default="test",
        help="Temporal scope of features to predict on (default: 'test').",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Specific model version timestamp. Defaults to LATEST.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction return horizon in days (default: 1).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute prediction generation workflow."""
    args = parse_args(argv)
    try:
        service = PredictionService()
        model_name = args.model_name
        model_store_root = Path(settings.MODEL_ARTIFACT_PATH)
        aliases = {
            "ridge": "LinearModel",
            "linear": "LinearModel",
            "ols": "OLSModel",
            "lasso": "LassoModel",
            "xgboost": "XGBoostModel",
            "lightgbm": "LightGBMModel",
            "random_forest": "RandomForestModel",
            "gradient_boosting": "GradientBoostingModel",
        }
        if not (model_store_root / model_name).exists():
            resolved = aliases.get(model_name.lower())
            if resolved and (model_store_root / resolved).exists():
                model_name = resolved

        version = args.model_version
        if version and version.lower() == "latest":
            version = None

        predictions_df = service.run(
            model_name=model_name,
            model_version=version,
            scope=args.scope,
            asset_class=args.asset_class,
            timeframe=args.timeframe,
        )

        n_rows = len(predictions_df)
        n_symbols = predictions_df["symbol"].nunique() if n_rows > 0 else 0
        min_date = predictions_df["date"].min() if n_rows > 0 else "N/A"
        max_date = predictions_df["date"].max() if n_rows > 0 else "N/A"

        print("=" * 60)
        print(f"PREDICTION SUMMARY: {args.model_name} (Scope: {args.scope})")
        print(f"Total Rows:     {n_rows}")
        print(f"Total Symbols:  {n_symbols}")
        print(f"Date Range:     {min_date} to {max_date}")
        print("=" * 60)
        return 0
    except Exception:
        logger.exception("Prediction generation failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
